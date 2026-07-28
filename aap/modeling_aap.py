#   Copyright 2023 Haotian Liu
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import itertools
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchoring import asymmetric_anchor_objective
from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .utils import IMAGE_TOKEN_INDEX


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class AlignmentProjector(nn.Module):
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        return self.projector(x)


class AAPMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(AAPMetaModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained")

    def initialize_aap_modules(self, config):
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        cls_head = (
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(in_dim // 2, 3),
        )
        self.cls_head = nn.ModuleList([nn.Sequential(*cls_head)])
        self.class_prompt_projector = nn.Linear(3, out_dim)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True,
        )
        self.text_hidden_fcs.train()
        self.cls_head.train()
        self.class_prompt_projector.train()
        self.attention_layer.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        for param in self.cls_head.parameters():
            param.requires_grad = True
        for param in self.class_prompt_projector.parameters():
            param.requires_grad = True
        for param in self.attention_layer.parameters():
            param.requires_grad = True


class AAPModel(AAPMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(AAPModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_layer = getattr(
            self.config, "mm_vision_select_layer", -2
        )
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.config.vision_hidden_size = 256
        self.config.fc_hidden_size = 1408
        self.config.llm_input_size = 1024


class AAPForCausalLM(LlavaLlamaForCausalLM):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Using normal distribution instead of trunc_normal_ for simplicity
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m,
            (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d),
        ):
            if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(self, config, **kwargs):
        config.train_mask_decoder = kwargs.get(
            "train_mask_decoder", getattr(config, "train_mask_decoder", True)
        )
        config.out_dim = kwargs.get("out_dim", getattr(config, "out_dim", 256))
        config.target_layers = kwargs.get(
            "target_layers", getattr(config, "target_layers", [16])
        )
        if isinstance(config.target_layers, int):
            config.target_layers = [config.target_layers]
        config.anchor_loss_weight = kwargs.get(
            "anchor_loss_weight", getattr(config, "anchor_loss_weight", 3.0)
        )
        config.projector_dim = kwargs.get(
            "projector_dim", getattr(config, "projector_dim", 2048)
        )
        config.z_dim = kwargs.get("z_dim", getattr(config, "z_dim", 1024))
        config.anchor_loss = kwargs.get(
            "anchor_loss", getattr(config, "anchor_loss", True)
        )

        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_tower = kwargs.get(
            "vision_tower",
            getattr(config, "vision_tower", "openai/clip-vit-large-patch14"),
        )

        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        self.cls_loss_weight = kwargs.pop("cls_loss_weight", 2.0)
        self.mask_loss_weight = kwargs.pop("mask_loss_weight", 1.0)

        self.cls_token_idx = kwargs.pop("cls_token_idx")
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.model = AAPModel(config, **kwargs)
        self.model.initialize_aap_modules(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # P_psi in Eq. (6): project a scalar error map into SAM prompt space.
        self.discrepancy_projector = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
        )
        self.discrepancy_projector.train()
        for param in self.discrepancy_projector.parameters():
            param.requires_grad = True

        self.discrepancy_projector.apply(self._init_weights)
        self.discrepancy_scale = kwargs.pop("discrepancy_scale", 3.0)

        self.anchor_loss = config.anchor_loss
        self.anchor_loss_weight = config.anchor_loss_weight

        if self.anchor_loss:
            self.target_layers = config.target_layers
            self.projector_dim = config.projector_dim
            self.z_dim = config.z_dim
            self.alignment_projector = nn.ModuleList([
                AlignmentProjector(config.hidden_size, self.projector_dim, self.z_dim)
                for _ in self.target_layers
            ])


    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        cls_labels: torch.LongTensor,
        resize_list: List[tuple],
        original_size_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]

        (
            new_input_ids,
            position_ids,
            attention_masks,
            _,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_masks,
            past_key_values=None,
            labels=None,
            images=images_clip,
        )

        if batch_size != cls_labels.shape[0]:
            raise ValueError("Image and class-label batch sizes must match.")
        cls_token_mask = new_input_ids == self.cls_token_idx
        seg_token_mask = new_input_ids == self.seg_token_idx

        output = super().forward(
            input_ids=new_input_ids,
            attention_mask=attention_masks,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states

        assert len(self.model.cls_head) == 1
        last_hidden_state_cls = self.model.cls_head[0](output_hidden_states[-1])
        cls_result = last_hidden_state_cls[cls_token_mask]

        logits = cls_result
        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(logits, cls_labels)

        anchor_loss = cls_loss.new_zeros(())
        batch_discrepancy_maps = [None] * batch_size

        if self.anchor_loss:
            image_token_mask = new_input_ids == IMAGE_TOKEN_INDEX
            expected_patches = self.get_vision_tower().num_patches
            patch_counts = image_token_mask.sum(dim=-1)
            if not torch.all(patch_counts == expected_patches):
                raise ValueError(
                    f"Expected {expected_patches} image tokens per sample, "
                    f"received {patch_counts.tolist()}."
                )

            middle_states = torch.stack(
                [output.hidden_states[layer] for layer in self.target_layers],
                dim=1,
            )
            projected_states = torch.stack(
                [
                    projector(middle_states[:, index])
                    for index, projector in enumerate(self.alignment_projector)
                ],
                dim=1,
            )
            projected_patches = torch.stack(
                [
                    projected_states[index, :, image_token_mask[index], :]
                    for index in range(batch_size)
                ],
                dim=0,
            )

            truth_encoder = self.get_vision_tower()
            truth_encoder.eval()
            with torch.no_grad():
                truth_anchor = truth_encoder(images_clip)

            anchor_loss, batch_discrepancy_maps = asymmetric_anchor_objective(
                projected_patches,
                truth_anchor,
                cls_labels,
            )

        mask_bce_loss = mask_dice_loss = mask_loss = cls_loss.new_zeros(())
        num_masks = 0

        if (cls_labels == 2).any():
            last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states[-1])
            cls_projected = self.model.class_prompt_projector(cls_result)
            tampered_indices = torch.nonzero(cls_labels == 2, as_tuple=False).flatten()
            enhanced_prompts = []
            for batch_index in tampered_indices.tolist():
                seg_embeddings = last_hidden_state[batch_index][
                    seg_token_mask[batch_index]
                ]
                if seg_embeddings.numel() == 0:
                    raise ValueError("Tampered samples must contain a [SEG] token.")
                query = cls_projected[batch_index].unsqueeze(0)
                attention, _ = self.model.attention_layer(
                    query=query,
                    key=seg_embeddings,
                    value=seg_embeddings,
                )
                enhanced_prompts.append(
                    (batch_index, seg_embeddings + attention)
                )

            pred_masks = []
            gt_masks = []
            for batch_index, prompt_embedding in enhanced_prompts:
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=prompt_embedding.unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(prompt_embedding.dtype)

                discrepancy_prompt = torch.zeros_like(dense_embeddings)
                patch_map = batch_discrepancy_maps[batch_index]
                if patch_map is not None:
                    num_patches_side = int(patch_map.shape[0] ** 0.5)
                    if num_patches_side ** 2 != patch_map.shape[0]:
                        raise ValueError("The patch error map must form a square grid.")
                    error_map = patch_map.view(
                        1, 1, num_patches_side, num_patches_side
                    )
                    error_map = F.interpolate(
                        error_map,
                        size=image_embeddings.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    discrepancy_prompt = self.discrepancy_projector(
                        error_map.to(
                            dense_embeddings.device,
                            dtype=dense_embeddings.dtype,
                        )
                    ) * self.discrepancy_scale

                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[batch_index].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings + discrepancy_prompt,
                    multimask_output=False,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[batch_index],
                    original_size=original_size_list[batch_index],
                )
                pred_masks.append(pred_mask[:, 0])
                gt_masks.append(masks_list[batch_index])

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                    "logits": logits,
                    "discrepancy_maps": batch_discrepancy_maps,
                    "cls_hidden_state": cls_result,
                }
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx].float()
                pred_mask = pred_masks[batch_idx].float()
                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]
            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss
        if not inference and seg_token_mask.sum() == 0:
            dummy = cls_loss.new_zeros(())
            for p in itertools.chain(
                self.model.visual_model.mask_decoder.parameters(),
                self.model.text_hidden_fcs.parameters(),
                self.model.class_prompt_projector.parameters(),
                self.model.attention_layer.parameters(),
                self.discrepancy_projector.parameters(),
            ):
                dummy = dummy + p.sum() * 0.0
            mask_loss = mask_loss + dummy

        loss = (
            self.cls_loss_weight * cls_loss
            + self.mask_loss_weight * mask_loss
            + self.anchor_loss_weight * anchor_loss
        )

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "cls_loss": cls_loss,
            "anchor_loss": anchor_loss,
            "logits": logits,
            "discrepancy_maps": batch_discrepancy_maps,
            "cls_hidden_state": cls_result,
        }
