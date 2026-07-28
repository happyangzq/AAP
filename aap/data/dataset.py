"""Dataset and collation for three-class image-forgery detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from aap.llava import conversation as conversation_lib
from aap.llava.mm_utils import tokenizer_image_token
from aap.segment_anything.utils.transforms import ResizeLongestSide
from aap.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collate_fn(batch, *, tokenizer, use_mm_start_end=True):
    """Create the multimodal tensors consumed by :class:`AAPForCausalLM`."""
    conversations = [sample["conversation"] for sample in batch]
    if use_mm_start_end:
        image_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conversations = [
            conversation.replace(DEFAULT_IMAGE_TOKEN, image_token)
            for conversation in conversations
        ]

    token_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        token_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    truncate_length = tokenizer.model_max_length - 255
    input_ids = input_ids[:, :truncate_length]
    attention_masks = attention_masks[:, :truncate_length]

    return {
        "images": torch.stack([sample["image"] for sample in batch]),
        "images_clip": torch.stack([sample["image_clip"] for sample in batch]),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "masks_list": [sample["mask"] for sample in batch],
        "cls_labels": torch.tensor(
            [sample["class_label"] for sample in batch],
            dtype=torch.long,
        ),
        "resize_list": [sample["input_size"] for sample in batch],
        "original_size_list": [sample["original_size"] for sample in batch],
        "inference": False,
    }


class ForgeryDataset(torch.utils.data.Dataset):
    """Read real, fully synthetic, and tampered images from one split."""

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_size = 1024

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        split="train",
        precision="fp32",
        image_size=1024,
    ):
        del tokenizer, precision
        if image_size != self.image_size:
            raise ValueError("AAP uses SAM's fixed 1024-pixel input size.")
        self.split_dir = Path(base_image_dir) / split
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        class_directories = ("real", "full_synthetic", "tampered")
        for directory in class_directories:
            path = self.split_dir / directory
            if not path.is_dir():
                raise ValueError(f"Required directory does not exist: {path}")

        image_groups = [
            self._list_images(self.split_dir / directory)
            for directory in class_directories
        ]
        self.images = [path for group in image_groups for path in group]
        self.cls_labels = [
            label
            for label, group in enumerate(image_groups)
            for _ in group
        ]

        mask_dir = self.split_dir / "masks"
        if image_groups[2] and not mask_dir.is_dir():
            raise ValueError(f"Required directory does not exist: {mask_dir}")
        for image_path in image_groups[2]:
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise ValueError(f"Mask not found for tampered image: {image_path}")

        counts = [len(group) for group in image_groups]
        print(
            f"{split}: {counts[0]} real, {counts[1]} full synthetic, "
            f"{counts[2]} tampered"
        )

    @staticmethod
    def _list_images(directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.images)

    def _preprocess_sam_image(self, image) -> tuple[torch.Tensor, tuple[int, int]]:
        resized = self.transform.apply_image(image)
        input_size = resized.shape[:2]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        pad_height = self.image_size - tensor.shape[-2]
        pad_width = self.image_size - tensor.shape[-1]
        if pad_height < 0 or pad_width < 0:
            raise ValueError("SAM input exceeds the configured 1024-pixel canvas.")
        return F.pad(tensor, (0, pad_width, 0, pad_height)), input_size

    @staticmethod
    def _response(class_label: int) -> str:
        if class_label == 0:
            return "[CLS] The image is real"
        if class_label == 1:
            return "[CLS] The image is fully synthetic"
        return "[CLS] The image is tampered [SEG]"

    def __getitem__(self, index):
        image_path = self.images[index]
        class_label = self.cls_labels[index]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        image_clip = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"][0]
        image_sam, input_size = self._preprocess_sam_image(image)

        mask = torch.zeros((1, *original_size), dtype=torch.float32)
        if class_label == 2:
            mask_path = self.split_dir / "masks" / f"{image_path.stem}_mask.png"
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise ValueError(f"Unable to read mask: {mask_path}")
            if mask_image.shape != original_size:
                mask_image = cv2.resize(
                    mask_image,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = torch.from_numpy(mask_image > 127).float().unsqueeze(0)

        conversation = conversation_lib.default_conversation.copy()
        conversation.append_message(
            conversation.roles[0],
            (
                f"{DEFAULT_IMAGE_TOKEN}\nClassify this image as real, "
                "fully synthetic, or tampered. If it is tampered, segment "
                "the manipulated region."
            ),
        )
        conversation.append_message(
            conversation.roles[1],
            self._response(class_label),
        )

        return {
            "image": image_sam,
            "image_clip": image_clip,
            "conversation": conversation.get_prompt(),
            "mask": mask,
            "class_label": class_label,
            "input_size": input_size,
            "original_size": original_size,
        }
