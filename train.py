"""Train and evaluate the Asymmetric Anchoring Paradigm (AAP)."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from aap.data.batch_sampler import ClassBalancedBatchSampler
from aap.data.dataset import ForgeryDataset, collate_fn
from aap.llava import conversation as conversation_lib
from aap.modeling_aap import AAPForCausalLM
from aap.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    dict_to_cuda,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate AAP")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        default=0,
        type=int,
    )
    parser.add_argument("--version", default="xinlai/LISA-7B-v1")
    parser.add_argument(
        "--vision-tower",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="./runs/aap")
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--val-splits", nargs="+", default=["test"])

    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--image-size", default=1024, type=int)
    parser.add_argument("--model-max-length", default=512, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--val-batch-size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")

    parser.add_argument("--target-layer", default=16, type=int)
    parser.add_argument("--projector-dim", default=2048, type=int)
    parser.add_argument("--anchor-dim", default=1024, type=int)
    parser.add_argument("--anchor-loss-weight", default=3.0, type=float)
    parser.add_argument("--classification-loss-weight", default=1.0, type=float)
    parser.add_argument("--segmentation-loss-weight", default=1.0, type=float)
    parser.add_argument("--dice-loss-weight", default=1.0, type=float)
    parser.add_argument("--bce-loss-weight", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def torch_dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]


def build_tokenizer(args: argparse.Namespace):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["[CLS]", "[SEG]"])
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
        special_tokens=True,
    )
    return tokenizer


def find_lora_targets(model, requested: str) -> list[str]:
    excluded = {
        "visual_model",
        "vision_tower",
        "mm_projector",
        "text_hidden_fcs",
        "cls_head",
        "class_prompt_projector",
        "attention_layer",
        "alignment_projector",
        "discrepancy_projector",
    }
    requested_names = requested.split(",")
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(part in name for part in excluded):
            continue
        if any(part in name for part in requested_names):
            targets.add(name)
    return sorted(targets)


def build_model(args: argparse.Namespace, tokenizer):
    dtype = torch_dtype(args.precision)
    model = AAPForCausalLM.from_pretrained(
        args.version,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        cls_loss_weight=args.classification_loss_weight,
        mask_loss_weight=args.segmentation_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        cls_token_idx=tokenizer("[CLS]", add_special_tokens=False).input_ids[0],
        seg_token_idx=tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
        vision_pretrained=args.sam_checkpoint,
        vision_tower=args.vision_tower,
        use_mm_start_end=True,
        anchor_loss=True,
        anchor_loss_weight=args.anchor_loss_weight,
        target_layers=[args.target_layer],
        projector_dim=args.projector_dim,
        z_dim=args.anchor_dim,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=dtype, device=args.local_rank)
    vision_tower.requires_grad_(False)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if args.lora_r > 0:
        targets = find_lora_targets(model, args.lora_target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=targets,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    model.resize_token_embeddings(len(tokenizer))
    for name, parameter in model.named_parameters():
        if "lm_head" in name:
            parameter.requires_grad = False
        if any(
            component in name
            for component in (
                "embed_tokens",
                "mm_projector",
                "mask_decoder",
                "text_hidden_fcs",
                "cls_head",
                "class_prompt_projector",
                "attention_layer",
                "alignment_projector",
                "discrepancy_projector",
            )
        ):
            parameter.requires_grad = True
    return model


def make_loader(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    batch_size: int,
):
    dataset = ForgeryDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split=split,
        precision=args.precision,
        image_size=args.image_size,
    )
    sampler = ClassBalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_mm_start_end=True,
        ),
    )
    if len(loader) == 0:
        raise ValueError(f"No complete batches available for split '{split}'.")
    return loader


def prepare_batch(batch: dict, precision: str) -> dict:
    batch = dict_to_cuda(batch)
    dtype = torch_dtype(precision)
    batch["images"] = batch["images"].to(dtype=dtype)
    batch["images_clip"] = batch["images_clip"].to(dtype=dtype)
    return batch


def train_epoch(engine, loader, args, writer, epoch: int) -> None:
    engine.train()
    totals = {"loss": 0.0, "cls_loss": 0.0, "mask_loss": 0.0, "anchor_loss": 0.0}
    for step, batch in enumerate(loader):
        batch = prepare_batch(batch, args.precision)
        output = engine(**batch)
        engine.backward(output["loss"])
        engine.step()
        for key in totals:
            totals[key] += float(output[key].detach())

    if dist.get_rank() == 0:
        for key, total in totals.items():
            writer.add_scalar(f"train/{key}", total / len(loader), epoch)


@torch.no_grad()
def evaluate(engine, loader, args) -> dict[str, float | list]:
    engine.eval()
    device = torch.device("cuda", args.local_rank)
    correct = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    confusion = torch.zeros((3, 3), device=device)
    intersection = torch.zeros((), device=device)
    union = torch.zeros((), device=device)

    for batch in loader:
        batch = prepare_batch(batch, args.precision)
        batch["inference"] = True
        output = engine(**batch)
        predictions = output["logits"].argmax(dim=-1)
        labels = batch["cls_labels"]
        correct += (predictions == labels).sum()
        total += labels.numel()
        for target, prediction in zip(labels, predictions):
            confusion[target.long(), prediction.long()] += 1

        if (labels == 2).any():
            for predicted_mask, target_mask in zip(
                output["pred_masks"],
                output["gt_masks"],
            ):
                predicted = predicted_mask.sigmoid() > 0.5
                target = target_mask > 0.5
                intersection += torch.logical_and(predicted, target).sum()
                union += torch.logical_or(predicted, target).sum()

    for value in (correct, total, confusion, intersection, union):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

    class_f1 = []
    for class_index in range(3):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(
            float((2 * true_positive / denominator).cpu())
            if denominator > 0
            else 0.0
        )

    return {
        "accuracy": float((correct / total.clamp_min(1)).cpu()),
        "macro_f1": sum(class_f1) / len(class_f1),
        "class_f1": class_f1,
        "localization_iou": float((intersection / union.clamp_min(1)).cpu()),
        "confusion_matrix": confusion.cpu().tolist(),
    }


def deepspeed_config(args: argparse.Namespace) -> dict:
    return {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "zero_optimization": {"stage": 2},
        "gradient_clipping": 1.0,
    }


def main() -> None:
    args = parse_args()
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume with a trained checkpoint.")

    deepspeed.init_distributed()
    torch.manual_seed(args.seed + dist.get_rank())
    torch.cuda.set_device(args.local_rank)

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    tokenizer = build_tokenizer(args)
    model = build_model(args, tokenizer)
    train_loader = None if args.eval_only else make_loader(
        args,
        tokenizer,
        split="train",
        batch_size=args.batch_size,
    )
    validation_loaders = {
        split: make_loader(
            args,
            tokenizer,
            split=split,
            batch_size=args.val_batch_size,
        )
        for split in args.val_splits
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=deepspeed_config(args),
    )
    if args.resume:
        engine.load_checkpoint(args.resume)

    output_dir = Path(args.output_dir)
    writer = None
    if dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir)
    dist.barrier()

    start_epoch = 0 if not args.eval_only else args.epochs
    for epoch in range(start_epoch, args.epochs):
        train_epoch(engine, train_loader, args, writer, epoch)
        for split, loader in validation_loaders.items():
            metrics = evaluate(engine, loader, args)
            if dist.get_rank() == 0:
                print(json.dumps({"epoch": epoch + 1, "split": split, **metrics}))
                for name in ("accuracy", "macro_f1", "localization_iou"):
                    writer.add_scalar(f"{split}/{name}", metrics[name], epoch)
        engine.save_checkpoint(str(output_dir), tag=f"epoch-{epoch + 1}")

    if args.eval_only:
        for split, loader in validation_loaders.items():
            metrics = evaluate(engine, loader, args)
            if dist.get_rank() == 0:
                print(json.dumps({"split": split, **metrics}, indent=2))

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
