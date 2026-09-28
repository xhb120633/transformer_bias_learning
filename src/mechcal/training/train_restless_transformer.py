"""Fit one pooled causal Transformer using loss only on choice tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.training.restless_dataset import RestlessTranscriptDataset, causal_batch

CHOICE_IDS = (1, 2, 3, 4)


def mask_reward_sessions(tokens, fraction, mask_token_id, generator):
    """Hide entire reward channels for a random subset; never change choices."""
    if not 0 <= fraction <= 1:
        raise ValueError("mask fraction must be in [0, 1]")
    changed = tokens.clone()
    rows = torch.randperm(len(tokens), generator=generator)[:round(len(tokens) * fraction)]
    changed[rows, 2::2] = mask_token_id
    return changed


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _choice_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., list(CHOICE_IDS)]


@torch.no_grad()
def evaluate(
    model: CausalTransformer,
    loader: DataLoader,
    dataset: RestlessTranscriptDataset,
    device: torch.device,
    precision: str,
    reward_mask_token_id: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_full_nll = 0.0
    total_choice_nll = 0.0
    total_correct = 0
    total_count = 0
    strata: dict[str, list[float]] = {}
    for tokens, masks, indices in loader:
        if reward_mask_token_id is not None:
            tokens = tokens.clone()
            tokens[:, 2::2] = reward_mask_token_id
        inputs, targets, target_mask = causal_batch(
            tokens.to(device), masks.to(device)
        )
        with _autocast(device, precision):
            logits = model(inputs)
        selected_logits = logits[target_mask].float()
        selected_targets = targets[target_mask]
        full_losses = F.cross_entropy(
            selected_logits, selected_targets, reduction="none"
        )
        choice_losses = F.cross_entropy(
            _choice_logits(selected_logits), selected_targets - 1, reduction="none"
        )
        predictions = _choice_logits(selected_logits).argmax(dim=-1) + 1
        total_full_nll += full_losses.sum().item()
        total_choice_nll += choice_losses.sum().item()
        total_correct += (predictions == selected_targets).sum().item()
        total_count += selected_targets.numel()

        per_session = choice_losses.view(len(tokens), -1).mean(dim=1).cpu().numpy()
        for data_index, loss in zip(indices.numpy(), per_session, strict=True):
            condition = str(dataset.audit.condition_name[data_index])
            sign = int(dataset.audit.kernel_sign[data_index])
            strata.setdefault(f"{condition}/sign_{sign:+d}", []).append(float(loss))

    return {
        "full_vocab_choice_nll": total_full_nll / total_count,
        "choice_nll": total_choice_nll / total_count,
        "choice_accuracy": total_correct / total_count,
        "n_choice_targets": total_count,
        "choice_nll_by_stratum": {
            key: float(np.mean(values)) for key, values in sorted(strata.items())
        },
    }


def _save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def train(
    config_path: Path,
    output_dir: Path,
    condition_override: str | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    condition = condition_override or raw.get("condition")
    if not condition:
        raise ValueError(
            "condition is required: train reward_dominant, balanced, and "
            "choice_dominant as separate models"
        )
    raw["condition"] = condition
    if seed_override is not None:
        raw["seed"] = seed_override
    output_dir.mkdir(parents=True)
    _save_json(output_dir / "resolved_config.json", raw)

    seed = int(raw["seed"])
    _seed_everything(seed)
    requested_device = str(raw.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    precision = str(raw.get("precision", "bf16"))
    data_root = Path(raw["data_root"])
    limits = raw.get("limits", {})
    train_data = RestlessTranscriptDataset(
        data_root, "train", condition, limits.get("train")
    )
    val_data = RestlessTranscriptDataset(data_root, "val", condition, limits.get("val"))
    defer_test = bool(raw.get("defer_test_evaluation", False))
    test_data = None if defer_test else RestlessTranscriptDataset(
        data_root, "test", condition, limits.get("test")
    )
    if raw.get("input_mode", "full") == "choice_only":
        from mechcal.analysis.reward_ablation import choice_only_ablation
        for dataset in (train_data, val_data, test_data):
            if dataset is not None:
                dataset.tokens, dataset.choice_target_mask = choice_only_ablation(dataset.tokens)
    loader_args = {
        "batch_size": int(raw["batch_size"]),
        "num_workers": int(raw.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    mask_generator = torch.Generator().manual_seed(seed + 100000)
    mask_fraction = float(raw.get("reward_mask_fraction", 0.0))
    mask_token_id = int(raw.get("reward_mask_token_id", 106))
    if not 0 <= mask_fraction <= 1:
        raise ValueError("reward_mask_fraction must be in [0, 1]")
    if mask_fraction and (raw.get("input_mode", "full") != "full"
                          or not 106 <= mask_token_id < raw["model"]["vocab_size"]):
        raise ValueError("reward masking requires full transcripts and a new reserved token")
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator, **loader_args
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = None if defer_test else DataLoader(test_data, shuffle=False, **loader_args)

    if raw.get("model_type", "transformer") == "gru":
        from mechcal.models.gru import CausalGRU, GRUConfig
        model_config = GRUConfig(**raw["model"])
        model = CausalGRU(model_config).to(device)
    else:
        model_config = TransformerConfig(**raw["model"])
        model = CausalTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(raw["learning_rate"]),
        weight_decay=float(raw.get("weight_decay", 0.01)),
    )
    max_epochs = int(raw["max_epochs"])
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(raw.get("warmup_steps", 0))

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and precision == "fp16")
    )
    history: list[dict[str, float | int]] = []
    best_nll = float("inf")
    best_epoch = 0
    patience = int(raw.get("early_stopping_patience", max_epochs))
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for tokens, masks, _ in train_loader:
            if mask_fraction:
                tokens = mask_reward_sessions(tokens, mask_fraction, mask_token_id, mask_generator)
            inputs, targets, target_mask = causal_batch(
                tokens.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                logits = model(inputs)
                loss = F.cross_entropy(logits[target_mask], targets[target_mask])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(raw.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = int(target_mask.sum().item())
            train_loss_sum += loss.item() * count
            train_count += count
            global_step += 1

        val_metrics = evaluate(model, val_loader, val_data, device, precision)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "step": global_step,
            "train_full_vocab_choice_nll": train_loss_sum / train_count,
            "val_full_vocab_choice_nll": val_metrics["full_vocab_choice_nll"],
            "val_choice_nll": val_metrics["choice_nll"],
            "val_choice_accuracy": val_metrics["choice_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["choice_nll"] < best_nll - float(raw.get("min_delta", 1e-4)):
            best_nll = val_metrics["choice_nll"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate(model, val_loader, val_data, device, precision)
    test_metrics = None if defer_test else evaluate(model, test_loader, test_data, device, precision)
    metrics = {
        "condition": condition,
        "best_epoch": best_epoch,
        "uniform_choice_nll": math.log(4.0),
        "validation": val_metrics,
        "test": test_metrics,
        "test_evaluation_deferred": defer_test,
        "dataset_sizes": {
            "train": len(train_data), "val": len(val_data), "test": None if defer_test else len(test_data)
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_pid": os.getpid(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "precision": precision,
        },
    }
    if raw.get("evaluate_reward_mask", False):
        metrics["validation_masked"] = evaluate(model, val_loader, val_data, device, precision, mask_token_id)
        if not defer_test:
            metrics["test_masked"] = evaluate(model, test_loader, test_data, device, precision, mask_token_id)
    _save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--condition",
        choices=("reward_dominant", "balanced", "choice_dominant"),
        help="Required unless set in config; each condition is trained separately.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the config seed without changing any other setting.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.config, args.output, args.condition, args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
