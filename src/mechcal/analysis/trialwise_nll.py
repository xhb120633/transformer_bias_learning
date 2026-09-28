"""Trial-wise model-versus-oracle NLL for fixed-noise restless models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.training.restless_dataset import RestlessTranscriptDataset, causal_batch


TRIAL_BINS = ((1, 10), (11, 25), (26, 50), (51, 100), (101, 150), (151, 200))


def _load_oracle_arrays(
    root: Path, split: str, ordered_session_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    by_session: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(root.glob(f"{split}_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            for index, session_id in enumerate(shard["session_id"]):
                by_session[str(session_id)] = (
                    shard["choice_probability"][index],
                    shard["action"][index],
                )
    probabilities = []
    actions = []
    for session_id in ordered_session_ids:
        probability, action = by_session[str(session_id)]
        probabilities.append(probability)
        actions.append(action)
    return np.stack(probabilities), np.stack(actions)


@torch.no_grad()
def _model_losses(
    checkpoint_path: Path,
    dataset: RestlessTranscriptDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = CausalTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    losses = np.empty((len(dataset), 200), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for tokens, masks, indices in loader:
        inputs, targets, target_mask = causal_batch(
            tokens.to(device), masks.to(device)
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(inputs)
        selected_logits = logits[target_mask].float().view(len(tokens), 200, -1)
        selected_targets = targets[target_mask].view(len(tokens), 200)
        batch_losses = F.cross_entropy(
            selected_logits[..., 1:5].reshape(-1, 4),
            (selected_targets - 1).reshape(-1),
            reduction="none",
        ).view(len(tokens), 200)
        losses[indices.numpy()] = batch_losses.cpu().numpy()
    return losses


def _mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=axis)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
    return mean, sem


def analyze(
    data_root: Path,
    runs: dict[str, Path],
    output_dir: Path,
    batch_size: int = 128,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(device_name)
    rows: list[dict[str, int | float | str]] = []
    summary: dict[str, Any] = {}
    curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for condition, run_dir in runs.items():
        dataset = RestlessTranscriptDataset(data_root, "test", condition)
        model_loss = _model_losses(
            run_dir / "best.pt", dataset, batch_size, device
        )
        probabilities, actions = _load_oracle_arrays(
            data_root, "test", dataset.audit.session_id
        )
        chosen_probability = np.take_along_axis(
            probabilities, actions[..., None], axis=2
        ).squeeze(-1)
        oracle_loss = -np.log(chosen_probability)
        summary[condition] = {}

        for sign in (-1, 1):
            selected = dataset.audit.kernel_sign == sign
            sign_model = model_loss[selected]
            sign_oracle = oracle_loss[selected]
            model_mean, model_sem = _mean_sem(sign_model)
            oracle_mean, oracle_sem = _mean_sem(sign_oracle)
            curves[(condition, sign)] = (model_mean, oracle_mean)
            for trial in range(model_loss.shape[1]):
                rows.append(
                    {
                        "condition": condition,
                        "kernel_sign": sign,
                        "trial": trial + 1,
                        "n_sessions": int(selected.sum()),
                        "model_nll": float(model_mean[trial]),
                        "model_sem": float(model_sem[trial]),
                        "oracle_nll": float(oracle_mean[trial]),
                        "oracle_sem": float(oracle_sem[trial]),
                        "excess_nll": float(model_mean[trial] - oracle_mean[trial]),
                    }
                )
            bin_summary = {}
            for start, stop in TRIAL_BINS:
                selection = slice(start - 1, stop)
                model_value = float(sign_model[:, selection].mean())
                oracle_value = float(sign_oracle[:, selection].mean())
                bin_summary[f"{start:03d}-{stop:03d}"] = {
                    "model_nll": model_value,
                    "oracle_nll": oracle_value,
                    "excess_nll": model_value - oracle_value,
                }
            summary[condition][f"sign_{sign:+d}"] = {
                "n_sessions": int(selected.sum()),
                "overall_model_nll": float(sign_model.mean()),
                "overall_oracle_nll": float(sign_oracle.mean()),
                "bins": bin_summary,
            }

    with (output_dir / "trialwise_nll.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    conditions = list(runs)
    figure, axes = plt.subplots(2, len(conditions), figsize=(13, 6), sharex=True, sharey=True)
    kernel = np.ones(10) / 10
    for column, condition in enumerate(conditions):
        for row_index, sign in enumerate((1, -1)):
            axis = axes[row_index, column]
            model_curve, oracle_curve = curves[(condition, sign)]
            axis.plot(
                np.arange(10, 201), np.convolve(model_curve, kernel, mode="valid"),
                label="Transformer", color="#1f77b4", linewidth=1.8
            )
            axis.plot(
                np.arange(10, 201), np.convolve(oracle_curve, kernel, mode="valid"),
                label="Oracle", color="#d62728", linewidth=1.5
            )
            axis.axhline(np.log(4), color="#777777", linestyle="--", linewidth=0.8)
            axis.set_title(f"{condition.replace('_', ' ')} / sign {sign:+d}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Trial")
            if column == 0:
                axis.set_ylabel("Choice NLL (10-trial mean)")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "trialwise_nll.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--run", action="append", required=True,
        help="Condition and run directory as condition=path; repeat three times.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        condition, path = item.split("=", 1)
        runs[condition] = Path(path)
    result = analyze(args.data_root, runs, args.output, args.batch_size, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
