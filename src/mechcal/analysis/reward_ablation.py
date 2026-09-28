"""Measure held-out choice-NLL effects of removing participant reward history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.training.restless_dataset import RestlessTranscriptDataset, causal_batch


CONDITIONS = ("reward_dominant", "balanced", "choice_dominant")


def donor_reward_ablation(tokens: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
    """Replace reward slots with another session while preserving all choices."""
    if tokens.ndim != 2 or tokens.shape[1] % 2 != 1:
        raise ValueError("tokens must have shape [sessions, 1 + 2 * trials]")
    permutation = np.asarray(permutation)
    if sorted(permutation.tolist()) != list(range(len(tokens))):
        raise ValueError("permutation must contain every session exactly once")
    if np.any(permutation == np.arange(len(tokens))):
        raise ValueError("reward donors must be a derangement")
    result = tokens.clone()
    result[:, 2::2] = tokens[torch.from_numpy(permutation).long(), 2::2]
    return result


def neutral_reward_ablation(tokens: torch.Tensor, reward_token: int = 55) -> torch.Tensor:
    """Replace every reward with REWARD_50 while preserving sequence geometry."""
    result = tokens.clone()
    result[:, 2::2] = reward_token
    return result


def choice_only_ablation(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Delete every reward token, yielding BOS followed by all observed choices."""
    if tokens.ndim != 2 or tokens.shape[1] % 2 != 1:
        raise ValueError("tokens must have shape [sessions, 1 + 2 * trials]")
    choice_only = torch.cat((tokens[:, :1], tokens[:, 1::2]), dim=1)
    choice_mask = torch.ones_like(choice_only, dtype=torch.bool)
    choice_mask[:, 0] = False
    return choice_only, choice_mask


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("a donor derangement requires at least two sessions")
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            return permutation


@torch.no_grad()
def _choice_losses(
    model: CausalTransformer,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    device: torch.device,
    zero_reward_embeddings: bool = False,
) -> np.ndarray:
    losses = []
    for start in range(0, len(tokens), batch_size):
        batch_tokens = tokens[start : start + batch_size].to(device)
        batch_masks = masks[start : start + batch_size].to(device)
        inputs, targets, target_mask = causal_batch(batch_tokens, batch_masks)
        zero_mask = None
        if zero_reward_embeddings:
            zero_mask = torch.zeros_like(inputs, dtype=torch.bool)
            zero_mask[:, 2::2] = True
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(inputs, zero_token_embedding_mask=zero_mask)
        selected_logits = logits[target_mask].float()
        selected_targets = targets[target_mask]
        loss = F.cross_entropy(
            selected_logits[:, 1:5], selected_targets - 1, reduction="none"
        )
        losses.append(loss.view(len(batch_tokens), -1).cpu().numpy())
    return np.concatenate(losses)


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(
    data_root: Path,
    runs: dict[str, Path],
    output_dir: Path,
    n_donors: int = 20,
    n_bootstrap: int = 10_000,
    seed: int = 7103,
    batch_size: int = 128,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(device_name)
    rng = np.random.default_rng(seed)
    bootstrap_rng = np.random.default_rng(seed + 1)
    summary: dict[str, Any] = {}
    session_effects: dict[str, dict[str, np.ndarray]] = {}
    raw_outputs: dict[str, np.ndarray] = {}

    # Identical participant ordering makes the same donor mappings valid across
    # all three paired conditions.
    reference = RestlessTranscriptDataset(data_root, "test", CONDITIONS[0])
    permutations = [_derangement(len(reference), rng) for _ in range(n_donors)]

    for condition in CONDITIONS:
        dataset = RestlessTranscriptDataset(data_root, "test", condition)
        if not np.array_equal(
            reference.audit.base_participant_id, dataset.audit.base_participant_id
        ):
            raise RuntimeError("conditions do not contain the same ordered participants")
        checkpoint = torch.load(
            runs[condition] / "best.pt", map_location=device, weights_only=True
        )
        model = CausalTransformer(TransformerConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        intact = _choice_losses(
            model, dataset.tokens, dataset.choice_target_mask, batch_size, device
        )
        donor_losses = []
        for permutation in permutations:
            ablated = donor_reward_ablation(dataset.tokens, permutation)
            donor_losses.append(
                _choice_losses(
                    model, ablated, dataset.choice_target_mask, batch_size, device
                )
            )
        donor = np.mean(donor_losses, axis=0)
        neutral = _choice_losses(
            model,
            neutral_reward_ablation(dataset.tokens),
            dataset.choice_target_mask,
            batch_size,
            device,
        )
        choice_only_tokens, choice_only_mask = choice_only_ablation(dataset.tokens)
        deleted = _choice_losses(
            model, choice_only_tokens, choice_only_mask, batch_size, device
        )
        placeholder = _choice_losses(
            model,
            dataset.tokens,
            dataset.choice_target_mask,
            batch_size,
            device,
            zero_reward_embeddings=True,
        )
        session_effects[condition] = {
            "donor": donor.mean(axis=1) - intact.mean(axis=1),
            "neutral50": neutral.mean(axis=1) - intact.mean(axis=1),
            "deleted": deleted.mean(axis=1) - intact.mean(axis=1),
            "placeholder": placeholder.mean(axis=1) - intact.mean(axis=1),
        }
        raw_outputs[f"{condition}_intact"] = intact
        raw_outputs[f"{condition}_donor"] = donor
        raw_outputs[f"{condition}_neutral50"] = neutral
        raw_outputs[f"{condition}_deleted"] = deleted
        raw_outputs[f"{condition}_placeholder"] = placeholder
        summary[condition] = {}
        for stratum, selected in (
            ("all", np.ones(len(dataset), dtype=bool)),
            ("sign_+1", dataset.audit.kernel_sign == 1),
            ("sign_-1", dataset.audit.kernel_sign == -1),
        ):
            intact_session = intact[selected].mean(axis=1)
            donor_session = donor[selected].mean(axis=1)
            neutral_session = neutral[selected].mean(axis=1)
            donor_effect = donor_session - intact_session
            neutral_effect = neutral_session - intact_session
            deleted_session = deleted[selected].mean(axis=1)
            deleted_effect = deleted_session - intact_session
            placeholder_session = placeholder[selected].mean(axis=1)
            placeholder_effect = placeholder_session - intact_session
            donor_ci = _bootstrap_ci(donor_effect, bootstrap_rng, n_bootstrap)
            neutral_ci = _bootstrap_ci(neutral_effect, bootstrap_rng, n_bootstrap)
            deleted_ci = _bootstrap_ci(deleted_effect, bootstrap_rng, n_bootstrap)
            placeholder_ci = _bootstrap_ci(
                placeholder_effect, bootstrap_rng, n_bootstrap
            )
            summary[condition][stratum] = {
                "n_sessions": int(selected.sum()),
                "intact_nll": float(intact_session.mean()),
                "donor_ablation_nll": float(donor_session.mean()),
                "delta_nll_donor": float(donor_effect.mean()),
                "delta_nll_donor_ci95": list(donor_ci),
                "neutral50_ablation_nll": float(neutral_session.mean()),
                "delta_nll_neutral50": float(neutral_effect.mean()),
                "delta_nll_neutral50_ci95": list(neutral_ci),
                "choice_only_nll": float(deleted_session.mean()),
                "delta_nll_choice_only": float(deleted_effect.mean()),
                "delta_nll_choice_only_ci95": list(deleted_ci),
                "placeholder_nll": float(placeholder_session.mean()),
                "delta_nll_placeholder": float(placeholder_effect.mean()),
                "delta_nll_placeholder_ci95": list(placeholder_ci),
                "delta_nll_donor_by_trial": (donor[selected] - intact[selected])
                .mean(axis=0)
                .tolist(),
            }

    paired_contrasts: dict[str, Any] = {}
    for first, second in (
        ("reward_dominant", "balanced"),
        ("balanced", "choice_dominant"),
        ("reward_dominant", "choice_dominant"),
    ):
        contrast_name = f"{first}_minus_{second}"
        paired_contrasts[contrast_name] = {}
        for stratum, selected in (
            ("all", np.ones(len(reference), dtype=bool)),
            ("sign_+1", reference.audit.kernel_sign == 1),
            ("sign_-1", reference.audit.kernel_sign == -1),
        ):
            paired_contrasts[contrast_name][stratum] = {}
            for ablation in ("donor", "neutral50", "deleted", "placeholder"):
                difference = (
                    session_effects[first][ablation][selected]
                    - session_effects[second][ablation][selected]
                )
                paired_contrasts[contrast_name][stratum][ablation] = {
                    "mean_delta_nll_difference": float(difference.mean()),
                    "ci95": list(
                        _bootstrap_ci(difference, bootstrap_rng, n_bootstrap)
                    ),
                }
    summary["paired_condition_contrasts"] = paired_contrasts

    np.savez_compressed(
        output_dir / "trial_losses.npz",
        participant_id=reference.audit.base_participant_id,
        kernel_sign=reference.audit.kernel_sign,
        donor_permutations=np.stack(permutations),
        **raw_outputs,
    )

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.19
    for axis, stratum in zip(axes, ("sign_+1", "sign_-1"), strict=True):
        donor_values = [summary[c][stratum]["delta_nll_donor"] for c in CONDITIONS]
        neutral_values = [
            summary[c][stratum]["delta_nll_neutral50"] for c in CONDITIONS
        ]
        deleted_values = [
            summary[c][stratum]["delta_nll_choice_only"] for c in CONDITIONS
        ]
        placeholder_values = [
            summary[c][stratum]["delta_nll_placeholder"] for c in CONDITIONS
        ]
        axis.bar(x - 1.5 * width, donor_values, width, label="Donor rewards")
        axis.bar(x - 0.5 * width, neutral_values, width, label="All REWARD_50")
        axis.bar(x + 0.5 * width, placeholder_values, width, label="Zero placeholder")
        axis.bar(x + 1.5 * width, deleted_values, width, label="Delete rewards")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["reward", "balanced", "choice"])
        axis.set_title(stratum.replace("_", " "))
        axis.set_xlabel("Training condition")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Reward-removal effect, delta NLL")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "reward_ablation.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-donors", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7103)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        condition, path = item.split("=", 1)
        runs[condition] = Path(path)
    missing = set(CONDITIONS) - set(runs)
    if missing:
        raise ValueError(f"missing run paths for {sorted(missing)}")
    result = analyze(
        args.data_root,
        runs,
        args.output,
        args.n_donors,
        args.n_bootstrap,
        args.seed,
        args.batch_size,
        args.device,
    )
    concise = {
        condition: {
            stratum: {
                "delta_nll_donor": result[condition][stratum]["delta_nll_donor"],
                "delta_nll_donor_ci95": result[condition][stratum][
                    "delta_nll_donor_ci95"
                ],
                "delta_nll_neutral50": result[condition][stratum][
                    "delta_nll_neutral50"
                ],
                "delta_nll_choice_only": result[condition][stratum][
                    "delta_nll_choice_only"
                ],
                "delta_nll_choice_only_ci95": result[condition][stratum][
                    "delta_nll_choice_only_ci95"
                ],
                "delta_nll_placeholder": result[condition][stratum][
                    "delta_nll_placeholder"
                ],
                "delta_nll_placeholder_ci95": result[condition][stratum][
                    "delta_nll_placeholder_ci95"
                ],
            }
            for stratum in ("all", "sign_+1", "sign_-1")
        }
        for condition in CONDITIONS
    }
    concise["paired_condition_contrasts"] = result["paired_condition_contrasts"]
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
