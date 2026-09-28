"""Unify donor-reward effects across oracle, Llama, and small Transformers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mechcal.analysis.restless_recovery import load_split


CONDITIONS = ("reward_dominant", "balanced", "choice_dominant")
LABELS = ("Reward", "Balanced", "Choice")
STRATA = ("all", "sign_+1", "sign_-1")


def _ci(values: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    idx = rng.integers(0, len(values), size=(n, len(values)))
    return np.quantile(values[idx].mean(axis=1), [0.025, 0.975]).tolist()


def _rolling(values: np.ndarray, width: int = 10) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _sign_lookup(dataset: Path, split: str) -> dict[str, dict[str, int]]:
    data = load_split(dataset, split)
    result: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        selected = data["condition_name"] == condition
        result[condition] = {
            str(pid): int(sign)
            for pid, sign in zip(
                data["base_participant_id"][selected],
                data["kernel_sign"][selected],
                strict=True,
            )
        }
    return result


def _load_raw_predictor(
    root: Path, nested: bool
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for condition in CONDITIONS:
        path = root / condition / "trial_losses.npz" if nested else root / f"{condition}.npz"
        with np.load(path) as z:
            participants_key = "participants"
            result[condition] = {
                "participants": z[participants_key].astype(str),
                "intact": z["intact_nll"].astype(float),
                "donor": z["donor_nll"].astype(float).mean(axis=0),
            }
    return result


def _load_small(root_pattern: str, seeds: list[int]) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    result = {}
    for seed in seeds:
        path = Path(root_pattern.format(seed=seed)) / "trial_losses.npz"
        with np.load(path) as z:
            result[seed] = {}
            participants = z["participant_id"].astype(str)
            signs = z["kernel_sign"].astype(int)
            for condition in CONDITIONS:
                result[seed][condition] = {
                    "participants": participants,
                    "signs": signs,
                    "intact": z[f"{condition}_intact"].astype(float),
                    "donor": z[f"{condition}_donor"].astype(float),
                }
    return result


def analyze(
    dataset: Path,
    oracle_root: Path,
    llama_root: Path,
    small_pattern: str,
    seeds: list[int],
    split: str,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], np.ndarray]]:
    rng = np.random.default_rng(seed)
    signs_by_id = _sign_lookup(dataset, split)
    raw = {
        "generator_oracle": _load_raw_predictor(oracle_root, nested=False),
        "llama_sft": _load_raw_predictor(llama_root, nested=True),
    }
    small = _load_small(small_pattern, seeds)
    report: dict[str, Any] = {
        "schema_version": 1,
        "primary_estimand": "mean per-choice NLL(donor reward history) - NLL(intact history)",
        "participant_bootstrap_replicates": bootstrap,
        "small_transformer_training_seeds": seeds,
        "predictors": {},
    }
    curves: dict[tuple[str, str, str], np.ndarray] = {}

    for predictor, conditions in raw.items():
        report["predictors"][predictor] = {}
        effects: dict[str, dict[str, np.ndarray]] = {}
        for condition in CONDITIONS:
            values = conditions[condition]
            delta = values["donor"] - values["intact"]
            signs = np.asarray(
                [signs_by_id[condition][pid] for pid in values["participants"]]
            )
            effects[condition] = {}
            report["predictors"][predictor][condition] = {}
            for stratum in STRATA:
                selected = np.ones(len(signs), bool) if stratum == "all" else signs == int(stratum[-2:])
                session = delta[selected].mean(axis=1)
                effects[condition][stratum] = session
                curves[(predictor, condition, stratum)] = delta[selected].mean(axis=0)
                report["predictors"][predictor][condition][stratum] = {
                    "n_sessions": int(selected.sum()),
                    "delta_nll": float(session.mean()),
                    "participant_bootstrap_ci95": _ci(session, rng, bootstrap),
                    "trial_1_delta_nll": float(delta[selected, 0].mean()),
                }
        contrasts = {}
        for left, right in ((CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[1], CONDITIONS[2]), (CONDITIONS[0], CONDITIONS[2])):
            contrasts[f"{left}_minus_{right}"] = {}
            for stratum in STRATA:
                difference = effects[left][stratum] - effects[right][stratum]
                contrasts[f"{left}_minus_{right}"][stratum] = {
                    "mean": float(difference.mean()),
                    "participant_bootstrap_ci95": _ci(difference, rng, bootstrap),
                }
        report["predictors"][predictor]["paired_condition_contrasts"] = contrasts

    predictor = "small_transformer"
    report["predictors"][predictor] = {}
    seed_effects: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        report["predictors"][predictor][condition] = {}
        seed_effects[condition] = {}
        for stratum in STRATA:
            estimates, trial_curves = [], []
            for train_seed in seeds:
                values = small[train_seed][condition]
                delta = values["donor"] - values["intact"]
                signs = values["signs"]
                selected = np.ones(len(signs), bool) if stratum == "all" else signs == int(stratum[-2:])
                estimates.append(float(delta[selected].mean()))
                trial_curves.append(delta[selected].mean(axis=0))
            estimates_array = np.asarray(estimates)
            trial_array = np.stack(trial_curves)
            seed_effects[condition][stratum] = estimates_array
            curves[(predictor, condition, stratum)] = trial_array
            report["predictors"][predictor][condition][stratum] = {
                "n_training_seeds": len(seeds),
                "per_seed_delta_nll": dict(zip(map(str, seeds), estimates, strict=True)),
                "delta_nll_mean_across_seeds": float(estimates_array.mean()),
                "delta_nll_sd_across_seeds": float(estimates_array.std(ddof=1)),
                "delta_nll_range_across_seeds": [float(estimates_array.min()), float(estimates_array.max())],
                "trial_1_delta_nll_mean_across_seeds": float(trial_array[:, 0].mean()),
            }
    contrasts = {}
    for left, right in ((CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[1], CONDITIONS[2]), (CONDITIONS[0], CONDITIONS[2])):
        contrasts[f"{left}_minus_{right}"] = {}
        for stratum in STRATA:
            difference = seed_effects[left][stratum] - seed_effects[right][stratum]
            contrasts[f"{left}_minus_{right}"][stratum] = {
                "per_seed": dict(zip(map(str, seeds), difference.tolist(), strict=True)),
                "mean_across_seeds": float(difference.mean()),
                "sd_across_seeds": float(difference.std(ddof=1)),
                "all_seeds_same_direction": bool(np.all(difference > 0) or np.all(difference < 0)),
            }
    report["predictors"][predictor]["paired_condition_contrasts"] = contrasts
    return report, curves


def make_figures(report: dict[str, Any], curves: dict[tuple[str, str, str], np.ndarray], output: Path) -> None:
    colors = {"reward_dominant": "#C44E52", "balanced": "#4C72B0", "choice_dominant": "#55A868"}
    predictors = ("generator_oracle", "llama_sft", "small_transformer")
    titles = ("Generator oracle", "Llama SFT", "Small Transformer (5 seeds)")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    x = np.arange(3)
    for ax, predictor, title in zip(axes, predictors, titles, strict=True):
        for offset, stratum, marker in ((-0.12, "sign_+1", "o"), (0.12, "sign_-1", "s")):
            ys, errs = [], []
            for condition in CONDITIONS:
                cell = report["predictors"][predictor][condition][stratum]
                if predictor == "small_transformer":
                    y, err = cell["delta_nll_mean_across_seeds"], cell["delta_nll_sd_across_seeds"]
                else:
                    y = cell["delta_nll"]
                    lo, hi = cell["participant_bootstrap_ci95"]
                    err = (hi - lo) / 2
                ys.append(y); errs.append(err)
            ax.errorbar(x + offset, ys, yerr=errs, fmt=marker, capsize=3, label=stratum.replace("sign_", "kernel "))
        ax.set_xticks(x, LABELS, rotation=20)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel("Donor $\\Delta$NLL" if ax is axes[0] else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure3_kernel_sign_calibration.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for row, (predictor, title) in enumerate(zip(predictors, titles, strict=True)):
        for col, stratum in enumerate(("sign_+1", "sign_-1")):
            ax = axes[row, col]
            for condition, label in zip(CONDITIONS, LABELS, strict=True):
                curve = curves[(predictor, condition, stratum)]
                if curve.ndim == 2:
                    mean = curve.mean(axis=0); sd = curve.std(axis=0, ddof=1)
                    ax.fill_between(np.arange(1, len(mean) + 1), _rolling(mean - sd), _rolling(mean + sd), color=colors[condition], alpha=.14)
                else:
                    mean = curve
                ax.plot(np.arange(1, len(mean) + 1), _rolling(mean), color=colors[condition], label=label, lw=1.7)
            ax.axhline(0, color="black", lw=.6)
            ax.set_title(f"{title}; kernel {'+' if col == 0 else '-'}")
            ax.set_ylabel("10-trial mean $\\Delta$NLL")
            if row == 2: ax.set_xlabel("Trial")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure4_trialwise_accumulation.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--oracle-root", required=True, type=Path)
    parser.add_argument("--llama-root", required=True, type=Path)
    parser.add_argument("--small-pattern", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report, curves = analyze(args.dataset, args.oracle_root, args.llama_root, args.small_pattern, args.seeds, args.split, args.bootstrap, args.seed)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["predictor", "condition", "stratum", "delta_nll", "uncertainty"])
        for predictor in ("generator_oracle", "llama_sft", "small_transformer"):
            for condition in CONDITIONS:
                for stratum in STRATA:
                    cell = report["predictors"][predictor][condition][stratum]
                    if predictor == "small_transformer": value, uncertainty = cell["delta_nll_mean_across_seeds"], cell["delta_nll_sd_across_seeds"]
                    else: value, uncertainty = cell["delta_nll"], cell["participant_bootstrap_ci95"]
                    writer.writerow([predictor, condition, stratum, value, json.dumps(uncertainty)])
    make_figures(report, curves, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
