"""Sign-stratified behavior and pooled recovery for restless-bandit data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def load_split(condition_dir: Path, split: str) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for file in sorted(condition_dir.glob(f"{split}_*.npz")):
        with np.load(file) as shard:
            for key in shard.files:
                chunks.setdefault(key, []).append(shard[key])
    if not chunks:
        raise FileNotFoundError(f"no {split} shards found in {condition_dir}")
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def behavior_summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    actions = data["action"]
    repeats = actions[:, 1:] == actions[:, :-1]
    probabilities = np.take_along_axis(
        data["choice_probability"], actions[..., None], axis=2
    ).squeeze(-1)
    return {
        "n_sessions": int(actions.shape[0]),
        "mean_reward": float(data["reward"].mean()),
        "optimal_choice_rate": float((actions == data["optimal_action"]).mean()),
        "repeat_rate": float(repeats.mean()),
        "oracle_nll_per_choice": float(-np.log(np.clip(probabilities, 1e-8, 1)).mean()),
        "lambda_mean": float(data["lambda_choice"].mean()),
        "alpha_mean": float(data["alpha"].mean()),
        "beta_reward_mean": float(data["beta_reward"].mean()),
        "beta_kernel_mean": float(data["beta_kernel"].mean()),
        "lapse_mean": float(data["lapse"].mean()),
    }


def _features(
    actions: np.ndarray, rewards: np.ndarray, alpha: float, q_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    n_sessions, n_trials = actions.shape
    q = np.full((n_sessions, 4), 50.0, dtype=np.float64)
    z_q = np.empty((n_sessions, n_trials, 4), dtype=np.float64)
    kernel = np.zeros_like(z_q)
    rows = np.arange(n_sessions)
    for trial in range(n_trials):
        z_q[:, trial] = (q - q.mean(axis=1, keepdims=True)) / q_scale
        if trial > 0:
            kernel[rows, trial, actions[:, trial - 1]] = 1.0
        chosen = actions[:, trial]
        q[rows, chosen] += alpha * (rewards[:, trial] - q[rows, chosen])
    return z_q, kernel


def pooled_raw_recovery(data: dict[str, np.ndarray], q_scale: float) -> dict[str, float]:
    actions = data["action"].astype(np.int64)
    rewards = data["reward"].astype(np.float64)
    rows = np.arange(actions.shape[0])[:, None]
    trials = np.arange(actions.shape[1])[None, :]

    def objective(parameters: np.ndarray) -> float:
        alpha, beta_reward, beta_kernel, lapse = parameters
        z_q, kernel = _features(actions, rewards, alpha, q_scale)
        logits = beta_reward * z_q + beta_kernel * kernel
        log_structured = logits - logsumexp(logits, axis=2, keepdims=True)
        structured = np.exp(log_structured)
        probability = (1.0 - lapse) * structured + lapse / 4.0
        chosen_probability = probability[rows, trials, actions]
        return float(-np.log(np.clip(chosen_probability, 1e-8, 1)).mean())

    starts = (
        np.asarray([0.25, 1.5, 1.0, 0.05]),
        np.asarray([0.25, 1.5, -1.0, 0.05]),
        np.asarray([0.50, 0.5, 0.0, 0.10]),
    )
    bounds = ((0.02, 0.98), (0.0, 5.0), (-5.0, 5.0), (0.0, 0.25))
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    best = min(fits, key=lambda result: result.fun)
    alpha, beta_reward, beta_kernel, lapse = best.x
    denominator = beta_reward + abs(beta_kernel)
    return {
        "alpha": float(alpha),
        "beta_reward": float(beta_reward),
        "beta_kernel": float(beta_kernel),
        "lambda_choice": float(abs(beta_kernel) / denominator),
        "lapse": float(lapse),
        "nll_per_choice": float(best.fun),
    }


def _safe_correlation(true: np.ndarray, estimated: np.ndarray) -> float | None:
    if np.std(true) < 1e-10 or np.std(estimated) < 1e-10:
        return None
    return float(np.corrcoef(true, estimated)[0, 1])


def individual_raw_recovery(
    data: dict[str, np.ndarray], q_scale: float, max_sessions: int
) -> dict[str, Any]:
    n_sessions = min(max_sessions, data["action"].shape[0])
    estimates = np.empty((n_sessions, 5), dtype=np.float64)
    for index in range(n_sessions):
        subset = {key: value[index : index + 1] for key, value in data.items()}
        fit = pooled_raw_recovery(subset, q_scale)
        estimates[index] = [
            fit["alpha"],
            fit["beta_reward"],
            fit["beta_kernel"],
            fit["lambda_choice"],
            fit["lapse"],
        ]
    true = np.column_stack(
        [
            data["alpha"][:n_sessions],
            data["beta_reward"][:n_sessions],
            data["beta_kernel"][:n_sessions],
            data["lambda_choice"][:n_sessions],
            data["lapse"][:n_sessions],
        ]
    )
    names = ("alpha", "beta_reward", "beta_kernel", "lambda_choice", "lapse")
    report: dict[str, Any] = {
        "n_sessions": n_sessions,
        "kernel_sign_accuracy": float(
            np.mean(np.sign(estimates[:, 2]) == np.sign(true[:, 2]))
        ),
    }
    for column, name in enumerate(names):
        report[name] = {
            "true_mean": float(true[:, column].mean()),
            "estimate_mean": float(estimates[:, column].mean()),
            "mae": float(np.mean(np.abs(estimates[:, column] - true[:, column]))),
            "correlation": _safe_correlation(true[:, column], estimates[:, column]),
        }
    return report


def analyze(root: Path, split: str, max_individual_sessions: int) -> dict[str, Any]:
    if (root / "metadata.json").exists() and list(root.glob(f"{split}_*.npz")):
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        data = load_split(root, split)
        report: dict[str, Any] = {}
        for condition_name in sorted(np.unique(data["condition_name"]).tolist()):
            condition_mask = data["condition_name"] == condition_name
            condition_report: dict[str, Any] = {}
            for sign, label in ((1, "perseveration"), (-1, "alternation")):
                mask = condition_mask & (data["kernel_sign"] == sign)
                subset = {key: value[mask] for key, value in data.items()}
                condition_report[label] = {
                    "behavior": behavior_summary(subset),
                    "pooled_raw_recovery": pooled_raw_recovery(
                        subset, float(metadata["q_pairwise_scale"])
                    ),
                }
            if max_individual_sessions > 0:
                subset = {key: value[condition_mask] for key, value in data.items()}
                condition_report["individual_raw_recovery"] = individual_raw_recovery(
                    subset,
                    float(metadata["q_pairwise_scale"]),
                    max_individual_sessions,
                )
            report[condition_name] = condition_report
        return report

    report: dict[str, Any] = {}
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = json.loads((condition_dir / "metadata.json").read_text(encoding="utf-8"))
        data = load_split(condition_dir, split)
        condition_report: dict[str, Any] = {}
        for sign, label in ((1, "perseveration"), (-1, "alternation")):
            mask = data["kernel_sign"] == sign
            subset = {key: value[mask] for key, value in data.items()}
            condition_report[label] = {
                "behavior": behavior_summary(subset),
                "pooled_raw_recovery": pooled_raw_recovery(
                    subset, float(metadata["q_pairwise_scale"])
                ),
            }
        condition_report["individual_raw_recovery"] = individual_raw_recovery(
            data, float(metadata["q_pairwise_scale"]), max_individual_sessions
        )
        report[condition_dir.name] = condition_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-individual-sessions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.dataset, args.split, args.max_individual_sessions)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
