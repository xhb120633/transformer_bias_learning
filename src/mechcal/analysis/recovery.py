"""Audit observable behavior and parameter recovery in generated datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def load_split(condition_dir: Path, split: str) -> dict[str, np.ndarray]:
    files = sorted(condition_dir.glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {split} shards found in {condition_dir}")
    chunks: dict[str, list[np.ndarray]] = {}
    for file in files:
        with np.load(file) as shard:
            for key in shard.files:
                chunks.setdefault(key, []).append(shard[key])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def behavior_summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    actions = data["action"]
    rewards = data["reward"]
    repeats = actions[:, 1:] == actions[:, :-1]
    previous_rewards = rewards[:, :-1]
    win_mask = previous_rewards == 1
    loss_mask = ~win_mask
    optimal = actions == data["high_reward_option"]
    probabilities = np.clip(data["choice_probability"], 1e-8, 1 - 1e-8)
    oracle_nll = -np.mean(
        actions * np.log(probabilities) + (1 - actions) * np.log(1 - probabilities)
    )
    return {
        "n_sessions": int(actions.shape[0]),
        "n_trials": int(actions.shape[1]),
        "choice_1_rate": float(actions.mean()),
        "obtained_reward_rate": float(rewards.mean()),
        "optimal_choice_rate": float(optimal.mean()),
        "repeat_rate": float(repeats.mean()),
        "win_stay_rate": float(repeats[win_mask].mean()),
        "lose_stay_rate": float(repeats[loss_mask].mean()),
        "win_minus_lose_stay": float(repeats[win_mask].mean() - repeats[loss_mask].mean()),
        "oracle_nll": float(oracle_nll),
    }


def pooled_oracle_state_recovery(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit choice on saved true z(Q) and K; this is a generator audit."""

    y = data["action"].reshape(-1).astype(np.float64)
    x = np.column_stack(
        [
            np.ones_like(y),
            data["z_q"].reshape(-1),
            data["choice_kernel"].reshape(-1),
        ]
    )

    def objective(beta: np.ndarray) -> float:
        logits = x @ beta
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    result = minimize(objective, np.zeros(3), method="BFGS")
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError(f"pooled recovery failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "beta_reward": float(result.x[1]),
        "beta_choice": float(result.x[2]),
        "nll": float(result.fun),
    }


def _features_from_transcript(actions: np.ndarray, rewards: np.ndarray, alpha: float, value_scale: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([0.5, 0.5], dtype=np.float64)
    z_q = np.empty(actions.shape[0], dtype=np.float64)
    kernel = np.empty(actions.shape[0], dtype=np.float64)
    previous_action: int | None = None
    for trial, (action, reward) in enumerate(zip(actions, rewards)):
        z_q[trial] = (q[1] - q[0]) / value_scale
        kernel[trial] = 0 if previous_action is None else (1 if previous_action == 1 else -1)
        q[action] += alpha * (reward - q[action])
        previous_action = int(action)
    return z_q, kernel


def fit_individual_transcript(actions: np.ndarray, rewards: np.ndarray, value_scale: float) -> tuple[np.ndarray, float]:
    """Fit alpha, reward coefficient, choice coefficient, and lapse."""

    y = actions.astype(np.float64)

    def objective(parameters: np.ndarray) -> float:
        alpha, beta_reward, beta_choice, lapse = parameters
        z_q, kernel = _features_from_transcript(actions, rewards, alpha, value_scale)
        probability = (1.0 - lapse) * expit(beta_reward * z_q + beta_choice * kernel) + 0.5 * lapse
        probability = np.clip(probability, 1e-8, 1 - 1e-8)
        return float(-np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability)))

    starts = (
        np.asarray([0.20, 1.0, 1.0, 0.05]),
        np.asarray([0.45, 0.5, 1.5, 0.10]),
        np.asarray([0.10, 1.5, 0.5, 0.02]),
    )
    bounds = ((0.02, 0.98), (0.0, 4.0), (0.0, 4.0), (0.0, 0.25))
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    best = min(fits, key=lambda result: result.fun)
    return best.x, float(best.fun)


def _batch_features_from_transcripts(
    actions: np.ndarray, rewards: np.ndarray, alpha: float, value_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    n_sessions, n_trials = actions.shape
    q = np.full((n_sessions, 2), 0.5, dtype=np.float64)
    z_q = np.empty((n_sessions, n_trials), dtype=np.float64)
    kernel = np.zeros((n_sessions, n_trials), dtype=np.float64)
    kernel[:, 1:] = np.where(actions[:, :-1] == 1, 1.0, -1.0)
    rows = np.arange(n_sessions)
    for trial in range(n_trials):
        z_q[:, trial] = (q[:, 1] - q[:, 0]) / value_scale
        chosen = actions[:, trial]
        q[rows, chosen] += alpha * (rewards[:, trial] - q[rows, chosen])
    return z_q, kernel


def pooled_raw_transcript_recovery(
    data: dict[str, np.ndarray], theta: float, value_scale: float
) -> dict[str, float | None]:
    """Fit one behavioral model to raw sessions without using saved latents."""

    actions = data["action"].astype(np.int64)
    rewards = data["reward"].astype(np.float64)
    y = actions.astype(np.float64)

    def objective(parameters: np.ndarray) -> float:
        alpha, beta_reward, beta_choice, lapse = parameters
        z_q, kernel = _batch_features_from_transcripts(actions, rewards, alpha, value_scale)
        probability = (1.0 - lapse) * expit(beta_reward * z_q + beta_choice * kernel) + 0.5 * lapse
        probability = np.clip(probability, 1e-8, 1 - 1e-8)
        return float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))

    starts = (
        np.asarray([0.20, 1.0, 1.0, 0.05]),
        np.asarray([0.40, 0.5, 1.5, 0.10]),
        np.asarray([0.10, 1.5, 0.5, 0.02]),
    )
    bounds = ((0.02, 0.98), (0.0, 4.0), (0.0, 4.0), (0.0, 0.25))
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    best = min(fits, key=lambda result: result.fun)
    alpha, beta_reward, beta_choice, lapse = best.x
    denominator = beta_reward + beta_choice
    return {
        "alpha": float(alpha),
        "beta_reward": float(beta_reward),
        "beta_choice": float(beta_choice),
        "lapse": float(lapse),
        "choice_reliance": float(beta_choice / denominator) if denominator > 0 else None,
        "true_alpha_mean": float(data["alpha"].mean()),
        "true_beta_reward_mean": float(np.mean(data["scale"] * np.cos(theta))),
        "true_beta_choice_mean": float(np.mean(data["scale"] * np.sin(theta))),
        "true_lapse_mean": float(data["lapse"].mean()),
        "nll_per_choice": float(best.fun),
    }


def _safe_correlation(true: np.ndarray, estimated: np.ndarray) -> float | None:
    if np.std(true) < 1e-10 or np.std(estimated) < 1e-10:
        return None
    return float(np.corrcoef(true, estimated)[0, 1])


def individual_recovery(
    data: dict[str, np.ndarray], theta: float, value_scale: float, max_sessions: int
) -> dict[str, Any]:
    n_sessions = min(max_sessions, data["action"].shape[0])
    estimates = np.empty((n_sessions, 4), dtype=np.float64)
    nll = np.empty(n_sessions, dtype=np.float64)
    for index in range(n_sessions):
        estimates[index], nll[index] = fit_individual_transcript(
            data["action"][index], data["reward"][index], value_scale
        )
    true = np.column_stack(
        [
            data["alpha"][:n_sessions],
            data["scale"][:n_sessions] * np.cos(theta),
            data["scale"][:n_sessions] * np.sin(theta),
            data["lapse"][:n_sessions],
        ]
    )
    names = ("alpha", "beta_reward", "beta_choice", "lapse")
    metrics: dict[str, Any] = {"n_sessions": n_sessions, "mean_fitted_nll": float(nll.mean())}
    for index, name in enumerate(names):
        metrics[name] = {
            "true_mean": float(true[:, index].mean()),
            "estimate_mean": float(estimates[:, index].mean()),
            "mae": float(np.mean(np.abs(estimates[:, index] - true[:, index]))),
            "correlation": _safe_correlation(true[:, index], estimates[:, index]),
            "boundary_rate": float(
                np.mean(
                    (estimates[:, index] <= (0.0201 if index == 0 else 0.0001))
                    | (estimates[:, index] >= (0.9799 if index == 0 else (0.2499 if index == 3 else 3.9999)))
                )
            ),
        }
    return metrics


def analyze_dataset(root: Path, split: str, max_individual_sessions: int) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = json.loads((condition_dir / "metadata.json").read_text(encoding="utf-8"))
        theta = float(metadata["generator"]["theta"])
        value_scale = float(metadata["generator"]["value_scale"])
        data = load_split(condition_dir, split)
        report[condition_dir.name] = {
            "theta": theta,
            "behavior": behavior_summary(data),
            "pooled_oracle_state_recovery": pooled_oracle_state_recovery(data),
            "pooled_raw_transcript_recovery": pooled_raw_transcript_recovery(
                data, theta, value_scale
            ),
            "individual_raw_transcript_recovery": individual_recovery(
                data, theta, value_scale, max_individual_sessions
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-individual-sessions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_dataset(args.dataset, args.split, args.max_individual_sessions)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
