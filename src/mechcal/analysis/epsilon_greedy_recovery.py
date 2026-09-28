"""Choice-blind parameter recovery for the exact epsilon-greedy RL+kernel model.

The known 10% decision-noise rule is used.  Because multiplying all utilities
by a positive policy scale cannot change an epsilon-greedy argmax, policy scale
is structurally unidentified; the estimable policy parameters are learning
rate, signed kernel direction, and the reward/kernel mixing ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mechcal.analysis.restless_recovery import load_split


CONDITIONS = ("reward_dominant", "balanced", "choice_dominant")


def _states(actions: np.ndarray, rewards: np.ndarray, alpha: float, q_scale: float) -> tuple[np.ndarray, np.ndarray]:
    n_trials = len(actions)
    q = np.full(4, 50.0, dtype=np.float64)
    z_q = np.empty((n_trials, 4), dtype=np.float64)
    kernel = np.zeros_like(z_q)
    for trial in range(n_trials):
        z_q[trial] = (q - q.mean()) / q_scale
        if trial:
            kernel[trial, actions[trial - 1]] = 1.0
        action = actions[trial]
        q[action] += alpha * (rewards[trial] - q[action])
    return z_q, kernel


def _candidate_nll(
    actions: np.ndarray,
    z_q: np.ndarray,
    kernel: np.ndarray,
    lambdas: np.ndarray,
    signs: np.ndarray,
    decision_noise: float,
    stop: int,
) -> np.ndarray:
    logits = (
        (1.0 - lambdas)[:, None, None] * z_q[None, :stop]
        + (signs * lambdas)[:, None, None] * kernel[None, :stop]
    )
    winners = np.isclose(
        logits, logits.max(axis=2, keepdims=True), rtol=0.0, atol=1e-12
    )
    chosen = winners[:, np.arange(stop), actions[:stop]]
    probability = decision_noise / 4.0 + chosen * (
        (1.0 - decision_noise) / winners.sum(axis=2)
    )
    return -np.log(np.clip(probability, 1e-12, 1.0)).mean(axis=1)


def fit_session(
    actions: np.ndarray,
    rewards: np.ndarray,
    q_scale: float,
    decision_noise: float,
    fit_trials: int,
) -> dict[str, float | int]:
    """Coarse-to-fine exact maximum-likelihood grid fit for one session."""

    coarse_alpha = np.linspace(0.04, 0.60, 29)
    coarse_lambda = np.linspace(0.01, 0.99, 50)
    lambda_grid = np.tile(coarse_lambda, 2)
    sign_grid = np.repeat(np.asarray([-1.0, 1.0]), len(coarse_lambda))
    candidates: list[tuple[float, float, float, float]] = []
    for alpha in coarse_alpha:
        z_q, kernel = _states(actions, rewards, float(alpha), q_scale)
        nll = _candidate_nll(
            actions, z_q, kernel, lambda_grid, sign_grid, decision_noise, fit_trials
        )
        best = float(nll.min())
        for index in np.flatnonzero(np.isclose(nll, best, atol=1e-12, rtol=0.0)):
            candidates.append((best, float(alpha), float(lambda_grid[index]), float(sign_grid[index])))
    global_best = min(row[0] for row in candidates)
    coarse_best = [row for row in candidates if abs(row[0] - global_best) <= 1e-12]
    center_alpha = float(np.median([row[1] for row in coarse_best]))
    center_lambda = float(np.median([row[2] for row in coarse_best]))
    preferred_sign = float(np.sign(np.median([row[3] for row in coarse_best])))
    if preferred_sign == 0:
        preferred_sign = coarse_best[0][3]

    fine_alpha = np.unique(np.clip(np.linspace(center_alpha - 0.03, center_alpha + 0.03, 31), 0.01, 0.99))
    fine_lambda = np.unique(np.clip(np.linspace(center_lambda - 0.04, center_lambda + 0.04, 41), 0.001, 0.999))
    fine_rows: list[tuple[float, float, float]] = []
    for alpha in fine_alpha:
        z_q, kernel = _states(actions, rewards, float(alpha), q_scale)
        nll = _candidate_nll(
            actions,
            z_q,
            kernel,
            fine_lambda,
            np.full(len(fine_lambda), preferred_sign),
            decision_noise,
            fit_trials,
        )
        for index in range(len(fine_lambda)):
            fine_rows.append((float(nll[index]), float(alpha), float(fine_lambda[index])))
    fine_best_nll = min(row[0] for row in fine_rows)
    tied = [row for row in fine_rows if abs(row[0] - fine_best_nll) <= 1e-12]
    alpha_hat = float(np.median([row[1] for row in tied]))
    lambda_hat = float(np.median([row[2] for row in tied]))

    z_q, kernel = _states(actions, rewards, alpha_hat, q_scale)
    full_trial_nll = _candidate_nll(
        actions,
        z_q,
        kernel,
        np.asarray([lambda_hat]),
        np.asarray([preferred_sign]),
        decision_noise,
        len(actions),
    )[0]
    heldout_trial_nll = _candidate_nll(
        actions[fit_trials:],
        z_q[fit_trials:],
        kernel[fit_trials:],
        np.asarray([lambda_hat]),
        np.asarray([preferred_sign]),
        decision_noise,
        len(actions) - fit_trials,
    )[0]
    return {
        "alpha": alpha_hat,
        "lambda_choice": lambda_hat,
        "kernel_sign": int(preferred_sign),
        "fit_nll": fine_best_nll,
        "heldout_nll": float(heldout_trial_nll),
        "full_nll": float(full_trial_nll),
        "n_tied_fine_grid_points": len(tied),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze(dataset: Path, split: str, fit_trials: int, max_sessions: int | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    if metadata["generator"]["policy_mode"] != "epsilon_greedy":
        raise ValueError("dataset is not epsilon-greedy")
    data = load_split(dataset, split)
    if max_sessions is not None:
        data = {key: value[:max_sessions] for key, value in data.items()}
    n_sessions, n_trials = data["action"].shape
    if not 1 < fit_trials < n_trials:
        raise ValueError("fit_trials must leave a nonempty held-out suffix")
    estimates = np.empty((n_sessions, 6), dtype=np.float64)
    for index in range(n_sessions):
        fit = fit_session(
            data["action"][index].astype(np.int64),
            data["reward"][index].astype(np.float64),
            float(metadata["q_pairwise_scale"]),
            float(data["decision_noise"][index]),
            fit_trials,
        )
        estimates[index] = [
            fit["alpha"], fit["lambda_choice"], fit["kernel_sign"],
            fit["fit_nll"], fit["heldout_nll"], fit["n_tied_fine_grid_points"],
        ]
    true_alpha = data["alpha"].astype(float)
    true_lambda = data["lambda_choice"].astype(float)
    true_sign = data["kernel_sign"].astype(int)
    predicted_condition = np.where(estimates[:, 1] < 0.35, "reward_dominant", np.where(estimates[:, 1] > 0.65, "choice_dominant", "balanced"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "correctly specified epsilon-greedy delta-rule plus signed one-back choice kernel",
        "fit_protocol": f"individual maximum likelihood on trials 1-{fit_trials}; evaluate trials {fit_trials + 1}-{n_trials}",
        "known_decision_noise": float(metadata["generator"]["decision_noise"]),
        "policy_scale_identifiable": False,
        "policy_scale_note": "Positive utility scaling does not change an epsilon-greedy argmax.",
        "n_sessions": n_sessions,
        "alpha": {"mae": float(np.mean(np.abs(estimates[:, 0] - true_alpha))), "correlation": _corr(estimates[:, 0], true_alpha)},
        "lambda_choice": {"mae": float(np.mean(np.abs(estimates[:, 1] - true_lambda))), "correlation": _corr(estimates[:, 1], true_lambda)},
        "kernel_sign_accuracy": float(np.mean(estimates[:, 2] == true_sign)),
        "condition_accuracy_from_fitted_lambda": float(np.mean(predicted_condition == data["condition_name"])),
        "heldout_nll_mean": float(estimates[:, 4].mean()),
        "true_oracle_heldout_nll_mean": float(-np.log(np.take_along_axis(data["choice_probability"][:, fit_trials:], data["action"][:, fit_trials:, None], axis=2).squeeze(-1)).mean()),
        "median_tied_fine_grid_points": float(np.median(estimates[:, 5])),
        "by_condition": {},
    }
    for condition in CONDITIONS:
        mask = data["condition_name"] == condition
        report["by_condition"][condition] = {
            "n_sessions": int(mask.sum()),
            "true_lambda_mean": float(true_lambda[mask].mean()),
            "fitted_lambda_mean": float(estimates[mask, 1].mean()),
            "lambda_mae": float(np.mean(np.abs(estimates[mask, 1] - true_lambda[mask]))),
            "kernel_sign_accuracy": float(np.mean(estimates[mask, 2] == true_sign[mask])),
            "condition_accuracy": float(np.mean(predicted_condition[mask] == condition)),
            "heldout_nll": float(estimates[mask, 4].mean()),
        }
    arrays = {
        "base_participant_id": data["base_participant_id"],
        "condition_name": data["condition_name"],
        "true_alpha": true_alpha,
        "true_lambda_choice": true_lambda,
        "true_kernel_sign": true_sign,
        "fitted_alpha": estimates[:, 0],
        "fitted_lambda_choice": estimates[:, 1],
        "fitted_kernel_sign": estimates[:, 2].astype(np.int8),
        "fit_nll": estimates[:, 3],
        "heldout_nll": estimates[:, 4],
        "n_tied_fine_grid_points": estimates[:, 5].astype(np.int32),
        "predicted_condition": predicted_condition,
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fit-trials", type=int, default=150)
    parser.add_argument("--max-sessions", type=int)
    args = parser.parse_args()
    report, arrays = analyze(args.dataset, args.split, args.fit_trials, args.max_sessions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_estimates.npz", **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
