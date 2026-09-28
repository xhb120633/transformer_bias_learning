"""Measure matched-donor reward effects under the known epsilon-greedy generator.

The observed action history is held fixed.  At each trial the recipient's Q
state is updated with the reward observed by a deranged donor at that same
trial, while all recipient policy parameters remain fixed.  This exactly
matches the natural-token donor intervention used for the Llama adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mechcal.analysis.restless_recovery import load_split


CONDITIONS = ("reward_dominant", "balanced", "choice_dominant")


def oracle_trial_nll(
    actions: np.ndarray,
    rewards: np.ndarray,
    alpha: np.ndarray,
    beta_reward: np.ndarray,
    beta_kernel: np.ndarray,
    q_scale: float,
    decision_noise: np.ndarray,
) -> np.ndarray:
    """Return trial NLLs after recursively updating Q from supplied rewards."""

    n_sessions, n_trials = actions.shape
    if rewards.shape != actions.shape:
        raise ValueError("actions and rewards must have identical shape")
    q = np.full((n_sessions, 4), 50.0, dtype=np.float64)
    rows = np.arange(n_sessions)
    losses = np.empty((n_sessions, n_trials), dtype=np.float64)
    for trial in range(n_trials):
        z_q = (q - q.mean(axis=1, keepdims=True)) / q_scale
        kernel = np.zeros_like(q)
        if trial:
            kernel[rows, actions[:, trial - 1]] = 1.0
        logits = beta_reward[:, None] * z_q + beta_kernel[:, None] * kernel
        winners = np.isclose(
            logits, logits.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12
        )
        n_winners = winners.sum(axis=1)
        chosen_is_winner = winners[rows, actions[:, trial]]
        probability = decision_noise / 4.0
        probability = probability + chosen_is_winner * (1.0 - decision_noise) / n_winners
        losses[:, trial] = -np.log(np.clip(probability, 1e-12, 1.0))
        chosen = actions[:, trial]
        q[rows, chosen] += alpha * (rewards[:, trial] - q[rows, chosen])
    return losses


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def _ordered_condition(
    data: dict[str, np.ndarray], condition: str, participants: np.ndarray
) -> dict[str, np.ndarray]:
    mask = data["condition_name"] == condition
    indices = {
        str(participant): index
        for index, participant in enumerate(data["base_participant_id"])
        if mask[index]
    }
    if set(indices) != set(participants.tolist()):
        raise AssertionError(f"participant mismatch for {condition}")
    order = np.asarray([indices[str(participant)] for participant in participants])
    return {key: value[order] for key, value in data.items()}


def analyze(
    dataset: Path,
    donor_root: Path,
    split: str = "test",
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260904,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    q_scale = float(metadata["q_pairwise_scale"])
    data = load_split(dataset, split)
    reports: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_mappings: np.ndarray | None = None
    reference_participants: np.ndarray | None = None

    for condition_index, condition in enumerate(CONDITIONS):
        with np.load(donor_root / condition / "trial_losses.npz") as donor_file:
            participants = donor_file["participants"].astype(str)
            mappings = donor_file["donor_mappings"].astype(np.int64)
        if reference_mappings is None:
            reference_mappings = mappings
            reference_participants = participants
        elif not np.array_equal(mappings, reference_mappings) or not np.array_equal(
            participants, reference_participants
        ):
            raise AssertionError("Llama donor mappings differ across conditions")

        ordered = _ordered_condition(data, condition, participants)
        kwargs = dict(
            actions=ordered["action"].astype(np.int64),
            alpha=ordered["alpha"].astype(np.float64),
            beta_reward=ordered["beta_reward"].astype(np.float64),
            beta_kernel=ordered["beta_kernel"].astype(np.float64),
            q_scale=q_scale,
            decision_noise=ordered["decision_noise"].astype(np.float64),
        )
        intact = oracle_trial_nll(rewards=ordered["reward"].astype(float), **kwargs)
        stored_probability = np.take_along_axis(
            ordered["choice_probability"], ordered["action"][..., None], axis=2
        ).squeeze(-1)
        stored = -np.log(np.clip(stored_probability, 1e-12, 1.0))
        max_error = float(np.max(np.abs(intact - stored)))
        if max_error > 2e-6:
            raise AssertionError(f"oracle replay mismatch for {condition}: {max_error}")

        donor = np.empty((len(mappings), *intact.shape), dtype=np.float32)
        for repeat, mapping in enumerate(mappings):
            donor[repeat] = oracle_trial_nll(
                rewards=ordered["reward"][mapping].astype(float), **kwargs
            )
        intact_session = intact.mean(axis=1)
        donor_session = donor.mean(axis=(0, 2))
        delta = donor_session - intact_session
        relative = delta / intact_session
        reports[condition] = {
            "condition": condition,
            "intervention": "matched-trial deranged-donor chosen rewards",
            "n_sessions": int(len(participants)),
            "n_trials": int(intact.shape[1]),
            "n_derangements": int(len(mappings)),
            "self_donors": int(np.sum(mappings == np.arange(len(participants)))),
            "intact_nll": float(intact_session.mean()),
            "donor_nll": float(donor_session.mean()),
            "delta_nll": float(delta.mean()),
            "delta_nll_ci95": _bootstrap_ci(
                delta, bootstrap_seed + 10 * condition_index, n_bootstrap
            ),
            "relative_delta": float(relative.mean()),
            "relative_delta_ci95": _bootstrap_ci(
                relative, bootstrap_seed + 10 * condition_index + 1, n_bootstrap
            ),
            "intact_replay_max_abs_nll_error": max_error,
        }
        arrays[condition] = {
            "participants": participants,
            "donor_mappings": mappings,
            "intact_nll": intact.astype(np.float32),
            "donor_nll": donor,
            "session_delta": delta.astype(np.float32),
        }

    contrasts: dict[str, Any] = {}
    for left, right in (("reward_dominant", "balanced"), ("reward_dominant", "choice_dominant"), ("balanced", "choice_dominant")):
        difference = arrays[left]["session_delta"] - arrays[right]["session_delta"]
        key = f"{left}_minus_{right}"
        contrasts[key] = {
            "mean": float(difference.mean()),
            "ci95": _bootstrap_ci(difference, bootstrap_seed + 100 + len(contrasts), n_bootstrap),
        }
    report = {
        "schema_version": 1,
        "dataset": str(dataset.resolve()),
        "donor_mapping_source": str(donor_root.resolve()),
        "q_pairwise_scale": q_scale,
        "conditions": reports,
        "paired_condition_contrasts": contrasts,
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--donor-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    report, arrays = analyze(
        args.dataset, args.donor_root, args.split, args.n_bootstrap
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for condition, values in arrays.items():
        np.savez_compressed(args.output_dir / f"{condition}.npz", **values)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
