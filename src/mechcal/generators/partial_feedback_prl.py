"""Chosen-only two-armed probabilistic reversal-learning generator.

One generated session represents one synthetic participant.  Choices are
produced by a mixture of a Rescorla-Wagner value difference and a one-step
choice kernel.  Rewards are observed and used to update only the chosen arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TOKEN_VOCAB = {
    "BOS": 0,
    "CHOICE_0": 1,
    "CHOICE_1": 2,
    "REWARD_0": 3,
    "REWARD_1": 4,
}


@dataclass(frozen=True)
class PRLConfig:
    """Configuration for one mechanism condition."""

    n_trials: int = 160
    reward_prob_high: float = 0.8
    reward_prob_low: float = 0.2
    reversal_centers: tuple[int, ...] = (40, 80, 120)
    reversal_jitter: int = 7
    min_block_length: int = 25
    alpha_mean: float = 0.25
    alpha_sd: float = 0.04
    scale_mean: float = 1.5
    scale_log_sd: float = 0.10
    lapse_mean: float = 0.05
    lapse_sd: float = 0.01
    value_scale: float = 0.25
    theta: float = np.pi / 4

    def validate(self) -> None:
        if self.n_trials < 2:
            raise ValueError("n_trials must be at least 2")
        if not 0 < self.reward_prob_low < self.reward_prob_high < 1:
            raise ValueError("reward probabilities must satisfy 0 < low < high < 1")
        if not 0 <= self.theta <= np.pi / 2:
            raise ValueError("theta must be in [0, pi/2]")
        if not 0 < self.value_scale:
            raise ValueError("value_scale must be positive")
        if not 0 <= self.lapse_mean <= 1 or self.lapse_sd < 0:
            raise ValueError("lapse parameters are invalid")
        if not 0 < self.alpha_mean < 1 or self.alpha_sd < 0:
            raise ValueError("alpha parameters are invalid")
        if self.scale_mean <= 0 or self.scale_log_sd < 0:
            raise ValueError("scale parameters are invalid")
        if self.reversal_jitter < 0 or self.min_block_length < 1:
            raise ValueError("reversal settings are invalid")
        if tuple(sorted(self.reversal_centers)) != self.reversal_centers:
            raise ValueError("reversal_centers must be strictly ordered")
        earliest = [c - self.reversal_jitter for c in self.reversal_centers]
        latest = [c + self.reversal_jitter for c in self.reversal_centers]
        if earliest and (earliest[0] < self.min_block_length):
            raise ValueError("first reversal can violate min_block_length")
        if latest and self.n_trials - latest[-1] < self.min_block_length:
            raise ValueError("last reversal can violate min_block_length")
        for left, right in zip(latest[:-1], earliest[1:]):
            if right - left < self.min_block_length:
                raise ValueError("reversal windows can violate min_block_length")


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _draw_reversals(config: PRLConfig, rng: np.random.Generator) -> np.ndarray:
    if not config.reversal_centers:
        return np.empty(0, dtype=np.int16)
    offsets = rng.integers(
        -config.reversal_jitter,
        config.reversal_jitter + 1,
        size=len(config.reversal_centers),
    )
    return np.asarray(config.reversal_centers, dtype=np.int16) + offsets.astype(np.int16)


def tokenize_session(actions: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Serialize a session and mark token positions that are choice targets.

    The returned mask is aligned to token positions.  A causal-LM trainer can
    shift both arrays by one and set labels to -100 wherever the shifted mask
    is false.
    """

    if actions.ndim != 1 or rewards.shape != actions.shape:
        raise ValueError("actions and rewards must be one-dimensional and aligned")
    n_trials = actions.shape[0]
    tokens = np.empty(1 + 2 * n_trials, dtype=np.int16)
    choice_target_mask = np.zeros_like(tokens, dtype=bool)
    tokens[0] = TOKEN_VOCAB["BOS"]
    tokens[1::2] = np.where(
        actions == 0, TOKEN_VOCAB["CHOICE_0"], TOKEN_VOCAB["CHOICE_1"]
    )
    tokens[2::2] = np.where(
        rewards == 0, TOKEN_VOCAB["REWARD_0"], TOKEN_VOCAB["REWARD_1"]
    )
    choice_target_mask[1::2] = True
    return tokens, choice_target_mask


def _generate_one(config: PRLConfig, rng: np.random.Generator) -> dict[str, np.ndarray | float | int]:
    reversals = _draw_reversals(config, rng)
    alpha = float(np.clip(rng.normal(config.alpha_mean, config.alpha_sd), 0.02, 0.98))
    scale = float(config.scale_mean * np.exp(rng.normal(0.0, config.scale_log_sd)))
    lapse = float(np.clip(rng.normal(config.lapse_mean, config.lapse_sd), 0.0, 0.25))

    actions = np.empty(config.n_trials, dtype=np.int8)
    rewards = np.empty(config.n_trials, dtype=np.int8)
    high_reward_option = np.empty(config.n_trials, dtype=np.int8)
    q_values = np.empty((config.n_trials, 2), dtype=np.float32)
    delta_q = np.empty(config.n_trials, dtype=np.float32)
    z_q = np.empty(config.n_trials, dtype=np.float32)
    choice_kernel = np.empty(config.n_trials, dtype=np.int8)
    policy_logit = np.empty(config.n_trials, dtype=np.float32)
    choice_probability = np.empty(config.n_trials, dtype=np.float32)

    q = np.asarray([0.5, 0.5], dtype=np.float64)
    previous_action: int | None = None
    high_option = int(rng.integers(0, 2))
    reversal_set = set(int(x) for x in reversals)

    for trial in range(config.n_trials):
        if trial in reversal_set:
            high_option = 1 - high_option

        kernel = 0 if previous_action is None else (1 if previous_action == 1 else -1)
        current_delta_q = float(q[1] - q[0])
        current_z_q = current_delta_q / config.value_scale
        logit = scale * (
            np.cos(config.theta) * current_z_q
            + np.sin(config.theta) * kernel
        )
        structured_probability = _sigmoid(float(logit))
        probability = (1.0 - lapse) * structured_probability + lapse * 0.5
        action = int(rng.random() < probability)
        reward_probability = (
            config.reward_prob_high if action == high_option else config.reward_prob_low
        )
        reward = int(rng.random() < reward_probability)

        q_values[trial] = q
        delta_q[trial] = current_delta_q
        z_q[trial] = current_z_q
        choice_kernel[trial] = kernel
        policy_logit[trial] = logit
        choice_probability[trial] = probability
        actions[trial] = action
        rewards[trial] = reward
        high_reward_option[trial] = high_option

        q[action] += alpha * (reward - q[action])
        previous_action = action

    tokens, choice_target_mask = tokenize_session(actions, rewards)
    return {
        "action": actions,
        "reward": rewards,
        "tokens": tokens,
        "choice_target_mask": choice_target_mask,
        "high_reward_option": high_reward_option,
        "q_value": q_values,
        "delta_q": delta_q,
        "z_q": z_q,
        "choice_kernel": choice_kernel,
        "policy_logit": policy_logit,
        "choice_probability": choice_probability,
        "reversal_positions": reversals,
        "alpha": alpha,
        "scale": scale,
        "lapse": lapse,
    }


def generate_sessions(config: PRLConfig, n_sessions: int, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic, independent synthetic-participant sessions."""

    config.validate()
    if n_sessions < 1:
        raise ValueError("n_sessions must be positive")
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(n_sessions)
    sessions = [_generate_one(config, np.random.default_rng(child)) for child in children]

    keys = sessions[0].keys()
    result: dict[str, np.ndarray] = {}
    for key in keys:
        result[key] = np.stack([np.asarray(session[key]) for session in sessions])
    result["session_id"] = np.asarray(
        [f"seed{seed:010d}_session{i:07d}" for i in range(n_sessions)]
    )
    return result


def config_to_dict(config: PRLConfig) -> dict[str, Any]:
    result = dict(config.__dict__)
    result["reversal_centers"] = list(config.reversal_centers)
    return result

