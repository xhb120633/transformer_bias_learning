"""Four-armed chosen-only restless-bandit transcript generator.

The environment follows fixed decaying Gaussian random-walk schedules.  Each
synthetic participant has a fixed learning rate, policy scale, and signed
choice-kernel weight. Mechanism conditions vary only the absolute
reward-versus-kernel weighting in the choice policy. The policy can use either
the legacy softmax-plus-lapse rule or a fixed-noise epsilon-greedy rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


BOS_TOKEN_ID = 0
CHOICE_TOKEN_OFFSET = 1
REWARD_TOKEN_OFFSET = 5


def token_vocab(reward_min: int = 0, reward_max: int = 100) -> dict[str, int]:
    vocab = {"BOS": BOS_TOKEN_ID}
    vocab.update({f"CHOICE_{arm}": CHOICE_TOKEN_OFFSET + arm for arm in range(4)})
    vocab.update(
        {
            f"REWARD_{reward}": REWARD_TOKEN_OFFSET + reward - reward_min
            for reward in range(reward_min, reward_max + 1)
        }
    )
    return vocab


@dataclass(frozen=True)
class RestlessConfig:
    n_arms: int = 4
    n_trials: int = 200
    reward_center: float = 50.0
    ou_decay: float = 0.9836
    process_noise_sd: float = 2.8
    observation_noise_sd: float = 4.0
    reward_min: int = 0
    reward_max: int = 100
    initial_means: tuple[float, ...] = (35.0, 45.0, 55.0, 65.0)
    alpha_mean: float = 0.25
    alpha_concentration: float = 24.0
    policy_scale_median: float = 2.25
    policy_scale_log_sd: float = 0.15
    lapse_mean: float = 0.05
    lapse_concentration: float = 100.0
    policy_mode: str = "softmax_lapse"
    decision_noise: float = 0.10
    kernel_positive_probability: float = 0.5

    def validate(self) -> None:
        if self.n_arms != 4:
            raise ValueError("the current token schema requires exactly four arms")
        if self.n_trials < 2:
            raise ValueError("n_trials must be at least 2")
        if len(self.initial_means) != self.n_arms:
            raise ValueError("initial_means must contain one value per arm")
        if not 0 < self.ou_decay < 1:
            raise ValueError("ou_decay must be in (0, 1)")
        if self.process_noise_sd <= 0 or self.observation_noise_sd < 0:
            raise ValueError("noise scales are invalid")
        if self.reward_min >= self.reward_max:
            raise ValueError("reward_min must be less than reward_max")
        if not self.reward_min <= min(self.initial_means) <= max(self.initial_means) <= self.reward_max:
            raise ValueError("initial means must lie inside the reward range")
        if not 0 < self.alpha_mean < 1 or self.alpha_concentration <= 0:
            raise ValueError("alpha distribution is invalid")
        if self.policy_scale_median <= 0 or self.policy_scale_log_sd < 0:
            raise ValueError("policy scale distribution is invalid")
        if not 0 < self.lapse_mean < 1 or self.lapse_concentration <= 0:
            raise ValueError("lapse distribution is invalid")
        if self.policy_mode not in {"softmax_lapse", "epsilon_greedy"}:
            raise ValueError("policy_mode must be softmax_lapse or epsilon_greedy")
        if not 0 <= self.decision_noise < 1:
            raise ValueError("decision_noise must be in [0, 1)")
        if not 0 <= self.kernel_positive_probability <= 1:
            raise ValueError("kernel_positive_probability must be in [0, 1]")


@dataclass(frozen=True)
class MechanismCondition:
    name: str
    lambda_low: float
    lambda_high: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("condition name cannot be empty")
        if not 0 <= self.lambda_low <= self.lambda_high <= 1:
            raise ValueError("lambda range must lie in [0, 1]")


def _beta_draw(mean: float, concentration: float, rng: np.random.Generator) -> float:
    return float(rng.beta(mean * concentration, (1.0 - mean) * concentration))


def generate_reward_schedule(config: RestlessConfig, seed: int) -> np.ndarray:
    """Generate a fixed latent mean-payoff trajectory with shape [T, 4]."""

    config.validate()
    rng = np.random.default_rng(seed)
    current = np.asarray(config.initial_means, dtype=np.float64).copy()
    rng.shuffle(current)
    schedule = np.empty((config.n_trials, config.n_arms), dtype=np.float32)
    for trial in range(config.n_trials):
        schedule[trial] = current
        innovation = rng.normal(0.0, config.process_noise_sd, size=config.n_arms)
        current = config.reward_center + config.ou_decay * (current - config.reward_center) + innovation
        current = np.clip(current, config.reward_min, config.reward_max)
    return schedule


def generate_potential_rewards(
    config: RestlessConfig, latent_schedule: np.ndarray, seed: int
) -> np.ndarray:
    """Generate all four potential rewards before any participant acts."""

    if latent_schedule.shape != (config.n_trials, config.n_arms):
        raise ValueError("latent schedule has the wrong shape")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
    rewards = np.rint(
        rng.normal(latent_schedule, config.observation_noise_sd)
    )
    return np.clip(rewards, config.reward_min, config.reward_max).astype(np.int16)


def tokenize_session(
    actions: np.ndarray, rewards: np.ndarray, reward_min: int = 0, reward_max: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    if actions.ndim != 1 or rewards.shape != actions.shape:
        raise ValueError("actions and rewards must be one-dimensional and aligned")
    if np.any((actions < 0) | (actions >= 4)):
        raise ValueError("actions must be in [0, 3]")
    if np.any((rewards < reward_min) | (rewards > reward_max)):
        raise ValueError("rewards are outside the tokenized range")
    tokens = np.empty(1 + 2 * len(actions), dtype=np.int16)
    target_mask = np.zeros_like(tokens, dtype=bool)
    tokens[0] = BOS_TOKEN_ID
    tokens[1::2] = CHOICE_TOKEN_OFFSET + actions
    tokens[2::2] = REWARD_TOKEN_OFFSET + rewards - reward_min
    target_mask[1::2] = True
    return tokens, target_mask


def calibrate_q_pairwise_scale(
    config: RestlessConfig,
    schedule_seeds: Sequence[int],
    n_participants: int,
    seed: int,
    burn_in: int = 10,
) -> float:
    """Freeze a reference Q scale using random-policy calibration sessions.

    The scale is the mean absolute pairwise Q difference after burn-in, so a
    unit reward coefficient corresponds to a typical pairwise logit contrast
    comparable to the one-step kernel contrast of one.
    """

    config.validate()
    if not schedule_seeds or n_participants < 1:
        raise ValueError("calibration requires schedules and participants")
    schedules = {seed_: generate_reward_schedule(config, seed_) for seed_ in schedule_seeds}
    roots = np.random.SeedSequence(seed).spawn(n_participants)
    differences: list[np.ndarray] = []
    arm_pairs = np.asarray(list(combinations(range(config.n_arms), 2)), dtype=np.int8)
    for index, root in enumerate(roots):
        parameter_seed, observation_seed, choice_seed = root.spawn(3)
        parameter_rng = np.random.default_rng(parameter_seed)
        observation_rng = np.random.default_rng(observation_seed)
        choice_rng = np.random.default_rng(choice_seed)
        alpha = _beta_draw(config.alpha_mean, config.alpha_concentration, parameter_rng)
        schedule = schedules[int(schedule_seeds[index % len(schedule_seeds)])]
        q = np.full(config.n_arms, config.reward_center, dtype=np.float64)
        session_differences = []
        for trial in range(config.n_trials):
            if trial >= burn_in:
                session_differences.append(
                    np.abs(q[arm_pairs[:, 0]] - q[arm_pairs[:, 1]])
                )
            action = int(choice_rng.integers(0, config.n_arms))
            reward = float(
                np.clip(
                    np.rint(
                        observation_rng.normal(
                            float(schedule[trial, action]), config.observation_noise_sd
                        )
                    ),
                    config.reward_min,
                    config.reward_max,
                )
            )
            q[action] += alpha * (reward - q[action])
        differences.append(np.concatenate(session_differences))
    scale = float(np.mean(np.concatenate(differences)))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Q-scale calibration produced an invalid value")
    return scale


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _generate_one(
    config: RestlessConfig,
    condition: MechanismCondition,
    schedule: np.ndarray,
    potential_rewards: np.ndarray | None,
    q_pairwise_scale: float,
    root_seed: np.random.SeedSequence,
) -> dict[str, np.ndarray | float | int]:
    parameter_seed, observation_seed, choice_seed = root_seed.spawn(3)
    parameter_rng = np.random.default_rng(parameter_seed)
    observation_rng = np.random.default_rng(observation_seed)
    choice_rng = np.random.default_rng(choice_seed)

    alpha = _beta_draw(config.alpha_mean, config.alpha_concentration, parameter_rng)
    policy_scale = float(
        config.policy_scale_median
        * np.exp(parameter_rng.normal(0.0, config.policy_scale_log_sd))
    )
    lapse = (
        _beta_draw(config.lapse_mean, config.lapse_concentration, parameter_rng)
        if config.policy_mode == "softmax_lapse"
        else config.decision_noise
    )
    lambda_choice = float(parameter_rng.uniform(condition.lambda_low, condition.lambda_high))
    kernel_sign = 1 if parameter_rng.random() < config.kernel_positive_probability else -1
    beta_reward = policy_scale * (1.0 - lambda_choice)
    beta_kernel = kernel_sign * policy_scale * lambda_choice

    actions = np.empty(config.n_trials, dtype=np.int8)
    rewards = np.empty(config.n_trials, dtype=np.int16)
    q_values = np.empty((config.n_trials, config.n_arms), dtype=np.float32)
    z_q = np.empty_like(q_values)
    choice_kernel = np.zeros_like(q_values)
    policy_logits = np.empty_like(q_values)
    choice_probabilities = np.empty_like(q_values)
    optimal_action = np.argmax(schedule, axis=1).astype(np.int8)

    q = np.full(config.n_arms, config.reward_center, dtype=np.float64)
    previous_action: int | None = None
    for trial in range(config.n_trials):
        current_z_q = (q - np.mean(q)) / q_pairwise_scale
        kernel = np.zeros(config.n_arms, dtype=np.float64)
        if previous_action is not None:
            kernel[previous_action] = 1.0
        logits = beta_reward * current_z_q + beta_kernel * kernel
        if config.policy_mode == "softmax_lapse":
            structured_probability = _softmax(logits)
            probability = (
                (1.0 - lapse) * structured_probability + lapse / config.n_arms
            )
        else:
            # Split the greedy mass across exact ties. This keeps the initial
            # all-equal Q state unbiased instead of favoring arm zero.
            winners = np.isclose(logits, np.max(logits), rtol=0.0, atol=1e-12)
            structured_probability = winners.astype(np.float64) / winners.sum()
            probability = (
                (1.0 - config.decision_noise) * structured_probability
                + config.decision_noise / config.n_arms
            )
        action = int(choice_rng.choice(config.n_arms, p=probability))
        if potential_rewards is None:
            reward = int(
                np.clip(
                    np.rint(
                        observation_rng.normal(
                            float(schedule[trial, action]), config.observation_noise_sd
                        )
                    ),
                    config.reward_min,
                    config.reward_max,
                )
            )
        else:
            reward = int(potential_rewards[trial, action])

        q_values[trial] = q
        z_q[trial] = current_z_q
        choice_kernel[trial] = kernel
        policy_logits[trial] = logits
        choice_probabilities[trial] = probability
        actions[trial] = action
        rewards[trial] = reward

        q[action] += alpha * (reward - q[action])
        previous_action = action

    tokens, choice_target_mask = tokenize_session(
        actions, rewards, config.reward_min, config.reward_max
    )
    return {
        "action": actions,
        "reward": rewards,
        "tokens": tokens,
        "choice_target_mask": choice_target_mask,
        "latent_reward_mean": schedule.astype(np.float32),
        "potential_reward": (
            potential_rewards.astype(np.int16)
            if potential_rewards is not None
            else np.full((config.n_trials, config.n_arms), -1, dtype=np.int16)
        ),
        "optimal_action": optimal_action,
        "q_value": q_values,
        "z_q": z_q,
        "choice_kernel": choice_kernel,
        "policy_logit": policy_logits,
        "choice_probability": choice_probabilities,
        "alpha": alpha,
        "policy_scale": policy_scale,
        "lapse": lapse,
        "decision_noise": config.decision_noise,
        "lambda_choice": lambda_choice,
        "kernel_sign": kernel_sign,
        "beta_reward": beta_reward,
        "beta_kernel": beta_kernel,
    }


def generate_sessions(
    config: RestlessConfig,
    condition: MechanismCondition,
    schedules: Mapping[int, np.ndarray],
    n_sessions: int,
    seed: int,
    q_pairwise_scale: float,
    potential_rewards: Mapping[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    config.validate()
    condition.validate()
    if n_sessions < 1 or not schedules or q_pairwise_scale <= 0:
        raise ValueError("invalid generation arguments")
    schedule_seeds = np.asarray(sorted(schedules), dtype=np.int64)
    assignments = np.resize(schedule_seeds, n_sessions)
    assignment_rng = np.random.default_rng(seed + 99173)
    assignment_rng.shuffle(assignments)
    roots = np.random.SeedSequence(seed).spawn(n_sessions)
    sessions = [
        _generate_one(
            config,
            condition,
            schedules[int(schedule_seed)],
            (
                potential_rewards[int(schedule_seed)]
                if potential_rewards is not None
                else None
            ),
            q_pairwise_scale,
            root,
        )
        for schedule_seed, root in zip(assignments, roots)
    ]
    result = {
        key: np.stack([np.asarray(session[key]) for session in sessions])
        for key in sessions[0]
    }
    result["schedule_seed"] = assignments
    result["session_id"] = np.asarray(
        [f"seed{seed:010d}_participant{i:07d}" for i in range(n_sessions)]
    )
    return result


def config_to_dict(config: RestlessConfig) -> dict[str, Any]:
    result = dict(config.__dict__)
    result["initial_means"] = list(config.initial_means)
    return result
