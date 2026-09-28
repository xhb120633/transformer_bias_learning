"""Controlled spatial bandit: GP reward exploitation mixed with local search.

Inspired by Wu et al. (2018), not a reproduction of its GP-UCB model.
Positions are row-major (row, column), Manhattan distance in grid cells.
Each round starts with one free observation and resets the learner.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class SpatialConfig:
    grid_size: int = 5
    rounds: int = 8
    trials: int = 20
    environment_length: float = 1.2
    reward_mean: float = 50.0
    reward_sd: float = 15.0
    observation_sd: float = 2.0
    lapse: float = 0.1


def coordinates(size):
    return np.array([(r, c) for r in range(size) for c in range(size)], dtype=float)


def rbf(coords, length):
    if length <= 0:
        raise ValueError('length must be positive')
    return np.exp(-np.square(coords[:, None] - coords[None]).sum(-1)/(2*length**2))


def softmax(x):
    y = np.exp(x - np.max(x))
    return y/y.sum()


class GPLearner:
    def __init__(self, coords, length, noise_variance):
        self.mean = np.zeros(len(coords))
        self.cov = rbf(coords, length)
        self.noise_variance = noise_variance

    def observe(self, arm, standardized_reward):
        column = self.cov[:, arm].copy()
        denominator = self.cov[arm, arm] + self.noise_variance
        self.mean += column/denominator * (standardized_reward-self.mean[arm])
        self.cov -= np.outer(column, column)/denominator
        self.cov = (self.cov+self.cov.T)/2


def policy(mean, coords, last, weight, reward_temperature, local_temperature, lapse):
    if not 0 <= weight <= 1 or not 0 <= lapse <= 1:
        raise ValueError('weight and lapse must lie in [0, 1]')
    if min(reward_temperature, local_temperature) <= 0:
        raise ValueError('temperatures must be positive')
    reward_p = softmax(mean/reward_temperature)
    local_p = softmax(-np.abs(coords-coords[last]).sum(-1)/local_temperature)
    return (1-lapse)*(weight*reward_p+(1-weight)*local_p)+lapse/len(coords)


def learner(config, params):
    return GPLearner(coordinates(config.grid_size), params['gp_length'],
                     (config.observation_sd/config.reward_sd)**2)


def replay(config, params, weight, cue_arm, cue_reward, actions, rewards):
    """One-step predictions with observed choices fixed, never sampled anew."""
    gp = learner(config, params)
    coords = coordinates(config.grid_size)
    gp.observe(cue_arm, (cue_reward-config.reward_mean)/config.reward_sd)
    last = cue_arm
    predictions = []
    for arm, reward in zip(actions, rewards):
        predictions.append(policy(gp.mean, coords, last, weight,
                                  params['reward_temperature'], params['local_temperature'], config.lapse))
        gp.observe(int(arm), (reward-config.reward_mean)/config.reward_sd)
        last = int(arm)
    return np.array(predictions)


def simulate_subject(config, params, weight, seed):
    """Independent streams keep landscapes/noise/choice uniforms matched over w."""
    coords = coordinates(config.grid_size)
    arms = len(coords)
    chol = np.linalg.cholesky(rbf(coords, config.environment_length)+1e-10*np.eye(arms))
    fields = {k: [] for k in ('actions','rewards','probabilities','landscape','cue_arm','cue_reward','environment_seed')}
    for round_id in range(config.rounds):
        env_seed = np.random.SeedSequence([seed, round_id, 0])
        env_rng = np.random.default_rng(env_seed)
        choice_rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 1]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 2]))
        landscape = config.reward_mean+config.reward_sd*(chol@env_rng.normal(size=arms))
        cue_arm = int(env_rng.integers(arms))
        cue_reward = landscape[cue_arm]+noise_rng.normal(0, config.observation_sd)
        # Potential rewards for ALL arms are audit-only; reveal only selected arm.
        noise = noise_rng.normal(0, config.observation_sd, (config.trials, arms))
        uniforms = choice_rng.random(config.trials)
        gp = learner(config, params)
        gp.observe(cue_arm, (cue_reward-config.reward_mean)/config.reward_sd)
        last = cue_arm
        actions, rewards, probabilities = [], [], []
        for t in range(config.trials):
            p = policy(gp.mean, coords, last, weight, params['reward_temperature'],
                       params['local_temperature'], config.lapse)
            arm = min(int(np.searchsorted(np.cumsum(p), uniforms[t], side='right')), arms-1)
            reward = float(landscape[arm]+noise[t, arm])
            actions.append(arm); rewards.append(reward); probabilities.append(p)
            gp.observe(arm, (reward-config.reward_mean)/config.reward_sd)
            last = arm
        for k,v in dict(actions=actions,rewards=rewards,probabilities=probabilities,
                        landscape=landscape,cue_arm=cue_arm,cue_reward=cue_reward,
                        environment_seed=env_seed.generate_state(4)).items():
            fields[k].append(v)
    return {k:np.asarray(v) for k,v in fields.items()}


def observable_record(config, participant, episode):
    """No latent map, policy probabilities, parameters, or condition in prompt."""
    coords = coordinates(config.grid_size).astype(int)
    rounds = []
    for b in range(config.rounds):
        rounds.append(dict(round=b,initial_position=coords[episode['cue_arm'][b]].tolist(),
                           initial_reward=float(episode['cue_reward'][b]),
                           choices=coords[episode['actions'][b]].tolist(),
                           rewards=episode['rewards'][b].tolist()))
    return dict(participant=participant,grid_size=config.grid_size,rounds=rounds)
