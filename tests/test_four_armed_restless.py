import json

import numpy as np

from mechcal.analysis.restless_recovery import _features
from mechcal.data.build_pooled_restless_dataset import build_dataset as build_pooled_dataset
from mechcal.data.build_restless_dataset import build_dataset
from mechcal.generators.four_armed_restless import (
    MechanismCondition,
    RestlessConfig,
    generate_potential_rewards,
    generate_reward_schedule,
    generate_sessions,
    tokenize_session,
)


def _small_config() -> RestlessConfig:
    return RestlessConfig(n_trials=24)


def test_reward_schedule_is_fixed_by_seed_and_restless():
    config = _small_config()
    first = generate_reward_schedule(config, 17)
    second = generate_reward_schedule(config, 17)
    other = generate_reward_schedule(config, 18)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (24, 4)
    assert np.any(np.diff(first, axis=0) != 0)


def test_potential_rewards_are_fixed_for_every_arm_before_choice():
    config = _small_config()
    latent = generate_reward_schedule(config, 17)
    first = generate_potential_rewards(config, latent, 17)
    second = generate_potential_rewards(config, latent, 17)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (24, 4)
    assert np.all((first >= 0) & (first <= 100))


def test_policy_weights_follow_condition_and_signed_kernel():
    config = _small_config()
    condition = MechanismCondition("balanced", 0.4, 0.6)
    schedules = {1: generate_reward_schedule(config, 1)}
    data = generate_sessions(config, condition, schedules, 200, 21, q_pairwise_scale=10.0)
    recovered_lambda = np.abs(data["beta_kernel"]) / (
        data["beta_reward"] + np.abs(data["beta_kernel"])
    )
    np.testing.assert_allclose(recovered_lambda, data["lambda_choice"], atol=1e-7)
    assert np.all((recovered_lambda >= 0.4) & (recovered_lambda <= 0.6))
    assert set(np.unique(data["kernel_sign"]).tolist()) == {-1, 1}
    assert np.all(np.sign(data["beta_kernel"]) == data["kernel_sign"])


def test_epsilon_greedy_policy_has_exactly_ten_percent_decision_noise():
    config = RestlessConfig(
        n_trials=24, policy_mode="epsilon_greedy", decision_noise=0.10
    )
    condition = MechanismCondition("balanced", 0.4, 0.6)
    schedules = {1: generate_reward_schedule(config, 1)}
    data = generate_sessions(config, condition, schedules, 12, 27, q_pairwise_scale=10.0)
    probabilities = data["choice_probability"]
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-7)
    # Trial zero is a four-way tie; subsequent unique winners receive .925 and
    # every non-winner .025. Exact later ties are allowed and split greedy mass.
    np.testing.assert_allclose(probabilities[:, 0], 0.25, atol=1e-7)
    for row in probabilities.reshape(-1, config.n_arms):
        assert any(
            np.allclose(
                np.sort(row),
                np.sort(
                    np.asarray(
                        [0.025 + (0.9 / winner_count)] * winner_count
                        + [0.025] * (config.n_arms - winner_count)
                    )
                ),
                atol=1e-7,
            )
            for winner_count in range(1, config.n_arms + 1)
        )
    np.testing.assert_allclose(data["lapse"], 0.10)


def test_only_chosen_q_value_updates():
    config = _small_config()
    condition = MechanismCondition("reward", 0.1, 0.3)
    schedules = {1: generate_reward_schedule(config, 1)}
    data = generate_sessions(config, condition, schedules, 3, 22, q_pairwise_scale=10.0)
    for session in range(3):
        for trial in range(1, config.n_trials):
            previous_action = int(data["action"][session, trial - 1])
            for arm in range(config.n_arms):
                if arm != previous_action:
                    assert data["q_value"][session, trial, arm] == data["q_value"][session, trial - 1, arm]


def test_full_session_tokenization_and_choice_mask():
    actions = np.asarray([0, 3, 1], dtype=np.int8)
    rewards = np.asarray([0, 100, 47], dtype=np.int16)
    tokens, mask = tokenize_session(actions, rewards)
    np.testing.assert_array_equal(tokens, [0, 1, 5, 4, 105, 2, 52])
    np.testing.assert_array_equal(mask, [False, True, False, True, False, True, False])


def test_raw_transcript_features_reconstruct_saved_q_state():
    config = RestlessConfig(n_trials=24, alpha_concentration=1_000_000.0)
    condition = MechanismCondition("balanced", 0.5, 0.5)
    schedules = {1: generate_reward_schedule(config, 1)}
    data = generate_sessions(config, condition, schedules, 3, 29, q_pairwise_scale=10.0)
    # Reconstruct each session with its realized alpha because participants vary.
    for index in range(3):
        z_q, kernel = _features(
            data["action"][index : index + 1],
            data["reward"][index : index + 1],
            float(data["alpha"][index]),
            10.0,
        )
        np.testing.assert_allclose(z_q[0], data["z_q"][index], atol=1e-6)
        np.testing.assert_array_equal(kernel[0], data["choice_kernel"][index])


def test_builder_uses_disjoint_schedule_splits_and_paired_participants(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 31
splits: {train: 8, val: 4, test: 4}
shard_size: 20
calibration_participants: 4
conditions:
  - {name: low, lambda_range: [0.1, 0.2]}
  - {name: high, lambda_range: [0.8, 0.9]}
schedule_seeds:
  calibration: [90]
  train: [10, 11]
  val: [20]
  test: [30]
generator:
  n_trials: 20
""".strip(),
        encoding="utf-8",
    )
    created = build_dataset(config_path, tmp_path / "out")
    assert [path.name for path in created] == ["low", "high"]
    low_metadata = json.loads((created[0] / "metadata.json").read_text(encoding="utf-8"))
    all_schedule_seeds = [
        set(low_metadata["splits"][split]["schedule_seeds"])
        for split in ("train", "val", "test")
    ]
    assert all_schedule_seeds[0].isdisjoint(all_schedule_seeds[1])
    assert all_schedule_seeds[0].isdisjoint(all_schedule_seeds[2])
    assert all_schedule_seeds[1].isdisjoint(all_schedule_seeds[2])

    with np.load(created[0] / "train_000.npz") as low, np.load(
        created[1] / "train_000.npz"
    ) as high:
        np.testing.assert_array_equal(low["session_id"], high["session_id"])
        np.testing.assert_array_equal(low["schedule_seed"], high["schedule_seed"])
        np.testing.assert_allclose(low["alpha"], high["alpha"])
        np.testing.assert_allclose(low["policy_scale"], high["policy_scale"])
        np.testing.assert_array_equal(low["kernel_sign"], high["kernel_sign"])
        assert low["tokens"].shape == (8, 41)


def test_pooled_builder_has_unique_paired_schedules_and_no_condition_input(tmp_path):
    config_path = tmp_path / "pooled.yaml"
    config_path.write_text(
        """
seed: 41
base_participants: {train: 4, val: 2, test: 2}
shard_size: 100
calibration_participants: 2
conditions:
  - {name: low, lambda_range: [0.1, 0.2]}
  - {name: middle, lambda_range: [0.4, 0.6]}
  - {name: high, lambda_range: [0.8, 0.9]}
schedule_seeds:
  calibration: {start: 900, count: 2}
  train: {start: 100, count: 4}
  val: {start: 200, count: 2}
  test: {start: 300, count: 2}
generator:
  n_trials: 20
""".strip(),
        encoding="utf-8",
    )
    output = build_pooled_dataset(config_path, tmp_path / "pooled")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pooling"]["conditions_are_pooled"] is True
    assert metadata["pooling"]["condition_is_model_input"] is False
    assert metadata["model_input_allowlist"] == ["tokens", "choice_target_mask"]
    assert metadata["splits"]["train"]["n_unique_schedules"] == 4
    assert metadata["splits"]["train"]["n_transcripts"] == 12

    with np.load(output / "train_000.npz") as shard:
        assert len(np.unique(shard["session_id"])) == 12
        assert len(np.unique(shard["schedule_seed"])) == 4
        for base_id in np.unique(shard["base_participant_id"]):
            rows = np.flatnonzero(shard["base_participant_id"] == base_id)
            assert len(rows) == 3
            np.testing.assert_array_equal(
                shard["schedule_seed"][rows],
                np.repeat(shard["schedule_seed"][rows[0]], 3),
            )
            for key in (
                "latent_reward_mean",
                "potential_reward",
                "alpha",
                "policy_scale",
                "lapse",
                "kernel_sign",
            ):
                np.testing.assert_allclose(
                    shard[key][rows],
                    np.repeat(shard[key][rows[0]][None, ...], 3, axis=0),
                )
            assert set(shard["condition_name"][rows].tolist()) == {
                "low",
                "middle",
                "high",
            }
            assert len(np.unique(shard["lambda_choice"][rows])) == 3

        chosen_potential = np.take_along_axis(
            shard["potential_reward"], shard["action"][..., None], axis=2
        ).squeeze(-1)
        np.testing.assert_array_equal(chosen_potential, shard["reward"])
