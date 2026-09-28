"""Build the primary pooled restless-bandit dataset with unique schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mechcal.data.build_restless_dataset import (
    _conditions,
    _load_config,
    _make_generator_config,
    _write_shards,
)
from mechcal.generators.four_armed_restless import (
    calibrate_q_pairwise_scale,
    config_to_dict,
    generate_potential_rewards,
    generate_reward_schedule,
    generate_sessions,
    token_vocab,
)


def _expand_seed_spec(spec: Any) -> list[int]:
    if isinstance(spec, list):
        return [int(value) for value in spec]
    if isinstance(spec, dict) and set(spec) == {"start", "count"}:
        start = int(spec["start"])
        count = int(spec["count"])
        if count < 1:
            raise ValueError("schedule seed count must be positive")
        return list(range(start, start + count))
    raise ValueError("seed spec must be a list or {start, count}")


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _concat_and_shuffle(
    condition_data: list[dict[str, np.ndarray]], seed: int
) -> dict[str, np.ndarray]:
    pooled = {
        key: np.concatenate([data[key] for data in condition_data], axis=0)
        for key in condition_data[0]
    }
    permutation = np.random.default_rng(seed).permutation(len(pooled["session_id"]))
    return {key: value[permutation] for key, value in pooled.items()}


def build_dataset(config_path: Path, output_root: Path) -> Path:
    raw = _load_config(config_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    config = _make_generator_config(raw)
    config.validate()
    conditions = _conditions(raw)
    splits = {name: int(value) for name, value in raw["base_participants"].items()}
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("base_participants must contain train, val, and test")
    seed_specs = raw["schedule_seeds"]
    required = {"calibration", "train", "val", "test"}
    if set(seed_specs) != required:
        raise ValueError(f"schedule_seeds must contain {sorted(required)}")
    schedule_seeds = {name: _expand_seed_spec(spec) for name, spec in seed_specs.items()}
    for split in ("train", "val", "test"):
        if len(schedule_seeds[split]) != splits[split]:
            raise ValueError(f"{split} requires one unique schedule seed per base participant")
    flattened = [seed for values in schedule_seeds.values() for seed in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("all calibration and split schedule seeds must be unique")

    base_seed = int(raw.get("seed", 2027))
    shard_size = int(raw.get("shard_size", 250))
    calibration_participants = int(raw.get("calibration_participants", 256))
    q_scale = calibrate_q_pairwise_scale(
        config,
        schedule_seeds["calibration"],
        calibration_participants,
        base_seed + 700_000,
    )
    output_root.mkdir(parents=True)
    split_metadata: dict[str, Any] = {}

    for split_index, split in enumerate(("train", "val", "test")):
        split_schedule_seeds = schedule_seeds[split]
        latent_schedules = {
            schedule_seed: generate_reward_schedule(config, schedule_seed)
            for schedule_seed in split_schedule_seeds
        }
        potential_rewards = {
            schedule_seed: generate_potential_rewards(
                config, latent_schedules[schedule_seed], schedule_seed
            )
            for schedule_seed in split_schedule_seeds
        }
        participant_seed = base_seed + split_index * 10_000
        generated_by_condition = []
        for condition_index, condition in enumerate(conditions):
            data = generate_sessions(
                config,
                condition,
                latent_schedules,
                splits[split],
                participant_seed,
                q_scale,
                potential_rewards=potential_rewards,
            )
            base_ids = data["session_id"].copy()
            data["base_participant_id"] = base_ids
            data["session_id"] = np.asarray(
                [f"{base_id}_{condition.name}" for base_id in base_ids]
            )
            data["condition_index"] = np.full(
                splits[split], condition_index, dtype=np.int8
            )
            data["condition_name"] = np.full(
                splits[split], condition.name
            )
            generated_by_condition.append(data)

        pooled = _concat_and_shuffle(
            generated_by_condition, base_seed + 500_000 + split_index
        )
        files = _write_shards(pooled, output_root, split, shard_size)
        split_metadata[split] = {
            "n_base_participants": splits[split],
            "n_transcripts": len(pooled["session_id"]),
            "n_unique_schedules": len(np.unique(pooled["schedule_seed"])),
            "participant_seed": participant_seed,
            "schedule_seed_range": [
                min(split_schedule_seeds),
                max(split_schedule_seeds),
            ],
            "schedule_seed_sha256": _hash_ids(
                [str(value) for value in split_schedule_seeds]
            ),
            "files": files,
        }

    metadata = {
        "schema_version": 3,
        "task": "four_armed_partial_feedback_restless_bandit",
        "training_protocol": "pooled_offline_full_session_causal_lm",
        "feedback": "chosen_only",
        "loss": "choice_tokens_only",
        "pooling": {
            "conditions_are_pooled": True,
            "condition_is_model_input": False,
            "schedule_id_is_model_input": False,
            "base_participant_id_is_model_input": False,
            "pairing": "Each base participant has three condition transcripts sharing the exact latent schedule, potential rewards, and non-mechanism parameters.",
        },
        "tokens_per_trial": 2,
        "sequence_length": 1 + 2 * config.n_trials,
        "conditions": [
            {
                "index": index,
                "name": condition.name,
                "lambda_range": [condition.lambda_low, condition.lambda_high],
            }
            for index, condition in enumerate(conditions)
        ],
        "generator": config_to_dict(config),
        "q_pairwise_scale": q_scale,
        "q_scale_calibration": {
            "definition": "mean absolute pairwise Q difference after 10 trials under a random policy",
            "n_unique_schedule_seeds": len(schedule_seeds["calibration"]),
            "n_participants": calibration_participants,
            "seed": base_seed + 700_000,
        },
        "token_vocab": token_vocab(config.reward_min, config.reward_max),
        "splits": split_metadata,
        "model_input_allowlist": ["tokens", "choice_target_mask"],
        "audit_only_fields": [
            "condition_index",
            "condition_name",
            "schedule_seed",
            "base_participant_id",
            "latent_reward_mean",
            "potential_reward",
            "optimal_action",
            "q_value",
            "z_q",
            "choice_kernel",
            "policy_logit",
            "choice_probability",
            "alpha",
            "policy_scale",
            "lapse",
            "decision_noise",
            "lambda_choice",
            "kernel_sign",
            "beta_reward",
            "beta_kernel",
        ],
    }
    with (output_root / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_dataset(args.config, args.output))


if __name__ == "__main__":
    main()
