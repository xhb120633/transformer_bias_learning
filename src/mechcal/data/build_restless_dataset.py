"""Build split-safe four-armed restless-bandit transcript datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mechcal.generators.four_armed_restless import (
    MechanismCondition,
    RestlessConfig,
    calibrate_q_pairwise_scale,
    config_to_dict,
    generate_reward_schedule,
    generate_sessions,
    token_vocab,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    return raw


def _make_generator_config(raw: dict[str, Any]) -> RestlessConfig:
    values = dict(raw.get("generator", {}))
    if "initial_means" in values:
        values["initial_means"] = tuple(values["initial_means"])
    valid = {field.name for field in fields(RestlessConfig)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown generator fields: {sorted(unknown)}")
    return RestlessConfig(**values)


def _conditions(raw: dict[str, Any]) -> list[MechanismCondition]:
    conditions = []
    for item in raw.get("conditions", []):
        conditions.append(
            MechanismCondition(
                name=str(item["name"]),
                lambda_low=float(item["lambda_range"][0]),
                lambda_high=float(item["lambda_range"][1]),
            )
        )
    if not conditions:
        raise ValueError("at least one condition is required")
    for condition in conditions:
        condition.validate()
    return conditions


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _write_shards(
    data: dict[str, np.ndarray], output: Path, split: str, shard_size: int
) -> list[str]:
    files = []
    for shard_index, start in enumerate(range(0, len(data["session_id"]), shard_size)):
        stop = min(start + shard_size, len(data["session_id"]))
        filename = f"{split}_{shard_index:03d}.npz"
        np.savez_compressed(
            output / filename,
            **{key: value[start:stop] for key, value in data.items()},
        )
        files.append(filename)
    return files


def build_dataset(config_path: Path, output_root: Path) -> list[Path]:
    raw = _load_config(config_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    config = _make_generator_config(raw)
    config.validate()
    conditions = _conditions(raw)
    splits = raw["splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("splits must contain exactly train, val, and test")
    schedule_seed_config = raw["schedule_seeds"]
    required_seed_sets = {"calibration", "train", "val", "test"}
    if set(schedule_seed_config) != required_seed_sets:
        raise ValueError(f"schedule_seeds must contain {sorted(required_seed_sets)}")
    seed_sets = {
        name: [int(value) for value in values]
        for name, values in schedule_seed_config.items()
    }
    flattened = [seed for values in seed_sets.values() for seed in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("schedule seeds must be disjoint across calibration and splits")

    base_seed = int(raw.get("seed", 2027))
    shard_size = int(raw.get("shard_size", 250))
    calibration_participants = int(raw.get("calibration_participants", 256))
    q_pairwise_scale = calibrate_q_pairwise_scale(
        config,
        seed_sets["calibration"],
        calibration_participants,
        base_seed + 700_000,
    )
    schedules_by_split = {
        split: {
            schedule_seed: generate_reward_schedule(config, schedule_seed)
            for schedule_seed in seed_sets[split]
        }
        for split in ("train", "val", "test")
    }

    output_root.mkdir(parents=True)
    created = []
    for condition in conditions:
        condition_dir = output_root / condition.name
        condition_dir.mkdir()
        split_metadata: dict[str, Any] = {}
        seen_session_ids: set[str] = set()
        for split_index, split in enumerate(("train", "val", "test")):
            split_seed = base_seed + split_index * 10_000
            data = generate_sessions(
                config,
                condition,
                schedules_by_split[split],
                int(splits[split]),
                split_seed,
                q_pairwise_scale,
            )
            ids = [str(value) for value in data["session_id"]]
            if seen_session_ids.intersection(ids):
                raise RuntimeError("session IDs overlap across splits")
            seen_session_ids.update(ids)
            files = _write_shards(data, condition_dir, split, shard_size)
            split_metadata[split] = {
                "n_sessions": int(splits[split]),
                "participant_seed": split_seed,
                "schedule_seeds": seed_sets[split],
                "session_id_sha256": _hash_ids(ids),
                "files": files,
            }

        metadata = {
            "schema_version": 2,
            "task": "four_armed_partial_feedback_restless_bandit",
            "training_protocol": "offline_full_session_causal_lm",
            "feedback": "chosen_only",
            "loss": "choice_tokens_only",
            "tokens_per_trial": 2,
            "sequence_length": 1 + 2 * config.n_trials,
            "condition": {
                "name": condition.name,
                "lambda_range": [condition.lambda_low, condition.lambda_high],
                "lambda_definition": "abs(beta_kernel) / (beta_reward + abs(beta_kernel))",
            },
            "generator": config_to_dict(config),
            "q_pairwise_scale": q_pairwise_scale,
            "q_scale_calibration": {
                "definition": "mean absolute pairwise Q difference after 10 trials under a random policy",
                "schedule_seeds": seed_sets["calibration"],
                "n_participants": calibration_participants,
                "seed": base_seed + 700_000,
            },
            "token_vocab": token_vocab(config.reward_min, config.reward_max),
            "splits": split_metadata,
            "latent_array_warning": "latent_reward_mean, optimal_action, q_value, z_q, choice_kernel, policy_logit, choice_probability, and participant parameters are audit-only and must not be model inputs",
        }
        with (condition_dir / "metadata.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        created.append(condition_dir)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for path in build_dataset(args.config, args.output):
        print(path)


if __name__ == "__main__":
    main()

