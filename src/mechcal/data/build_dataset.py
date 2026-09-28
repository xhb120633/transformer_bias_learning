"""CLI for building sharded offline PRL transcript datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mechcal.generators.partial_feedback_prl import (
    PRLConfig,
    TOKEN_VOCAB,
    config_to_dict,
    generate_sessions,
)


def _condition_name(theta: float) -> str:
    return f"theta_{theta:.6f}".replace(".", "p")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    return raw


def _make_prl_config(raw: dict[str, Any], theta: float) -> PRLConfig:
    generator = dict(raw.get("generator", {}))
    generator["theta"] = theta
    if "reversal_centers" in generator:
        generator["reversal_centers"] = tuple(generator["reversal_centers"])
    valid = {field.name for field in fields(PRLConfig)}
    unknown = set(generator) - valid
    if unknown:
        raise ValueError(f"unknown generator fields: {sorted(unknown)}")
    return PRLConfig(**generator)


def _write_shards(data: dict[str, np.ndarray], output: Path, split: str, shard_size: int) -> list[str]:
    files: list[str] = []
    n_sessions = len(data["session_id"])
    for shard_index, start in enumerate(range(0, n_sessions, shard_size)):
        stop = min(start + shard_size, n_sessions)
        filename = f"{split}_{shard_index:03d}.npz"
        np.savez_compressed(output / filename, **{key: value[start:stop] for key, value in data.items()})
        files.append(filename)
    return files


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def build_dataset(config_path: Path, output_root: Path) -> list[Path]:
    raw = _load_config(config_path)
    splits = raw.get("splits", {"train": 2000, "val": 500, "test": 500})
    theta_values = [float(value) for value in raw.get("theta_values", [0, math.pi / 4, math.pi / 2])]
    base_seed = int(raw.get("seed", 2026))
    shard_size = int(raw.get("shard_size", 1000))
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("splits must contain exactly train, val, and test")

    created: list[Path] = []
    for condition_index, theta in enumerate(theta_values):
        config = _make_prl_config(raw, theta)
        condition_dir = output_root / _condition_name(theta)
        if condition_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing directory: {condition_dir}")
        condition_dir.mkdir(parents=True)
        split_metadata: dict[str, Any] = {}
        seen_ids: set[str] = set()

        for split_index, split in enumerate(("train", "val", "test")):
            n_sessions = int(splits[split])
            split_seed = base_seed + condition_index * 100_000 + split_index * 10_000
            data = generate_sessions(config, n_sessions=n_sessions, seed=split_seed)
            ids = [str(value) for value in data["session_id"]]
            if seen_ids.intersection(ids):
                raise RuntimeError("session IDs overlap across splits")
            seen_ids.update(ids)
            files = _write_shards(data, condition_dir, split, shard_size)
            split_metadata[split] = {
                "n_sessions": n_sessions,
                "seed": split_seed,
                "session_id_sha256": _hash_ids(ids),
                "files": files,
            }

        metadata = {
            "schema_version": 1,
            "task": "two_armed_partial_feedback_prl",
            "training_protocol": "offline_full_session_causal_lm",
            "feedback": "chosen_only",
            "loss": "choice_tokens_only",
            "tokens_per_trial": 2,
            "sequence_length": 1 + 2 * config.n_trials,
            "token_vocab": TOKEN_VOCAB,
            "generator": config_to_dict(config),
            "noise_sources": {
                "reward": "Bernoulli outcome noise",
                "choice": "Bernoulli sampling from the policy probability",
                "lapse": "uniform-choice mixture with recorded session-level rate",
                "participant_heterogeneity": "recorded alpha and scale variation",
                "reversal_timing": "session-level jitter around configured centers",
            },
            "splits": split_metadata,
            "latent_array_warning": "q_value, delta_q, z_q, choice_kernel, policy_logit, and choice_probability are evaluation-only and must not be model inputs",
        }
        with (condition_dir / "metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        created.append(condition_dir)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    created = build_dataset(args.config, args.output)
    for path in created:
        print(path)


if __name__ == "__main__":
    main()

