"""Export restless-bandit sessions as Centaur-style SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARM_LABELS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    paths = sorted(root.glob(f"{split}_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards found under {root}")
    keys = ("tokens", "condition_name", "base_participant_id", "schedule_seed")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            for key in keys:
                arrays[key].append(shard[key])
    return {key: np.concatenate(values) for key, values in arrays.items()}


def _cover_story(n_trials: int) -> str:
    return "\n".join(
        [
            "In this task, you repeatedly choose among four slot machines labeled A, B, C, and D.",
            "You choose a slot machine by pressing its corresponding key.",
            "After every choice, you receive a reward between 0 and 100 points from the chosen machine.",
            "The machines can change over time, so past rewards may not remain the same.",
            "Your goal is to earn as many points as possible.",
            f"You will play one game consisting of {n_trials} trials.",
            "Game 1:",
        ]
    )


def transcript_to_text(tokens: np.ndarray, vocab: dict[str, int]) -> str:
    if tokens.ndim != 1 or len(tokens) < 3 or (len(tokens) - 1) % 2:
        raise ValueError("expected BOS followed by alternating choice/reward tokens")
    if int(tokens[0]) != int(vocab["BOS"]):
        raise ValueError("transcript does not begin with BOS")
    n_trials = (len(tokens) - 1) // 2
    lines = [_cover_story(n_trials)]
    for trial in range(n_trials):
        choice_id = int(tokens[1 + 2 * trial])
        reward_id = int(tokens[2 + 2 * trial])
        choice_matches = [
            arm for arm in range(4) if choice_id == int(vocab[f"CHOICE_{arm}"])
        ]
        if len(choice_matches) != 1:
            raise ValueError(f"invalid choice token {choice_id} at trial {trial}")
        reward_matches = [
            reward
            for reward in range(101)
            if reward_id == int(vocab[f"REWARD_{reward}"])
        ]
        if len(reward_matches) != 1:
            raise ValueError(f"invalid reward token {reward_id} at trial {trial}")
        lines.append(
            f"You press <<{ARM_LABELS[choice_matches[0]]}>> and receive "
            f"{reward_matches[0]} points."
        )
    return "\n".join(lines)


def export_centaur_jsonl(dataset_root: Path, output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    metadata_path = dataset_root / "metadata.json"
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocab = {key: int(value) for key, value in metadata["token_vocab"].items()}
    expected_trials = int(metadata["generator"]["n_trials"])
    conditions = [entry["name"] for entry in metadata["conditions"]]
    output_root.mkdir(parents=True)

    files: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        files[condition] = {}
        for split in ("train", "val", "test"):
            data = _load_split(dataset_root, split)
            selected = np.flatnonzero(data["condition_name"] == condition)
            selected = selected[
                np.argsort(data["base_participant_id"][selected].astype(str))
            ]
            output_path = output_root / f"{condition}_{split}.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for index in selected:
                    text = transcript_to_text(data["tokens"][index], vocab)
                    if text.count("<<") != expected_trials or text.count(">>") != expected_trials:
                        raise AssertionError("choice delimiter count does not match trial count")
                    record = {
                        "text": text,
                        "participant": str(data["base_participant_id"][index]),
                        "condition": condition,
                        "split": split,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            files[condition][split] = {
                "path": output_path.name,
                "n_transcripts": int(len(selected)),
                "n_choice_spans_per_transcript": expected_trials,
                "sha256": _sha256(output_path),
            }

    manifest = {
        "schema_version": 1,
        "source_dataset": str(dataset_root.resolve()),
        "source_metadata_sha256": _sha256(metadata_path),
        "base_model": "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "format": "Centaur-style natural language with choices enclosed by << and >>",
        "loss": "tokens strictly inside choice delimiters only",
        "condition_is_in_prompt": False,
        "schedule_id_is_in_prompt": False,
        "n_trials": expected_trials,
        "files": files,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(export_centaur_jsonl(args.dataset_root, args.output_root))


if __name__ == "__main__":
    main()
