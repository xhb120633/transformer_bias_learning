"""Export Centaur-style natural choice-history-only SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mechcal.analysis.evaluate_centaur_history_only import history_only_text


CONDITIONS = ("reward_dominant", "balanced", "choice_dominant")
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_choice_only(source_root: Path, output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    output_root.mkdir(parents=True)
    files: dict[str, dict[str, object]] = {}
    participant_sets: dict[str, dict[str, list[str]]] = {}
    for condition in CONDITIONS:
        files[condition] = {}
        participant_sets[condition] = {}
        for split in SPLITS:
            source_path = source_root / f"{condition}_{split}.jsonl"
            output_path = output_root / source_path.name
            records = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
            ]
            participants: list[str] = []
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    transformed = history_only_text(record["text"])
                    if transformed.count("<<") != 200 or transformed.count(">>") != 200:
                        raise AssertionError("choice span count changed during export")
                    record = dict(record)
                    record["text"] = transformed
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    participants.append(str(record["participant"]))
            participant_sets[condition][split] = participants
            files[condition][split] = {
                "path": output_path.name,
                "n_transcripts": len(records),
                "n_choice_spans_per_transcript": 200,
                "sha256": _sha256(output_path),
                "source_sha256": _sha256(source_path),
            }

    for split in SPLITS:
        reference = participant_sets[CONDITIONS[0]][split]
        for condition in CONDITIONS[1:]:
            if participant_sets[condition][split] != reference:
                raise AssertionError(
                    f"participant pairing differs for {condition} {split}"
                )

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "format": "minimal natural instruction plus complete choice history",
        "reward_information_present": False,
        "condition_is_in_prompt": False,
        "loss": "tokens strictly inside choice delimiters only",
        "n_trials": 200,
        "participant_pairing_identical_across_conditions": True,
        "files": files,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(export_choice_only(args.source_root, args.output_root))


if __name__ == "__main__":
    main()
