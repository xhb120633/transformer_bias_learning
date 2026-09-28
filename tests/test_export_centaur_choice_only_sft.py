import json

from mechcal.data.export_centaur_choice_only_sft import (
    CONDITIONS,
    SPLITS,
    export_choice_only,
)


def test_choice_only_export_preserves_pairing_and_removes_rewards(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    choices = ("A", "B", "C", "D") * 50
    text = "\n".join(
        ["Full task instruction"]
        + [
            f"You press <<{choice}>> and receive {trial % 101} points."
            for trial, choice in enumerate(choices)
        ]
    )
    for condition in CONDITIONS:
        for split in SPLITS:
            path = source / f"{condition}_{split}.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": text,
                        "participant": f"{split}_p0",
                        "condition": condition,
                        "split": split,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    output = tmp_path / "choice_only"
    manifest_path = export_choice_only(source, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["reward_information_present"] is False
    assert manifest["participant_pairing_identical_across_conditions"] is True
    record = json.loads((output / "balanced_train.jsonl").read_text(encoding="utf-8"))
    assert record["text"].count("<<") == 200
    assert "reward" not in record["text"].lower()
    assert "points" not in record["text"].lower()
