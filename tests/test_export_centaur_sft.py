import json

import numpy as np

from mechcal.data.export_centaur_sft import export_centaur_jsonl, transcript_to_text


def _vocab():
    vocab = {"BOS": 0}
    vocab.update({f"CHOICE_{arm}": 1 + arm for arm in range(4)})
    vocab.update({f"REWARD_{reward}": 5 + reward for reward in range(101)})
    return vocab


def test_transcript_to_text_uses_centaur_choice_spans():
    tokens = np.asarray([0, 1, 12, 4, 105])
    text = transcript_to_text(tokens, _vocab())
    assert "<<A>>" in text
    assert "receive 7 points" in text
    assert "<<D>>" in text
    assert "receive 100 points" in text
    assert text.count("<<") == 2


def test_export_separates_conditions_without_putting_condition_in_text(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    metadata = {
        "generator": {"n_trials": 2},
        "conditions": [
            {"name": "reward_dominant"},
            {"name": "balanced"},
            {"name": "choice_dominant"},
        ],
        "token_vocab": _vocab(),
    }
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split_index, split in enumerate(("train", "val", "test")):
        np.savez_compressed(
            source / f"{split}_000.npz",
            tokens=np.asarray(
                [
                    [0, 1, 5 + split_index, 2, 6 + split_index],
                    [0, 2, 7 + split_index, 3, 8 + split_index],
                    [0, 3, 9 + split_index, 4, 10 + split_index],
                ]
            ),
            condition_name=np.asarray(
                ["reward_dominant", "balanced", "choice_dominant"]
            ),
            base_participant_id=np.asarray(["p0", "p0", "p0"]),
            schedule_seed=np.asarray([10, 10, 10]),
        )
    output = tmp_path / "export"
    manifest_path = export_centaur_jsonl(source, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["condition_is_in_prompt"] is False
    assert manifest["files"]["balanced"]["train"]["n_transcripts"] == 1
    record = json.loads((output / "balanced_train.jsonl").read_text(encoding="utf-8"))
    assert record["condition"] == "balanced"
    assert "balanced" not in record["text"]
    assert record["text"].count("<<") == 2
