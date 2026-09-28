"""Evaluate a Centaur adapter with intact versus meaningless reward tokens."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REWARD_PATTERN = re.compile(r"and receive (\d+) points\.")


def _reward_positions(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    spans = [match.span(1) for match in REWARD_PATTERN.finditer(text)]
    positions: list[int] = []
    for span_start, span_end in spans:
        matched = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < span_end and end > span_start
        ]
        if not matched:
            raise ValueError(f"reward span {span_start}:{span_end} has no token")
        positions.extend(matched)
    if len(spans) != 200:
        raise ValueError(f"expected 200 reward spans, found {len(spans)}")
    return positions


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _choice_losses(model, input_ids, attention_mask, labels) -> np.ndarray:
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1].float()
    targets = labels[:, 1:]
    selected = targets.ne(-100)
    losses = F.cross_entropy(logits[selected], targets[selected], reduction="none")
    return losses.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--placeholder-token", default="<|reserved_special_token_0|>"
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import unsloth  # noqa: F401
    from trl import DataCollatorForCompletionOnlyLM
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    placeholder_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    if placeholder_id is None or placeholder_id == tokenizer.unk_token_id:
        raise ValueError(f"unknown placeholder token: {args.placeholder_token}")

    left_id = tokenizer(" <<").input_ids[1:]
    right_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=left_id,
        instruction_template=right_id,
        tokenizer=tokenizer,
    )

    records = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        records = records[: args.limit]
    participants: list[str] = []
    intact_losses: list[np.ndarray] = []
    placeholder_losses: list[np.ndarray] = []
    deleted_losses: list[np.ndarray] = []
    reward_token_counts: list[int] = []
    device = next(model.parameters()).device

    for index, record in enumerate(records):
        text = record["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
        reward_positions = _reward_positions(text, offsets)
        features = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        batch = collator([features])
        if int(batch["labels"].ne(-100).sum()) != 200:
            raise ValueError("expected exactly 200 supervised choice tokens")
        intact_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        placeholder_ids = intact_ids.clone()
        placeholder_ids[0, reward_positions] = int(placeholder_id)
        if placeholder_ids.shape != intact_ids.shape:
            raise AssertionError("placeholder changed sequence shape")
        keep = torch.ones(intact_ids.shape[1], dtype=torch.bool, device=device)
        keep[reward_positions] = False
        deleted_ids = intact_ids[:, keep]
        deleted_attention_mask = attention_mask[:, keep]
        deleted_labels = labels[:, keep]
        if int(deleted_labels.ne(-100).sum()) != 200:
            raise AssertionError("reward deletion changed the choice targets")

        intact_losses.append(
            _choice_losses(model, intact_ids, attention_mask, labels)
        )
        placeholder_losses.append(
            _choice_losses(model, placeholder_ids, attention_mask, labels)
        )
        deleted_losses.append(
            _choice_losses(
                model, deleted_ids, deleted_attention_mask, deleted_labels
            )
        )
        participants.append(str(record["participant"]))
        reward_token_counts.append(len(reward_positions))
        if (index + 1) % 25 == 0:
            print(f"evaluated {index + 1}/{len(records)}", flush=True)

    intact = np.stack(intact_losses)
    placeholder = np.stack(placeholder_losses)
    deleted = np.stack(deleted_losses)
    session_intact = intact.mean(axis=1)
    session_placeholder = placeholder.mean(axis=1)
    session_deleted = deleted.mean(axis=1)
    delta = session_placeholder - session_intact
    relative = delta / session_intact
    deletion_delta = session_deleted - session_intact
    deletion_relative = deletion_delta / session_intact
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "trial_losses.npz",
        participants=np.asarray(participants),
        intact_nll=intact,
        placeholder_nll=placeholder,
        deleted_nll=deleted,
    )
    summary = {
        "condition": args.condition,
        "adapter": str(args.adapter),
        "test_file": str(args.test_file),
        "n_sessions": len(records),
        "n_trials": int(intact.shape[1]),
        "placeholder_token": args.placeholder_token,
        "placeholder_token_id": int(placeholder_id),
        "sequence_shape_preserved": True,
        "deletion_sequence_shape_preserved": False,
        "reward_tokens_per_session_min": int(min(reward_token_counts)),
        "reward_tokens_per_session_max": int(max(reward_token_counts)),
        "intact_nll": float(session_intact.mean()),
        "placeholder_nll": float(session_placeholder.mean()),
        "delta_nll": float(delta.mean()),
        "delta_nll_ci95": _bootstrap_ci(delta, args.bootstrap_seed, args.n_bootstrap),
        "relative_delta": float(relative.mean()),
        "relative_delta_ci95": _bootstrap_ci(
            relative, args.bootstrap_seed + 1, args.n_bootstrap
        ),
        "deleted_nll": float(session_deleted.mean()),
        "deletion_delta_nll": float(deletion_delta.mean()),
        "deletion_delta_nll_ci95": _bootstrap_ci(
            deletion_delta, args.bootstrap_seed + 2, args.n_bootstrap
        ),
        "deletion_relative_delta": float(deletion_relative.mean()),
        "deletion_relative_delta_ci95": _bootstrap_ci(
            deletion_relative, args.bootstrap_seed + 3, args.n_bootstrap
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
