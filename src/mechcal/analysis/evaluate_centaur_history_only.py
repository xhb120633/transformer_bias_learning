"""Evaluate Centaur adapters on intact versus minimal history-only prompts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TRIAL_PATTERN = re.compile(
    r"^You press <<([ABCD])>> and receive (\d+) points\.$", re.MULTILINE
)
HISTORY_ONLY_INSTRUCTION = "\n".join(
    [
        "You repeatedly choose among four options labeled A, B, C, and D.",
        "In each trial, you select an option by pressing its corresponding key.",
    ]
)


def history_only_text(text: str, expected_trials: int = 200) -> str:
    """Remove task semantics and feedback while preserving ordered choices."""
    choices = [match.group(1) for match in TRIAL_PATTERN.finditer(text)]
    if len(choices) != expected_trials:
        raise ValueError(
            f"expected {expected_trials} choice/reward trials, found {len(choices)}"
        )
    lines = [HISTORY_ONLY_INSTRUCTION]
    lines.extend(f"You press <<{choice}>>." for choice in choices)
    transformed = "\n".join(lines)
    forbidden = ("slot machine", "reward", "points", "game", "goal")
    lowered = transformed.lower()
    if any(term in lowered for term in forbidden):
        raise AssertionError("history-only prompt retained task or feedback semantics")
    return transformed


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
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
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

    left_id = tokenizer(" <<").input_ids[1:]
    right_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=left_id,
        instruction_template=right_id,
        tokenizer=tokenizer,
    )

    records = [
        json.loads(line)
        for line in args.test_file.read_text(encoding="utf-8").splitlines()
    ]
    if args.limit is not None:
        records = records[: args.limit]
    participants: list[str] = []
    intact_losses: list[np.ndarray] = []
    history_losses: list[np.ndarray] = []
    intact_lengths: list[int] = []
    history_lengths: list[int] = []
    device = next(model.parameters()).device

    for index, record in enumerate(records):
        texts = (record["text"], history_only_text(record["text"]))
        condition_losses: list[np.ndarray] = []
        lengths: list[int] = []
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )
            batch = collator(
                [
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    }
                ]
            )
            if int(batch["labels"].ne(-100).sum()) != 200:
                raise ValueError("expected exactly 200 supervised choice tokens")
            condition_losses.append(
                _choice_losses(
                    model,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
            )
            lengths.append(int(batch["input_ids"].shape[1]))
        intact_losses.append(condition_losses[0])
        history_losses.append(condition_losses[1])
        intact_lengths.append(lengths[0])
        history_lengths.append(lengths[1])
        participants.append(str(record["participant"]))
        if (index + 1) % 25 == 0:
            print(f"evaluated {index + 1}/{len(records)}", flush=True)

    intact = np.stack(intact_losses)
    history = np.stack(history_losses)
    session_intact = intact.mean(axis=1)
    session_history = history.mean(axis=1)
    delta = session_history - session_intact
    relative = delta / session_intact
    chance_nll = math.log(4.0)
    intact_headroom = chance_nll - session_intact
    retained_headroom = (chance_nll - session_history) / intact_headroom

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "trial_losses.npz",
        participants=np.asarray(participants),
        intact_nll=intact,
        history_only_nll=history,
    )
    summary = {
        "condition": args.condition,
        "adapter": str(args.adapter),
        "test_file": str(args.test_file),
        "n_sessions": len(records),
        "n_trials": int(intact.shape[1]),
        "ablation": "minimal_instruction_plus_complete_choice_history_only",
        "removed": [
            "slot_machine_cover_story",
            "reward_goal",
            "restless_dynamics_description",
            "trial_count_and_game_label",
            "all_reward_feedback",
        ],
        "preserved": ["choice_alphabet", "choice_order", "all_choice_targets"],
        "sequence_shape_preserved": False,
        "intact_tokens_min": int(min(intact_lengths)),
        "intact_tokens_max": int(max(intact_lengths)),
        "history_only_tokens_min": int(min(history_lengths)),
        "history_only_tokens_max": int(max(history_lengths)),
        "chance_nll": chance_nll,
        "intact_nll": float(session_intact.mean()),
        "history_only_nll": float(session_history.mean()),
        "delta_nll": float(delta.mean()),
        "delta_nll_ci95": _bootstrap_ci(delta, args.bootstrap_seed, args.n_bootstrap),
        "relative_delta": float(relative.mean()),
        "relative_delta_ci95": _bootstrap_ci(
            relative, args.bootstrap_seed + 1, args.n_bootstrap
        ),
        "retained_predictive_headroom": float(retained_headroom.mean()),
        "retained_predictive_headroom_ci95": _bootstrap_ci(
            retained_headroom, args.bootstrap_seed + 2, args.n_bootstrap
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
