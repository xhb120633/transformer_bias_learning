"""Evaluate a choice-history-only Centaur-style adapter on held-out sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _choice_losses(model, input_ids, attention_mask, labels, choice_ids=None):
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1].float()
    targets = labels[:, 1:]
    selected = targets.ne(-100)
    losses = F.cross_entropy(logits[selected], targets[selected], reduction="none")
    if choice_ids is not None:
        if not torch.isin(targets[selected], torch.tensor(choice_ids, device=targets.device)).all():
            raise ValueError('supervised token is not one of ABCD')
        probabilities = logits[selected].log_softmax(-1)[:, choice_ids].exp()
        actions = (targets[selected][:, None] == torch.tensor(choice_ids, device=targets.device)).long().argmax(-1)
        return (losses.cpu().numpy().astype(np.float32), probabilities.cpu().numpy(), actions.cpu().numpy())
    return losses.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-choice-probabilities", action="store_true")
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
    if not records:
        raise ValueError("test file is empty")

    participants: list[str] = []
    trial_losses: list[np.ndarray] = []
    sequence_lengths: list[int] = []
    device = next(model.parameters()).device
    probabilities = []
    actions = []
    choice_ids = None
    if args.save_choice_probabilities:
        encoded_choices = [tokenizer.encode(a, add_special_tokens=False) for a in 'ABCD']
        if any(len(ids) != 1 for ids in encoded_choices):
            raise ValueError('ABCD must each be a single token')
        choice_ids = [ids[0] for ids in encoded_choices]
    for index, record in enumerate(records):
        text = record["text"]
        if "receive" in text or "points" in text:
            raise ValueError("choice-only transcript contains reward language")
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
        n_targets = int(batch["labels"].ne(-100).sum())
        if n_targets != 200:
            raise ValueError(f"expected 200 supervised choice tokens, found {n_targets}")
        prediction = _choice_losses(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                choice_ids,
            )
        if choice_ids is None:
            trial_losses.append(prediction)
        else:
            loss, probability, action = prediction
            trial_losses.append(loss)
            probabilities.append(probability)
            actions.append(action)
        participants.append(str(record["participant"]))
        sequence_lengths.append(len(encoded["input_ids"]))
        if (index + 1) % 25 == 0:
            print(f"evaluated {index + 1}/{len(records)}", flush=True)

    losses = np.stack(trial_losses)
    session_nll = losses.mean(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "trial_losses.npz",
        participants=np.asarray(participants),
        choice_only_nll=losses,
        **({'intact_choice_probabilities': np.stack(probabilities), 'actions': np.stack(actions)} if probabilities else {}),
    )
    summary = {
        "condition": args.condition,
        "adapter": str(args.adapter),
        "test_file": str(args.test_file),
        "intervention": "separately trained choice-history-only predictor",
        "n_sessions": len(records),
        "n_trials": int(losses.shape[1]),
        "choice_only_nll": float(session_nll.mean()),
        "choice_only_nll_ci95": _bootstrap_ci(
            session_nll, args.bootstrap_seed, args.n_bootstrap
        ),
        "sequence_tokens_min": int(min(sequence_lengths)),
        "sequence_tokens_max": int(max(sequence_lengths)),
        "reward_information_present": False,
        "choice_targets_per_session": 200,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
