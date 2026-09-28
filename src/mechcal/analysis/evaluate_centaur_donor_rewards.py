"""Evaluate Centaur adapters after marginal-preserving donor-reward replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechcal.analysis.evaluate_centaur_reward_placeholder import (
    _bootstrap_ci,
    _choice_losses,
    _reward_positions,
)


def donor_derangements(
    n_sessions: int, n_derangements: int, seed: int
) -> np.ndarray:
    """Return reproducible permutations with no participant mapped to itself."""
    if n_sessions < 2:
        raise ValueError("donor replacement requires at least two sessions")
    rng = np.random.default_rng(seed)
    identity = np.arange(n_sessions)
    result = np.empty((n_derangements, n_sessions), dtype=np.int64)
    for repeat in range(n_derangements):
        for _ in range(10_000):
            candidate = rng.permutation(n_sessions)
            if np.all(candidate != identity):
                result[repeat] = candidate
                break
        else:
            raise RuntimeError("failed to sample a donor derangement")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--donor-seed", type=int, default=20260901)
    parser.add_argument("--n-derangements", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-choice-probabilities", action="store_true")
    parser.add_argument("--intact-only", action="store_true")
    parser.add_argument("--local-trials", default=None,
                        help="Comma-separated one-based trials; replace only the preceding reward independently")
    args = parser.parse_args()
    local_trials = [int(t) for t in args.local_trials.replace(':', ',').split(',')] if args.local_trials else None
    if local_trials and (len(set(local_trials)) != len(local_trials) or any(t < 1 or t > 200 for t in local_trials)):
        raise ValueError('local trials must be unique and between 1 and 200')
    n_targets_out = len(local_trials) if local_trials else 200
    if args.intact_only:
        args.n_derangements = 0
    if local_trials and not args.save_choice_probabilities:
        raise ValueError('local analysis requires probability export')

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
    mappings = donor_derangements(
        len(records), args.n_derangements, args.donor_seed
    )

    prepared: list[dict[str, torch.Tensor | list[int]]] = []
    reward_token_ids: list[np.ndarray] = []
    participants: list[str] = []
    sequence_lengths: list[int] = []
    for record in records:
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
        reward_positions = _reward_positions(record["text"], offsets)
        if len(reward_positions) != 200:
            raise ValueError(
                "donor evaluation requires exactly one token per reward value; "
                f"found {len(reward_positions)}"
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
        prepared.append(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "reward_positions": reward_positions,
            }
        )
        reward_token_ids.append(
            batch["input_ids"][0, reward_positions].cpu().numpy().astype(np.int64)
        )
        participants.append(str(record["participant"]))
        sequence_lengths.append(int(batch["input_ids"].shape[1]))

    reward_tokens = np.stack(reward_token_ids)
    # Every derangement is a permutation, so the complete test-set reward-token
    # marginal is preserved exactly rather than only in expectation.
    for mapping in mappings:
        if not np.array_equal(
            np.sort(reward_tokens[mapping].reshape(-1)),
            np.sort(reward_tokens.reshape(-1)),
        ):
            raise AssertionError("donor mapping changed the reward-token marginal")

    device = next(model.parameters()).device
    probabilities = {}
    if args.save_choice_probabilities:
        choice_ids = [tokenizer.encode(a, add_special_tokens=False) for a in 'ABCD']
        if any(len(ids) != 1 for ids in choice_ids):
            raise ValueError('ABCD must each tokenize to one token')
        choice_ids = [ids[0] for ids in choice_ids]
        probabilities['intact_choice_probabilities'] = np.empty((len(records), n_targets_out, 4), np.float32)
        probabilities['donor_choice_probabilities'] = np.empty((args.n_derangements, len(records), n_targets_out, 4), np.float32)

    def evaluate(ids, mask, targets, key, index):
        if not args.save_choice_probabilities:
            return _choice_losses(model, ids, mask, targets)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
            chosen = targets[:, 1:]
            selected = chosen.ne(-100)
            scores = logits[selected].float()
            if not torch.isin(chosen[selected], torch.tensor(choice_ids, device=device)).all():
                raise ValueError('supervised tokens are not ABCD')
            logp = scores.log_softmax(-1)
            probabilities[key][index] = logp[:, choice_ids].exp().cpu().numpy()
            return (-logp.gather(1, chosen[selected].unsqueeze(1)).squeeze(1)).cpu().numpy()
    intact = np.empty((len(records), n_targets_out), dtype=np.float32)
    donor = np.empty(
        (args.n_derangements, len(records), n_targets_out), dtype=np.float32
    )
    for index, item in enumerate(prepared):
        input_ids = item["input_ids"].to(device)  # type: ignore[union-attr]
        attention_mask = item["attention_mask"].to(device)  # type: ignore[union-attr]
        labels = item["labels"].to(device)  # type: ignore[union-attr]
        reward_positions = item["reward_positions"]
        if local_trials:
            choice_positions = labels[0].ne(-100).nonzero().flatten().tolist()
            for ti, trial in enumerate(local_trials):
                pos = choice_positions[trial-1]
                prefix = input_ids[:, :pos+1]
                mask = attention_mask[:, :pos+1]
                target = torch.full_like(prefix, -100)
                target[:, -1] = labels[:, pos]
                intact[index, ti] = evaluate(prefix, mask, target, 'intact_choice_probabilities', (index, slice(ti,ti+1)))[0]
                for repeat in range(args.n_derangements):
                    if trial == 1:
                        donor[repeat,index,ti] = intact[index,ti]
                        probabilities['donor_choice_probabilities'][repeat,index,ti] = probabilities['intact_choice_probabilities'][index,ti]
                        continue
                    donor_ids = prefix.clone()
                    rp = reward_positions[trial-2]
                    assert rp < pos
                    donor_ids[0,rp] = int(reward_tokens[mappings[repeat,index],trial-2])
                    assert int((donor_ids != prefix).sum()) <= 1
                    donor[repeat,index,ti] = evaluate(donor_ids,mask,target,'donor_choice_probabilities',(repeat,index,slice(ti,ti+1)))[0]
            if (index+1) % 10 == 0: print(f'local evaluated {index+1}/{len(records)}',flush=True)
            continue
        intact[index] = evaluate(input_ids, attention_mask, labels, 'intact_choice_probabilities', index)
        for repeat in range(args.n_derangements):
            donor_ids = input_ids.clone()
            donor_index = int(mappings[repeat, index])
            donor_ids[0, reward_positions] = torch.as_tensor(
                reward_tokens[donor_index], device=device, dtype=donor_ids.dtype
            )
            if donor_ids.shape != input_ids.shape:
                raise AssertionError("donor replacement changed sequence shape")
            donor[repeat, index] = evaluate(
                donor_ids, attention_mask, labels, 'donor_choice_probabilities', (repeat, index)
            )
        if (index + 1) % 10 == 0:
            print(f"evaluated {index + 1}/{len(records)}", flush=True)

    session_intact = intact.mean(axis=1)
    if args.save_choice_probabilities:
        for values in probabilities.values():
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(-1) > 1.00001):
                raise AssertionError('invalid ABCD full-vocabulary probabilities')
        if (not local_trials or local_trials[0] == 1) and not np.allclose(
            probabilities['donor_choice_probabilities'][:, :, 0],
            probabilities['intact_choice_probabilities'][None, :, 0],
            atol=1e-5, rtol=1e-4,
        ):
            raise AssertionError('donor changes trial-1 choice distribution')
    if args.intact_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output_dir / 'trial_losses.npz', participants=np.asarray(participants), intact_nll=intact, **probabilities)
        (args.output_dir / 'summary.json').write_text(json.dumps({'condition':args.condition,'test_file':str(args.test_file),'adapter':str(args.adapter),'intact_nll':float(intact.mean()),'n_sessions':len(records),'n_trials':n_targets_out,'mode':'intact calibration export'},indent=2)+'\n')
        return
    donor_mean = donor.mean(axis=0)
    session_donor = donor_mean.mean(axis=1)
    delta = session_donor - session_intact
    relative = delta / session_intact

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "trial_losses.npz",
        participants=np.asarray(participants),
        donor_mappings=mappings,
        intact_nll=intact,
        donor_nll=donor,
        trial_indices=np.asarray(local_trials or list(range(1,201))),
        **probabilities,
    )
    summary = {
        "condition": args.condition,
        "adapter": str(args.adapter),
        "test_file": str(args.test_file),
        "n_sessions": len(records),
        "n_trials": int(intact.shape[1]),
        "n_derangements": args.n_derangements,
        "donor_seed": args.donor_seed,
        "intervention": "independent preceding-reward replacement" if local_trials else "matched-trial donor reward token replacement",
        "trial_indices": local_trials or list(range(1,201)),
        "sequence_shape_preserved": True,
        "choice_positions_preserved": True,
        "reward_token_support_preserved": True,
        "test_set_reward_marginal_preserved_exactly": True,
        "self_donors": int(np.sum(mappings == np.arange(len(records)))),
        "sequence_tokens_min": int(min(sequence_lengths)),
        "sequence_tokens_max": int(max(sequence_lengths)),
        "intact_nll": float(session_intact.mean()),
        "donor_nll": float(session_donor.mean()),
        "delta_nll": float(delta.mean()),
        "delta_nll_ci95": _bootstrap_ci(
            delta, args.bootstrap_seed, args.n_bootstrap
        ),
        "relative_delta": float(relative.mean()),
        "relative_delta_ci95": _bootstrap_ci(
            relative, args.bootstrap_seed + 1, args.n_bootstrap
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
