"""Evaluate fixed full-input GRUs under the seven-weight reward edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mechcal.models.gru import CausalGRU, GRUConfig
from run_reference_followup import WEIGHTS, load


ROOT = Path('outputs/weight_neural_full_20260916')
OUT = Path('outputs/weight_operation_sweep_gru_20260924')


@torch.inference_mode()
def probabilities(model, tokens, operation, batch_size=32):
    result = []
    for batch in tokens.split(batch_size):
        x = batch[:, :-1].cuda()
        embeddings = model.embedding(x)
        if operation == 'zero_embedding':
            embeddings[:, 2::2] = 0
        states, _ = model.rnn(embeddings)
        logits = model.head(states).float()
        positions = slice(0, 200) if operation == 'delete' else slice(0, 400, 2)
        result.append(logits[:, positions, 1:5].softmax(-1).cpu().numpy())
    return np.concatenate(result)


def nll(p, action):
    return -np.log(np.take_along_axis(p, action[..., None], axis=-1)[..., 0])


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    max_first_error = 0.
    for w in WEIGHTS:
        code = f'{round(w*100):03d}'
        data = load(w)
        actions = data['action'][:, 150:]
        tokens = torch.tensor(data['tokens'].astype(np.int64))
        assert tokens.shape == (250, 401)
        for seed in (11, 22, 33):
            run = ROOT / f'gru_reward_w{code}_full_seed{seed}'
            target = OUT / (run.name + '.npz')
            ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            model = CausalGRU(GRUConfig(**ck['model_config'])).cuda().eval()
            model.load_state_dict(ck['model_state'])
            original = probabilities(model, tokens, 'intact')
            with np.load(run / 'assay_full.npz') as saved:
                np.testing.assert_array_equal(saved['participants'], data['base_participant_id'])
                np.testing.assert_allclose(original, saved['intact_probabilities'], atol=2e-6)
                donor = saved['donor_probabilities'].astype(float)
            changed = tokens.clone()
            changed[:, 2::2] = 55  # reward 50 token, as in the Transformer sweep
            neutral = probabilities(model, changed, 'neutral50')
            zero = probabilities(model, tokens, 'zero_embedding')
            deleted_tokens = torch.cat((tokens[:, :1], tokens[:, 1::2]), dim=1)
            deletion = probabilities(model, deleted_tokens, 'delete')
            assert original.shape == neutral.shape == zero.shape == deletion.shape == (250, 200, 4)
            for edited in (neutral, zero, deletion):
                max_first_error = max(max_first_error, float(np.max(np.abs(edited[:, 0] - original[:, 0]))))
            base = nll(original[:, 150:], actions).mean(1)
            donor_loss = nll(donor, actions[None]).mean((0, 2))
            metrics = dict(donor_delta_nll=donor_loss - base)
            for name, p in [('neutral50', neutral), ('zero_embedding', zero), ('delete', deletion)]:
                metrics[name + '_delta_nll'] = nll(p[:, 150:], actions).mean(1) - base
            np.savez_compressed(target, participants=data['base_participant_id'], intact_nll=base,
                                first_choice_max_error=max_first_error, **metrics)
            records.append({'run': run.name, 'checkpoint_sha256': hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()})
            print('COMPLETE', run.name, {k: round(float(v.mean()), 5) for k, v in metrics.items()}, flush=True)
            del model
            torch.cuda.empty_cache()
    assert len(records) == 21 and max_first_error < .003
    (OUT/'COMPLETE.json').write_text(json.dumps({'split': 'val', 'test_used': False,
        'weights': list(WEIGHTS), 'seeds': [11,22,33], 'runs': 21,
        'score_trials': '151-200', 'max_first_choice_error': max_first_error,
        'checkpoints': records}, indent=2))


if __name__ == '__main__':
    main()
