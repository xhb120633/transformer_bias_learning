import copy
import re
import torch
from mechcal.training.spatial_transcripts import encode_map, render_map, number, CHOICE_IDS, VOCAB
from mechcal.models.gru import CausalGRU, GRUConfig
from mechcal.models.causal_transformer import CausalTransformer, TransformerConfig


def episode():
    return {'initial_position': [2, 3], 'initial_reward': -1.1234567890123456,
            'choices': [[i//5, i%5] for i in range(20)], 'rewards': [i+.23456789012345 for i in range(20)]}


def test_roundtrip_and_masks():
    ep = episode()
    assert float(number(ep['initial_reward'])) == ep['initial_reward']
    for mode in ('full', 'choice_only'):
        ids, mask = encode_map(ep, mode)
        assert len(ids) == len(mask) and sum(mask) == 20
        assert [i for i, m in zip(ids, mask) if m] == CHOICE_IDS[:20]
        assert re.findall(r'<<([A-Y])>>', render_map(ep, mode)) == list('ABCDEFGHIJKLMNOPQRST')


def test_reward_visibility_and_causal_prefix():
    ep, other = episode(), episode()
    other['rewards'] = [r+100 for r in ep['rewards']]
    other['initial_reward'] += 100
    assert encode_map(ep, 'choice_only') == encode_map(other, 'choice_only')
    assert render_map(ep, 'choice_only') == render_map(other, 'choice_only')
    for t in range(20):
        other = copy.deepcopy(ep)
        other['choices'][t:] = [[4, 4]]*(20-t)
        other['rewards'][t:] = [999.]*(20-t)
        for mode in ('full', 'choice_only'):
            tokens, mask = encode_map(ep, mode)
            new_tokens, new_mask = encode_map(other, mode)
            index = [i for i,m in enumerate(mask) if m][t]
            assert tokens[:index] == new_tokens[:index]


def test_models_are_causal():
    torch.set_num_threads(2)
    models = [CausalGRU(GRUConfig(vocab_size=len(VOCAB), embedding_dim=8, hidden_size=16, num_layers=1, dropout=0)),
              CausalTransformer(TransformerConfig(vocab_size=len(VOCAB), max_sequence_length=12,
                    d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0))]
    x = torch.randint(1, len(VOCAB), (2, 12))
    y = x.clone(); y[:, 6:] = (y[:, 6:]+1) % len(VOCAB)
    for model in models:
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x)[:, :6], model(y)[:, :6], atol=1e-6)
