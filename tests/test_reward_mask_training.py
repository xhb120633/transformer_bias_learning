import torch
from mechcal.training.train_restless_transformer import mask_reward_sessions


def test_session_mask_preserves_choice_positions_and_source():
    tokens = torch.tensor([[0, 1, 55, 2, 66]]).repeat(8, 1)
    before = tokens.clone()
    result = mask_reward_sessions(tokens, .5, 106, torch.Generator().manual_seed(11))
    assert torch.equal(tokens, before)
    assert torch.equal(result[:, :1], tokens[:, :1])
    assert torch.equal(result[:, 1::2], tokens[:, 1::2])
    hidden = (result[:, 2::2] == 106).all(1)
    assert hidden.sum() == 4
    assert torch.equal(result[~hidden], tokens[~hidden])
    assert torch.equal(result, mask_reward_sessions(tokens, .5, 106, torch.Generator().manual_seed(11)))


def test_mask_endpoints():
    t = torch.tensor([[0, 1, 55, 2, 66]])
    g = torch.Generator().manual_seed(1)
    assert torch.equal(t, mask_reward_sessions(t, 0, 106, g))
    assert (mask_reward_sessions(t, 1, 106, g)[:, 2::2] == 106).all()
