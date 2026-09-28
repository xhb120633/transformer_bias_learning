import torch
import torch.nn.functional as F

from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.training.restless_dataset import causal_batch


def _tiny_model(dropout: float = 0.0) -> CausalTransformer:
    return CausalTransformer(
        TransformerConfig(
            vocab_size=106,
            max_sequence_length=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=dropout,
        )
    )


def test_causal_batch_supervises_only_shifted_choice_positions() -> None:
    tokens = torch.tensor([[0, 1, 10, 2, 11]])
    mask = torch.tensor([[False, True, False, True, False]])
    inputs, targets, target_mask = causal_batch(tokens, mask)
    assert inputs.tolist() == [[0, 1, 10, 2]]
    assert targets[target_mask].tolist() == [1, 2]
    assert target_mask.tolist() == [[True, False, True, False]]


def test_future_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(2)
    model = _tiny_model().eval()
    original = torch.tensor([[0, 1, 10, 2, 11, 3, 12]])
    changed = original.clone()
    changed[:, 4:] = torch.tensor([[99, 4, 100]])
    with torch.no_grad():
        first = model(original)
        second = model(changed)
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-6)


def test_zero_embedding_placeholder_preserves_shape_and_positions() -> None:
    torch.manual_seed(7)
    model = _tiny_model().eval()
    tokens = torch.tensor([[0, 1, 10, 2, 11]])
    mask = torch.tensor([[False, False, True, False, True]])
    with torch.no_grad():
        intact = model(tokens)
        placeholder = model(tokens, zero_token_embedding_mask=mask)
    assert placeholder.shape == intact.shape
    assert not torch.allclose(placeholder, intact)


def test_tiny_model_can_overfit_choice_targets() -> None:
    torch.manual_seed(3)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokens = torch.tensor([[0, 1, 10, 2, 11, 1, 12]]).repeat(8, 1)
    mask = torch.tensor([[False, True, False, True, False, True, False]]).repeat(8, 1)
    inputs, targets, target_mask = causal_batch(tokens, mask)
    for _ in range(40):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits[target_mask], targets[target_mask])
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.15
