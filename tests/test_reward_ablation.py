import numpy as np
import pytest
import torch

from mechcal.analysis.reward_ablation import (
    choice_only_ablation,
    donor_reward_ablation,
    neutral_reward_ablation,
)


def test_donor_ablation_changes_only_reward_slots() -> None:
    tokens = torch.tensor(
        [[0, 1, 10, 2, 11], [0, 3, 20, 4, 21], [0, 2, 30, 1, 31]]
    )
    ablated = donor_reward_ablation(tokens, np.asarray([1, 2, 0]))
    assert ablated[:, 0].tolist() == tokens[:, 0].tolist()
    assert ablated[:, 1::2].tolist() == tokens[:, 1::2].tolist()
    assert ablated[:, 2::2].tolist() == [[20, 21], [30, 31], [10, 11]]


def test_donor_ablation_rejects_self_donors() -> None:
    tokens = torch.tensor([[0, 1, 10], [0, 2, 20]])
    with pytest.raises(ValueError, match="derangement"):
        donor_reward_ablation(tokens, np.asarray([0, 1]))


def test_neutral_ablation_preserves_choices_and_geometry() -> None:
    tokens = torch.tensor([[0, 1, 10, 2, 11]])
    ablated = neutral_reward_ablation(tokens)
    assert ablated.tolist() == [[0, 1, 55, 2, 55]]


def test_choice_only_ablation_deletes_rewards_and_marks_all_choices() -> None:
    tokens = torch.tensor([[0, 1, 10, 2, 11, 4, 12]])
    ablated, mask = choice_only_ablation(tokens)
    assert ablated.tolist() == [[0, 1, 2, 4]]
    assert mask.tolist() == [[False, True, True, True]]
