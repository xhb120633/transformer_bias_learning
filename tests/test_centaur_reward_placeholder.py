from mechcal.analysis.evaluate_centaur_reward_placeholder import _reward_positions


def test_reward_positions_find_exactly_200_value_slots() -> None:
    text = "\n".join(
        f"You press <<A>> and receive {trial % 101} points." for trial in range(200)
    )
    offsets = [(index, index + 1) for index in range(len(text))]
    positions = _reward_positions(text, offsets)
    expected = sum(len(str(trial % 101)) for trial in range(200))
    assert len(positions) == expected
    assert all(text[index].isdigit() for index in positions)
