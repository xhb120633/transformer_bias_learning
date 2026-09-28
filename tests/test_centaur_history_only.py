from mechcal.analysis.evaluate_centaur_history_only import history_only_text


def test_history_only_keeps_ordered_choices_and_removes_task_information() -> None:
    choices = ("A", "D", "B", "C") * 50
    trials = "\n".join(
        f"You press <<{choice}>> and receive {index % 101} points."
        for index, choice in enumerate(choices)
    )
    original = "\n".join(
        [
            "In this task, you repeatedly choose among four slot machines labeled A, B, C, and D.",
            "The machines can change over time.",
            "Your goal is to earn as many points as possible.",
            trials,
        ]
    )

    transformed = history_only_text(original)

    assert transformed.count("<<") == 200
    assert transformed.count(">>") == 200
    assert transformed.splitlines()[2:6] == [
        "You press <<A>>.",
        "You press <<D>>.",
        "You press <<B>>.",
        "You press <<C>>.",
    ]
    for forbidden in ("slot", "reward", "points", "change", "goal", "game"):
        assert forbidden not in transformed.lower()
