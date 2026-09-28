import numpy as np

from mechcal.analysis.evaluate_centaur_donor_rewards import donor_derangements


def test_donor_derangements_are_reproducible_permutations_without_self_donors() -> None:
    first = donor_derangements(25, 20, 17)
    second = donor_derangements(25, 20, 17)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 25)
    identity = np.arange(25)
    for mapping in first:
        np.testing.assert_array_equal(np.sort(mapping), identity)
        assert np.all(mapping != identity)
