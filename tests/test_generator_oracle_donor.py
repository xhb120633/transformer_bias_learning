import numpy as np

from mechcal.analysis.generator_oracle_donor import oracle_trial_nll


def test_reward_changes_propagate_only_after_the_changed_trial() -> None:
    actions = np.asarray([[0, 0, 0]], dtype=np.int64)
    common = dict(
        actions=actions,
        alpha=np.asarray([0.5]),
        beta_reward=np.asarray([1.0]),
        beta_kernel=np.asarray([0.0]),
        q_scale=10.0,
        decision_noise=np.asarray([0.1]),
    )
    high = oracle_trial_nll(rewards=np.asarray([[100, 100, 100]]), **common)
    low = oracle_trial_nll(rewards=np.asarray([[0, 100, 100]]), **common)
    assert high[0, 0] == low[0, 0]
    assert high[0, 1] < low[0, 1]
    assert np.all(np.isfinite(high))
