import numpy as np
import pandas as pd
import math
import pytest

from validation.deflated_sharpe import (
    EULER_GAMMA,
    NORMAL,
    annualized_sharpe,
    compute_deflated_sharpe,
    sample_kurtosis,
    sample_skew,
)
from validation.pbo_cscv import compute_pbo


def test_deflated_sharpe_rewards_persistent_positive_returns():
    returns = np.array([0.010, -0.002] * 160)

    result = compute_deflated_sharpe(returns, n_trials=1)

    assert result.sharpe > 0
    assert result.probability > 0.95


def test_deflated_sharpe_penalizes_multiple_trials():
    returns = np.array([0.004, -0.002] * 160)

    one_trial = compute_deflated_sharpe(returns, n_trials=1)
    trial_sharpes = np.linspace(-0.2, 1.2, 100)
    many_trials = compute_deflated_sharpe(
        returns, n_trials=100, trial_sharpes=trial_sharpes,
    )

    assert many_trials.expected_max_sharpe > one_trial.expected_max_sharpe
    assert many_trials.probability < one_trial.probability


def test_deflated_sharpe_matches_cross_trial_reference_formula():
    returns = np.array([0.006, -0.003, 0.002, -0.001] * 100)
    trial_sharpes = np.array([-0.1, 0.2, 0.4, 0.7, 1.0])
    result = compute_deflated_sharpe(
        returns, n_trials=5, trial_sharpes=trial_sharpes,
    )

    n = len(trial_sharpes)
    cross_std = float(np.std(trial_sharpes, ddof=1))
    expected_hurdle = float(np.mean(trial_sharpes)) + cross_std * (
        (1 - EULER_GAMMA) * NORMAL.inv_cdf(1 - 1 / n)
        + EULER_GAMMA * NORMAL.inv_cdf(1 - 1 / (n * math.e))
    )
    sr = annualized_sharpe(returns)
    sr_period = sr / math.sqrt(252)
    variance_term = (
        1 - sample_skew(returns) * sr_period
        + ((sample_kurtosis(returns) - 1) / 4) * sr_period ** 2
    )
    sampling_se = math.sqrt(variance_term / (len(returns) - 1)) * math.sqrt(252)
    expected_z = (sr - expected_hurdle) / sampling_se

    assert result.trial_sharpe_std == pytest.approx(cross_std)
    assert result.expected_max_sharpe == pytest.approx(expected_hurdle)
    assert result.deflated_sharpe == pytest.approx(expected_z)


def test_deflated_sharpe_rejects_fake_multiple_testing_without_trial_dispersion():
    with pytest.raises(ValueError, match="requires trial_sharpes"):
        compute_deflated_sharpe([0.01, -0.002, 0.004], n_trials=22)


def test_pbo_cscv_returns_probability_and_split_details():
    idx = pd.date_range("2024-01-01", periods=32, freq="D")
    returns = pd.DataFrame(
        {
            "steady": [0.001, 0.002, -0.001, 0.001] * 8,
            "boom_bust": [0.010] * 16 + [-0.010] * 16,
            "late": [-0.004] * 16 + [0.006] * 16,
            "flat": [0.0, 0.001, 0.0, -0.001] * 8,
        },
        index=idx,
    )

    result = compute_pbo(returns, n_splits=8)

    assert 0 <= result.pbo <= 1
    assert result.n_trials == 4
    assert result.n_observations == 32
    assert len(result.logits) == math.comb(8, 4) == 70
    assert len(result.split_results) == 70


def test_pbo_requires_identical_finite_trial_dates():
    a = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    b = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-02", periods=2))
    with pytest.raises(ValueError, match="identical, finite"):
        compute_pbo({"a": a, "b": b}, n_splits=2)

