import numpy as np
import pandas as pd
import pytest

from strategies.optimized_v85 import SURGE_PRO_PARAMS
from surge_pro_ablation_runner import DEFAULT_GRID, load_grid
from surge_pro_gap_off_runner import (
    TRIAL_ID,
    compare_equity_returns,
    gap_off_contract,
)


def test_gap_off_contract_changes_only_dynamic_gap_filter():
    grid, _ = load_grid(DEFAULT_GRID)
    trial, parameters = gap_off_contract(grid)
    expected = dict(SURGE_PRO_PARAMS)
    expected["dynamic_gap_filter"] = False

    assert trial["trial_id"] == TRIAL_ID
    assert parameters["strong_tiers"] == [
        list(row) for row in expected["strong_tiers"]
    ]
    assert {
        key: value for key, value in parameters.items()
        if key != "strong_tiers"
    } == {
        key: value for key, value in expected.items()
        if key != "strong_tiers"
    }
    assert parameters["dynamic_gap_filter"] is False


def test_compare_equity_returns_uses_strict_calendar_and_tolerance():
    dates = pd.bdate_range("2024-01-02", periods=8)
    reference = pd.DataFrame(
        {"Equity": 1_000_000 * np.cumprod([1.0, 1.01, 0.99, 1.02] * 2)},
        index=dates,
    )
    matched = compare_equity_returns(reference, reference.copy())
    assert matched["daily_returns_numerically_equal"]
    assert matched["max_abs_return_difference"] == pytest.approx(0.0)

    shifted = reference.copy()
    shifted.index = shifted.index + pd.offsets.BDay(1)
    mismatch = compare_equity_returns(reference, shifted)
    assert not mismatch["calendar_equal"]
    assert not mismatch["daily_returns_numerically_equal"]
