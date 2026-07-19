import numpy as np
import pandas as pd

from strategies.base import ExecConfig
from strategies.optimized_v85 import GUARD_PARAMS, SURGE_PRO_PARAMS
from surge_pro_family_comparison import (
    VARIANT_ORDER,
    analyze_family,
    build_variant_engine,
    variant_parameters,
)


def test_family_variant_parameters_bind_the_three_canonical_versions():
    assert VARIANT_ORDER == ("momentum_v85", "mom_guard", "mom_surge_pro")
    assert variant_parameters("mom_guard") == GUARD_PARAMS
    assert variant_parameters("mom_surge_pro") == SURGE_PRO_PARAMS
    v85 = variant_parameters("momentum_v85")
    assert v85["sl_atr"] == 3.0
    assert v85["hold_days"] == 20
    assert v85["regime_sizing"] is False


def test_family_engine_mapping_preserves_version_differences():
    execution = ExecConfig()
    v85 = build_variant_engine("momentum_v85", execution)
    guard = build_variant_engine("mom_guard", execution)
    surge = build_variant_engine("mom_surge_pro", execution)

    assert v85.sl_atr_mult == 3.0
    assert guard.sl_atr_mult == surge.sl_atr_mult == 3.5
    assert v85.max_hold_days == guard.max_hold_days == 20
    assert surge.max_hold_days == 25
    assert not v85.regime_sizing
    assert not guard.regime_sizing
    assert surge.regime_sizing
    assert guard.corr_select_max == 0.70
    assert surge.corr_select_max == 0.0


def test_family_analysis_compares_all_three_and_keeps_statistics_diagnostic():
    rng = np.random.default_rng(20260719)
    dates = pd.bdate_range("2024-01-01", periods=160)
    returns = pd.DataFrame({
        "momentum_v85": rng.normal(0.0002, 0.005, len(dates)),
        "mom_guard": rng.normal(0.0004, 0.0045, len(dates)),
        "mom_surge_pro": rng.normal(0.0006, 0.004, len(dates)),
    }, index=dates)
    trial_results = []
    for name in VARIANT_ORDER:
        series = returns[name]
        trial_results.append({
            "trial_id": name,
            "metrics": {
                "ann_return": float(series.mean() * 252),
                "sharpe": float(series.mean() / series.std() * np.sqrt(252)),
                "max_drawdown_pct": -0.2,
            },
        })

    analysis = analyze_family(trial_results, returns)

    assert set(analysis["metrics"]) == set(VARIANT_ORDER)
    assert set(analysis["pairwise"]) == {
        "guard_minus_v85",
        "surge_pro_minus_guard",
        "surge_pro_minus_v85",
    }
    assert analysis["diagnostic_pbo"]["evaluated_splits"] == 70
    assert set(analysis["diagnostic_dsr"]) == set(VARIANT_ORDER)
    assert "does not replace" in analysis["scope"]
