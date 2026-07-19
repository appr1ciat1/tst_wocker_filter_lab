import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from surge_pro_ablation_runner import (
    DEFAULT_GRID,
    analyze_trials,
    bundle_identity,
    compare_runs,
    load_grid,
    normalize_vix,
    payload_sha256,
    persist_vix,
    read_json,
    run_grid,
    verify_bundle,
    vix_sha256,
    write_json,
)
from twstk.data.contract import build_manifest, freeze_snapshot, validate_panel


def test_grid_hash_is_canonical_and_stable():
    grid, grid_hash = load_grid(DEFAULT_GRID)
    reordered = json.loads(json.dumps(grid, sort_keys=True))

    assert len(grid["trials"]) == 22
    assert grid_hash == payload_sha256(reordered)


def test_vix_normalization_and_hash_ignore_input_row_order():
    dates = pd.to_datetime(["2024-01-03", "2024-01-02"])
    raw = pd.DataFrame({"VIX": [14.0, 13.0]}, index=dates)
    normalized = normalize_vix(raw)

    assert list(normalized.index) == sorted(dates)
    assert vix_sha256(normalized) == vix_sha256(normalized.sort_index())


def test_bundle_identity_binds_panel_vix_grid_and_period():
    period = {
        "start_inclusive": "2019-01-01",
        "end_inclusive": "2026-07-17",
        "download_end_exclusive": "2026-07-18",
    }
    tickers = ["0050", "2330"]
    original = bundle_identity("panel", "vix", "grid", period, tickers)

    assert original != bundle_identity("panel-2", "vix", "grid", period, tickers)
    assert original != bundle_identity("panel", "vix-2", "grid", period, tickers)
    assert original != bundle_identity("panel", "vix", "grid-2", period, tickers)
    assert original != bundle_identity(
        "panel", "vix", "grid", period, list(reversed(tickers)),
    )


def test_vix_persistence_hashes_the_round_tripped_values(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=3)
    original = pd.Series(
        [13.123456789012345, 14.987654321098765, 15.000000000000002],
        index=dates,
        name="VIX",
    )
    persisted = persist_vix(original, tmp_path / "vix.csv")

    assert vix_sha256(persisted) == vix_sha256(
        normalize_vix(
            pd.read_csv(tmp_path / "vix.csv", index_col=0, parse_dates=True)
        )
    )


def test_grid_loader_rejects_a_semantic_neighborhood_typo(tmp_path):
    grid, _ = load_grid(DEFAULT_GRID)
    grid["trials"][0]["strong_vix_max"] = 24.0
    path = tmp_path / "bad-grid.json"
    write_json(path, grid)

    with pytest.raises(ValueError, match="exact 2x3x3"):
        load_grid(path)


def test_formal_analysis_uses_baseline_and_all_70_cscv_splits():
    grid, _ = load_grid(DEFAULT_GRID)
    rng = np.random.default_rng(20260719)
    dates = pd.bdate_range("2024-01-01", periods=160)
    returns = {}
    trial_results = []

    for number, trial in enumerate(grid["trials"]):
        trial_id = trial["trial_id"]
        mean = 0.00015 + number * 0.000002
        if trial_id == grid["baseline_trial_id"]:
            mean = 0.0012
        elif trial_id == grid["matched_no_sizing_trial_id"]:
            mean = 0.0001
        series = pd.Series(
            mean + rng.normal(0.0, 0.004, len(dates)),
            index=dates,
        )
        returns[trial_id] = series
        sharpe = float(series.mean() / series.std() * np.sqrt(252))
        trial_results.append({
            "trial_id": trial_id,
            "metrics": {
                "ann_return": float(mean * 252),
                "sharpe": sharpe,
                "max_drawdown_pct": -0.2,
            },
        })

    analysis = analyze_trials(trial_results, pd.DataFrame(returns), grid)

    assert analysis["baseline_trial_id"] == grid["baseline_trial_id"]
    assert len(analysis["pbo"]["split_results"]) == 70
    assert analysis["baseline_dsr"]["n_trials"] == 22
    assert analysis["best_trial_id_diagnostic_only"] in returns
    assert set(analysis["matched_effects"]) == {
        "strong_tiers_increment_at_baseline",
        "base_regime_sizing_increment",
        "all_sizing_layers_increment",
        "dynamic_gap_increment_at_baseline",
        "dynamic_gap_increment_without_tiers",
        "dynamic_gap_increment_without_sizing",
    }


def test_runner_executes_all_22_trials_from_a_frozen_bundle(
    tmp_path,
    monkeypatch,
):
    grid, grid_hash = load_grid(DEFAULT_GRID)
    dates = pd.bdate_range(end=grid["period"]["end_inclusive"], periods=100)
    columns = ["0050", "2330", "2317"]
    close = pd.DataFrame(100.0, index=dates, columns=columns)
    panel = {
        "Close": close,
        "Open": close.copy(),
        "High": close + 1.0,
        "Low": close - 1.0,
        "Volume": pd.DataFrame(1_000_000.0, index=dates, columns=columns),
    }
    contract = validate_panel(
        panel,
        grid["period"]["end_inclusive"],
        scheduled=True,
        key_tickers=("0050",),
    )
    assert contract.ok

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    vix = persist_vix(
        pd.Series(15.0, index=dates, name="VIX"),
        bundle / "vix.csv",
    )
    write_json(bundle / "grid.json", grid)
    manifest = build_manifest(
        panel,
        grid["period"]["end_inclusive"],
        provider="synthetic-test",
        auto_adjust=True,
        contract=contract,
    )
    manifest.update({
        "period": grid["period"],
        "grid_id": grid["grid_id"],
        "grid_sha256": grid_hash,
        "vix_sha256": vix_sha256(vix),
        "ticker_order": columns,
        "vix_injection_policy": (
            "inject frozen VIX when regime_sizing=true; pass None when false "
            "to preserve the strategy's no-sizing semantics"
        ),
        "statistics_protocol": {
            "pbo": "test protocol",
            "dsr": "test protocol",
            "best_trial": "diagnostic_only",
        },
    })
    manifest["bundle_id"] = bundle_identity(
        manifest["panel_sha256"],
        manifest["vix_sha256"],
        manifest["grid_sha256"],
        manifest["period"],
        columns,
    )
    freeze_snapshot(panel, bundle, manifest)

    def forbid_network(*args, **kwargs):
        raise AssertionError("run phase must not call yfinance")

    monkeypatch.setattr(yf, "download", forbid_network)
    output = tmp_path / "run"
    summary = run_grid(
        bundle,
        output,
        "pytest-independent-run",
        allow_dirty=True,
    )

    assert summary["trial_count"] == 22
    assert len(summary["analysis"]["pbo"]["split_results"]) == 70
    assert summary["analysis"]["decision"] == "reject"
    assert summary["record_status"] == "provisional"
    assert (output / "experiment_registry.sqlite").is_file()
    assert (output / "run_manifest.json").is_file()
    with sqlite3.connect(output / "experiment_registry.sqlite") as connection:
        decisions = {
            row[0] for row in connection.execute(
                "SELECT decision FROM experiments"
            )
        }
    assert decisions == {"provisional"}

    tampered = read_json(bundle / "manifest.json")
    tampered["ticker_order"] = list(reversed(columns))
    write_json(bundle / "manifest.json", tampered)
    with pytest.raises(RuntimeError, match="ticker order"):
        verify_bundle(bundle)
    write_json(bundle / "manifest.json", manifest)

    second_output = tmp_path / "run-2"
    second_summary = run_grid(
        bundle,
        second_output,
        "pytest-independent-run-2",
        allow_dirty=True,
    )
    assert second_summary["record_status"] == "provisional"

    comparison_path = tmp_path / "comparison.json"
    comparison = compare_runs(output, second_output, comparison_path)
    assert comparison["reproduced"]
    assert comparison["formal_decision"] == "reject"
    assert (tmp_path / "comparison.sqlite").is_file()
    with sqlite3.connect(tmp_path / "comparison.sqlite") as connection:
        decisions = {
            row[0] for row in connection.execute(
                "SELECT decision FROM experiments"
            )
        }
    assert decisions == {"reject"}

    with pytest.raises(RuntimeError, match="distinct directories"):
        compare_runs(output, output, tmp_path / "self-comparison.json")
