"""Independent full-period SURGE PRO run with dynamic gap filtering disabled."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.experiment_registry import (
    ExperimentRegistry,
    daily_returns_from_equity,
    trial_record,
)
from strategies.base import ExecConfig, MarketData
from strategies.momentum_v85 import MomentumV85
from strategies.optimized_v85 import SURGE_PRO_PARAMS, _build_engine
from strategy.ai_strategy import build_liquid_universe
from strategy.risk_metrics import compute_risk_metrics
from surge_pro_ablation_runner import (
    BENCHMARK_TICKER,
    FIELDS,
    ROOT,
    UNIVERSE_LOOKBACK,
    UNIVERSE_TOP_N,
    clean_metrics,
    dependency_versions,
    file_sha256,
    git_state,
    payload_sha256,
    read_json,
    strategy_parameters_for_trial,
    utc_now,
    validate_trade_ledger,
    verified_run_output,
    verify_bundle,
    write_json,
)
from surge_pro_family_comparison import (
    professional_diagnostics,
    professional_metrics_row,
)


TRIAL_ID = "rs1_tiers1_gap0_vix28_scale190"
STRATEGY_ID = "mom_surge_pro_gap_off"


def gap_off_contract(
    grid: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    trial = next(
        (row for row in grid["trials"] if row["trial_id"] == TRIAL_ID),
        None,
    )
    if trial is None:
        raise RuntimeError(f"frozen grid is missing {TRIAL_ID}")
    parameters = strategy_parameters_for_trial(grid, trial)
    expected = dict(SURGE_PRO_PARAMS)
    expected["dynamic_gap_filter"] = False
    if payload_sha256(parameters) != payload_sha256(expected):
        raise RuntimeError(
            "gap-off trial is no longer SURGE PRO with only "
            "dynamic_gap_filter disabled"
        )
    return trial, parameters


def compare_equity_returns(
    reference_equity: pd.DataFrame,
    current_equity: pd.DataFrame,
    *,
    tolerance: float = 1e-15,
) -> dict[str, Any]:
    reference = reference_equity["Equity"].astype(float).pct_change().dropna()
    current = current_equity["Equity"].astype(float).pct_change().dropna()
    calendar_equal = reference.index.equals(current.index)
    max_abs_difference = (
        float(np.max(np.abs(reference.to_numpy() - current.to_numpy())))
        if calendar_equal and len(reference) else float("inf")
    )
    values_equal = bool(
        calendar_equal
        and np.allclose(
            reference.to_numpy(),
            current.to_numpy(),
            rtol=0.0,
            atol=tolerance,
        )
    )
    return {
        "calendar_equal": calendar_equal,
        "daily_returns_numerically_equal": values_equal,
        "max_abs_return_difference": max_abs_difference,
        "return_tolerance": tolerance,
        "observations": int(len(current)),
    }


def compare_ablation_reference(
    reference_dir: str | Path,
    bundle_manifest: dict[str, Any],
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    reference, _, reference_summary = verified_run_output(reference_dir)
    if reference_summary["bundle_id"] != bundle_manifest["bundle_id"]:
        raise RuntimeError("gap-off reference used a different data bundle")

    reference_trial = reference / "trials" / TRIAL_ID
    reference_result = read_json(reference_trial / "result.json")
    reference_parameters = reference_result["parameters"]["strategy_parameters"]
    if reference_parameters != parameters:
        raise RuntimeError("gap-off reference used different strategy parameters")

    reference_equity = pd.read_csv(
        reference_trial / "equity.csv",
        index_col=0,
        parse_dates=True,
    )
    return_match = compare_equity_returns(reference_equity, equity)
    metric_keys = (
        "ann_return",
        "ann_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown_pct",
        "total_trades",
        "win_rate",
        "profit_factor",
    )
    metric_checks = {
        key: metrics[key] == reference_result["metrics"][key]
        for key in metric_keys
    }
    result = {
        "reference_path": str(reference),
        "reference_experiment_id": reference_summary["experiment_id"],
        "reference_git_commit": reference_summary["git_commit"],
        "reference_trial_id": TRIAL_ID,
        "same_bundle": True,
        **return_match,
        "metric_checks": metric_checks,
        "equity_csv_byte_identical": (
            file_sha256(output / "equity.csv")
            == file_sha256(reference_trial / "equity.csv")
        ),
        "trades_csv_byte_identical": (
            file_sha256(output / "trades.csv")
            == file_sha256(reference_trial / "trades.csv")
        ),
    }
    result["matched"] = bool(
        return_match["daily_returns_numerically_equal"]
        and all(metric_checks.values())
        and result["equity_csv_byte_identical"]
        and result["trades_csv_byte_identical"]
    )
    if not result["matched"]:
        raise RuntimeError("independent gap-off run did not reproduce its reference")
    return result


def run_gap_off(
    bundle_dir: str | Path,
    output_dir: str | Path,
    runner_id: str,
    *,
    reference_ablation_run: str | Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    panel, vix, bundle_manifest, grid = verify_bundle(bundle_dir)
    trial, parameters = gap_off_contract(grid)
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("independent gap-off run requires a clean worktree")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_schema_version": 1,
        "status": "running",
        "record_status": "independent_reproduction_only",
        "runner_id": runner_id,
        "started_at": utc_now(),
        "git": git,
        "dependencies": dependency_versions(),
        "bundle_path": str(Path(bundle_dir).resolve()),
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "grid_sha256": bundle_manifest["grid_sha256"],
        "trial_id": TRIAL_ID,
        "parameter_contract_sha256": payload_sha256(parameters),
        "runner_sha256": file_sha256(__file__),
        "command": " ".join(sys.argv),
    }
    write_json(output / "run_manifest.json", run_manifest)

    close_all = panel["Close"]
    stock_columns = [
        str(column)
        for column in close_all.columns
        if str(column) != BENCHMARK_TICKER
    ]
    stock_panel = {
        field: panel[field].reindex(columns=stock_columns)
        for field in FIELDS
    }
    universe = build_liquid_universe(
        stock_panel["Close"],
        stock_panel["Volume"],
        top_n=UNIVERSE_TOP_N,
        lookback=UNIVERSE_LOOKBACK,
    )
    data = MarketData(
        close=stock_panel["Close"],
        open=stock_panel["Open"],
        high=stock_panel["High"],
        low=stock_panel["Low"],
        volume=stock_panel["Volume"],
        market_close=close_all[BENCHMARK_TICKER],
        universe_mask=universe,
    )
    signals = MomentumV85().prepare(data)
    execution = ExecConfig(**grid["execution"])

    print(f"[gap-off {runner_id}] running {TRIAL_ID}", flush=True)
    started = time.perf_counter()
    engine = _build_engine(parameters, execution)
    trades, equity = engine.run(
        signals.total_score,
        data.close,
        data.open,
        data.high,
        data.low,
        signals.ma_long,
        top_k=execution.top_k,
        threshold=execution.threshold,
        market_close=data.market_close,
        vol_df=data.volume,
        universe_mask=data.universe_mask,
        vix_series=vix,
    )
    if equity.empty or "Equity" not in equity:
        raise RuntimeError("gap-off run produced no equity curve")
    trade_quality = validate_trade_ledger(trades, TRIAL_ID)
    metrics = clean_metrics(
        compute_risk_metrics(equity, trades, execution.initial_capital)
    )
    professional = clean_metrics(
        professional_diagnostics(equity, trades, data.market_close)
    )
    effective_config = {
        "strategy_id": STRATEGY_ID,
        "source_trial_id": TRIAL_ID,
        "strategy_parameters": parameters,
        "execution": grid["execution"],
        "period": grid["period"],
        "universe": {
            "top_n": UNIVERSE_TOP_N,
            "lookback": UNIVERSE_LOOKBACK,
            "ticker_order": bundle_manifest["ticker_order"],
        },
        "bundle_id": bundle_manifest["bundle_id"],
        "vix_policy": bundle_manifest["vix_injection_policy"],
    }
    result = {
        "trial_id": TRIAL_ID,
        "strategy_id": STRATEGY_ID,
        "parameters": effective_config,
        "metrics": metrics,
        "professional": professional,
        "trade_ledger_quality": trade_quality,
        "elapsed_seconds": time.perf_counter() - started,
    }

    equity.to_csv(
        output / "equity.csv",
        index_label="Date",
        float_format="%.10f",
        lineterminator="\n",
    )
    trades.to_csv(
        output / "trades.csv",
        index=False,
        float_format="%.10f",
        lineterminator="\n",
    )
    pd.DataFrame({
        STRATEGY_ID: equity["Equity"].pct_change().dropna(),
    }).to_csv(
        output / "daily_returns.csv",
        index_label="Date",
        float_format="%.17g",
        lineterminator="\n",
    )
    pd.DataFrame([
        professional_metrics_row(STRATEGY_ID, professional, metrics),
    ]).to_csv(
        output / "professional_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    write_json(output / "result.json", result)

    reference_match = compare_ablation_reference(
        reference_ablation_run,
        bundle_manifest,
        parameters,
        metrics,
        equity,
        output,
    )
    registry_trial = trial_record(
        trial_id=TRIAL_ID,
        parameters=effective_config,
        metrics={**metrics, "professional": professional},
        daily_returns=daily_returns_from_equity(equity),
        decision="independent_reproduction_only",
    )
    registry = ExperimentRegistry(output / "experiment_registry.sqlite")
    experiment_id = registry.record_experiment(
        source="surge_pro_gap_off_runner.py",
        strategy_version=f"{STRATEGY_ID}@{git['commit']}",
        hypothesis=(
            "Independently reproduce full-period SURGE PRO with only "
            "dynamic_gap_filter disabled."
        ),
        parameter_space={"predeclared_single_trial": trial},
        number_of_trials=1,
        in_sample_period=(
            f"{grid['period']['start_inclusive']}.."
            f"{grid['period']['end_inclusive']}"
        ),
        metrics={
            "core": metrics,
            "professional": professional,
            "reference_match": reference_match,
            "bundle_id": bundle_manifest["bundle_id"],
        },
        daily_returns=daily_returns_from_equity(equity),
        max_drawdown=metrics["max_drawdown_pct"],
        sharpe=metrics["sharpe"],
        decision="independent_reproduction_only",
        command=" ".join(sys.argv),
        notes=(
            "Single predeclared reproduction only. PBO/DSR remain governed "
            "by the frozen 22-trial ablation."
        ),
        trials=[registry_trial],
        repo_root=ROOT,
        data_paths=[
            Path(bundle_dir) / "panel.pkl",
            Path(bundle_dir) / "vix.csv",
            Path(bundle_dir) / "grid.json",
        ],
        config=effective_config,
    )
    summary = {
        "record_status": "independent_reproduction_only",
        "experiment_id": experiment_id,
        "runner_id": runner_id,
        "git_commit": git["commit"],
        "bundle_id": bundle_manifest["bundle_id"],
        "trial_id": TRIAL_ID,
        "return_observations": int(len(equity) - 1),
        "metrics": metrics,
        "professional": professional,
        "trade_ledger_quality": trade_quality,
        "reference_match": reference_match,
        "statistical_scope": (
            "single predeclared reproduction; no standalone PBO or DSR"
        ),
    }
    write_json(output / "summary.json", summary)
    run_manifest.update({
        "status": "complete",
        "completed_at": utc_now(),
        "experiment_id": experiment_id,
        "output_hashes": {
            name: file_sha256(output / name)
            for name in (
                "summary.json",
                "result.json",
                "equity.csv",
                "trades.csv",
                "daily_returns.csv",
                "professional_metrics.csv",
                "experiment_registry.sqlite",
            )
        },
    })
    write_json(output / "run_manifest.json", run_manifest)
    print(
        "[gap-off] complete "
        f"CAGR={metrics['ann_return']:.2%} "
        f"Sharpe={metrics['sharpe']:.3f} "
        f"MDD={metrics['max_drawdown_pct']:.2%}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner-id", required=True)
    parser.add_argument("--reference-ablation-run", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_gap_off(
        args.bundle,
        args.output,
        args.runner_id,
        reference_ablation_run=args.reference_ablation_run,
        allow_dirty=args.allow_dirty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
