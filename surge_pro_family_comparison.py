"""Full-period v8.5 / GUARD / SURGE PRO comparison on one frozen bundle."""

from __future__ import annotations

import argparse
import math
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
from strategies.optimized_v85 import GUARD_PARAMS, SURGE_PRO_PARAMS, _build_engine
from strategy.ai_strategy import build_liquid_universe
from strategy.event_backtest import EventDrivenBacktester
from strategy.risk_metrics import compute_risk_metrics
from surge_pro_ablation_runner import (
    BENCHMARK_TICKER,
    FIELDS,
    ROOT,
    UNIVERSE_LOOKBACK,
    UNIVERSE_TOP_N,
    aligned_returns,
    clean_metrics,
    dependency_versions,
    file_sha256,
    git_state,
    payload_sha256,
    read_json,
    utc_now,
    verified_run_output,
    verify_bundle,
    write_json,
)
from validation.deflated_sharpe import annualized_sharpe, compute_deflated_sharpe
from validation.pbo_cscv import compute_pbo


VARIANT_ORDER = ("momentum_v85", "mom_guard", "mom_surge_pro")


def build_v85_engine(execution: ExecConfig) -> EventDrivenBacktester:
    """Mirror MomentumV85.run_engine while reusing the common prepared signal."""
    params = MomentumV85.DEFAULTS
    return EventDrivenBacktester(
        tp_sl_mode="atr",
        tp_atr_mult=params["tp_atr"],
        sl_atr_mult=params["sl_atr"],
        max_hold_days=params["hold_days"],
        initial_capital=execution.initial_capital,
        regime_filter=params["regime_filter"],
        gap_filter_atr=params["gap_filter_atr"],
        hybrid_tiered=False,
        buy_cost=execution.buy_cost,
        sell_cost=execution.sell_cost,
        slippage=execution.slippage,
        corr_select_max=0.0,
        corr_select_window=60,
        corr_select_cap=1,
    )


def variant_parameters(name: str) -> dict[str, Any]:
    if name == "momentum_v85":
        return {
            **MomentumV85.DEFAULTS,
            "position_size": 0.10,
            "dynamic_topk": False,
            "dynamic_gap_filter": False,
            "regime_sizing": False,
        }
    if name == "mom_guard":
        return dict(GUARD_PARAMS)
    if name == "mom_surge_pro":
        return dict(SURGE_PRO_PARAMS)
    raise KeyError(name)


def build_variant_engine(name: str, execution: ExecConfig) -> EventDrivenBacktester:
    if name == "momentum_v85":
        return build_v85_engine(execution)
    return _build_engine(variant_parameters(name), execution)


def yearly_statistics(returns: pd.Series) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for year, values in returns.groupby(returns.index.year):
        equity = (1.0 + values).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows[str(int(year))] = {
            "return": float(equity.iloc[-1] - 1.0),
            "sharpe": float(annualized_sharpe(values)),
            "max_drawdown": float(drawdown.min()),
            "observations": int(len(values)),
        }
    return rows


def analyze_family(
    trial_results: list[dict[str, Any]],
    returns: pd.DataFrame,
) -> dict[str, Any]:
    if list(returns.columns) != list(VARIANT_ORDER):
        raise RuntimeError("family returns are not in the frozen variant order")
    by_id = {row["trial_id"]: row for row in trial_results}
    if set(by_id) != set(VARIANT_ORDER):
        raise RuntimeError("family results do not contain all three variants")

    trial_sharpes = [
        annualized_sharpe(returns[name])
        for name in VARIANT_ORDER
    ]
    pbo = compute_pbo(returns, n_splits=8)
    if len(pbo.split_results) != 70:
        raise RuntimeError(f"expected 70 CSCV splits, found {len(pbo.split_results)}")
    dsr = {
        name: compute_deflated_sharpe(
            returns[name],
            n_trials=len(VARIANT_ORDER),
            trial_sharpes=trial_sharpes,
        )
        for name in VARIANT_ORDER
    }

    def delta(right: str, left: str) -> dict[str, float]:
        right_metrics = by_id[right]["metrics"]
        left_metrics = by_id[left]["metrics"]
        return {
            "ann_return_delta": (
                float(right_metrics["ann_return"])
                - float(left_metrics["ann_return"])
            ),
            "sharpe_delta": (
                float(right_metrics["sharpe"])
                - float(left_metrics["sharpe"])
            ),
            "max_drawdown_delta": (
                float(right_metrics["max_drawdown_pct"])
                - float(left_metrics["max_drawdown_pct"])
            ),
        }

    ranking = sorted(
        VARIANT_ORDER,
        key=lambda name: float(by_id[name]["metrics"]["sharpe"]),
        reverse=True,
    )
    return {
        "scope": (
            "pre-declared three-strategy horizontal comparison; diagnostic "
            "PBO/DSR does not replace the SURGE PRO 22-trial acceptance gate"
        ),
        "ranking_by_sharpe": ranking,
        "metrics": {
            name: by_id[name]["metrics"]
            for name in VARIANT_ORDER
        },
        "pairwise": {
            "guard_minus_v85": delta("mom_guard", "momentum_v85"),
            "surge_pro_minus_guard": delta("mom_surge_pro", "mom_guard"),
            "surge_pro_minus_v85": delta("mom_surge_pro", "momentum_v85"),
        },
        "diagnostic_pbo": {
            "pbo": pbo.pbo,
            "n_splits": pbo.n_splits,
            "evaluated_splits": len(pbo.split_results),
            "n_trials": pbo.n_trials,
            "n_observations": pbo.n_observations,
            "split_results": pbo.split_results,
        },
        "diagnostic_dsr": {
            name: {
                "probability": result.probability,
                "z_score": result.deflated_sharpe,
                "expected_max_sharpe": result.expected_max_sharpe,
                "sharpe": result.sharpe,
            }
            for name, result in dsr.items()
        },
        "yearly": {
            name: yearly_statistics(returns[name])
            for name in VARIANT_ORDER
        },
    }


def compare_surge_reference(
    reference_dir: str | Path,
    bundle_manifest: dict[str, Any],
    trial_results: list[dict[str, Any]],
    returns: pd.DataFrame,
) -> dict[str, Any]:
    reference, _, reference_summary = verified_run_output(reference_dir)
    if reference_summary["bundle_id"] != bundle_manifest["bundle_id"]:
        raise RuntimeError("SURGE reference used a different data bundle")
    baseline_id = reference_summary["analysis"]["baseline_trial_id"]
    reference_returns = pd.read_csv(
        reference / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )[baseline_id]
    current_returns = returns["mom_surge_pro"]
    calendar_equal = reference_returns.index.equals(current_returns.index)
    max_abs_return_difference = (
        float(
            np.max(np.abs(
                reference_returns.to_numpy(dtype=float)
                - current_returns.to_numpy(dtype=float)
            ))
        )
        if calendar_equal else float("inf")
    )
    values_equal = (
        calendar_equal
        and np.allclose(
            reference_returns.to_numpy(dtype=float),
            current_returns.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        )
    )
    current_metrics = next(
        row["metrics"] for row in trial_results
        if row["trial_id"] == "mom_surge_pro"
    )
    reference_metrics = reference_summary["analysis"]["baseline_metrics"]
    metric_keys = ("ann_return", "sharpe", "max_drawdown_pct", "total_trades")
    metric_checks = {
        key: current_metrics[key] == reference_metrics[key]
        for key in metric_keys
    }
    result = {
        "reference_path": str(reference),
        "reference_experiment_id": reference_summary["experiment_id"],
        "reference_git_commit": reference_summary["git_commit"],
        "same_bundle": True,
        "calendar_equal": calendar_equal,
        "daily_returns_numerically_equal": values_equal,
        "max_abs_return_difference": max_abs_return_difference,
        "return_tolerance": 1e-15,
        "metric_checks": metric_checks,
        "matched": values_equal and all(metric_checks.values()),
    }
    if not result["matched"]:
        raise RuntimeError("family SURGE PRO does not reproduce ablation baseline")
    return result


def run_family(
    bundle_dir: str | Path,
    output_dir: str | Path,
    runner_id: str,
    *,
    reference_ablation_run: str | Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    panel, vix, bundle_manifest, grid = verify_bundle(bundle_dir)
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("formal family comparison requires a clean git worktree")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_schema_version": 1,
        "status": "running",
        "record_status": "comparison_only",
        "runner_id": runner_id,
        "started_at": utc_now(),
        "git": git,
        "dependencies": dependency_versions(),
        "bundle_path": str(Path(bundle_dir).resolve()),
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "grid_sha256": bundle_manifest["grid_sha256"],
        "family_protocol_sha256": payload_sha256({
            "variants": list(VARIANT_ORDER),
            "parameters": {
                name: variant_parameters(name)
                for name in VARIANT_ORDER
            },
            "execution": grid["execution"],
            "universe_top_n": UNIVERSE_TOP_N,
            "universe_lookback": UNIVERSE_LOOKBACK,
        }),
        "runner_sha256": file_sha256(__file__),
        "command": " ".join(sys.argv),
    }
    write_json(output / "run_manifest.json", run_manifest)

    close_all = panel["Close"]
    stock_columns = [
        str(column) for column in close_all.columns
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

    equities: dict[str, pd.DataFrame] = {}
    trial_results: list[dict[str, Any]] = []
    for number, name in enumerate(VARIANT_ORDER, 1):
        print(f"[family {runner_id}] {number}/3 {name}", flush=True)
        started = time.perf_counter()
        engine = build_variant_engine(name, execution)
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
            vix_series=vix if name == "mom_surge_pro" else None,
        )
        metrics = clean_metrics(
            compute_risk_metrics(equity, trades, execution.initial_capital)
        )
        parameters = {
            "strategy_id": name,
            "strategy_parameters": variant_parameters(name),
            "execution": grid["execution"],
            "period": grid["period"],
            "universe": {
                "top_n": UNIVERSE_TOP_N,
                "lookback": UNIVERSE_LOOKBACK,
                "ticker_order": bundle_manifest["ticker_order"],
            },
            "bundle_id": bundle_manifest["bundle_id"],
            "vix_policy": (
                bundle_manifest["vix_injection_policy"]
                if name == "mom_surge_pro"
                else "no VIX dependency"
            ),
        }
        row = {
            "trial_id": name,
            "parameters": parameters,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trial_dir = output / "trials" / name
        trial_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(
            trial_dir / "equity.csv",
            index_label="Date",
            float_format="%.10f",
            lineterminator="\n",
        )
        trades.to_csv(
            trial_dir / "trades.csv",
            index=False,
            float_format="%.10f",
            lineterminator="\n",
        )
        write_json(trial_dir / "result.json", row)
        equities[name] = equity
        trial_results.append(row)

    returns = aligned_returns(equities)
    analysis = analyze_family(trial_results, returns)
    reference_match = None
    if reference_ablation_run is not None:
        reference_match = compare_surge_reference(
            reference_ablation_run,
            bundle_manifest,
            trial_results,
            returns,
        )

    pd.DataFrame([
        {
            "strategy": name,
            "ann_return": analysis["metrics"][name]["ann_return"],
            "sharpe": analysis["metrics"][name]["sharpe"],
            "max_drawdown_pct": analysis["metrics"][name]["max_drawdown_pct"],
            "total_trades": analysis["metrics"][name]["total_trades"],
            "win_rate": analysis["metrics"][name]["win_rate"],
        }
        for name in VARIANT_ORDER
    ]).to_csv(
        output / "strategies.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    returns.to_csv(
        output / "daily_returns.csv",
        index_label="Date",
        float_format="%.17g",
        lineterminator="\n",
    )
    yearly_rows = []
    for name in VARIANT_ORDER:
        for year, values in analysis["yearly"][name].items():
            yearly_rows.append({"strategy": name, "year": year, **values})
    pd.DataFrame(yearly_rows).to_csv(
        output / "yearly.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    registry_trials = [
        trial_record(
            trial_id=row["trial_id"],
            parameters=row["parameters"],
            metrics=row["metrics"],
            daily_returns=daily_returns_from_equity(equities[row["trial_id"]]),
            decision="comparison_only",
        )
        for row in trial_results
    ]
    registry = ExperimentRegistry(output / "experiment_registry.sqlite")
    experiment_id = registry.record_experiment(
        source="surge_pro_family_comparison.py",
        strategy_version=f"v85-guard-surge-pro@{git['commit']}",
        hypothesis=(
            "Compare v8.5, GUARD, and SURGE PRO under one corrected event "
            "clock and frozen data bundle."
        ),
        parameter_space={
            name: variant_parameters(name)
            for name in VARIANT_ORDER
        },
        number_of_trials=3,
        in_sample_period=(
            f"{grid['period']['start_inclusive']}.."
            f"{grid['period']['end_inclusive']}"
        ),
        metrics={
            "family_analysis": analysis,
            "surge_reference_match": reference_match,
            "bundle_id": bundle_manifest["bundle_id"],
        },
        daily_returns=daily_returns_from_equity(equities["mom_surge_pro"]),
        max_drawdown=analysis["metrics"]["mom_surge_pro"]["max_drawdown_pct"],
        sharpe=analysis["metrics"]["mom_surge_pro"]["sharpe"],
        deflated_sharpe=analysis["diagnostic_dsr"]["mom_surge_pro"]["probability"],
        pbo=analysis["diagnostic_pbo"]["pbo"],
        decision="comparison_only",
        command=" ".join(sys.argv),
        notes=(
            "Horizontal diagnostic comparison only; the 22-trial SURGE PRO "
            "acceptance record remains authoritative."
        ),
        trials=registry_trials,
        repo_root=ROOT,
        data_paths=[
            Path(bundle_dir) / "panel.pkl",
            Path(bundle_dir) / "vix.csv",
            Path(bundle_dir) / "grid.json",
        ],
        config={
            "comparison_id": "v85-vs-guard-vs-surge-pro",
            "bundle_id": bundle_manifest["bundle_id"],
            "execution": grid["execution"],
        },
    )
    summary = {
        "record_status": "comparison_only",
        "experiment_id": experiment_id,
        "runner_id": runner_id,
        "git_commit": git["commit"],
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "trial_count": 3,
        "return_observations": int(len(returns)),
        "surge_reference_match": reference_match,
        "analysis": analysis,
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
                "strategies.csv",
                "daily_returns.csv",
                "yearly.csv",
                "experiment_registry.sqlite",
            )
        },
    })
    write_json(output / "run_manifest.json", run_manifest)
    print(
        "[family] complete "
        + " | ".join(
            f"{name}: CAGR={analysis['metrics'][name]['ann_return']:.2%}, "
            f"Sharpe={analysis['metrics'][name]['sharpe']:.3f}"
            for name in VARIANT_ORDER
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner-id", required=True)
    parser.add_argument("--reference-ablation-run")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_family(
        args.bundle,
        args.output,
        args.runner_id,
        reference_ablation_run=args.reference_ablation_run,
        allow_dirty=args.allow_dirty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
