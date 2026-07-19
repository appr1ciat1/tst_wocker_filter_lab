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


def longest_true_run(values: pd.Series) -> int:
    mask = values.fillna(False).astype(bool)
    if not mask.any():
        return 0
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max())


def professional_diagnostics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_close: pd.Series,
) -> dict[str, Any]:
    """Compute institutional-style risk, trade-edge, and benchmark diagnostics."""
    equity_series = equity["Equity"].astype(float).sort_index()
    if not np.isfinite(equity_series.to_numpy(dtype=float)).all():
        raise RuntimeError("equity curve contains non-finite values")
    returns = equity_series.pct_change().dropna()
    if len(returns) < 3:
        raise RuntimeError("professional diagnostics require at least three returns")

    cagr = float(
        (equity_series.iloc[-1] / equity_series.iloc[0])
        ** (252.0 / max(len(returns), 1))
        - 1.0
    )
    cumulative_peak = equity_series.cummax()
    drawdown = equity_series / cumulative_peak - 1.0
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown.to_numpy()))))
    current_drawdown = float(drawdown.iloc[-1])
    max_underwater_days = longest_true_run(drawdown < 0.0)

    trough_date = drawdown.idxmin()
    peak_date = equity_series.loc[:trough_date].idxmax()
    peak_value = float(equity_series.loc[peak_date])
    after_trough = equity_series.loc[trough_date:]
    recovered_rows = after_trough[after_trough >= peak_value]
    recovery_date = recovered_rows.index[0] if not recovered_rows.empty else None
    index_positions = {
        date: position for position, date in enumerate(equity_series.index)
    }
    recovery_days = (
        index_positions[recovery_date] - index_positions[peak_date]
        if recovery_date is not None else None
    )

    q05 = float(returns.quantile(0.05))
    q95 = float(returns.quantile(0.95))
    tail_losses = returns[returns <= q05]
    historical_var_95 = float(-q05)
    historical_cvar_95 = float(-tail_losses.mean())
    tail_ratio = float(q95 / abs(q05)) if q05 < 0 else float("nan")
    positive_sum = float(returns[returns > 0].sum())
    negative_sum = float(abs(returns[returns < 0].sum()))
    omega_zero = positive_sum / negative_sum if negative_sum > 0 else float("inf")

    rolling_mean = returns.rolling(252).mean()
    rolling_std = returns.rolling(252).std()
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()

    monthly = equity_series.resample("ME").last().pct_change().dropna()
    annual = (1.0 + returns).groupby(returns.index.year).prod() - 1.0

    if not trades.empty and "Return_Pct" not in trades:
        raise RuntimeError("trade ledger is missing Return_Pct")
    trade_returns = (
        pd.to_numeric(trades["Return_Pct"], errors="coerce")
        if not trades.empty else pd.Series(dtype=float)
    )
    invalid_trade_returns = int(
        (~np.isfinite(trade_returns.to_numpy(dtype=float))).sum()
    )
    if invalid_trade_returns:
        raise RuntimeError(
            f"trade ledger contains {invalid_trade_returns} non-finite returns"
        )
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float("inf")
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    breakeven_win_rate = (
        1.0 / (1.0 + payoff_ratio)
        if math.isfinite(payoff_ratio) and payoff_ratio > 0 else 0.0
    )
    kelly_fraction = (
        win_rate - (1.0 - win_rate) / payoff_ratio
        if math.isfinite(payoff_ratio) and payoff_ratio > 0 else 0.0
    )
    ordered_trades = trades.copy()
    if not ordered_trades.empty and "Exit_Date" in ordered_trades:
        ordered_trades["_exit"] = pd.to_datetime(ordered_trades["Exit_Date"])
        ordered_trades = ordered_trades.sort_values("_exit", kind="stable")
    ordered_returns = (
        pd.to_numeric(ordered_trades["Return_Pct"], errors="coerce")
        if not ordered_trades.empty else pd.Series(dtype=float)
    )
    max_consecutive_wins = longest_true_run(ordered_returns > 0)
    max_consecutive_losses = longest_true_run(ordered_returns <= 0)
    top_10_profit_concentration = (
        float(wins.nlargest(min(10, len(wins))).sum() / gross_profit)
        if gross_profit > 0 else 0.0
    )
    holding = (
        pd.to_numeric(trades["Days_Held"], errors="coerce")
        if "Days_Held" in trades else pd.Series(np.nan, index=trades.index)
    )
    avg_holding_winners = (
        float(holding.loc[trade_returns[trade_returns > 0].index].mean())
        if len(wins) else 0.0
    )
    avg_holding_losers = (
        float(holding.loc[trade_returns[trade_returns <= 0].index].mean())
        if len(losses) else 0.0
    )

    benchmark_returns = benchmark_close.astype(float).sort_index().pct_change()
    paired = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    benchmark_variance = float(paired["benchmark"].var(ddof=1))
    beta = (
        float(paired["strategy"].cov(paired["benchmark"]) / benchmark_variance)
        if benchmark_variance > 0 else 0.0
    )
    alpha_annual = float(
        (paired["strategy"].mean() - beta * paired["benchmark"].mean()) * 252
    )
    correlation = float(paired["strategy"].corr(paired["benchmark"]))
    active = paired["strategy"] - paired["benchmark"]
    tracking_error = float(active.std(ddof=1) * np.sqrt(252))
    information_ratio = (
        float(active.mean() / active.std(ddof=1) * np.sqrt(252))
        if active.std(ddof=1) > 0 else 0.0
    )
    up = paired["benchmark"] > 0
    down = paired["benchmark"] < 0
    upside_capture = (
        float(paired.loc[up, "strategy"].mean() / paired.loc[up, "benchmark"].mean())
        if up.any() else 0.0
    )
    downside_capture = (
        float(
            paired.loc[down, "strategy"].mean()
            / paired.loc[down, "benchmark"].mean()
        )
        if down.any() else 0.0
    )

    return {
        "data_quality": {
            "trade_records": int(len(trades)),
            "finite_trade_returns": int(len(trade_returns)),
            "invalid_trade_returns": invalid_trade_returns,
            "finite_equity_observations": int(
                np.isfinite(equity_series.to_numpy(dtype=float)).sum()
            ),
            "equity_observations": int(len(equity_series)),
        },
        "trade_edge": {
            "win_rate": win_rate,
            "average_winner": avg_win,
            "average_loser": avg_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
            "expectancy_per_trade": (
                float(trade_returns.mean()) if len(trade_returns) else 0.0
            ),
            "median_trade_return": (
                float(trade_returns.median()) if len(trade_returns) else 0.0
            ),
            "breakeven_win_rate": breakeven_win_rate,
            "win_rate_edge": win_rate - breakeven_win_rate,
            "kelly_fraction_theoretical": kelly_fraction,
            "best_trade": float(trade_returns.max()) if len(trade_returns) else 0.0,
            "worst_trade": float(trade_returns.min()) if len(trade_returns) else 0.0,
            "trade_return_5pct": (
                float(trade_returns.quantile(0.05)) if len(trade_returns) else 0.0
            ),
            "trade_return_95pct": (
                float(trade_returns.quantile(0.95)) if len(trade_returns) else 0.0
            ),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "top_10_profit_concentration": top_10_profit_concentration,
            "average_holding_days_winners": avg_holding_winners,
            "average_holding_days_losers": avg_holding_losers,
            "trades_per_year": float(len(trade_returns) / (len(returns) / 252.0)),
        },
        "tail_and_drawdown": {
            "cagr_from_first_equity": cagr,
            "historical_var_95_daily": historical_var_95,
            "historical_cvar_95_daily": historical_cvar_95,
            "tail_ratio": tail_ratio,
            "omega_ratio_zero": omega_zero,
            "return_skew": float(returns.skew()),
            "excess_kurtosis": float(returns.kurt()),
            "ulcer_index": ulcer_index,
            "current_drawdown": current_drawdown,
            "max_underwater_trading_days": max_underwater_days,
            "max_drawdown_peak_date": pd.Timestamp(peak_date).strftime("%Y-%m-%d"),
            "max_drawdown_trough_date": pd.Timestamp(trough_date).strftime("%Y-%m-%d"),
            "max_drawdown_recovery_date": (
                pd.Timestamp(recovery_date).strftime("%Y-%m-%d")
                if recovery_date is not None else None
            ),
            "max_drawdown_recovery_trading_days": recovery_days,
            "max_drawdown_recovered": recovery_date is not None,
        },
        "stability": {
            "positive_month_ratio": float((monthly > 0).mean()),
            "positive_year_ratio": float((annual > 0).mean()),
            "worst_month": float(monthly.min()),
            "best_month": float(monthly.max()),
            "worst_year": float(annual.min()),
            "best_year": float(annual.max()),
            "annual_return_std": float(annual.std(ddof=1)),
            "rolling_252d_sharpe_median": (
                float(rolling_sharpe.median()) if len(rolling_sharpe) else None
            ),
            "rolling_252d_sharpe_worst": (
                float(rolling_sharpe.min()) if len(rolling_sharpe) else None
            ),
            "rolling_252d_sharpe_positive_ratio": (
                float((rolling_sharpe > 0).mean()) if len(rolling_sharpe) else None
            ),
        },
        "benchmark_relative": {
            "benchmark": BENCHMARK_TICKER,
            "beta": beta,
            "alpha_annual_arithmetic": alpha_annual,
            "correlation": correlation,
            "tracking_error_annual": tracking_error,
            "information_ratio": information_ratio,
            "upside_capture": upside_capture,
            "downside_capture": downside_capture,
            "paired_observations": int(len(paired)),
        },
    }


def professional_metrics_row(
    strategy: str,
    diagnostics: dict[str, Any],
    core_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core = core_metrics or {}
    trade = diagnostics["trade_edge"]
    risk = diagnostics["tail_and_drawdown"]
    stability = diagnostics["stability"]
    relative = diagnostics["benchmark_relative"]
    quality = diagnostics["data_quality"]
    return {
        "strategy": strategy,
        "trade_records": quality["trade_records"],
        "invalid_trade_returns": quality["invalid_trade_returns"],
        "ann_return": core.get("ann_return"),
        "ann_volatility": core.get("ann_volatility"),
        "sharpe": core.get("sharpe"),
        "sortino": core.get("sortino"),
        "calmar": core.get("calmar"),
        "max_drawdown_pct": core.get("max_drawdown_pct"),
        "payoff_ratio": trade["payoff_ratio"],
        "profit_factor": trade["profit_factor"],
        "expectancy_per_trade": trade["expectancy_per_trade"],
        "win_rate": trade["win_rate"],
        "average_winner": trade["average_winner"],
        "average_loser": trade["average_loser"],
        "median_trade_return": trade["median_trade_return"],
        "best_trade": trade["best_trade"],
        "worst_trade": trade["worst_trade"],
        "breakeven_win_rate": trade["breakeven_win_rate"],
        "win_rate_edge": trade["win_rate_edge"],
        "kelly_fraction_theoretical": trade["kelly_fraction_theoretical"],
        "max_consecutive_losses": trade["max_consecutive_losses"],
        "top_10_profit_concentration": trade["top_10_profit_concentration"],
        "historical_var_95_daily": risk["historical_var_95_daily"],
        "historical_cvar_95_daily": risk["historical_cvar_95_daily"],
        "tail_ratio": risk["tail_ratio"],
        "omega_ratio_zero": risk["omega_ratio_zero"],
        "return_skew": risk["return_skew"],
        "excess_kurtosis": risk["excess_kurtosis"],
        "ulcer_index": risk["ulcer_index"],
        "current_drawdown": risk["current_drawdown"],
        "max_underwater_trading_days": risk["max_underwater_trading_days"],
        "max_drawdown_recovery_trading_days": (
            risk["max_drawdown_recovery_trading_days"]
        ),
        "positive_month_ratio": stability["positive_month_ratio"],
        "positive_year_ratio": stability["positive_year_ratio"],
        "annual_return_std": stability["annual_return_std"],
        "rolling_252d_sharpe_median": stability["rolling_252d_sharpe_median"],
        "rolling_252d_sharpe_worst": stability["rolling_252d_sharpe_worst"],
        "rolling_252d_sharpe_positive_ratio": (
            stability["rolling_252d_sharpe_positive_ratio"]
        ),
        "beta_0050": relative["beta"],
        "alpha_annual_arithmetic_0050": relative["alpha_annual_arithmetic"],
        "correlation_0050": relative["correlation"],
        "tracking_error_annual_0050": relative["tracking_error_annual"],
        "information_ratio_0050": relative["information_ratio"],
        "upside_capture_0050": relative["upside_capture"],
        "downside_capture_0050": relative["downside_capture"],
    }


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
        "professional": {
            name: by_id[name].get("professional", {})
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
        professional = clean_metrics(
            professional_diagnostics(equity, trades, data.market_close)
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
            "professional": professional,
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
    pd.DataFrame([
        professional_metrics_row(
            name,
            analysis["professional"][name],
            analysis["metrics"][name],
        )
        for name in VARIANT_ORDER
    ]).to_csv(
        output / "professional_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    registry_trials = [
        trial_record(
            trial_id=row["trial_id"],
            parameters=row["parameters"],
            metrics={
                **row["metrics"],
                "professional": row["professional"],
            },
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
                "professional_metrics.csv",
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
