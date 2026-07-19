"""Reproducible full-period SURGE PRO layer ablation runner.

The runner has three deliberately separate phases:

1. ``prepare`` downloads and freezes one Taiwan OHLCV panel plus one VIX series.
2. ``run`` executes the frozen 22-trial grid without any network access.
3. ``compare`` verifies that two independent runs used the same code/data/grid
   and produced byte-identical daily returns.

Formal acceptance is based on the pre-declared baseline trial. The post-hoc
best trial is reported only as a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from preflight import BENCHMARK_TICKER, load_extended_tickers
from research.experiment_registry import (
    ExperimentRegistry,
    coerce_jsonable,
    daily_returns_from_equity,
    trial_record,
)
from strategies.base import ExecConfig, MarketData
from strategies.momentum_v85 import MomentumV85
from strategies.optimized_v85 import SURGE_PRO_PARAMS, _build_engine
from strategy.ai_strategy import build_liquid_universe, fetch_panel_data
from strategy.risk_metrics import compute_risk_metrics
from twstk.data.contract import (
    FIELDS,
    build_manifest,
    freeze_snapshot,
    load_snapshot,
    validate_panel,
)
from validation.deflated_sharpe import annualized_sharpe, compute_deflated_sharpe
from validation.pbo_cscv import compute_pbo


ROOT = Path(__file__).resolve().parent
DEFAULT_GRID = ROOT / "research" / "grids" / "surge_pro_layer_ablation_v1.json"
UNIVERSE_TOP_N = 60
UNIVERSE_LOOKBACK = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        coerce_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            coerce_jsonable(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(target)


def git_state(repo_root: Path = ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "yfinance": yf.__version__,
    }


def load_grid(path: str | os.PathLike[str] = DEFAULT_GRID) -> tuple[dict[str, Any], str]:
    grid = read_json(path)
    trials = grid.get("trials", [])
    ids = [trial.get("trial_id") for trial in trials]
    expected = int(grid.get("acceptance", {}).get("complete_trials", -1))
    if len(trials) != expected or expected != 22:
        raise ValueError(f"grid must contain exactly 22 trials, found {len(trials)}")
    if len(ids) != len(set(ids)) or any(not trial_id for trial_id in ids):
        raise ValueError("grid trial IDs must be non-empty and unique")
    for required in ("baseline_trial_id", "matched_no_sizing_trial_id"):
        if grid.get(required) not in ids:
            raise ValueError(f"{required} is not present in the grid")
    acceptance = grid["acceptance"]
    if acceptance["cscv_partitions"] != 8 or acceptance["expected_cscv_splits"] != 70:
        raise ValueError("formal grid requires 8 CSCV partitions and 70 splits")

    neighborhood = [trial for trial in trials if trial.get("group") == "tier_neighborhood"]
    no_tiers = [trial for trial in trials if trial.get("group") == "no_strong_tiers"]
    no_sizing = [trial for trial in trials if trial.get("group") == "no_regime_sizing"]
    expected_neighborhood = {
        (gap, vix, scale)
        for gap in (False, True)
        for vix in (25.0, 28.0, 31.0)
        for scale in (1.7, 1.8, 1.9)
    }
    actual_neighborhood = {
        (
            trial.get("dynamic_gap_filter"),
            trial.get("strong_vix_max"),
            trial.get("max_regime_scale"),
        )
        for trial in neighborhood
    }
    if len(neighborhood) != 18 or actual_neighborhood != expected_neighborhood:
        raise ValueError("grid must contain the exact 2x3x3 tier neighborhood")
    if (
        len(no_tiers) != 2
        or {trial.get("dynamic_gap_filter") for trial in no_tiers} != {False, True}
        or any(not trial.get("regime_sizing") for trial in no_tiers)
        or any(trial.get("strong_tiers") is not None for trial in no_tiers)
    ):
        raise ValueError("grid must contain exactly two matched no-tier trials")
    if (
        len(no_sizing) != 2
        or {trial.get("dynamic_gap_filter") for trial in no_sizing} != {False, True}
        or any(trial.get("regime_sizing") for trial in no_sizing)
        or any(trial.get("strong_tiers") is not None for trial in no_sizing)
    ):
        raise ValueError("grid must contain exactly two matched no-sizing trials")

    effective_configs = []
    for trial in trials:
        effective = {
            **grid["fixed_strategy_parameters"],
            **{
                key: trial[key]
                for key in (
                    "regime_sizing",
                    "strong_tiers",
                    "dynamic_gap_filter",
                    "strong_vix_max",
                    "max_regime_scale",
                )
            },
        }
        effective_configs.append(payload_sha256(effective))
    if len(effective_configs) != len(set(effective_configs)):
        raise ValueError("grid contains duplicate effective configurations")

    baseline = next(
        trial for trial in trials
        if trial["trial_id"] == grid["baseline_trial_id"]
    )
    baseline_effective = {
        **grid["fixed_strategy_parameters"],
        **{
            key: baseline[key]
            for key in (
                "regime_sizing",
                "strong_tiers",
                "dynamic_gap_filter",
                "strong_vix_max",
                "max_regime_scale",
            )
        },
    }
    if payload_sha256(baseline_effective) != payload_sha256(SURGE_PRO_PARAMS):
        raise ValueError("baseline trial does not match canonical SURGE_PRO_PARAMS")
    return grid, payload_sha256(grid)


def vix_payload(series: pd.Series) -> list[dict[str, Any]]:
    clean = series.astype(float).sort_index()
    return [
        {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "value": float(value)}
        for date, value in clean.items()
    ]


def vix_sha256(series: pd.Series) -> str:
    return payload_sha256(vix_payload(series))


def persist_vix(series: pd.Series, path: str | os.PathLike[str]) -> pd.Series:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    series.sort_index().rename("VIX").to_frame().to_csv(
        target,
        index_label="Date",
        float_format="%.17g",
        lineterminator="\n",
    )
    persisted = pd.read_csv(target, index_col=0, parse_dates=True)
    return normalize_vix(persisted)


def normalize_vix(raw: pd.DataFrame) -> pd.Series:
    if raw.empty:
        raise RuntimeError("VIX download returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("VIX download is missing Close")
        close = raw["Close"]
        series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    elif "Close" in raw.columns:
        series = raw["Close"]
    elif "VIX" in raw.columns:
        series = raw["VIX"]
    else:
        raise RuntimeError("VIX download is missing Close")

    series = pd.Series(series, dtype=float).dropna()
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    series.index = index.normalize()
    series = series.groupby(level=0).last().sort_index()
    if series.empty or series.index.duplicated().any():
        raise RuntimeError("VIX series is empty or has duplicate dates")
    if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
        raise RuntimeError("VIX series contains non-finite or non-positive values")
    series.name = "VIX"
    return series


def download_vix(start: str, end_exclusive: str) -> pd.Series:
    raw = yf.download(
        "^VIX",
        start=pd.Timestamp(start),
        end=pd.Timestamp(end_exclusive),
        auto_adjust=False,
        progress=False,
    )
    series = normalize_vix(raw)
    if series.index[0] > pd.Timestamp(start) + pd.Timedelta(days=7):
        raise RuntimeError(f"VIX begins too late: {series.index[0].date()}")
    expected_last = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    if series.index[-1] < expected_last - pd.Timedelta(days=4):
        raise RuntimeError(f"VIX ends too early: {series.index[-1].date()}")
    return series


def invalid_ohlc_mask(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = panel["Close"]
    valid = (
        close.notna()
        & panel["Open"].notna()
        & panel["High"].notna()
        & panel["Low"].notna()
    )
    upper = np.maximum(close, panel["Open"])
    lower = np.minimum(close, panel["Open"])
    high_close_enough = pd.DataFrame(
        np.isclose(panel["High"], upper, rtol=1e-12, atol=1e-10),
        index=close.index,
        columns=close.columns,
    )
    low_close_enough = pd.DataFrame(
        np.isclose(panel["Low"], lower, rtol=1e-12, atol=1e-10),
        index=close.index,
        columns=close.columns,
    )
    return (
        valid
        & (
            ((panel["High"] < upper) & ~high_close_enough)
            | ((panel["Low"] > lower) & ~low_close_enough)
        )
    )


def mask_invalid_tradable_bars(
    panel: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Mask vendor-impossible OHLC bars instead of inventing executable prices."""
    cleaned = {field: frame.copy() for field, frame in panel.items()}
    invalid = invalid_ohlc_mask(cleaned)
    anomalies: list[dict[str, Any]] = []
    for row, column in np.argwhere(invalid.to_numpy()):
        date = cleaned["Close"].index[row]
        ticker = str(cleaned["Close"].columns[column])
        anomalies.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "open": float(cleaned["Open"].iloc[row, column]),
            "high": float(cleaned["High"].iloc[row, column]),
            "low": float(cleaned["Low"].iloc[row, column]),
            "close": float(cleaned["Close"].iloc[row, column]),
            "action": "Open/High/Low/Volume masked; Close retained for marking",
        })
    for field in ("Open", "High", "Low", "Volume"):
        cleaned[field] = cleaned[field].mask(invalid)
    return cleaned, anomalies


def validate_research_panel(panel: dict[str, pd.DataFrame], as_of: str) -> None:
    contract = validate_panel(
        panel,
        as_of,
        scheduled=True,
        min_completeness=0.90,
        min_column_coverage=0.50,
        key_tickers=(BENCHMARK_TICKER,),
        require_volume=True,
    )
    if not contract.ok:
        raise RuntimeError(contract.summary())

    close = panel["Close"]
    if close.index.duplicated().any() or close.columns.duplicated().any():
        raise RuntimeError("Close has duplicate dates or tickers")
    if not close.index.is_monotonic_increasing:
        raise RuntimeError("Close dates are not monotonic")
    for field in FIELDS:
        frame = panel.get(field)
        if frame is None or frame.empty:
            raise RuntimeError(f"{field} is empty")
        if not frame.index.equals(close.index):
            raise RuntimeError(f"{field} dates differ from Close")
        if set(frame.columns) != set(close.columns):
            raise RuntimeError(f"{field} tickers differ from Close")
    if (panel["Volume"].dropna() < 0).any().any():
        raise RuntimeError("Volume contains negative values")

    if invalid_ohlc_mask(panel).any().any():
        raise RuntimeError("OHLC price relationships are invalid")

    for ticker in close.columns:
        first_valid = close[ticker].first_valid_index()
        if first_valid is None:
            raise RuntimeError(f"{ticker} has no valid Close observations")
        joint = pd.DataFrame({
            field: panel[field].loc[first_valid:, ticker]
            for field in FIELDS
        })
        finite_prices = np.isfinite(
            joint[["Close", "Open", "High", "Low"]].to_numpy(dtype=float)
        ).all(axis=1)
        finite_volume = np.isfinite(joint["Volume"].to_numpy(dtype=float))
        joint_coverage = float(np.mean(finite_prices & finite_volume))
        if joint_coverage < 0.90:
            raise RuntimeError(
                f"{ticker} joint OHLCV coverage after first listing is "
                f"{joint_coverage:.1%}, below 90%"
            )


def bundle_identity(
    panel_hash: str,
    vix_hash: str,
    grid_hash: str,
    period: dict[str, str],
    ticker_order: list[str],
) -> str:
    return payload_sha256({
        "panel_sha256": panel_hash,
        "vix_sha256": vix_hash,
        "grid_sha256": grid_hash,
        "period": period,
        "ticker_order": ticker_order,
        "universe_top_n": UNIVERSE_TOP_N,
        "universe_lookback": UNIVERSE_LOOKBACK,
    })


def prepare_bundle(
    grid_path: str | os.PathLike[str],
    bundle_dir: str | os.PathLike[str],
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    grid, grid_hash = load_grid(grid_path)
    prepared_by_git = git_state()
    if prepared_by_git["dirty"] and not allow_dirty:
        raise RuntimeError("formal bundle preparation requires a clean git worktree")
    if os.environ.get("TWSTK_SNAPSHOT"):
        raise RuntimeError(
            "TWSTK_SNAPSHOT must be unset during formal bundle preparation"
        )
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        _, _, manifest, _ = verify_bundle(bundle)
        return manifest
    if bundle.exists() and any(bundle.iterdir()):
        raise RuntimeError(f"refusing non-empty bundle directory without manifest: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)

    period = grid["period"]
    tickers = load_extended_tickers(ROOT / "ai_report.py")
    fetch_list = list(dict.fromkeys([BENCHMARK_TICKER, *tickers]))
    print(f"[prepare] downloading {len(fetch_list)} Taiwan symbols", flush=True)
    close, open_, high, low, volume = fetch_panel_data(
        fetch_list,
        start_date=period["start_inclusive"],
        end_date=period["download_end_exclusive"],
    )
    raw_panel = {
        "Close": close,
        "Open": open_,
        "High": high,
        "Low": low,
        "Volume": volume,
    }
    panel, ohlcv_anomalies = mask_invalid_tradable_bars(raw_panel)
    if ohlcv_anomalies:
        print(
            f"[prepare] masked {len(ohlcv_anomalies)} vendor-impossible "
            "OHLC bar(s) as untradable",
            flush=True,
        )
    validate_research_panel(panel, period["end_inclusive"])

    print("[prepare] downloading and freezing ^VIX", flush=True)
    downloaded_vix = download_vix(
        period["start_inclusive"],
        period["download_end_exclusive"],
    )
    vix_path = bundle / "vix.csv"
    vix = persist_vix(downloaded_vix, vix_path)
    write_json(bundle / "grid.json", grid)

    contract = validate_panel(
        panel,
        period["end_inclusive"],
        scheduled=True,
        min_completeness=0.90,
        min_column_coverage=0.50,
        key_tickers=(BENCHMARK_TICKER,),
    )
    manifest = build_manifest(
        panel,
        period["end_inclusive"],
        provider="yfinance",
        auto_adjust=True,
        contract=contract,
    )
    manifest.update({
        "bundle_schema_version": 1,
        "bundle_created_at": utc_now(),
        "period": period,
        "grid_id": grid["grid_id"],
        "grid_sha256": grid_hash,
        "vix_symbol": "^VIX",
        "vix_auto_adjust": False,
        "vix_first_date": vix.index[0].strftime("%Y-%m-%d"),
        "vix_last_date": vix.index[-1].strftime("%Y-%m-%d"),
        "vix_observations": int(len(vix)),
        "vix_sha256": vix_sha256(vix),
        "ticker_order": [str(ticker) for ticker in close.columns],
        "universe_top_n": UNIVERSE_TOP_N,
        "universe_lookback": UNIVERSE_LOOKBACK,
        "vix_injection_policy": (
            "inject frozen VIX when regime_sizing=true; pass None when false "
            "to preserve the strategy's no-sizing semantics"
        ),
        "ohlcv_anomaly_policy": (
            "Vendor bars with High < max(Open,Close) or "
            "Low > min(Open,Close), beyond float tolerance, retain Close for "
            "marking and mask Open/High/Low/Volume as untradable."
        ),
        "ohlcv_anomalies": ohlcv_anomalies,
        "statistics_protocol": {
            "pbo": (
                "CSCV S=8; all C(8,4)=70 directional train/test splits; "
                "annualized arithmetic Sharpe; average OOS tie ranks"
            ),
            "dsr": (
                "pre-declared baseline; hurdle centered on the mean of all "
                "22 observed annualized trial Sharpes; cross-trial std ddof=1"
            ),
            "best_trial": "diagnostic_only",
        },
        "prepared_by_git": prepared_by_git,
        "dependencies": dependency_versions(),
    })
    manifest["bundle_id"] = bundle_identity(
        manifest["panel_sha256"],
        manifest["vix_sha256"],
        manifest["grid_sha256"],
        manifest["period"],
        manifest["ticker_order"],
    )
    freeze_snapshot(panel, bundle, manifest)
    verify_bundle(bundle)
    print(
        f"[prepare] bundle={manifest['bundle_id']} "
        f"panel={manifest['panel_sha256']} vix={manifest['vix_sha256']}",
        flush=True,
    )
    return manifest


def verify_bundle(
    bundle_dir: str | os.PathLike[str],
) -> tuple[dict[str, pd.DataFrame], pd.Series, dict[str, Any], dict[str, Any]]:
    bundle = Path(bundle_dir)
    manifest = read_json(bundle / "manifest.json")
    grid = read_json(bundle / "grid.json")
    _, grid_hash = load_grid(bundle / "grid.json")
    if grid_hash != manifest.get("grid_sha256"):
        raise RuntimeError("bundle grid hash mismatch")

    loaded = load_snapshot(bundle, verify_manifest=True)
    panel = dict(zip(FIELDS, loaded))
    actual_ticker_order = [str(ticker) for ticker in panel["Close"].columns]
    if actual_ticker_order != manifest.get("ticker_order"):
        raise RuntimeError(
            "bundle ticker order differs from the order bound in the manifest"
        )
    vix_frame = pd.read_csv(bundle / "vix.csv", index_col=0, parse_dates=True)
    if list(vix_frame.columns) != ["VIX"]:
        raise RuntimeError("bundle VIX CSV must contain exactly one VIX column")
    vix = normalize_vix(vix_frame)
    if vix_sha256(vix) != manifest.get("vix_sha256"):
        raise RuntimeError("bundle VIX hash mismatch")
    expected_id = bundle_identity(
        manifest["panel_sha256"],
        manifest["vix_sha256"],
        manifest["grid_sha256"],
        manifest["period"],
        actual_ticker_order,
    )
    if expected_id != manifest.get("bundle_id"):
        raise RuntimeError("bundle identity mismatch")
    validate_research_panel(panel, manifest["period"]["end_inclusive"])
    return panel, vix, manifest, grid


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return coerce_jsonable(metrics)


def strategy_parameters_for_trial(
    grid: dict[str, Any],
    trial: dict[str, Any],
) -> dict[str, Any]:
    return {
        **grid["fixed_strategy_parameters"],
        "regime_sizing": trial["regime_sizing"],
        "strong_tiers": trial["strong_tiers"],
        "dynamic_gap_filter": trial["dynamic_gap_filter"],
        "strong_vix_max": trial["strong_vix_max"],
        "max_regime_scale": trial["max_regime_scale"],
    }


def effective_trial_config(
    grid: dict[str, Any],
    trial: dict[str, Any],
    bundle_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_id": grid["strategy_id"],
        "grid_id": grid["grid_id"],
        "trial_id": trial["trial_id"],
        "strategy_parameters": strategy_parameters_for_trial(grid, trial),
        "execution": grid["execution"],
        "period": grid["period"],
        "universe": {
            "top_n": UNIVERSE_TOP_N,
            "lookback": UNIVERSE_LOOKBACK,
            "ticker_order": bundle_manifest["ticker_order"],
        },
        "engine_implicit_defaults": {
            "tp_atr": 4.0,
            "gap_filter_atr": 1.5,
            "regime_filter": True,
            "hybrid_tiered": False,
        },
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "vix_policy": bundle_manifest["vix_injection_policy"],
        "statistics_protocol": bundle_manifest["statistics_protocol"],
    }


def aligned_returns(equities: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = {
        trial_id: equity["Equity"].sort_index().pct_change().dropna()
        for trial_id, equity in equities.items()
    }
    frame = pd.DataFrame(returns).sort_index()
    if frame.empty or frame.isna().any().any():
        missing = frame.isna().sum()
        raise RuntimeError(
            "trial return calendars are not identical: "
            + ", ".join(f"{key}={int(value)}" for key, value in missing.items() if value)
        )
    if not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("trial returns contain non-finite values")
    return frame


def analyze_trials(
    trial_results: list[dict[str, Any]],
    returns: pd.DataFrame,
    grid: dict[str, Any],
) -> dict[str, Any]:
    by_id = {row["trial_id"]: row for row in trial_results}
    expected_ids = [trial["trial_id"] for trial in grid["trials"]]
    if list(returns.columns) != expected_ids:
        raise RuntimeError("return columns do not match frozen grid order")
    if set(by_id) != set(expected_ids):
        raise RuntimeError("trial results do not match frozen grid")

    pbo = compute_pbo(
        returns,
        n_splits=grid["acceptance"]["cscv_partitions"],
    )
    if len(pbo.split_results) != grid["acceptance"]["expected_cscv_splits"]:
        raise RuntimeError(
            f"expected 70 CSCV splits, computed {len(pbo.split_results)}"
        )

    sharpes = [annualized_sharpe(returns[trial_id]) for trial_id in expected_ids]
    baseline_id = grid["baseline_trial_id"]
    best_id = expected_ids[int(np.argmax(sharpes))]
    baseline_dsr = compute_deflated_sharpe(
        returns[baseline_id],
        n_trials=len(sharpes),
        trial_sharpes=sharpes,
    )
    best_dsr = compute_deflated_sharpe(
        returns[best_id],
        n_trials=len(sharpes),
        trial_sharpes=sharpes,
    )

    no_sizing_id = grid["matched_no_sizing_trial_id"]
    no_tiers_id = "rs1_tiers0_gap1_vix28_scale190"
    baseline_no_gap_id = "rs1_tiers1_gap0_vix28_scale190"
    no_tiers_no_gap_id = "rs1_tiers0_gap0_vix28_scale190"
    no_sizing_no_gap_id = "rs0_tiers0_gap0_vix28_scale190"
    baseline_metrics = by_id[baseline_id]["metrics"]
    no_sizing_metrics = by_id[no_sizing_id]["metrics"]
    no_tiers_metrics = by_id[no_tiers_id]["metrics"]
    rules = grid["acceptance"]
    checks = {
        "complete_trials": len(trial_results) == rules["complete_trials"],
        "pbo": math.isfinite(pbo.pbo) and pbo.pbo <= rules["max_pbo"],
        "baseline_dsr": (
            baseline_dsr.probability >= rules["min_baseline_dsr_probability"]
        ),
        "baseline_cagr_gt_no_sizing": (
            float(baseline_metrics["ann_return"])
            > float(no_sizing_metrics["ann_return"])
        ),
        "baseline_sharpe_gt_no_sizing": (
            float(baseline_metrics["sharpe"])
            > float(no_sizing_metrics["sharpe"])
        ),
    }

    def effect(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        return {
            "ann_return_delta": (
                float(left["ann_return"]) - float(right["ann_return"])
            ),
            "sharpe_delta": float(left["sharpe"]) - float(right["sharpe"]),
            "max_drawdown_delta": (
                float(left["max_drawdown_pct"])
                - float(right["max_drawdown_pct"])
            ),
        }

    return {
        "decision": "accept" if all(checks.values()) else "reject",
        "decision_scope": (
            "statistical candidate only; independent reproduction is required "
            "before registry promotion"
        ),
        "dsr_convention": (
            "hurdle center = mean of all 22 observed annualized trial Sharpes; "
            "cross-trial standard deviation uses ddof=1"
        ),
        "checks": checks,
        "pbo": asdict(pbo),
        "baseline_trial_id": baseline_id,
        "baseline_metrics": baseline_metrics,
        "baseline_dsr": asdict(baseline_dsr),
        "best_trial_id_diagnostic_only": best_id,
        "best_metrics_diagnostic_only": by_id[best_id]["metrics"],
        "best_dsr_diagnostic_only": asdict(best_dsr),
        "trial_sharpes": dict(zip(expected_ids, sharpes)),
        "matched_effects": {
            "strong_tiers_increment_at_baseline": effect(
                baseline_metrics,
                no_tiers_metrics,
            ),
            "base_regime_sizing_increment": effect(
                no_tiers_metrics,
                no_sizing_metrics,
            ),
            "all_sizing_layers_increment": effect(
                baseline_metrics,
                no_sizing_metrics,
            ),
            "dynamic_gap_increment_at_baseline": effect(
                baseline_metrics,
                by_id[baseline_no_gap_id]["metrics"],
            ),
            "dynamic_gap_increment_without_tiers": effect(
                no_tiers_metrics,
                by_id[no_tiers_no_gap_id]["metrics"],
            ),
            "dynamic_gap_increment_without_sizing": effect(
                no_sizing_metrics,
                by_id[no_sizing_no_gap_id]["metrics"],
            ),
        },
    }


def record_formal_experiment(
    output: Path,
    grid: dict[str, Any],
    bundle_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    trial_results: list[dict[str, Any]],
    equities: dict[str, pd.DataFrame],
    analysis: dict[str, Any],
) -> str:
    baseline_id = grid["baseline_trial_id"]
    baseline = next(row for row in trial_results if row["trial_id"] == baseline_id)
    registry_trials = [
        trial_record(
            trial_id=row["trial_id"],
            parameters=row["parameters"],
            metrics=row["metrics"],
            daily_returns=daily_returns_from_equity(equities[row["trial_id"]]),
            decision="provisional",
        )
        for row in trial_results
    ]
    registry = ExperimentRegistry(output / "experiment_registry.sqlite")
    experiment_id = (
        f"{grid['grid_id']}--{run_manifest['runner_id']}--"
        f"{run_manifest['started_at'].replace(':', '').replace('+', '_')}"
    )
    return registry.record_experiment(
        experiment_id=experiment_id,
        source="surge_pro_ablation_runner.py",
        strategy_version=(
            f"{grid['strategy_id']}@{run_manifest['git']['commit']}"
        ),
        hypothesis=grid["purpose"],
        parameter_space=grid["varied_parameters"],
        number_of_trials=len(trial_results),
        in_sample_period=(
            f"{grid['period']['start_inclusive']}.."
            f"{grid['period']['end_inclusive']}"
        ),
        metrics={
            "formal_analysis": analysis,
            "bundle_id": bundle_manifest["bundle_id"],
            "panel_sha256": bundle_manifest["panel_sha256"],
            "vix_sha256": bundle_manifest["vix_sha256"],
            "grid_sha256": bundle_manifest["grid_sha256"],
            "runner_id": run_manifest["runner_id"],
        },
        daily_returns=daily_returns_from_equity(equities[baseline_id]),
        max_drawdown=baseline["metrics"]["max_drawdown_pct"],
        sharpe=baseline["metrics"]["sharpe"],
        deflated_sharpe=analysis["baseline_dsr"]["probability"],
        pbo=analysis["pbo"]["pbo"],
        decision="provisional",
        command=" ".join(sys.argv),
        notes=(
            "Provisional full-period SURGE PRO 22-trial layer ablation; "
            "requires a second independent output and compare promotion. "
            "Best trial is diagnostic only."
        ),
        trials=registry_trials,
        repo_root=ROOT,
        data_paths=[
            Path(bundle_manifest["bundle_path"]) / "panel.pkl",
            Path(bundle_manifest["bundle_path"]) / "vix.csv",
            Path(bundle_manifest["bundle_path"]) / "grid.json",
        ],
        config=baseline["parameters"],
    )


def run_grid(
    bundle_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    runner_id: str,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    panel, vix, bundle_manifest, grid = verify_bundle(bundle_dir)
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("formal run requires a clean git worktree")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_schema_version": 1,
        "status": "running",
        "runner_id": runner_id,
        "started_at": utc_now(),
        "git": git,
        "dependencies": dependency_versions(),
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "grid_id": grid["grid_id"],
        "grid_sha256": bundle_manifest["grid_sha256"],
        "runner_sha256": file_sha256(__file__),
        "bundle_path": str(Path(bundle_dir).resolve()),
        "command": " ".join(sys.argv),
    }
    write_json(output / "run_manifest.json", run_manifest)

    close_all = panel["Close"]
    if BENCHMARK_TICKER not in close_all.columns:
        raise RuntimeError(f"bundle is missing benchmark {BENCHMARK_TICKER}")
    stock_columns = [
        str(ticker) for ticker in close_all.columns
        if str(ticker) != BENCHMARK_TICKER
    ]
    if not stock_columns:
        raise RuntimeError("bundle contains no stock tickers")
    market_close = close_all[BENCHMARK_TICKER].copy()
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
        market_close=market_close,
        universe_mask=universe,
    )
    signals = MomentumV85().prepare(data)
    execution = ExecConfig(**grid["execution"])

    trial_results: list[dict[str, Any]] = []
    equities: dict[str, pd.DataFrame] = {}
    total = len(grid["trials"])
    for number, trial in enumerate(grid["trials"], 1):
        trial_id = trial["trial_id"]
        print(f"[run {runner_id}] {number:02d}/{total} {trial_id}", flush=True)
        started = time.perf_counter()
        strategy_parameters = strategy_parameters_for_trial(grid, trial)
        parameters = effective_trial_config(grid, trial, bundle_manifest)
        engine = _build_engine(strategy_parameters, execution)
        frozen_vix = vix if strategy_parameters["regime_sizing"] else None
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
            vix_series=frozen_vix,
        )
        if equity.empty or "Equity" not in equity.columns:
            raise RuntimeError(f"{trial_id} produced no equity curve")
        metrics = clean_metrics(
            compute_risk_metrics(equity, trades, execution.initial_capital)
        )
        row = {
            "trial_id": trial_id,
            "group": trial["group"],
            "parameters": parameters,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trial_dir = output / "trials" / trial_id
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
        equities[trial_id] = equity
        trial_results.append(row)
        write_json(output / "checkpoint.json", {
            "bundle_id": bundle_manifest["bundle_id"],
            "grid_sha256": bundle_manifest["grid_sha256"],
            "runner_id": runner_id,
            "completed_trial_ids": [item["trial_id"] for item in trial_results],
        })

    returns = aligned_returns(equities)
    analysis = analyze_trials(trial_results, returns, grid)
    summary_rows = []
    for row in trial_results:
        metrics = row["metrics"]
        summary_rows.append({
            "trial_id": row["trial_id"],
            "group": row["group"],
            "ann_return": metrics["ann_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "total_trades": metrics["total_trades"],
        })
    pd.DataFrame(summary_rows).to_csv(
        output / "trials.csv",
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
    pd.DataFrame(analysis["pbo"]["split_results"]).to_csv(
        output / "pbo_splits.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    bundle_manifest_for_registry = {
        **bundle_manifest,
        "bundle_path": str(Path(bundle_dir).resolve()),
    }
    experiment_id = record_formal_experiment(
        output,
        grid,
        bundle_manifest_for_registry,
        run_manifest,
        trial_results,
        equities,
        analysis,
    )
    summary = {
        "grid_id": grid["grid_id"],
        "grid_sha256": bundle_manifest["grid_sha256"],
        "bundle_id": bundle_manifest["bundle_id"],
        "panel_sha256": bundle_manifest["panel_sha256"],
        "vix_sha256": bundle_manifest["vix_sha256"],
        "git_commit": git["commit"],
        "runner_id": runner_id,
        "experiment_id": experiment_id,
        "record_status": "provisional",
        "trial_count": len(trial_results),
        "return_observations": int(len(returns)),
        "analysis": analysis,
    }
    write_json(output / "summary.json", summary)

    run_manifest.update({
        "status": "complete",
        "completed_at": utc_now(),
        "experiment_id": experiment_id,
        "candidate_decision": analysis["decision"],
        "record_status": "provisional",
        "output_hashes": {
            name: file_sha256(output / name)
            for name in (
                "summary.json",
                "trials.csv",
                "daily_returns.csv",
                "pbo_splits.csv",
                "experiment_registry.sqlite",
            )
        },
    })
    write_json(output / "run_manifest.json", run_manifest)
    print(
        f"[run {runner_id}] complete decision={analysis['decision']} "
        f"PBO={analysis['pbo']['pbo']:.4f} "
        f"baseline_DSR={analysis['baseline_dsr']['probability']:.4f}",
        flush=True,
    )
    return summary


def verified_run_output(
    run_dir: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(run_dir).resolve()
    manifest = read_json(path / "run_manifest.json")
    summary = read_json(path / "summary.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {path}")
    if manifest.get("record_status") != "provisional":
        raise RuntimeError(f"single-run registry must remain provisional: {path}")
    for name, expected_hash in manifest.get("output_hashes", {}).items():
        actual_hash = file_sha256(path / name)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"declared output hash mismatch for {path / name}: "
                f"{expected_hash} != {actual_hash}"
            )
    bindings = {
        "runner_id": manifest.get("runner_id"),
        "bundle_id": manifest.get("bundle_id"),
        "grid_sha256": manifest.get("grid_sha256"),
        "panel_sha256": manifest.get("panel_sha256"),
        "vix_sha256": manifest.get("vix_sha256"),
        "git_commit": manifest.get("git", {}).get("commit"),
        "experiment_id": manifest.get("experiment_id"),
    }
    for key, value in bindings.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"run manifest/summary binding mismatch for {key}: {path}"
            )
    return path, manifest, summary


def consolidated_acceptance_record(
    left: Path,
    left_manifest: dict[str, Any],
    left_summary: dict[str, Any],
    right_manifest: dict[str, Any],
    right_summary: dict[str, Any],
    comparison: dict[str, Any],
    registry_path: str | os.PathLike[str],
) -> str:
    bundle = Path(left_manifest["bundle_path"])
    _, _, bundle_manifest, grid = verify_bundle(bundle)
    returns = pd.read_csv(
        left / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    if list(returns.columns) != [trial["trial_id"] for trial in grid["trials"]]:
        raise RuntimeError("consolidated returns do not match frozen grid order")
    decision = left_summary["analysis"]["decision"]
    registry_trials = []
    baseline_parameters = None
    for trial in grid["trials"]:
        trial_id = trial["trial_id"]
        result = read_json(left / "trials" / trial_id / "result.json")
        daily_returns = [
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "return": float(value),
            }
            for date, value in returns[trial_id].items()
        ]
        registry_trials.append(trial_record(
            trial_id=trial_id,
            parameters=result["parameters"],
            metrics=result["metrics"],
            daily_returns=daily_returns,
            decision=decision,
            notes="Promoted only after independent reproduction.",
        ))
        if trial_id == grid["baseline_trial_id"]:
            baseline_parameters = result["parameters"]
    if baseline_parameters is None:
        raise RuntimeError("baseline parameters missing during promotion")

    baseline_id = grid["baseline_trial_id"]
    baseline_metrics = left_summary["analysis"]["baseline_metrics"]
    baseline_daily_returns = [
        {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "return": float(value),
        }
        for date, value in returns[baseline_id].items()
    ]
    registry = ExperimentRegistry(registry_path)
    experiment_id = (
        f"{grid['grid_id']}--independent-acceptance--"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return registry.record_experiment(
        experiment_id=experiment_id,
        source="surge_pro_ablation_runner.py compare",
        strategy_version=f"{grid['strategy_id']}@{left_summary['git_commit']}",
        hypothesis=grid["purpose"],
        parameter_space=grid["varied_parameters"],
        number_of_trials=len(grid["trials"]),
        in_sample_period=(
            f"{grid['period']['start_inclusive']}.."
            f"{grid['period']['end_inclusive']}"
        ),
        metrics={
            "formal_analysis": left_summary["analysis"],
            "independent_reproduction": comparison,
            "left_experiment_id": left_summary["experiment_id"],
            "right_experiment_id": right_summary["experiment_id"],
            "bundle_id": left_summary["bundle_id"],
            "panel_sha256": left_summary["panel_sha256"],
            "vix_sha256": left_summary["vix_sha256"],
            "grid_sha256": left_summary["grid_sha256"],
        },
        daily_returns=baseline_daily_returns,
        max_drawdown=baseline_metrics["max_drawdown_pct"],
        sharpe=baseline_metrics["sharpe"],
        deflated_sharpe=left_summary["analysis"]["baseline_dsr"]["probability"],
        pbo=left_summary["analysis"]["pbo"]["pbo"],
        decision=decision,
        command=" ".join(sys.argv),
        notes=(
            "Consolidated formal record created only after two distinct runner "
            "IDs and output directories reproduced identical evidence."
        ),
        trials=registry_trials,
        repo_root=ROOT,
        data_paths=[
            bundle / "panel.pkl",
            bundle / "vix.csv",
            bundle / "grid.json",
        ],
        config=baseline_parameters,
    )


def compare_runs(
    left_dir: str | os.PathLike[str],
    right_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    registry_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    left, left_manifest, left_summary = verified_run_output(left_dir)
    right, right_manifest, right_summary = verified_run_output(right_dir)
    if left == right:
        raise RuntimeError("independent comparison requires two distinct directories")
    if (
        not left_summary.get("runner_id")
        or not right_summary.get("runner_id")
        or left_summary["runner_id"] == right_summary["runner_id"]
    ):
        raise RuntimeError("independent comparison requires two distinct runner IDs")

    identity_keys = (
        "grid_id",
        "grid_sha256",
        "bundle_id",
        "panel_sha256",
        "vix_sha256",
        "git_commit",
        "trial_count",
        "return_observations",
    )
    identity = {
        key: left_summary.get(key) == right_summary.get(key)
        for key in identity_keys
    }
    manifest_identity = {
        "runner_sha256": (
            left_manifest.get("runner_sha256")
            == right_manifest.get("runner_sha256")
        ),
        "dependencies": (
            left_manifest.get("dependencies")
            == right_manifest.get("dependencies")
        ),
        "git": left_manifest.get("git") == right_manifest.get("git"),
        "bundle_path": (
            Path(left_manifest["bundle_path"]).resolve()
            == Path(right_manifest["bundle_path"]).resolve()
        ),
    }
    evidence = {
        "daily_returns_byte_identical": (
            file_sha256(left / "daily_returns.csv")
            == file_sha256(right / "daily_returns.csv")
        ),
        "trials_byte_identical": (
            file_sha256(left / "trials.csv")
            == file_sha256(right / "trials.csv")
        ),
        "pbo_splits_byte_identical": (
            file_sha256(left / "pbo_splits.csv")
            == file_sha256(right / "pbo_splits.csv")
        ),
        "formal_analysis_identical": (
            left_summary["analysis"] == right_summary["analysis"]
        ),
    }
    result = {
        "comparison_schema_version": 1,
        "created_at": utc_now(),
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "left_runner_id": left_summary["runner_id"],
        "right_runner_id": right_summary["runner_id"],
        "identity_checks": identity,
        "manifest_checks": manifest_identity,
        "evidence_checks": evidence,
        "reproduced": (
            all(identity.values())
            and all(manifest_identity.values())
            and all(evidence.values())
        ),
    }
    if not result["reproduced"]:
        write_json(output_path, result)
        raise RuntimeError(f"independent runs differ; see {output_path}")
    result["comparison_core_sha256"] = payload_sha256(result)
    formal_registry = Path(
        registry_path
        if registry_path is not None
        else Path(output_path).with_suffix(".sqlite")
    ).resolve()
    experiment_id = consolidated_acceptance_record(
        left,
        left_manifest,
        left_summary,
        right_manifest,
        right_summary,
        result,
        formal_registry,
    )
    result.update({
        "formal_registry": str(formal_registry),
        "formal_experiment_id": experiment_id,
        "formal_decision": left_summary["analysis"]["decision"],
    })
    write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze panel + VIX + grid")
    prepare.add_argument("--grid", default=str(DEFAULT_GRID))
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--allow-dirty", action="store_true")

    run = subparsers.add_parser("run", help="run all 22 frozen trials")
    run.add_argument("--bundle", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--runner-id", required=True)
    run.add_argument("--allow-dirty", action="store_true")

    compare = subparsers.add_parser("compare", help="compare independent outputs")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--registry")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_bundle(args.grid, args.bundle, allow_dirty=args.allow_dirty)
    elif args.command == "run":
        run_grid(
            args.bundle,
            args.output,
            args.runner_id,
            allow_dirty=args.allow_dirty,
        )
    elif args.command == "compare":
        result = compare_runs(
            args.left,
            args.right,
            args.output,
            registry_path=args.registry,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
