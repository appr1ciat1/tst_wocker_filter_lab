"""Deflated Sharpe Ratio utilities.

The implementation follows the Bailey/Lopez de Prado approach at a practical
engineering level: estimate the uncertainty of an observed Sharpe ratio under
non-normal returns, adjust the hurdle for the number of tried configurations,
then return the probability that the observed Sharpe clears that hurdle.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
EULER_GAMMA = 0.5772156649015329
NORMAL = statistics.NormalDist()


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe: float
    benchmark_sharpe: float
    sharpe_std: float
    expected_max_sharpe: float
    deflated_sharpe: float
    probability: float
    n_observations: int
    n_trials: int
    skew: float
    kurtosis: float
    trial_sharpe_mean: float
    trial_sharpe_std: float


def _as_clean_returns(returns: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(returns) if not isinstance(returns, (pd.Series, np.ndarray)) else returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def annualized_sharpe(returns: Iterable[float] | pd.Series | np.ndarray,
                      annualization: int = TRADING_DAYS_PER_YEAR) -> float:
    arr = _as_clean_returns(returns)
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * math.sqrt(annualization))


def sample_skew(returns: np.ndarray) -> float:
    if len(returns) < 3:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(((returns - mean) / std) ** 3))


def sample_kurtosis(returns: np.ndarray) -> float:
    """Return Pearson kurtosis, where normal returns have kurtosis near 3."""
    if len(returns) < 4:
        return 3.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 3.0
    return float(np.mean(((returns - mean) / std) ** 4))


def expected_max_sharpe(
    sharpe_std: float,
    n_trials: int,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Expected maximum Sharpe after trying n independent configurations."""
    if n_trials <= 1 or sharpe_std <= 0:
        return benchmark_sharpe

    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    p1 = min(max(p1, 1e-12), 1 - 1e-12)
    p2 = min(max(p2, 1e-12), 1 - 1e-12)
    return benchmark_sharpe + sharpe_std * (
        (1.0 - EULER_GAMMA) * NORMAL.inv_cdf(p1)
        + EULER_GAMMA * NORMAL.inv_cdf(p2)
    )


def compute_deflated_sharpe(
    returns: Iterable[float] | pd.Series | np.ndarray,
    *,
    n_trials: int = 1,
    benchmark_sharpe: float = 0.0,
    annualization: int = TRADING_DAYS_PER_YEAR,
    trial_sharpes: Iterable[float] | np.ndarray | None = None,
    trial_mean_sharpe: float | None = None,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio probability for a return series.

    The returned ``deflated_sharpe`` is the z-score. ``probability`` is the
    normal CDF of that z-score and is usually the friendlier gate metric.
    """
    arr = _as_clean_returns(returns)
    trial_sr = (
        _as_clean_returns(trial_sharpes)
        if trial_sharpes is not None else np.array([], dtype=float)
    )
    if trial_sharpes is not None:
        if len(trial_sr) == 0:
            raise ValueError("trial_sharpes must contain at least one finite Sharpe")
        if n_trials not in (1, len(trial_sr)):
            raise ValueError(
                f"n_trials={n_trials} does not match {len(trial_sr)} trial Sharpes"
            )
        n_trials = len(trial_sr)
    else:
        n_trials = max(1, int(n_trials))
        if n_trials > 1:
            raise ValueError(
                "formal DSR with n_trials > 1 requires trial_sharpes; "
                "the selected series sampling error is not cross-trial dispersion"
            )

    trial_sr_mean = (
        float(trial_mean_sharpe)
        if trial_mean_sharpe is not None
        else (float(np.mean(trial_sr)) if len(trial_sr) else float(benchmark_sharpe))
    )
    trial_sr_std = (
        float(np.std(trial_sr, ddof=1))
        if len(trial_sr) > 1 else 0.0
    )
    n_obs = len(arr)
    if n_obs < 3:
        return DeflatedSharpeResult(
            sharpe=0.0,
            benchmark_sharpe=benchmark_sharpe,
            sharpe_std=float("inf"),
            expected_max_sharpe=benchmark_sharpe,
            deflated_sharpe=float("-inf"),
            probability=0.0,
            n_observations=n_obs,
            n_trials=n_trials,
            skew=0.0,
            kurtosis=3.0,
            trial_sharpe_mean=trial_sr_mean,
            trial_sharpe_std=trial_sr_std,
        )

    sr = annualized_sharpe(arr, annualization=annualization)
    sr_period = sr / math.sqrt(annualization)
    skew = sample_skew(arr)
    kurt = sample_kurtosis(arr)

    variance_term = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period ** 2)
    variance_term = max(variance_term, 1e-12)
    sharpe_std_period = math.sqrt(variance_term / max(n_obs - 1, 1))
    sharpe_std = sharpe_std_period * math.sqrt(annualization)
    # Multiple-testing hurdle 用「所有 trial 的 Sharpe 橫斷面離散度」；
    # selected series 的 sampling SE 只用在下方 PSR/DSR 分母。兩者不可混用。
    hurdle = (
        expected_max_sharpe(trial_sr_std, n_trials, trial_sr_mean)
        if n_trials > 1 else float(benchmark_sharpe)
    )

    if sharpe_std <= 0 or not math.isfinite(sharpe_std):
        z_score = float("inf") if sr > hurdle else float("-inf")
    else:
        z_score = (sr - hurdle) / sharpe_std

    probability = NORMAL.cdf(z_score) if math.isfinite(z_score) else (1.0 if z_score > 0 else 0.0)
    return DeflatedSharpeResult(
        sharpe=sr,
        benchmark_sharpe=benchmark_sharpe,
        sharpe_std=sharpe_std,
        expected_max_sharpe=hurdle,
        deflated_sharpe=z_score,
        probability=probability,
        n_observations=n_obs,
        n_trials=n_trials,
        skew=skew,
        kurtosis=kurt,
        trial_sharpe_mean=trial_sr_mean,
        trial_sharpe_std=trial_sr_std,
    )


def daily_returns_from_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Equity" in df.columns:
        return df["Equity"].pct_change().dropna()
    if len(df.columns) == 1:
        series = df.iloc[:, 0]
        return series.pct_change().dropna()
    raise ValueError(f"{path} must contain an Equity column or one price/equity column")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Deflated Sharpe Ratio from an equity CSV")
    parser.add_argument("--equity", required=True, help="CSV with an Equity column")
    parser.add_argument("--trials", type=int, default=1, help="Number of tested configurations")
    parser.add_argument("--benchmark-sharpe", type=float, default=0.0)
    parser.add_argument(
        "--trial-sharpes", nargs="*", type=float,
        help="All tested configurations' annualized Sharpes (required when --trials > 1)",
    )
    args = parser.parse_args()

    returns = daily_returns_from_csv(args.equity)
    result = compute_deflated_sharpe(
        returns,
        n_trials=args.trials,
        benchmark_sharpe=args.benchmark_sharpe,
        trial_sharpes=args.trial_sharpes,
    )
    print(f"Sharpe: {result.sharpe:.4f}")
    print(f"Expected max Sharpe hurdle: {result.expected_max_sharpe:.4f}")
    print(f"Deflated Sharpe z-score: {result.deflated_sharpe:.4f}")
    print(f"DSR probability: {result.probability:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

