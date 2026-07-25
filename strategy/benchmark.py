"""
Benchmark 模組：提供基準對比曲線

支援：
1. 0050 (台灣 50 ETF) Buy-and-Hold
2. 等權持有策略池內所有股票
3. Excess Return 計算
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def fetch_benchmark(ticker='0050', days=800, start_date=None, end_date=None):
    """
    下載 Benchmark 的每日收盤價。

    Parameters
    ----------
    ticker : str
        Benchmark 代號（預設 0050 = 台灣 50 ETF）
    days : int
        回溯天數。若提供 start_date，則忽略此參數。
    start_date, end_date : str or datetime, optional
        明確指定 benchmark 區間。

    Returns
    -------
    benchmark_equity : pd.Series
        以 1.0 為起始的 buy-and-hold 淨值曲線
    """
    if end_date is not None:
        end_dt = pd.Timestamp(end_date)
    else:
        end_dt = pd.Timestamp(datetime.today())

    if start_date is not None:
        start_dt = pd.Timestamp(start_date)
        range_label = f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}"
    else:
        start_dt = end_dt - timedelta(days=days)
        range_label = f"{days} 天"

    print(f"📈 下載 Benchmark: {ticker}.TW ({range_label})...")

    df = yf.download(f"{ticker}.TW", start=start_dt, end=end_dt, progress=False)

    if df.empty:
        print(f"   ⚠️ 無法下載 {ticker} 資料")
        return pd.Series(dtype=float)

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors='coerce').replace([np.inf, -np.inf], np.nan)
    dropped_count = int(close.isna().sum())
    close = close.dropna()
    if close.empty:
        print(f"   ⚠️ {ticker} 無有效收盤價資料")
        return pd.Series(dtype=float)

    if dropped_count:
        print(f"   ℹ️ 已忽略 {dropped_count} 筆無效收盤價")

    benchmark_equity = close / close.iloc[0]

    print(f"   ✅ Benchmark 下載完成: {close.index[0].strftime('%Y-%m-%d')}"
          f" → {close.index[-1].strftime('%Y-%m-%d')}")
    return benchmark_equity


def equal_weight_benchmark(close_df):
    """
    計算等權持有所有池內股票的淨值曲線。

    Parameters
    ----------
    close_df : pd.DataFrame
        收盤價矩陣

    Returns
    -------
    ew_equity : pd.Series
        等權持有淨值曲線（以 1.0 為起始）
    """
    daily_returns = close_df.pct_change()
    ew_return = daily_returns.mean(axis=1)  # 每日等權平均報酬
    ew_equity = (1 + ew_return).cumprod()
    ew_equity.iloc[0] = 1.0
    return ew_equity


def compute_excess_return(strategy_equity, benchmark_equity):
    """
    計算策略相對 Benchmark 的超額累積報酬。

    Parameters
    ----------
    strategy_equity : pd.Series
        策略淨值曲線
    benchmark_equity : pd.Series
        Benchmark 淨值曲線

    Returns
    -------
    excess : pd.Series
        累積超額報酬
    """
    # 對齊日期
    common_idx = strategy_equity.index.intersection(benchmark_equity.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    strat = strategy_equity.loc[common_idx]
    bench = benchmark_equity.loc[common_idx]

    # 累積超額
    strat_norm = strat / strat.iloc[0]
    bench_norm = bench / bench.iloc[0]
    excess = strat_norm - bench_norm

    return excess


def capm_vs_benchmark(strategy_equity, benchmark_equity, nw_lags=10):
    """對基準做 CAPM 迴歸，回傳 beta、年化 alpha、Newey-West t 值。

    為什麼報表需要這個（2026-07-25 稽核）
    ------------------------------------
    報表原本只並列絕對報酬。但 2019–2026 是台股最強的動量行情之一，
    0050 買進持有本身就有 30.5% 年化——絕對數字無法回答「這套機制到底
    有沒有加值」。

    更關鍵的是**基準要選對**：使用者已經人工挑好 116 檔股票池，那麼
    「什麼都不做」的真實替代方案不是 0050，而是**等權買進持有這 116 檔**。
    實測該池等權 Sharpe 1.68，高於任何一個主動策略。所以策略必須先贏過
    「只是抱著自己選的股票」，才談得上機制有價值。

    t 值用 Newey-West 修正（日策略報酬有自相關與異質變異，OLS t 值會高估）。
    實測差異不小：SURGE PRO 對 0050 的 alpha，OLS t=2.15（顯著）但
    NW t=1.95（不顯著）。

    Returns
    -------
    dict 或 None（資料不足時）：{beta, ann_alpha, t_alpha, r2, n_obs}
    """
    import numpy as np
    import pandas as pd

    if strategy_equity is None or benchmark_equity is None:
        return None
    s = strategy_equity['Equity'] if isinstance(strategy_equity, pd.DataFrame) \
        and 'Equity' in strategy_equity.columns else strategy_equity
    s = pd.Series(s).dropna().pct_change().dropna()
    b = pd.Series(benchmark_equity).dropna().pct_change().dropna()
    j = pd.concat([s.rename('s'), b.rename('b')], axis=1, join='inner').dropna()
    if len(j) < 60:
        return None

    y = j['s'].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(j)), j['b'].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape

    # Newey-West HAC 共變異數
    Xu = resid[:, None] * X
    S = Xu.T @ Xu
    for lag in range(1, min(nw_lags, n - 1) + 1):
        w = 1.0 - lag / (nw_lags + 1)
        A = Xu[lag:].T @ Xu[:-lag]
        S += w * (A + A.T)
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    var = np.diag(XtXi @ S @ XtXi * n / max(n - k, 1))
    se = np.sqrt(np.maximum(var, 1e-30))

    return {
        'beta': float(coef[1]),
        'ann_alpha': float(coef[0] * 252),
        't_alpha': float(coef[0] / se[0]),
        'r2': float(1 - resid.var() / y.var()) if y.var() > 0 else float('nan'),
        'n_obs': int(n),
    }
