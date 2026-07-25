"""回歸測試：前瞻訊號的逐筆配對基準（2026-07-25 稽核 B3）。

開發時抓到的兩個真 bug（測試即為防線）：
  1. 舊做法拿「0050 整段總漲幅」比「逐筆訊號平均報酬」——兩個量不可比。
  2. 我第一版把「池」誤寫成 `ohlc.items()`，那只含**曾被訊號選中的股票**
     （＝動量贏家），不是完整 116 檔池。基準被換成強得多的對照組，
     跑出 -2.51pp / t=-3.41 的假結論；修正後是 +0.44pp / t=+0.45。
"""

import numpy as np
import pandas as pd
import pytest

from forward_benchmark import _window_return, paired_compare, pool_window_return


def _px(base=100.0, drift=0.0, n=200):
    idx = pd.bdate_range("2026-01-01", periods=n)
    p = base * np.cumprod(np.full(n, 1 + drift))
    return pd.DataFrame({"open": p, "high": p * 1.01,
                         "low": p * 0.99, "close": p}, index=idx)


def test_window_return_matches_manual():
    px = _px(drift=0.01)
    d0, d1 = px.index[10], px.index[20]
    manual = float(px.loc[d1, "close"] / px.loc[d0, "open"] - 1)
    assert _window_return(px, d0, d1) == pytest.approx(manual, abs=1e-12)


def test_window_return_none_when_dates_missing():
    px = _px()
    assert _window_return(px, pd.Timestamp("1999-01-01"), px.index[5]) is None


def test_identical_series_gives_zero_excess():
    """策略與基準完全相同 → 超額必須為 0。"""
    bench = _px(drift=0.005)
    tr = pd.DataFrame([
        dict(strategy="X", signal_date=bench.index[i - 1], entry_date=bench.index[i],
             exit_date=bench.index[i + 10],
             net=_window_return(bench, bench.index[i], bench.index[i + 10]))
        for i in range(5, 120, 5)])
    st = paired_compare(tr, {"b": bench})["X"]["b"]
    assert st["excess"] == pytest.approx(0.0, abs=1e-12)


def test_constant_edge_is_detected_as_significant():
    bench = _px(drift=0.005)
    tr = pd.DataFrame([
        dict(strategy="X", signal_date=bench.index[i - 1], entry_date=bench.index[i],
             exit_date=bench.index[i + 10],
             net=_window_return(bench, bench.index[i], bench.index[i + 10]) + 0.02)
        for i in range(5, 120, 5)])
    st = paired_compare(tr, {"b": bench})["X"]["b"]
    assert st["excess"] == pytest.approx(0.02, abs=1e-9)
    assert st["significant"] and st["t_paired"] > 0


def test_same_day_signals_do_not_inflate_degrees_of_freedom():
    """★同一訊號日的多筆訊號必須先取平均。

    同日訊號彼此高度相關，逐筆當獨立樣本會灌水自由度、虛增 t 值。
    """
    bench = _px(drift=0.005)
    base = pd.DataFrame([
        dict(strategy="X", signal_date=bench.index[i - 1], entry_date=bench.index[i],
             exit_date=bench.index[i + 10],
             net=_window_return(bench, bench.index[i], bench.index[i + 10]) + 0.02)
        for i in range(5, 120, 5)])
    one = paired_compare(base, {"b": bench})["X"]["b"]
    five = paired_compare(pd.concat([base] * 5, ignore_index=True),
                          {"b": bench})["X"]["b"]
    assert five["n_pairs"] == one["n_pairs"] * 5
    assert five["n_signal_days"] == one["n_signal_days"]
    assert five["t_paired"] == pytest.approx(one["t_paired"], rel=1e-9)


def test_pool_return_is_mean_of_components():
    pool = {f"S{i}": _px(drift=0.002 * (i + 1)) for i in range(6)}
    idx = next(iter(pool.values())).index
    d0, d1 = idx[10], idx[30]
    manual = float(np.mean([_window_return(p, d0, d1) for p in pool.values()]))
    assert pool_window_return(pool, d0, d1) == pytest.approx(manual, abs=1e-12)


def test_pool_needs_enough_components():
    """成分太少時回 None，避免用 2~3 檔冒充『池』。"""
    tiny = {f"S{i}": _px() for i in range(3)}
    idx = next(iter(tiny.values())).index
    assert pool_window_return(tiny, idx[5], idx[10]) is None


def test_forward_record_uses_full_pool_not_signal_tickers():
    """★防止回歸成「用訊號股當基準」。

    那會把對照組換成動量贏家，讓策略看起來顯著落後（實測 -2.51pp、
    t=-3.41），是完全錯誤的結論。
    """
    src = open("forward_record.py", encoding="utf-8").read()
    assert "EXTENDED_TICKERS" in src, "池基準未使用完整股票池"
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    joined = "\n".join(code)
    assert "pool_px = {t: v for t, v in ohlc.items()}" not in joined, \
        "池基準退回成『曾被訊號選中的股票』"


def test_multi_strategy_grouping():
    bench = _px(drift=0.005)
    rows = []
    for st, edge in (("v85", 0.02), ("guard", -0.01)):
        for i in range(5, 100, 5):
            rows.append(dict(
                strategy=st, signal_date=bench.index[i - 1],
                entry_date=bench.index[i], exit_date=bench.index[i + 10],
                net=_window_return(bench, bench.index[i], bench.index[i + 10]) + edge))
    out = paired_compare(pd.DataFrame(rows), {"b": bench})
    assert set(out) == {"v85", "guard"}
    assert out["v85"]["b"]["excess"] > 0 > out["guard"]["b"]["excess"]
