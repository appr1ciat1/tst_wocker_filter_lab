"""回歸測試：C1 極簡基準模擬器（2026-07-25 稽核）。

這支模擬器是「執行層到底加分還扣分」的裁判，所以它自己必須先可信。
刻意用**已知答案的合成資料**驗證，而不是拿真實回測數字互相比對——
後者無法分辨「策略真的這樣」與「模擬器寫錯了」。

開發過程抓到的兩個真 bug（測試即為其防線）：
  1. 每次再平衡「全部出清再全部買回」，對留倉不動的持股也收費。
     單次全額換手 ≈ 0.985%，月頻 ×12 ≈ 11.8%/年 的假成本。
     指紋：Top-N 越大負 α 越大（實測 Top-60 α=-10.1%, t=-3.73）。
  2. 目標值未預留買入手續費 → 最後一檔永遠買不足 → 下次再補 → 棘輪。
"""

import numpy as np
import pandas as pd
import pytest

from minimal_baseline import perf, run_minimal

COLS = list("ABCDE")


def _frame(value, n=400):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(float(value), index=idx, columns=COLS)


def _universe(n=400):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(True, index=idx, columns=COLS)


def _flat_score():
    s = _frame(0.0)
    s["A"], s["B"], s["C"] = 3.0, 2.0, 1.0
    return s


def test_no_price_move_no_cost_leaves_equity_untouched():
    """零成本 + 價格不動 → 權益必須完全不變（無漂移、無洩漏）。"""
    c = _frame(100.0)
    eq, _ = run_minimal(c, c, _frame(1.0), _universe(), top_n=3,
                        slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    assert eq["Equity"].iloc[-1] == pytest.approx(1.0, abs=1e-12)


def test_single_pick_replicates_that_asset_exactly():
    """只買一檔且永遠不換 → 組合報酬必須等於該檔報酬。"""
    c = _frame(100.0)
    c["A"] = 100 * (1.001 ** np.arange(len(c)))
    o = c.shift(1).fillna(100.0)
    s = _frame(0.0)
    s["A"] = 1.0
    eq, log = run_minimal(c, o, s, _universe(), top_n=1,
                          slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    i0 = c.index.get_loc(pd.Timestamp(log[0]["date"]))
    asset = c["A"].iloc[-1] / o["A"].iloc[i0] - 1
    port = eq["Equity"].iloc[-1] - 1
    assert asset == pytest.approx(port, abs=1e-9)


def test_costs_actually_reduce_equity():
    c = _frame(100.0)
    c["A"] = 100 * (1.001 ** np.arange(len(c)))
    o = c.shift(1).fillna(100.0)
    s = _frame(0.0)
    s["A"] = 1.0
    free, _ = run_minimal(c, o, s, _universe(), top_n=1,
                          slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    paid, _ = run_minimal(c, o, s, _universe(), top_n=1, slippage=0.002)
    assert paid["Equity"].iloc[-1] < free["Equity"].iloc[-1]


def test_no_lookahead():
    """訊號只在最後一天翻轉 → 之前絕不可持有那一檔。"""
    c = _frame(100.0)
    s = _frame(0.0)
    s["A"] = 1.0
    s.iloc[-1] = [0, 9, 0, 0, 0]        # 最後一天才選 B
    _, log = run_minimal(c, c, s, _universe(), top_n=1,
                         slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    assert not any("B" in r["picks"] for r in log[:-1])


def test_deterministic():
    c = _frame(100.0)
    c["A"] = 100 * (1.001 ** np.arange(len(c)))
    o = c.shift(1).fillna(100.0)
    s = _frame(0.0)
    s["A"] = 1.0
    a, _ = run_minimal(c, o, s, _universe(), top_n=1, slippage=0.002)
    b, _ = run_minimal(c, o, s, _universe(), top_n=1, slippage=0.002)
    assert a["Equity"].equals(b["Equity"])


def test_unchanged_holdings_incur_no_repeat_cost_when_free():
    """★核心不變量：持股不變時，零成本下再平衡不得產生任何漂移。

    這是「只交易差額」的判準。若退回「全清全買」，此測試會立刻失敗。
    """
    c = _frame(100.0)
    eq, log = run_minimal(c, c, _flat_score(), _universe(), top_n=3,
                          slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    after = eq["Equity"].loc[pd.Timestamp(log[0]["date"]):]
    assert after.max() - after.min() == pytest.approx(0.0, abs=1e-12)


def test_unchanged_holdings_cost_is_one_off_not_recurring():
    """有成本時，持股不變的長期漂移必須是「一次性建倉」等級，不是每期收費。

    全清全買會是每次 ~0.985%，16 次累積 14.6%；只交易差額則是
    首次建倉 0.34% + 殘留 ~1e-5。門檻設 1e-3 足以區分兩者。
    """
    c = _frame(100.0)
    eq, log = run_minimal(c, c, _flat_score(), _universe(), top_n=3,
                          slippage=0.002)
    after = eq["Equity"].loc[pd.Timestamp(log[0]["date"]):]
    one_off = 1.0 - after.iloc[0]
    assert one_off == pytest.approx(1 - 1 / 1.001425 / 1.002, abs=1e-6), \
        "首次建倉成本應等於一次買入的手續費+滑價"
    assert after.max() - after.min() < 1e-3, \
        "持股不變卻持續扣費 → 可能退回了『全清全買』"


def test_larger_top_n_moves_toward_the_pool():
    """Top-N 放大時，組合應更接近等權池（β 下降、集中度下降）。

    這是當初抓到「全清全買」bug 的那個檢查：修好前，Top-N 越大
    負 α 越大（成本假象）；修好後應呈現合理的收斂方向。
    """
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    cols = [f"S{i}" for i in range(20)]
    px = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.015, (len(idx), len(cols))), axis=0),
        index=idx, columns=cols)
    score = px.pct_change(20).fillna(0)          # 20 日動量當訊號
    univ = pd.DataFrame(True, index=idx, columns=cols)
    pool = (1 + px.pct_change().mean(axis=1).fillna(0)).cumprod()

    betas = []
    for n in (3, 10, 20):
        eq, _ = run_minimal(px, px, score, univ, top_n=n, slippage=0.0,
                            buy_cost=0.0, sell_cost=0.0)
        j = pd.concat([eq["Equity"].pct_change().rename("s"),
                       pool.pct_change().rename("b")], axis=1,
                      sort=True).dropna()
        betas.append(np.polyfit(j["b"], j["s"], 1)[0])
    assert betas[0] > betas[-1], f"Top-N 放大時 β 未下降：{betas}"
    assert betas[-1] == pytest.approx(1.0, abs=0.15), \
        f"Top-N=全部時 β 應趨近 1，實得 {betas[-1]:.3f}"


def test_perf_metrics_are_sane():
    c = _frame(100.0)
    c["A"] = 100 * (1.02 ** np.arange(len(c)) ** 0.5)
    eq, _ = run_minimal(c, c, _flat_score(), _universe(), top_n=3,
                        slippage=0.0, buy_cost=0.0, sell_cost=0.0)
    m = perf(eq)
    assert m["max_drawdown_pct"] <= 0
    assert m["n_days"] > 0
    assert np.isfinite(m["sharpe"])
