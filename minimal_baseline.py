#!/usr/bin/env python3
"""
minimal_baseline — 極簡基準：池 + 動量排名 Top-N + 等權 + 月再平衡

為什麼需要這支（2026-07-25 稽核 C1）
------------------------------------
稽核發現：四個生產策略**沒有一個** Sharpe 贏過「等權抱著自己選的 116 檔」
（池 1.68 vs 最好的 SURGE 1.45），對池的 α 也全都不顯著（NW t 0.10~1.51）。
但同時，真實前瞻紀錄顯示**訊號層是有效的**（訊號 − 池 = +5.13pp/20日窗，
配對 t = 3.44）。

兩者合起來的推論是「edge 在訊號，被執行層吃掉」——但**從來沒有人正面測過**。
這支就是那個正面測試。

設計原則：單一變因
------------------
與生產**完全相同**（不可變）：
  · 同一個 116 檔池、同一個 Top-60 流動性 universe
  · 同一個訊號（strategy.ai_strategy.engineer_features 的 total_score）
  · 同一個事件時鐘（t-1 收盤出訊號 → t 開盤成交）
  · 同一組成本（買 0.1425% / 賣 0.4425% / 滑價）

**刻意移除**（＝被測對象，執行層）：
  · 停損 / 停利 / 移動停利 / 汰弱
  · regime 濾網 / 動態曝險 / 加碼分段
  · 動態 sizing（vol parity、rank weighted、gap sizing…）
  · 事件驅動進出 → 改成固定月再平衡

★刻意不重用 EventDrivenBacktester：它本身就是被測對象。這支必須簡單到
  可以逐行用眼睛驗證，否則「基準」本身就成了另一個不可信的黑箱。

用法
----
    python minimal_baseline.py --snapshot artifacts/snapshot_X/panel.pkl
    python minimal_baseline.py --snapshot ... --top-n 5 10 --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUY_COST = 0.001425
SELL_COST = 0.004425
TRADING_DAYS = 252


def run_minimal(close, open_, score, universe, *, top_n=7, slippage=0.002,
                buy_cost=BUY_COST, sell_cost=SELL_COST, freq="M",
                warmup=60):
    """等權 Top-N + 固定週期再平衡。回傳 (equity, rebalance_log)。

    事件時鐘（與修正後的生產引擎一致）：
      · 再平衡日 t：用 **t-1 收盤** 的訊號決定持股
      · 成交價 = **t 當日開盤** × (1 ± slippage)
      · 期間內不做任何事（無停損、無 regime、無加碼）

    權重採「目標等權」：每次再平衡後每檔權重 = 1/N。期間內隨價格漂移，
    不做期中調整——這是刻意的，任何期中干預都屬執行層。
    """
    dates = close.index
    if len(dates) <= warmup:
        raise ValueError("資料長度不足")

    # 再平衡日 = 每個週期的第一個交易日（warmup 之後）
    period = pd.Series(dates, index=dates).dt.to_period(freq)
    is_first = period != period.shift(1)
    rebal = [d for d in dates[warmup:] if is_first.loc[d]]
    if not rebal:
        raise ValueError("沒有任何再平衡日")

    cash = 1.0
    holdings: dict[str, float] = {}          # ticker -> 股數（分數股，尺度無關）
    equity, log = [], []

    def mark(day, px_row):
        v = cash
        for t, sh in holdings.items():
            p = px_row.get(t, np.nan)
            if not pd.isna(p):
                v += sh * p
        return v

    rebal_set = set(rebal)
    for i, day in enumerate(dates):
        if i < warmup:
            continue
        px_close = close.loc[day]

        if day in rebal_set:
            prev = dates[i - 1]
            s = score.loc[prev]                      # ★t-1 收盤訊號
            if universe is not None:
                s = s.where(universe.loc[prev], np.nan)
            px_open = open_.loc[day]
            # 只保留當日開盤價可用者（不可買賣沒有價格的東西）
            s = s[[c for c in s.index if not pd.isna(px_open.get(c, np.nan))]]
            picks = list(s.dropna().sort_values(ascending=False).head(top_n).index)

            if picks:
                # ★只交易差額，不做「全部出清再買回」。
                #   全額換手每次成本 ≈ 賣0.4425%+買0.1425%+來回滑價 ≈ 0.985%，
                #   月頻 ×12 ≈ 11.8%/年 —— 對留倉不動的持股收這筆費用是錯的，
                #   會讓 Top-N 越大時憑空產生越大的負 α（實測 Top-60 α=-10.1%，
                #   t=-3.73，正是這個 bug 的指紋）。
                def px(t):
                    p = px_open.get(t, np.nan)
                    if pd.isna(p):
                        p = px_close.get(t, np.nan)   # 停牌：以當日收盤估
                    return float(p) if not pd.isna(p) else None

                # 以「開盤價」計算再平衡當下的總值
                gross = cash
                for t, sh in holdings.items():
                    p = px(t)
                    if p is not None:
                        gross += sh * p
                # ★目標值必須預留買入手續費，否則按 gross/N 下單時，
                #   現金會在買最後一檔時用罄 → 該檔永遠買不足 → 下次再平衡
                #   又去補一點 → 形成棘輪，累積出假成本（實測 9.35e-06）。
                #   除以 (1+buy_cost) 等於保留手續費所需現金。
                target_val = gross / len(picks) / (1 + buy_cost)
                # 相對容差：避免浮點誤差造成無意義的微額換手。
                tol = max(gross * 1e-9, 1e-15)

                # 1) 先賣：不在名單內的全出，超重的賣掉超出部分
                for t, sh in list(holdings.items()):
                    p = px(t)
                    if p is None:
                        continue
                    want = target_val if t in picks else 0.0
                    cur = sh * p
                    if cur > want + tol:
                        sell_amt = cur - want
                        sell_sh = sell_amt / p
                        cash += sell_sh * p * (1 - slippage) * (1 - sell_cost)
                        holdings[t] = sh - sell_sh
                        if holdings[t] * p <= tol:
                            del holdings[t]

                # 2) 再買：不足的補到目標
                for t in picks:
                    p = px(t)
                    if p is None:
                        continue
                    cur = holdings.get(t, 0.0) * p
                    if cur < target_val - tol:
                        need = target_val - cur
                        buy_p = p * (1 + slippage)
                        amt = min(need, max(cash, 0.0) / (1 + buy_cost))
                        if amt <= 0:
                            continue
                        holdings[t] = holdings.get(t, 0.0) + amt / buy_p
                        cash -= amt * (1 + buy_cost)

                log.append({"date": str(day.date()), "picks": picks,
                            "equity": mark(day, px_close),
                            "gross": gross, "cash_after": cash})

        equity.append({"Date": day, "Equity": mark(day, px_close)})

    return pd.DataFrame(equity).set_index("Date"), log


def perf(equity, freq_days=TRADING_DAYS):
    r = equity["Equity"].pct_change().dropna()
    yrs = len(r) / freq_days
    tot = float(equity["Equity"].iloc[-1] / equity["Equity"].iloc[0] - 1)
    ann = (1 + tot) ** (1 / yrs) - 1
    vol = float(r.std() * np.sqrt(freq_days))
    eq = equity["Equity"]
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ann_return": ann, "ann_vol": vol,
            "sharpe": ann / vol if vol else float("nan"),
            "max_drawdown_pct": mdd, "total_return": tot, "n_days": len(r)}


def main() -> int:
    # ★只在當作腳本執行時包裝 stdout。放在模組層會讓任何 import 這支的
    #   程式（例如 pytest）的輸出擷取壞掉：I/O operation on closed file。
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, nargs="+", default=[5, 7, 10])
    ap.add_argument("--slippage", type=float, default=0.002)
    ap.add_argument("--freq", default="M", help="再平衡週期：M=月 Q=季 W=週")
    ap.add_argument("--universe-top", type=int, default=60)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from strategy.ai_strategy import build_liquid_universe, engineer_features
    from strategy.benchmark import capm_vs_benchmark

    panel = pickle.load(open(args.snapshot, "rb"))
    close = panel["Close"].loc[args.start:args.end]
    open_ = panel["Open"].loc[close.index]
    vol = panel["Volume"].loc[close.index]

    universe = build_liquid_universe(close, vol, top_n=args.universe_top)
    score, _ma, _atr, _s = engineer_features(close, vol, universe)

    pool_cols = [c for c in close.columns if c != "0050"]
    pool_ret = close[pool_cols].pct_change().mean(axis=1).fillna(0)
    pool_eq = (1 + pool_ret).cumprod()
    mkt = close["0050"].dropna() if "0050" in close.columns else None

    print("=" * 96)
    print("C1 極簡基準：池 + 動量 Top-N + 等權 + 再平衡（無停損／無 regime／無加碼）")
    print(f"快照 {args.snapshot}")
    print(f"期間 {close.index[0].date()} ~ {close.index[-1].date()}  "
          f"再平衡 {args.freq}  滑價 {args.slippage*1e4:.0f}bps")
    print("=" * 96)

    pool_perf = perf(pd.DataFrame({"Equity": pool_eq}))
    print(f"{'配置':<22}{'年化':>8}{'Sharpe':>8}{'MDD':>9}{'換股次數':>9}"
          f"{'β(池)':>8}{'α(池)':>9}{'NW t':>7}")
    print("-" * 96)
    print(f"{'116池等權抱著':<22}{pool_perf['ann_return']*100:7.1f}%"
          f"{pool_perf['sharpe']:8.2f}{pool_perf['max_drawdown_pct']*100:8.1f}%"
          f"{'—':>9}{'—':>8}{'—':>9}{'—':>7}")

    out = {"pool": pool_perf, "variants": {}}
    for n in args.top_n:
        eq, log = run_minimal(close, open_, score, universe, top_n=n,
                              slippage=args.slippage)
        p = perf(eq)
        cap = capm_vs_benchmark(eq, pool_eq)
        out["variants"][f"top{n}"] = {**p, "n_rebalance": len(log),
                                      "capm_vs_pool": cap}
        sig = "✅" if (cap and cap["t_alpha"] > 1.96 and cap["ann_alpha"] > 0) else "❌"
        print(f"{'極簡 Top-'+str(n):<22}{p['ann_return']*100:7.1f}%{p['sharpe']:8.2f}"
              f"{p['max_drawdown_pct']*100:8.1f}%{len(log):9}"
              f"{cap['beta']:8.2f}{cap['ann_alpha']*100:8.1f}%{cap['t_alpha']:7.2f} {sig}")

    if mkt is not None:
        mp = perf(pd.DataFrame({"Equity": mkt}))
        out["benchmark_0050"] = mp
        print(f"{'（參考）0050 抱著':<22}{mp['ann_return']*100:7.1f}%{mp['sharpe']:8.2f}"
              f"{mp['max_drawdown_pct']*100:8.1f}%")

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"\n📄 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
