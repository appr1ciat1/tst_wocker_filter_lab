#!/usr/bin/env python3
"""
wf_vs_pool — Walk-Forward + Monte Carlo，但對照組是「池」而不是策略自己

為什麼要重寫（2026-07-25 稽核 B2）
----------------------------------
既有 `walk_forward.py` 每折只算策略自己，回答的是「策略在不同時期穩不穩」。
但 C1 已經證明：**連「等權抱著自己選的 116 檔」都贏過所有策略的 Sharpe**
（池 Sharpe 高於四者全部）。在這個前提下，只驗證策略對自己的穩定性
沒有決策價值——穩定地輸給對照組仍然是輸。

所以每一折都同時算三條線：
    · 生產策略
    · C1 極簡基準（同訊號、無執行層）
    · 116 池等權（什麼都不做）
並直接回答：**策略在哪些時期曾經贏過池？**

Monte Carlo 同理：不 bootstrap 策略自己的權益曲線（那只會告訴你「這條曲線
的形狀有多不確定」），而是 bootstrap **配對差額**（策略 − 池的日報酬），
直接檢驗「超額」禁不禁得起重抽樣。用 block bootstrap 保留自相關。

★誠實標註：這是**固定參數**的分段檢驗，不是 train→select→test。
  它能偵測「策略在某些 regime 失效」，但**不能**偵測參數過擬合——
  後者需要 walk_forward_nested.py 那種巢狀迴圈。

用法
----
    python wf_vs_pool.py --snapshot artifacts/snapshot_X/panel.pkl
    python wf_vs_pool.py --snapshot ... --mc-runs 5000 --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 旗標逐字取自 .github/workflows/update_ai_report.yml
COMMON = ("--no-hybrid-tiered --consec-loss-limit 3 --capital 1000000 "
          "--start-date 2019-01-01 --eval-start 2019-01-01")
STRATEGIES = {
    "v8.5": "--no-regime-graduated --no-breadth-regime --tp-atr 4.0 --sl-atr 3.0 "
            "--hold-days 20 --top-k 7 --gap-filter 1.5",
    "GUARD": "--sl-atr 3.5 --regime-floor 0.0 --dynamic-topk --corr-select-max 0.7 "
             "--corr-select-window 60 --corr-select-cap 2",
    "SURGE": "--sl-atr 3.5 --hold-days 22 --position-size 0.10 --regime-floor 0.0 "
             "--dynamic-topk --dynamic-gap-filter --regime-sizing "
             "--strong-regime-mult 1.25 --strong-breadth-min 0.55 "
             "--strong-vix-max 25.0 --max-regime-scale 1.7 "
             '--strong-tiers "0.65,20,1.45;0.75,15,1.75"',
    "SURGE PRO": "--sl-atr 3.5 --hold-days 25 --position-size 0.10 --regime-floor 0.0 "
                 "--dynamic-topk --dynamic-gap-filter --regime-sizing "
                 "--strong-regime-mult 1.25 --strong-breadth-min 0.55 "
                 "--strong-vix-max 28.0 --max-regime-scale 1.9 "
                 '--strong-tiers "0.62,18,1.7;0.72,15,1.85"',
}


def fold_stats(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / TRADING_DAYS
    tot = float((1 + r).prod() - 1)
    ann = (1 + tot) ** (1 / yrs) - 1
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    eq = (1 + r).cumprod()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else np.nan,
            "mdd": float((eq / eq.cummax() - 1).min()), "n": len(r)}


def block_bootstrap_excess(diff: pd.Series, *, runs=2000, block=10, seed=0):
    """對配對差額做 block bootstrap，回傳年化超額的分布。

    bootstrap 的是 **策略 − 對照** 的日報酬序列（而非策略自己的權益），
    因為要檢驗的是「超額是否禁得起重抽樣」，不是「這條曲線多不確定」。
    block 保留自相關（日策略報酬有明顯自相關，iid 重抽會低估變異）。
    """
    x = diff.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < block * 3:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(runs)
    starts_max = n - block
    for i in range(runs):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        s = np.concatenate([x[st:st + block] for st in starts])[:n]
        out[i] = s.mean() * TRADING_DAYS          # 年化平均超額
    return out


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-07-17")
    ap.add_argument("--slippage", type=float, default=0.002)
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--mc-runs", type=int, default=2000)
    ap.add_argument("--mc-block", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import ai_report
    from minimal_baseline import run_minimal
    from strategy.ai_strategy import build_liquid_universe, engineer_features

    panel = pickle.load(open(args.snapshot, "rb"))
    close = panel["Close"].loc[args.start:args.end]
    op, hi, lo, vol = (panel[k].loc[close.index]
                       for k in ("Open", "High", "Low", "Volume"))
    vixp = Path(args.snapshot).parent / "vix.csv"
    vix = (pd.read_csv(vixp, index_col=0, parse_dates=True).iloc[:, 0]
           if vixp.exists() else None)

    univ = build_liquid_universe(close, vol, top_n=60)
    score, ma60, _atr, _s = engineer_features(close, vol, univ)
    mkt = close["0050"].dropna() if "0050" in close.columns else None

    # ── 三條線的日報酬 ──
    series: dict[str, pd.Series] = {}
    pool_cols = [c for c in close.columns if c != "0050"]
    series["116池等權"] = close[pool_cols].pct_change().mean(axis=1).dropna()
    if mkt is not None:
        series["0050"] = mkt.pct_change().dropna()

    eq_min, _ = run_minimal(close, op, score, univ, top_n=args.top_k,
                            slippage=args.slippage)
    series[f"極簡Top-{args.top_k}"] = eq_min["Equity"].pct_change().dropna()

    for name, flags in STRATEGIES.items():
        argv = sys.argv
        sys.argv = ["ai_report.py"] + shlex.split(
            f"{COMMON} {flags} --slippage {args.slippage} --end-date {args.end}")
        try:
            bt = ai_report.build_backtester_from_args(ai_report.parse_args())
        finally:
            sys.argv = argv
        _tr, eq = bt.run(score, close, op, hi, lo, ma60, top_k=args.top_k,
                         threshold=2.0, vol_df=vol, universe_mask=univ,
                         market_close=mkt, vix_series=vix)
        series[name] = eq["Equity"].pct_change().dropna()
        print(f"  {name} 完成", flush=True)

    aligned = pd.DataFrame(series).dropna()
    pool = aligned["116池等權"]
    names = [c for c in aligned.columns if c != "116池等權"]

    # ── Walk-Forward：逐年不重疊分段 ──
    print("\n" + "=" * 100)
    print("Walk-Forward（固定參數、逐年不重疊）— 每折同時算三條線")
    print("★這是 regime 穩定性檢驗，不是 train→select→test，無法偵測參數過擬合")
    print("=" * 100)
    years = sorted(aligned.index.year.unique())
    hdr = f"{'年度':<8}{'池':>9}" + "".join(f"{n:>12}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    wf_rows, beat = [], {n: 0 for n in names}
    for y in years:
        sub = aligned[aligned.index.year == y]
        if len(sub) < 40:
            continue
        ps = fold_stats(sub["116池等權"])
        row = {"year": int(y), "n_days": len(sub), "pool_ann": ps["ann"],
               "pool_sharpe": ps["sharpe"]}
        line = f"{y:<8}{ps['ann']*100:8.1f}%"
        for n in names:
            st = fold_stats(sub[n])
            row[f"{n}_ann"] = st["ann"]
            row[f"{n}_sharpe"] = st["sharpe"]
            win = st["ann"] > ps["ann"]
            beat[n] += win
            line += f"{st['ann']*100:10.1f}%{'✓' if win else ' '}"
        # ★不完整年度（暖身吃掉開頭、或最後一年尚未結束）年化會被放大，
        #   必須標註，否則 2019/2026 會被誤讀成與完整年度可比。
        if len(sub) < 200:
            line += f"   ⚠️不完整({len(sub)}日)"
            row["partial_year"] = True
        wf_rows.append(row)
        print(line)
    n_folds = len(wf_rows)
    print("-" * len(hdr))
    print(f"{'勝池折數':<8}{'—':>9}" + "".join(
        f"{str(beat[n])+'/'+str(n_folds):>12}" for n in names))

    # ── Monte Carlo：對「策略 − 池」做 block bootstrap ──
    print("\n" + "=" * 100)
    print(f"Monte Carlo（block bootstrap，block={args.mc_block} 日，"
          f"{args.mc_runs} 次）— 檢驗『超額報酬』而非曲線本身")
    print("=" * 100)
    print(f"{'配置':<14}{'年化超額':>10}{'5%':>10}{'95%':>10}{'>0 機率':>10}  判定")
    print("-" * 100)
    mc = {}
    for n in names:
        d = block_bootstrap_excess(aligned[n] - pool, runs=args.mc_runs,
                                   block=args.mc_block)
        if d is None:
            continue
        lo_, hi_ = np.percentile(d, [5, 95])
        p = float((d > 0).mean())
        mc[n] = {"mean": float(d.mean()), "p05": float(lo_),
                 "p95": float(hi_), "prob_positive": p}
        sig = "✅ 95%CI 全正" if lo_ > 0 else ("❌ CI 含 0" if hi_ > 0 else "❌ 全負")
        print(f"{n:<14}{d.mean()*100:9.1f}%{lo_*100:9.1f}%{hi_*100:9.1f}%"
              f"{p*100:9.1f}%  {sig}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"walk_forward": wf_rows, "beat_pool": beat, "n_folds": n_folds,
             "monte_carlo": mc}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"\n📄 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
