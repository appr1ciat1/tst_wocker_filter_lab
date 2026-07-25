#!/usr/bin/env python3
"""
forward_benchmark — 前瞻訊號的**逐筆配對基準比較**（2026-07-25 稽核 B3）

為什麼要重做基準比較
--------------------
`forward_record.py` 原本的基準是「同期間 0050 漲了多少」——比較的是
**整段期間的總漲幅** vs **逐筆訊號的平均報酬**。這兩個量不可比：
訊號是分散在不同時點、各持有 N 天的短窗；0050 是連續持有整段。

正確做法是**配對**：每一筆訊號都跟「同一個進出場窗」的基準比，
再對配對差額做檢定。實測差異極大——用配對法，v8.5 的訊號相對池
是 +5.13pp/窗、配對 t = 3.44（顯著）；用總漲幅法完全看不出來。

兩個基準
--------
  · 0050（市場）
  · **116 池等權**（＝「什麼都不做、抱著自己選的股票」）
    ★池才是真正的對照組。稽核已證實池等權 Sharpe 1.68 高於任何策略，
      所以「贏過 0050」不構成價值主張。

★這裡量測的是**訊號品質**（該不該看這檔），不是完整策略績效——
  它不含部位大小、實際停損停利的資金效果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window_return(px: pd.DataFrame, d0, d1) -> float | None:
    """同一個窗（d0 開盤 → d1 收盤）的報酬。取不到就回 None。"""
    try:
        if d0 not in px.index or d1 not in px.index:
            return None
        o = float(px.loc[d0, "open"])
        c = float(px.loc[d1, "close"])
        if not np.isfinite(o) or o <= 0 or not np.isfinite(c):
            return None
        return c / o - 1.0
    except Exception:
        return None


def pool_window_return(pool_px: dict[str, pd.DataFrame], d0, d1) -> float | None:
    """池等權在同一個窗的報酬＝各成分同窗報酬的平均。"""
    vals = [r for r in (_window_return(p, d0, d1) for p in pool_px.values())
            if r is not None]
    return float(np.mean(vals)) if len(vals) >= 5 else None


def paired_compare(trades: pd.DataFrame, benches: dict[str, object]) -> dict:
    """逐筆配對比較，回傳每個策略 × 每個基準的配對統計。

    Parameters
    ----------
    trades : 需含 strategy / entry_date / exit_date / net
    benches : {基準名: DataFrame(單一標的 OHLC) 或 dict[ticker->DataFrame](池)}

    配對 t 檢定以**訊號日**為單位（同一天的多筆訊號先取平均），
    因為同日訊號彼此高度相關，逐筆當獨立樣本會高估自由度。
    """
    if trades.empty:
        return {}
    out: dict = {}
    for strat, sub in trades.groupby("strategy"):
        rec: dict = {"n_trades": int(len(sub)),
                     "mean_net": float(sub["net"].mean()),
                     "win_rate": float((sub["net"] > 0).mean())}
        for bname, bdata in benches.items():
            rows = []
            for _, r in sub.iterrows():
                d0 = pd.Timestamp(r["entry_date"])
                d1 = pd.Timestamp(r["exit_date"])
                b = (pool_window_return(bdata, d0, d1)
                     if isinstance(bdata, dict) else _window_return(bdata, d0, d1))
                if b is None:
                    continue
                rows.append({"d": pd.Timestamp(r["signal_date"]),
                             "s": float(r["net"]), "b": b})
            if len(rows) < 10:
                rec[bname] = {"n": len(rows), "note": "配對樣本不足"}
                continue
            df = pd.DataFrame(rows)
            g = df.groupby("d")[["s", "b"]].mean()      # ★同日先取平均
            diff = g["s"] - g["b"]
            n = len(diff)
            sd = float(diff.std(ddof=1))
            t = float(diff.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")
            rec[bname] = {
                "n_pairs": int(len(df)), "n_signal_days": int(n),
                "strategy_mean": float(g["s"].mean()),
                "bench_mean": float(g["b"].mean()),
                "excess": float(diff.mean()),
                "t_paired": t,
                "significant": bool(abs(t) > 1.96) if np.isfinite(t) else False,
            }
        out[strat] = rec
    return out


def format_report(stats: dict) -> str:
    lines = []
    lines.append("=" * 92)
    lines.append("前瞻訊號 × 逐筆配對基準（進出場窗完全對齊；t 檢定以訊號日為單位）")
    lines.append("=" * 92)
    for strat, rec in sorted(stats.items()):
        lines.append(f"\n■ {strat}   訊號 {rec['n_trades']} 筆  "
                     f"平均淨報酬 {rec['mean_net']*100:+.2f}%  "
                     f"勝率 {rec['win_rate']*100:.1f}%")
        for bname, b in rec.items():
            if not isinstance(b, dict) or "excess" not in b:
                if isinstance(b, dict):
                    lines.append(f"    {bname:<12} {b.get('note', '')}（n={b.get('n', 0)}）")
                continue
            mark = "✅ 顯著" if b["significant"] else "❌ 不顯著"
            lines.append(
                f"    vs {bname:<10} 訊號 {b['strategy_mean']*100:+6.2f}%  "
                f"基準 {b['bench_mean']*100:+6.2f}%  "
                f"超額 {b['excess']*100:+6.2f}pp  "
                f"配對t {b['t_paired']:+5.2f}  {mark}  "
                f"(n={b['n_signal_days']} 訊號日)")
    lines.append("\n★量測的是訊號品質（該不該看這檔），不是完整策略績效。")
    return "\n".join(lines)


__all__ = ["paired_compare", "pool_window_return", "format_report"]
