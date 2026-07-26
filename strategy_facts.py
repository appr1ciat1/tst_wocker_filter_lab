"""strategy_facts — 四策略實測事實表：一律回讀報表，供 paper 頁與著陸頁共用。

獨立成模組是為了讓 build_index.py 能用它，而不必 import 笨重的 build_paper_page
（那會連帶拉進 twstk.backtest.engine，CI 的著陸頁步驟不需要）。
"""

from __future__ import annotations

import os
import re

# ── 四策略事實表：一律回讀報表，不在文案裡寫死數字 ────────────────
#
# 為什麼要這樣（2026-07-26 稽核）
# ------------------------------
# 這裡原本是一張手寫的「市場情境 → 最適策略」對照表。它壞了兩層：
#
#   1) 數字寫死在文案裡。修正同日時序穿越與零滑價後，四策略排名整個變了，
#      但文案不會跟著動 —— 跟 index.html 當初漂掉 5~9pp 是同一個失效模式，
#      那次已經用 build_index.py 回讀報表解決過一次了。
#
#   2) 比第 1 點嚴重：那張表宣稱「不同市況該選不同策略」，而全部證據只有
#      一條 8 年路徑、其中**只有一個空頭年（2022）**。n=1 推不出情境法則。
#      而且當時表內寫著「💥 崩盤 → SURGE（最防守）、崩盤年 OOS 最佳」——
#      SURGE 恰恰是 2022 最差的那一個（輸池 14.4pp），GUARD 才是最抗跌的。
#      錯在最危險的方向：叫人在崩盤時選最不抗跌的。
#
# 所以現在：數字全部從報表抽，**「最好／最差」由資料算出來**而不是寫死。
# 排名再怎麼變，這張表都不可能再說反話。抽不到就丟例外，不要靜默填舊值。

REPORTS = {
    "SURGE":     ("report_surge.html",     "#f59e0b"),
    "SURGE PRO": ("report_surge_pro.html", "#ef4444"),
    "GUARD":     ("report_guard.html",     "#10b981"),
    "v8.5":      ("report_v85.html",       "#3b82f6"),
}


def _flat(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _grab(text: str, pattern: str, field: str, src: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"{src} 內找不到「{field}」，報表格式可能已變更")
    return m.group(1)


def strategy_facts(root: str = ".") -> dict[str, dict]:
    """讀四份報表，回傳每個策略的實測指標。抓不到就 raise。"""
    facts: dict[str, dict] = {}
    for name, (fname, color) in REPORTS.items():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到報表 {fname}；請先跑 ai_report.py 產生")
        t = _flat(open(path, encoding="utf-8").read())
        y22 = re.search(r"2022 🐻 逆風年 ([+-][\d.]+)% ([+-][\d.]+)% ([+-][\d.]+)pp", t)
        if not y22:
            raise ValueError(f"{fname} 內找不到 2022 逐年對池列，報表格式可能已變更")
        facts[name] = {
            "color": color,
            "report": fname,
            "ann":    float(_grab(t, r"年化報酬.{0,60}?([+-]?\d+\.\d)%", "年化報酬", fname)),
            "mdd":    float(_grab(t, r"最大回撤.{0,60}?(-?\d+\.\d)%", "最大回撤", fname)),
            "calmar": float(_grab(t, r"Calmar.{0,60}?([\d.]+)", "Calmar", fname)),
            "sharpe": float(_grab(t, r"Sharpe.{0,60}?([\d.]+)", "Sharpe", fname)),
            "pool22": float(y22.group(1)),
            "y2022":  float(y22.group(2)),
            "d2022":  float(y22.group(3)),
            "alpha":  float(_grab(t, r"對池的年化 α ([+-][\d.]+)%", "對池 α", fname)),
            "nw_t":   float(_grab(t, r"α 的 Newey-West t ([+-]?[\d.]+)", "Newey-West t", fname)),
            # 「池等權 年化 / Sharpe +30.1% / 1.60」——要第二個數，不是年化那個。
            # 舊文案寫死 1.68 是更早一版的值，早就漂掉了。
            "pool_ann": float(_grab(t, r"池等權 年化 / Sharpe ([+-][\d.]+)% / [\d.]+",
                                    "池等權年化", fname)),
            "pool_sharpe": float(_grab(t, r"池等權 年化 / Sharpe [+-][\d.]+% / ([\d.]+)",
                                       "池等權 Sharpe", fname)),
        }
    return facts


def build_guide_html(root: str = ".", *, reproducibility_note: str = "") -> str:
    """四策略實測對照。**不做情境推薦**——證據只有一個空頭年，撐不起那種結論。

    reproducibility_note：呼叫端若知道自己頁面上還有「另一次執行」的數字
    （例如 paper 頁的摘要表來自它自己當天的回測，而本表讀自報表），
    務必傳入說明，否則同一頁會出現兩組打架的數字而讀者無從判斷。
    """
    f = strategy_facts(root)
    order = sorted(f, key=lambda k: -f[k]["ann"])

    best = {                                   # ★最優項用算的，不是用寫的
        "ann":    max(f, key=lambda k: f[k]["ann"]),
        "mdd":    max(f, key=lambda k: f[k]["mdd"]),      # MDD 是負值，越大越淺
        "calmar": max(f, key=lambda k: f[k]["calmar"]),
        "sharpe": max(f, key=lambda k: f[k]["sharpe"]),
        "d2022":  max(f, key=lambda k: f[k]["d2022"]),
    }
    worst22 = min(f, key=lambda k: f[k]["d2022"])
    pool_sharpe = f[order[0]]["pool_sharpe"]
    pool22 = f[order[0]]["pool22"]

    def cell(name, key, fmt):
        v = fmt.format(f[name][key])
        return f"<td><b>{v}</b></td>" if best.get(key) == name else f"<td>{v}</td>"

    rows = []
    for name in order:
        d = f[name]
        if name == worst22:
            tag = f"<span style='color:#ff6b6b'>（輸池 {abs(d['d2022']):.1f}pp，四者最差）</span>"
        elif name == best["d2022"]:
            tag = f"<span style='color:#94a3b8'>（輸池 {abs(d['d2022']):.1f}pp，四者最好）</span>"
        else:
            tag = f"<span style='color:#94a3b8'>（輸池 {abs(d['d2022']):.1f}pp）</span>"
        y22 = f"{d['y2022']:+.1f}%".replace("-", "−")
        rows.append(
            f"<tr><td><b style='color:{d['color']}'>{name}</b></td>"
            + cell(name, "ann", "{:+.1f}%").replace("-", "−")
            + cell(name, "mdd", "{:.1f}%").replace("-", "−")
            + cell(name, "calmar", "{:.2f}")
            + cell(name, "sharpe", "{:.2f}")
            + f"<td>{y22}{tag}</td></tr>"
        )

    # 效率前緣＝在任一欄拿到第一的策略；其餘為被支配者
    front = [n for n in order if n in set(best.values())]
    dominated = [n for n in order if n not in set(best.values())]
    span = lambda n: f"<b style='color:{f[n]['color']}'>{n}</b>"  # noqa: E731
    t_lo, t_hi = min(d["nw_t"] for d in f.values()), max(d["nw_t"] for d in f.values())
    s_lo, s_hi = min(d["sharpe"] for d in f.values()), max(d["sharpe"] for d in f.values())
    pp_lo = min(abs(d["d2022"]) for d in f.values())
    pp_hi = max(abs(d["d2022"]) for d in f.values())

    return (
        (f"<p style='color:#fbbf24;background:#422006;border-left:3px solid #f59e0b;"
           f"padding:8px 10px;margin:0 0 10px;font-size:.85rem;line-height:1.65'>{reproducibility_note}</p>"
           if reproducibility_note else "")
        + "<p style=\"color:#94a3b8;margin:0 0 10px;line-height:1.7\">"
        "四策略<b>選股訊號與弱勢去風險邏輯完全相同</b>，差別只在強勢時加碼的積極度與進場挑剔度。"
        "下表是<b>實際量到的數字</b>（直接讀自四份報表），<b>不是情境推薦</b>——"
        "全部證據只有一條 2019–2026 的路徑，其中<b>只有一個空頭年</b>，"
        "不足以支撐「什麼盤該用哪一套」這種結論。</p>"
        "<table><tr><th>策略</th><th>年化</th><th>MDD</th><th>Calmar</th><th>Sharpe</th>"
        "<th>2022（唯一空頭年）</th></tr>"
        + "".join(rows)
        + "<tr style='border-top:2px solid #475569'><td><b>池等權買進持有</b></td>"
        "<td colspan='3' style='color:#94a3b8'>（116 檔等權，不擇時不停損）</td>"
        f"<td><b style='color:#22d3ee'>{pool_sharpe:.2f}</b></td>"
        f"<td><b>{pool22:+.1f}%</b></td></tr>".replace("-", "−")
        + "</table>"
        "<p style='color:#cbd5e1;font-size:.88rem;margin:12px 0 0;line-height:1.7'>"
        f"在效率前緣上的只有：{'、'.join(span(n) for n in front)}。"
        + (f"{'、'.join(span(n) for n in dominated)} <b>沒有任何一項第一</b>——"
           "SURGE PRO 的分段加碼層在 ablation 中也未通過統計驗證。" if dominated else "")
        + "<br>⚠️ <b>三件事必須一起看</b>："
        f"四者 Sharpe {s_lo:.2f}~{s_hi:.2f}，<b>全低於池等權 {pool_sharpe:.2f}</b>；"
        f"對池 α 的 Newey-West t 只有 {t_lo:.2f}~{t_hi:.2f}（<b>全部不顯著</b>，門檻 1.96）；"
        f"而在唯一的空頭年 <b>四者全部輸給什麼都不做</b>（{pp_lo:.1f}~{pp_hi:.1f}pp）。"
        "「去風險機制會在崩盤時保護你」這個說法，目前的資料<b>不支持</b>（2026-07-26 稽核）。</p>"
    )
