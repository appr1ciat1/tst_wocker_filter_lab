"""回歸測試：對外敘事不得與報表數字脫節（2026-07-26 稽核）。

這是什麼事件的防線
------------------
修好同日時序穿越與零滑價後，四策略排名整個變了：SURGE PRO 從「全場最高」
掉到四項指標沒有一項第一。但**文案沒有跟著動**，於是網站同時掛著：

  · 標題「SURGE PRO 追最高報酬」        —— 實際 32.9% < SURGE 34.0%
  · 對照表「💥 崩盤 → SURGE（最防守）、崩盤年 OOS 最佳」
                                        —— SURGE 是 2022 最差的（輸池 14.4pp）
  · README「池等權 Sharpe 1.68」        —— 報表寫 1.60
  · README「2022 GUARD -21.3%」         —— 報表寫 -13.2%

第二條錯在最危險的方向：叫人在崩盤時選最不抗跌的那一個。

根因是**數字被手寫在文案裡**。index.html 早先漂掉 5~9pp 是同一個病，
當時用 build_index.py 回讀報表治好了，但這幾張表沒有納入。

所以本檔鎖三件事：
  1. 事實一律回讀報表（strategy_facts）；
  2. 「最好／最差」由資料算出來，資料一變、標記就跟著變——不可能再寫反；
  3. 對外檔案不得出現已作廢的宣稱。
"""

import os
import re

import pytest

import strategy_facts as SF

ROOT = os.path.dirname(os.path.abspath(__file__))


# ── 1. 事實表確實讀得到、且四份報表對池的描述一致 ──────────────

def test_reads_all_four_reports():
    f = SF.strategy_facts(ROOT)
    assert set(f) == {"SURGE", "SURGE PRO", "GUARD", "v8.5"}


def test_pool_metrics_identical_across_reports():
    """池是同一個池。四份報表若給出不同的池 Sharpe，代表有報表沒重建。"""
    f = SF.strategy_facts(ROOT)
    assert len({v["pool_sharpe"] for v in f.values()}) == 1
    assert len({v["pool22"] for v in f.values()}) == 1


def test_missing_report_raises_not_silently_defaults(tmp_path):
    """抓不到就 raise——靜默填舊值正是這次事件的成因。"""
    with pytest.raises((FileNotFoundError, ValueError)):
        SF.strategy_facts(str(tmp_path))


# ── 2. 最優／最差標記必須跟著資料走 ────────────────────────────

def _fake_report(*, ann, mdd, calmar, sharpe, y22, pool22, d22,
                 alpha=10.0, t=1.0, pool_ann=30.1, pool_sharpe=1.60):
    """最小合成報表。欄位順序要緊：Sharpe/Calmar 取的是第一個出現的。"""
    return (
        f"<p>年化報酬率 {ann:+.1f}%</p>"
        f"<p>最大回撤 {mdd:.1f}%</p>"
        f"<p>Calmar Ratio {calmar:.2f}</p>"
        f"<p>Sharpe Ratio {sharpe:.2f}</p>"
        f"<p>池等權 年化 / Sharpe {pool_ann:+.1f}% / {pool_sharpe:.2f}</p>"
        f"<p>對池的年化 α {alpha:+.1f}%</p>"
        f"<p>α 的 Newey-West t {t:.2f}</p>"
        f"<td>2022 🐻 逆風年</td><td>{pool22:+.1f}%</td><td>{y22:+.1f}%</td><td>{d22:+.1f}pp</td>"
    )


def _write_fake(tmp_path, spec):
    for name, (fname, _color) in SF.REPORTS.items():
        (tmp_path / fname).write_text(_fake_report(**spec[name]), encoding="utf-8")
    return str(tmp_path)


BASE = {
    "SURGE":     dict(ann=34.0, mdd=-29.5, calmar=1.15, sharpe=1.27,
                      y22=-22.8, pool22=-8.4, d22=-14.4),
    "SURGE PRO": dict(ann=32.9, mdd=-29.6, calmar=1.11, sharpe=1.25,
                      y22=-18.9, pool22=-8.4, d22=-10.4),
    "GUARD":     dict(ann=30.9, mdd=-21.0, calmar=1.47, sharpe=1.31,
                      y22=-13.2, pool22=-8.4, d22=-4.8),
    "v8.5":      dict(ann=27.5, mdd=-34.7, calmar=0.79, sharpe=1.15,
                      y22=-17.9, pool22=-8.4, d22=-9.4),
}


def test_worst_bear_year_label_follows_data(tmp_path):
    """★核心：把 2022 的好壞對調，「四者最差」必須跟著換人。

    這正是舊表寫反的地方——它把最差的標成最防守。
    """
    root = _write_fake(tmp_path, BASE)
    html = SF.build_guide_html(root)
    surge_row = html[html.index(">SURGE<"):html.index(">SURGE PRO<")]
    assert "四者最差" in surge_row, "SURGE 輸池 14.4pp，應標最差"

    swapped = {k: dict(v) for k, v in BASE.items()}
    swapped["SURGE"]["d22"], swapped["GUARD"]["d22"] = -4.8, -14.4
    swapped["SURGE"]["y22"], swapped["GUARD"]["y22"] = -13.2, -22.8
    html2 = SF.build_guide_html(_write_fake(tmp_path, swapped))
    guard_row = html2[html2.index(">GUARD<"):]
    assert "四者最差" in guard_row, "資料換了，最差標記卻沒跟著換——又寫死了"


def test_best_columns_follow_data(tmp_path):
    """年化最高者換人時，粗體也要換人。"""
    bumped = {k: dict(v) for k, v in BASE.items()}
    bumped["v8.5"]["ann"] = 99.9
    html = SF.build_guide_html(_write_fake(tmp_path, bumped))
    assert "<td><b>+99.9%</b></td>" in html
    assert "<td><b>+34.0%</b></td>" not in html


def test_no_regime_recommendation_table(tmp_path):
    """不得再出現情境推薦——證據只有一個空頭年，撐不起那種結論。"""
    html = SF.build_guide_html(_write_fake(tmp_path, BASE))
    for banned in ("市場情境", "最適策略", "最防守", "崩盤年 OOS 最佳"):
        assert banned not in html, f"情境推薦復活了：{banned}"


# ── 3. 對外檔案不得殘留已作廢的宣稱 ────────────────────────────

PUBLIC = ["index.html", "README.md", "paper_trading.html",
          "paper_trading_guard.html", "report_surge_pro.html",
          "stock_report.html", ".github/workflows/update_ai_report.yml"]

# 用 regex 而非純字串：paper 頁內嵌 chart.js 的價格陣列，
# 裸比對 "1.68" 會誤中 "241.68"。數值類宣稱一律加邊界。
RETRACTED = [
    "追最高報酬",                      # SURGE PRO 已非最高
    "崩盤年 OOS 最佳",                 # 指向 SURGE，而 SURGE 是 2022 最差
    "全期年化反而是 SURGE PRO 最高",
    "要榨乾回測優勢",
    "報酬／Sharpe／Calmar 全期最高",
    r"(?<![\d.])1\.68(?![\d])",        # 過期的池 Sharpe
    r"0\.10~1\.51",                    # 過期的 NW t 區間
]

# 稽核註解會刻意引述舊錯誤來說明它錯在哪；那是紀錄，不是宣稱。
_EXCUSE = ("已作廢", "更正", "舊註解", "寫反", "撤掉", "移除",
           "漂掉", "過期", "舊標題", "原本")


@pytest.mark.parametrize("rel", PUBLIC)
def test_public_files_free_of_retracted_claims(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f"{rel} 不存在（尚未產生）")
    src = open(path, encoding="utf-8", errors="replace").read()
    for claim in RETRACTED:
        pat = claim if claim.startswith("(") or "\\" in claim else re.escape(claim)
        for m in re.finditer(pat, src):
            ctx = src[max(0, m.start() - 260):m.start() + 90]
            assert any(k in ctx for k in _EXCUSE), (
                f"{rel} 仍在宣稱已作廢的「{m.group(0)}」（且未標示為更正）"
                f"\n上下文：…{re.sub(r'<[^>]+>', ' ', ctx)[-190:]}…")


def test_readme_headline_numbers_match_reports():
    """★README 的定位表必須與報表同步——這次就是它漂掉 6 個數字。"""
    f = SF.strategy_facts(ROOT)
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    pool_sharpe = next(iter(f.values()))["pool_sharpe"]
    assert f"{pool_sharpe:.2f}" in readme, "README 的池 Sharpe 與報表不符"
    for name, d in f.items():
        assert f"{d['ann']:.1f}%" in readme, f"README 缺 {name} 的年化 {d['ann']:.1f}%"
        assert f"{d['sharpe']:.2f}" in readme, f"README 缺 {name} 的 Sharpe"


def test_index_delegates_facts_table_to_generator():
    """index.html 的對照表必須是回填區，不能又變回手寫。"""
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    assert 'data-auto="facts"' in html, "index.html 的四策略對照表未接回填機制"
    assert "市場情境" not in html, "index.html 的情境推薦表復活了"
