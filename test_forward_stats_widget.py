"""回歸測試：paper 頁的「訊號 vs 對照組」配對區塊（2026-07-25 稽核 B5）。

為什麼這一列數字必須每天自動出現
--------------------------------
稽核已證實：**沒有任何策略的 Sharpe 贏過「等權抱著自己選的 116 檔」**
（池 Sharpe 高於四者全部），對池的 α 也全部不顯著。
所以「贏過 0050」不構成價值主張——池才是對照組。

這個區塊把「訊號到底有沒有贏過什麼都不做」放進每日 paper 頁，
不必有人手動去算，也不可能再被忽略。
"""

import pytest

from forward_stats_widget import _paired_block, render


def _rec(excess, t, significant, bench="池等權", strat="v85"):
    return {strat: {"n_trades": 100, bench: {
        "strategy_mean": 0.05, "bench_mean": 0.05 - excess,
        "excess": excess, "t_paired": t, "significant": significant,
        "n_signal_days": 40, "n_pairs": 200}}}


@pytest.mark.parametrize("stats", [
    {"blocks": [1]},                                   # 無此鍵
    {"blocks": [1], "paired_benchmark": {}},           # 空
    {"blocks": [1], "paired_benchmark": {"v85": {"n_trades": 3}}},  # 有策略但無基準
])
def test_missing_data_renders_nothing(stats):
    """顯示層 fail-soft：缺統計不該讓整頁生不出來。"""
    assert _paired_block(stats) == ""


@pytest.mark.parametrize("excess, t, sig, expect", [
    (0.05, 3.44, True, "顯著勝出"),
    (-0.03, -2.50, True, "顯著落後"),
    (0.004, 0.45, False, "不顯著"),
])
def test_verdict_wording(excess, t, sig, expect):
    html = _paired_block({"blocks": [1], "paired_benchmark": _rec(excess, t, sig)})
    assert expect in html


def test_significant_loss_is_not_hidden():
    """★顯著落後必須明講，不可只呈現對策略有利的那一面。"""
    html = _paired_block({"blocks": [1], "paired_benchmark": _rec(-0.03, -2.5, True)})
    assert "顯著落後" in html
    assert "-3.00pp" in html


def test_all_strategies_and_benchmarks_listed():
    pb = {s: {b: {"strategy_mean": 0.05, "bench_mean": 0.04, "excess": 0.01,
                  "t_paired": 1.2, "significant": False, "n_signal_days": 40}
              for b in ("池等權", "0050")}
          for s in ("v85", "guard", "surge", "surge_pro")}
    html = _paired_block({"blocks": [1], "paired_benchmark": pb})
    assert html.count("<tr>") - 1 == 8          # 4 策略 × 2 基準（扣表頭）
    for s in ("v85", "guard", "surge", "surge_pro"):
        assert s in html


def test_pool_is_described_as_the_real_alternative():
    """文案必須指明對照組是『等權抱著這 116 檔』，不是 0050。"""
    html = _paired_block({"blocks": [1], "paired_benchmark": _rec(0.01, 1.0, False)})
    assert "等權抱著這 116 檔" in html
    assert "不是 0050" in html


def test_significance_threshold_is_stated():
    html = _paired_block({"blocks": [1], "paired_benchmark": _rec(0.01, 1.0, False)})
    assert "1.96" in html


def test_render_is_safe_without_stats_file():
    assert render({}, path="/definitely/not/here.json") == ""


def test_render_includes_paired_block_when_present():
    stats = {"blocks": [{"name": "全部訊號（基準）", "n": 10, "avg": 0.05, "wr": 0.6}],
             "paired_benchmark": _rec(0.05, 3.44, True)}
    html = render(stats)
    assert "訊號 vs 對照組" in html


def test_forward_record_persists_paired_stats():
    """防回歸：forward_record 必須把配對統計寫進 forward_stats.json。"""
    src = open("forward_record.py", encoding="utf-8").read()
    assert 'stats["paired_benchmark"] = paired_stats' in src
    assert "paired_stats = {}" in src, "失敗時需有預設值，否則整條管線會炸"
