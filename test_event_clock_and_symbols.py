"""回歸測試：事件時鐘（t-1 close → t open）與市場後綴／快照完整性。

這些測試鎖住 2026-07-19 稽核找到的四個 P0。它們刻意用合成資料，
不碰網路，執行時間 < 1 秒。**修改 event_backtest 的當日順序前請先讀這裡。**

背景（為什麼這些測試存在）：
  · 原本 Step 2 用「今日收盤」算 current_equity，Step 3 卻拿它決定
    「今日開盤」的部位大小 → 開盤下單時偷看了當天收盤價。
  · 原本今天盤中停利的回款會立刻進 capital，同一天開盤就能拿去買別檔。
  · 原本 '5274.TWO' 被 replace('.TW','') 寫成 '5274O'，四檔上櫃股靜默走樣。
  · 原本 load_snapshot 缺股就默默丟掉，回測少了四檔卻毫無警告。
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from strategy.ai_strategy import strip_market_suffix
from strategy.event_backtest import EventDrivenBacktester
from twstk.data.contract import (
    FIELDS,
    SnapshotIntegrityError,
    SnapshotTickerError,
    build_manifest,
    freeze_snapshot,
    load_snapshot,
    validate_panel,
)


# ───────────────────────── 合成面板工具 ─────────────────────────

def _panel(tickers, n=85, close=100.0, high=101.0, low=99.0):
    dates = pd.bdate_range("2025-01-01", periods=n)

    def mk(v):
        return pd.DataFrame(float(v), index=dates, columns=list(tickers))

    return dates, mk(close), mk(close), mk(high), mk(low), mk(1_000_000.0)


def _gate(dates, tickers, when):
    """when = {ticker: day_index}；只有該日允許進場。"""
    g = pd.DataFrame(False, index=dates, columns=list(tickers))
    for t, day in when.items():
        g.loc[dates[day], t] = True
    return g


# ───────────────── P0-1：今日收盤不得影響今日開盤的部位 ─────────────────

def _size_b_given_a_close_on_entry_day(a_close_on_day70):
    """B 於第 70 日進場；只改變 A 在第 70 日的『收盤價』，回傳 B 的股數。

    A 的 high/low 不動，確保不會因價格跳動觸發 A 的停利／停損——
    這樣兩個情境的差異就只有「當日收盤價」這一個開盤時不可知的變數。
    """
    dates, close, open_, high, low, vol = _panel(["A", "B"])
    close.loc[dates[70], "A"] = a_close_on_day70

    score = pd.DataFrame(3.0, index=dates, columns=["A", "B"])
    ma = pd.DataFrame(50.0, index=dates, columns=["A", "B"])

    bt = EventDrivenBacktester(
        tp_sl_mode="fixed", tp_pct=5.0, sl_pct=0.9,   # 寬到不會出場
        max_hold_days=999, regime_filter=False, gap_filter_atr=0,
        initial_capital=1_000_000, position_size=0.10,
    )
    bt.run(
        score, close, open_, high, low, ma, top_k=2, threshold=2.0,
        vol_df=vol, entry_gate_df=_gate(dates, ["A", "B"], {"A": 65, "B": 70}),
    )
    assert "B" in bt.last_positions, "B 應於第 70 日建倉"
    return bt.last_positions["B"]["shares"]


def test_today_close_does_not_change_today_open_position_size():
    """今天的收盤價在開盤下單時還沒發生，不可用來決定今天的部位大小。"""
    flat = _size_b_given_a_close_on_entry_day(100.0)
    spiked = _size_b_given_a_close_on_entry_day(300.0)
    assert flat == pytest.approx(spiked), (
        f"B 的股數隨 A 的『當日收盤價』改變（{flat} → {spiked}），"
        f"代表開盤 sizing 偷看了當日收盤 = 時間穿越"
    )


def test_intraday_exit_proceeds_are_not_spendable_same_day():
    """今天盤中停利拿回的錢，不能在同一天的開盤買別檔。"""
    dates, close, open_, high, low, vol = _panel(["A", "B"])
    high.loc[dates[70], "A"] = 400.0          # A 於第 70 日盤中觸發停利
    score = pd.DataFrame(3.0, index=dates, columns=["A", "B"])
    ma = pd.DataFrame(50.0, index=dates, columns=["A", "B"])

    bt = EventDrivenBacktester(
        tp_sl_mode="fixed", tp_pct=0.10, sl_pct=0.9,
        max_hold_days=999, regime_filter=False, gap_filter_atr=0,
        initial_capital=100_000, position_size=0.95,   # A 幾乎吃光現金
    )
    trades, _ = bt.run(
        score, close, open_, high, low, ma, top_k=2, threshold=2.0,
        vol_df=vol, entry_gate_df=_gate(dates, ["A", "B"], {"A": 65, "B": 70}),
    )

    assert not trades.empty and (trades["Ticker"] == "A").any(), "A 應該有停利成交"
    assert "B" not in bt.last_positions, (
        "B 在 A 幾乎吃光現金的情況下仍於同日建倉，"
        "代表用到了 A 當日盤中才回來的賣出款"
    )


def test_intraday_exit_does_not_free_a_same_open_position_slot():
    """今天盤中才出場的 A，在今天開盤決策時仍占用持倉 slot。

    只凍結現金還不夠：若 Step 1 先讀今日 high/low 刪掉 A，Step 3 再以
    len(active_trades) 補進 B，即使完全沒動用賣出款，仍然是時間穿越。
    """
    dates, close, open_, high, low, vol = _panel(["A", "B"])
    high.loc[dates[70], "A"] = 400.0
    score = pd.DataFrame(3.0, index=dates, columns=["A", "B"])
    ma = pd.DataFrame(50.0, index=dates, columns=["A", "B"])

    bt = EventDrivenBacktester(
        tp_sl_mode="fixed", tp_pct=0.10, sl_pct=0.9,
        max_hold_days=999, regime_filter=False, gap_filter_atr=0,
        initial_capital=1_000_000, position_size=0.95,
    )
    trades, _ = bt.run(
        score, close, open_, high, low, ma, top_k=1, threshold=2.0,
        vol_df=vol, entry_gate_df=_gate(dates, ["A", "B"], {"A": 65, "B": 70}),
    )

    assert not trades.empty and (trades["Ticker"] == "A").any(), "A 應於第 70 日盤中停利"
    assert "B" not in bt.last_positions, (
        "A 的今日盤中出場釋放了同一個今日 open 的持倉 slot，代表 Step 3 "
        "利用了今日 high/low 才知道的資訊"
    )


def test_drawdown_gate_is_decided_before_the_open():
    """回撤卡必須在開盤前定案：當日盤中崩跌不得回頭擋當日開盤的進場。"""
    dates, close, open_, high, low, vol = _panel(["A", "B"])
    # 第 70 日 A 盤中與收盤同時崩跌；B 的進場決策不應被同一天的崩跌影響
    low.loc[dates[70], "A"] = 5.0
    close.loc[dates[70], "A"] = 5.0
    score = pd.DataFrame(3.0, index=dates, columns=["A", "B"])
    ma = pd.DataFrame(50.0, index=dates, columns=["A", "B"])

    bt = EventDrivenBacktester(
        tp_sl_mode="fixed", tp_pct=5.0, sl_pct=0.9,
        max_hold_days=999, regime_filter=False, gap_filter_atr=0,
        initial_capital=1_000_000, position_size=0.10,
        dd_pause_pct=0.02, dd_pause_days=5,
    )
    bt.run(
        score, close, open_, high, low, ma, top_k=2, threshold=2.0,
        vol_df=vol, entry_gate_df=_gate(dates, ["A", "B"], {"A": 65, "B": 70}),
    )
    assert "B" in bt.last_positions, (
        "B 未能於第 70 日建倉：回撤卡讀到了當日才發生的崩跌（同日 lookahead）"
    )


# ───────────────── P0-2：市場後綴剝除 ─────────────────

@pytest.mark.parametrize("raw, want", [
    ("2330.TW", "2330"),
    ("5274.TWO", "5274"),      # ★ 曾被 replace('.TW','') 寫成 '5274O'
    ("3529.TWO", "3529"),
    ("6547.TWO", "6547"),
    ("4743.TWO", "4743"),
    ("0050.TW", "0050"),
    ("00981A.TW", "00981A"),
    ("2330", "2330"),          # 無後綴不動
])
def test_strip_market_suffix(raw, want):
    assert strip_market_suffix(raw) == want


def test_no_upper_cabinet_ticker_grows_a_trailing_letter():
    """上櫃代號剝完後必須是純數字（或數字+法定字尾），不能多出 'O'。"""
    for code in ("5274", "3529", "6547", "4743"):
        assert strip_market_suffix(f"{code}.TWO") == code
        assert not strip_market_suffix(f"{code}.TWO").endswith("O")


def test_single_source_of_strip_logic():
    """剝除邏輯只能有一份：ai_strategy 必須是 twstk.data.symbols 的 re-export。"""
    from twstk.data import symbols as S
    assert strip_market_suffix is S.strip_market_suffix


# ───────────── 市場歸屬：唯一真相＝security_master，未知不得猜 ─────────────

@pytest.mark.parametrize("code, want", [
    ("5274", "tpex"), ("3529", "tpex"), ("6547", "tpex"), ("4743", "tpex"),
    ("2330", "twse"), ("0050", "twse"),
])
def test_market_of_uses_security_master(code, want):
    from twstk.data.symbols import market_of
    assert market_of(code) == want


def test_yahoo_symbol_returns_none_when_market_unknown():
    """★判不出市場時必須回 None，不可預設 .TW。

    原本 build_paper_page 在法人 API 失敗時 (`except → {}`) 會讓所有股票
    退化成 .TW，上櫃股因此產生死連結——「靜默降級成看起來合理的錯誤答案」。
    """
    from twstk.data.symbols import yahoo_symbol
    assert yahoo_symbol("9999") is None
    assert yahoo_symbol("5274") == "5274.TWO"
    assert yahoo_symbol("2330") == "2330.TW"


def test_yahoo_symbol_respects_explicit_suffix():
    from twstk.data.symbols import yahoo_symbol
    assert yahoo_symbol("5274.TWO") == "5274.TWO"
    assert yahoo_symbol("2330.TW") == "2330.TW"


def test_stock_link_never_emits_a_guessed_tw_link(monkeypatch):
    """市場未知時只給純文字，不給連結。"""
    import build_paper_page as bpp
    from twstk.data import symbols as S

    monkeypatch.setattr(S, "market_of", lambda t: None)
    monkeypatch.setattr(bpp, "_STOCK_META", {})
    html = bpp._stock_link("5274")
    assert "href" not in html, f"市場未知卻仍產生連結：{html}"
    assert "5274" in html


def test_stock_link_uses_two_for_upper_cabinet(monkeypatch):
    import build_paper_page as bpp

    monkeypatch.setattr(bpp, "_STOCK_META", {"5274": "信驊"})
    html = bpp._stock_link("5274")
    assert "quote/5274.TWO" in html, html
    assert "5274.TW'" not in html


# ───────────── 共用探測 helper ─────────────

def test_probe_tries_tw_then_two_and_stops_when_found():
    from twstk.data.symbols import probe_tw_then_two
    seen = []

    def fetch(symbol_map):
        seen.append(sorted(symbol_map.values()))
        return {t: s for t, s in symbol_map.items() if s.endswith(".TWO")}

    out = probe_tw_then_two(["5274"], fetch)
    assert out == {"5274": "5274.TWO"}
    assert seen == [["5274.TW"], ["5274.TWO"]], seen


def test_probe_does_not_refetch_tickers_already_found():
    from twstk.data.symbols import probe_tw_then_two
    calls = []

    def fetch(symbol_map):
        calls.append(sorted(symbol_map))
        return {t: 1.0 for t in symbol_map}      # 第一輪就全中

    out = probe_tw_then_two(["2330", "2317"], fetch)
    assert set(out) == {"2330", "2317"}
    assert calls == [["2317", "2330"]], "第一輪全中就不該再打第二輪"


def test_probe_warns_when_still_missing(capsys):
    from twstk.data.symbols import probe_tw_then_two
    out = probe_tw_then_two(["9999"], lambda m: {}, warn_label="無報價")
    assert out == {}
    assert "9999" in capsys.readouterr().out, "兩輪都抓不到必須出聲，不能靜默"


# ───────────── 兩層參數（ai_report CLI vs strategies plugin）不得漂移 ─────────────

def _cli_default(attr: str) -> str:
    """取 ai_report 的 argparse 預設值。

    起子行程執行，避免 import ai_report 把 yfinance 等重依賴拉進測試行程。
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.argv=['x']; import ai_report; "
        f"print(getattr(ai_report.parse_args(), {attr!r}))"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout.strip()


def test_slippage_default_agrees_across_param_layers():
    """ai_report 的 CLI 預設與 strategies/base.py 的 ExecConfig 預設必須一致。

    這兩層是**兩套獨立的參數組裝**（CLI → 發布報表；PARAMS dict → paper 頁）。
    consec_loss_limit 就已經漂移過（引擎 3 / CLI 99），靠 README 提醒人記得補旗標。
    滑價這一個至少用測試釘住，不要再多一個靠人記憶的坑。
    """
    from strategies.base import ExecConfig
    cli = _cli_default("slippage")
    assert cli, "取不到 ai_report --slippage 預設值"
    assert abs(float(cli) - ExecConfig().slippage) < 1e-12, (
        f"ai_report CLI 預設 slippage={cli} 與 ExecConfig.slippage="
        f"{ExecConfig().slippage} 不一致（兩層參數漂移）"
    )


def test_slippage_default_is_non_zero():
    """零滑價是很強的假設，不可再當預設。"""
    from strategies.base import ExecConfig
    assert ExecConfig().slippage > 0, "滑價預設不得為 0（稽核前所有發布績效都是零滑價）"


def test_event_engine_default_slippage_matches_execution_layer():
    """直接建立事件引擎也不得悄悄退回零滑價。"""
    from strategies.base import ExecConfig
    assert EventDrivenBacktester().slippage == pytest.approx(ExecConfig().slippage)


# ───────────────── P0-3／P0-4：快照與資料契約 fail-loud ─────────────────

def _write_snapshot(tmp_path, close_map, n=30):
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = pd.DataFrame({t: v for t, v in close_map.items()}, index=dates)
    panel = {f: close.copy() for f in FIELDS}
    pkl = tmp_path / "panel.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(panel, f)
    return pkl, dates


def test_load_snapshot_raises_when_requested_ticker_is_missing(tmp_path):
    pkl, _ = _write_snapshot(tmp_path, {"2330": 100.0, "2317": 50.0})
    with pytest.raises(SnapshotTickerError, match="缺少"):
        load_snapshot(pkl, tickers=["2330", "2317", "5274"])


def test_load_snapshot_raises_when_column_is_all_nan(tmp_path):
    """這正是 '5274O' bug 的形狀：base 欄在、但整段全空。"""
    pkl, _ = _write_snapshot(tmp_path, {"2330": 100.0, "5274": np.nan})
    with pytest.raises(SnapshotTickerError, match="無有效 Close"):
        load_snapshot(pkl, tickers=["2330", "5274"])


def test_load_snapshot_hints_at_suffix_mangling(tmp_path):
    """快照裡有 '5274O' 卻要 '5274' 時，錯誤訊息要指向後綴剝除。"""
    pkl, _ = _write_snapshot(tmp_path, {"2330": 100.0, "5274O": 100.0})
    with pytest.raises(SnapshotTickerError, match="市場後綴"):
        load_snapshot(pkl, tickers=["2330", "5274"])


def test_load_snapshot_strict_false_still_allows_deliberate_subset(tmp_path):
    pkl, _ = _write_snapshot(tmp_path, {"2330": 100.0, "2317": 50.0})
    close, *_ = load_snapshot(pkl, tickers=["2330", "9999"], strict=False)
    assert list(close.columns) == ["2330"]


def test_panel_hash_changes_when_a_ticker_label_changes():
    """值完全相同但 2330 被改名 FAKE，資料快照 hash 必須改變。"""
    dates = pd.bdate_range("2025-01-01", periods=30)
    close = pd.DataFrame({"2330": 100.0, "2317": 50.0}, index=dates)
    panel = {f: close.copy() for f in FIELDS}
    original = build_manifest(
        panel, dates[-1], provider="synthetic", auto_adjust=True,
    )["panel_sha256"]

    renamed = {
        field: frame.rename(columns={"2330": "FAKE"})
        for field, frame in panel.items()
    }
    renamed_hash = build_manifest(
        renamed, dates[-1], provider="synthetic", auto_adjust=True,
    )["panel_sha256"]
    assert renamed_hash != original

    reordered = {
        field: frame[["2317", "2330"]]
        for field, frame in panel.items()
    }
    reordered_hash = build_manifest(
        reordered, dates[-1], provider="synthetic", auto_adjust=True,
    )["panel_sha256"]
    assert reordered_hash == original, "只改欄位順序不應改 canonical panel hash"


def test_snapshot_manifest_detects_column_label_tampering(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=30)
    close = pd.DataFrame({"2330": 100.0, "2317": 50.0}, index=dates)
    panel = {f: close.copy() for f in FIELDS}
    manifest = build_manifest(
        panel, dates[-1], provider="synthetic", auto_adjust=True,
    )
    out = tmp_path / "snapshot"
    freeze_snapshot(panel, out, manifest)

    pkl = out / "panel.pkl"
    with open(pkl, "rb") as f:
        tampered = pickle.load(f)
    for field in FIELDS:
        tampered[field] = tampered[field].rename(columns={"2330": "FAKE"})
    with open(pkl, "wb") as f:
        pickle.dump(tampered, f)

    with pytest.raises(SnapshotIntegrityError, match="manifest 不一致"):
        load_snapshot(out, tickers=["FAKE", "2317"])


def test_contract_rejects_all_nan_column(tmp_path):
    """橫斷面完整度看不到的盲區：整欄全空必須讓契約 fail-closed。"""
    dates = pd.bdate_range("2025-01-01", periods=40)
    close = pd.DataFrame(
        {"0050": 100.0, "2330": 100.0, "2317": 100.0, "5274": np.nan},
        index=dates,
    )
    panel = {f: close.copy() for f in FIELDS}
    panel["Volume"] = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)

    result = validate_panel(
        panel, dates[-1].date(), scheduled=False, calendar="weekday-approx",
    )
    assert not result.ok
    assert any("整段無任何有效 Close" in r for r in result.reasons), result.reasons


def test_contract_passes_when_all_columns_are_covered(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=40)
    close = pd.DataFrame(
        {"0050": 100.0, "2330": 100.0, "5274": 100.0}, index=dates,
    )
    panel = {f: close.copy() for f in FIELDS}
    panel["Volume"] = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)

    result = validate_panel(
        panel, dates[-1].date(), scheduled=False, calendar="weekday-approx",
    )
    assert result.ok, result.reasons
