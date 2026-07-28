"""回歸測試：快照的外部序列凍結（2026-07-25 稽核 B4）。

問題背景
--------
即使設了 TWSTK_SNAPSHOT，兩個**會改變回測結果**的輸入仍走網路：
  · 0050（regime filter）── ai_report 另外呼叫 fetch_benchmark
  · VIX（regime_sizing / macro_regime）── 引擎在 run() 內自行下載
實測 VIX 用凍結 vs 即時，v8.5 年化差 5pp。
⇒ 只凍結 panel 不足以宣稱「當日 run 可重現」，panel_sha256 的出處宣稱不完整。

開發時踩到的兩個坑（測試即為防線）
  1. 對**記憶體物件**取 aux hash → CSV 往返改變浮點字串表示，
     「剛寫完就讀」就對不上。改為對**檔案內容**取 hash。
  2. 第一版從 `close_df` 取 0050 —— 但 close_df 已被 load_snapshot 依
     EXTENDED_TICKERS(115 檔) subset，0050 不在其中而被濾掉，修正完全沒生效。
     必須直接從快照面板讀。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from twstk.data.contract import (
    FIELDS,
    SnapshotIntegrityError,
    build_manifest,
    freeze_snapshot,
    load_aux_series,
    load_snapshot,
)


def _panel(n=300, cols=("0050", "2330", "2317")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return {f: pd.DataFrame(100.0, index=idx, columns=list(cols)) for f in FIELDS}, idx


def _freeze(tmp_path, aux=None, name="snap"):
    panel, idx = _panel()
    man = build_manifest(panel, "2025-02-28", provider="test", auto_adjust=True)
    out = str(tmp_path / name)
    freeze_snapshot(panel, out, man, aux_series=aux)
    return out, idx


def test_aux_round_trips_exactly(tmp_path):
    _, idx = _panel()
    vix = pd.Series(np.linspace(12, 30, len(idx)), index=idx)
    out, _ = _freeze(tmp_path, {"VIX": vix})
    got = load_aux_series(out, "VIX")
    assert got is not None and len(got) == len(vix)
    assert np.allclose(got.to_numpy(), vix.to_numpy())


def test_manifest_records_aux_names_and_hash(tmp_path):
    _, idx = _panel()
    out, _ = _freeze(tmp_path, {"VIX": pd.Series(20.0, index=idx)})
    man = json.loads((tmp_path / "snap" / "manifest.json").read_text(encoding="utf-8"))
    assert man["aux_series"] == ["VIX"]
    assert isinstance(man["aux_sha256"], str) and len(man["aux_sha256"]) == 64


def test_tampered_aux_is_rejected(tmp_path):
    """★竄改必須被擋——這是凍結的意義所在。"""
    _, idx = _panel()
    out, _ = _freeze(tmp_path, {"VIX": pd.Series(np.linspace(12, 30, len(idx)),
                                                 index=idx)})
    p = os.path.join(out, "aux_series.csv")
    txt = open(p, encoding="utf-8").read().replace("12.0", "99.0", 1)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(txt)
    with pytest.raises(SnapshotIntegrityError):
        load_aux_series(out, "VIX")


def test_old_snapshot_without_aux_returns_none(tmp_path):
    """舊快照沒有 aux 檔 → 安靜回 None，呼叫端據此退回即時下載並出聲。"""
    out, _ = _freeze(tmp_path, aux=None)
    assert load_aux_series(out, "VIX") is None


def test_unknown_series_name_returns_none(tmp_path):
    _, idx = _panel()
    out, _ = _freeze(tmp_path, {"VIX": pd.Series(20.0, index=idx)})
    assert load_aux_series(out, "NOT_THERE") is None


def test_panel_loading_unaffected_by_aux(tmp_path):
    _, idx = _panel()
    out, _ = _freeze(tmp_path, {"VIX": pd.Series(20.0, index=idx)})
    close, *_ = load_snapshot(os.path.join(out, "panel.pkl"),
                              tickers=["0050", "2330", "2317"])
    assert close.shape == (300, 3)


def test_aux_accepts_path_to_panel_pkl(tmp_path):
    """呼叫端習慣傳 panel.pkl 路徑（TWSTK_SNAPSHOT 就是它），必須也能用。"""
    _, idx = _panel()
    out, _ = _freeze(tmp_path, {"VIX": pd.Series(20.0, index=idx)})
    assert load_aux_series(os.path.join(out, "panel.pkl"), "VIX") is not None


def test_preflight_freezes_vix():
    src = open("preflight.py", encoding="utf-8").read()
    assert "aux_series=" in src and "^VIX" in src, "preflight 未凍結 VIX"


def test_ai_report_prefers_snapshot_for_market_and_vix():
    """★防回歸：0050 與 VIX 都必須優先走快照。

    尤其 0050 不可從 close_df 取——它已被 subset 掉 0050。
    """
    code = [ln for ln in open("ai_report.py", encoding="utf-8").read().splitlines()
            if not ln.lstrip().startswith("#")]
    src = "\n".join(code)
    assert "load_aux_series" in src, "未從快照讀 VIX"
    assert "vix_series=frozen_vix" in src, "未把凍結 VIX 傳給引擎"
    assert "tickers=['0050']" in src, "未從快照面板直接讀 0050"
    assert "market_close = close_df['0050']" not in src, \
        "0050 不可從 close_df 取（已被 EXTENDED_TICKERS subset 濾掉）"


def test_report_benchmark_row_also_comes_from_snapshot():
    """★報表上的「0050 年化 / Sharpe / MDD / 超額報酬 α」也必須凍結。

    Phase 3.5 的 regime 用 0050 早就走快照了，但 Phase 6 算給讀者看的那組
    benchmark 數字原本仍即時下載 —— 同一個 panel_sha256 會對到不同的 0050
    數字，而 README 與 index 又引用它。2026-07-28 補上。
    """
    code = [ln for ln in open("ai_report.py", encoding="utf-8").read().splitlines()
            if not ln.lstrip().startswith("#")]
    src = "\n".join(code)
    assert "_snapshot_benchmark('0050'" in src, "Phase 6 的 0050 未優先走快照"
    assert "def _snapshot_benchmark(" in src


def test_snapshot_benchmark_normalises_to_one(tmp_path, monkeypatch):
    """必須與 fetch_benchmark 同樣「起點歸一」，否則年化／MDD 會失真。"""
    import numpy as np

    import ai_report

    idx = pd.bdate_range("2024-01-01", periods=300)
    panel = {f: pd.DataFrame(1.0, index=idx, columns=["0050", "2330"]) for f in FIELDS}
    panel["Close"]["0050"] = np.linspace(50.0, 100.0, len(idx))
    man = build_manifest(panel, "2025-02-28", provider="test", auto_adjust=True)
    out = str(tmp_path / "snap")
    freeze_snapshot(panel, out, man)

    monkeypatch.setenv("TWSTK_SNAPSHOT", os.path.join(out, "panel.pkl"))
    s = ai_report._snapshot_benchmark("0050", None, None)
    assert s is not None
    assert s.iloc[0] == pytest.approx(1.0)
    assert s.iloc[-1] == pytest.approx(2.0)      # 50 → 100


def test_snapshot_benchmark_returns_none_when_absent(tmp_path, monkeypatch):
    """快照沒有這檔（如 00981A）→ 回 None，呼叫端照舊下載。"""
    import ai_report

    idx = pd.bdate_range("2024-01-01", periods=300)
    panel = {f: pd.DataFrame(1.0, index=idx, columns=["2330"]) for f in FIELDS}
    man = build_manifest(panel, "2025-02-28", provider="test", auto_adjust=True)
    out = str(tmp_path / "snap2")
    freeze_snapshot(panel, out, man)
    monkeypatch.setenv("TWSTK_SNAPSHOT", os.path.join(out, "panel.pkl"))
    assert ai_report._snapshot_benchmark("00981A", None, None) is None
