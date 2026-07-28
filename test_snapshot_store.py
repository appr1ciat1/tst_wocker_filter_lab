"""回歸測試：凍結快照的保存與取回（2026-07-26 稽核）。

為什麼這件事重要
----------------
實測：同一份程式、同一組參數、同一個回測視窗，只差下載日期一天，
v8.5 年化 27.5% → 18.79%。⇒ 發布數字只在「那一次下載」成立。

preflight 早就會凍結快照，但 panel.pkl 沒有任何地方留著（.gitignore 只放行
manifest.json）——留了收據沒留貨。snapshot_store 就是把貨留住的那一層。

本檔只測**離線**路徑（pack / unpack / verify / 命名 / 安全性）。
publish / fetch 要呼叫 gh，屬於傳輸層，不在單元測試範圍；
但它們的核心邏輯都走這裡測過的函式。
"""

import json
import os
import pickle
import tarfile

import pandas as pd
import pytest

import snapshot_store as S
from twstk.data.contract import (
    FIELDS,
    SnapshotIntegrityError,
    build_manifest,
    freeze_snapshot,
)


def _make_snapshot(tmp_path, name="snapshot_20260728", with_aux=True, n=40):
    idx = pd.bdate_range("2026-01-01", periods=n)
    cols = ["0050", "2330", "2317"]
    panel = {f: pd.DataFrame(100.0, index=idx, columns=cols) for f in FIELDS}
    man = build_manifest(panel, "2026-07-28", provider="test", auto_adjust=True)
    out = str(tmp_path / name)
    aux = {"VIX": pd.Series(18.0, index=idx)} if with_aux else None
    freeze_snapshot(panel, out, man, aux_series=aux)
    return out


# ── 命名：同一天的不同面板不可互相覆蓋 ──────────────────────────

def test_asset_name_includes_date_and_hash(tmp_path):
    snap = _make_snapshot(tmp_path)
    man = json.load(open(os.path.join(snap, "manifest.json"), encoding="utf-8"))
    name = S.asset_name(man)
    assert name.startswith("snapshot_2026-07-28_")
    assert name.endswith(".tar.gz")
    assert man["panel_sha256"][:8] in name


def test_same_day_different_panel_gets_different_name(tmp_path):
    """★同一天重跑若產生不同面板，兩份都要留得住。

    這正是本次事件的形狀：日期相同、資料被上游改寫。若只以日期命名，
    第二份會無聲覆蓋第一份，等於把證據弄丟。
    """
    a = _make_snapshot(tmp_path, "snapshot_20260728")
    idx = pd.bdate_range("2026-01-01", periods=40)
    panel2 = {f: pd.DataFrame(101.0, index=idx, columns=["0050", "2330", "2317"])
              for f in FIELDS}
    man2 = build_manifest(panel2, "2026-07-28", provider="test", auto_adjust=True)
    b = str(tmp_path / "snapshot_20260728_v2")
    freeze_snapshot(panel2, b, man2)

    na = S.asset_name(json.load(open(os.path.join(a, "manifest.json"), encoding="utf-8")))
    nb = S.asset_name(json.load(open(os.path.join(b, "manifest.json"), encoding="utf-8")))
    assert na != nb, "同日不同面板卻算出同一個檔名——會互相覆蓋"


# ── 往返 ────────────────────────────────────────────────────

def test_pack_unpack_verify_round_trip(tmp_path):
    snap = _make_snapshot(tmp_path)
    arc = S.pack(snap, str(tmp_path / "a.tar.gz"))
    got = S.unpack(arc, str(tmp_path / "out"))
    info = S.verify(got)
    assert info["as_of"] == "2026-07-28"
    assert info["n_tickers"] == 3
    assert info["aux"] == [{"name": "VIX", "n": 40}]


def test_round_trip_without_aux(tmp_path):
    """舊快照沒有 aux 檔，仍要能打包與驗證。"""
    snap = _make_snapshot(tmp_path, with_aux=False)
    got = S.unpack(S.pack(snap, str(tmp_path / "a.tar.gz")), str(tmp_path / "out"))
    assert S.verify(got)["aux"] == []


# ── 完整性：留下來的貨必須能證明沒被動過 ────────────────────────

def test_tampered_panel_is_rejected(tmp_path):
    """★沒有這一條，保存機制就沒有意義。"""
    snap = _make_snapshot(tmp_path)
    got = S.unpack(S.pack(snap, str(tmp_path / "a.tar.gz")), str(tmp_path / "out"))
    p = os.path.join(got, "panel.pkl")
    d = pickle.load(open(p, "rb"))
    d["Close"].iloc[0, 0] = 999999.0
    pickle.dump(d, open(p, "wb"), protocol=5)
    with pytest.raises(SnapshotIntegrityError):
        S.verify(got)


def test_tampered_aux_is_rejected(tmp_path):
    snap = _make_snapshot(tmp_path)
    got = S.unpack(S.pack(snap, str(tmp_path / "a.tar.gz")), str(tmp_path / "out"))
    p = os.path.join(got, "aux_series.csv")
    txt = open(p, encoding="utf-8").read().replace("18.0", "99.0", 1)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(txt)
    with pytest.raises(SnapshotIntegrityError):
        S.verify(got)


def test_manifest_claiming_absent_aux_is_rejected(tmp_path):
    """manifest 說有 VIX、檔案卻沒有 → 必須出聲，不可當作沒事。"""
    snap = _make_snapshot(tmp_path)
    got = S.unpack(S.pack(snap, str(tmp_path / "a.tar.gz")), str(tmp_path / "out"))
    os.remove(os.path.join(got, "aux_series.csv"))
    with pytest.raises((S.SnapshotStoreError, SnapshotIntegrityError)):
        S.verify(got)


# ── 失敗要吵，不要靜默 ──────────────────────────────────────

def test_pack_without_panel_raises(tmp_path):
    """只有 manifest（＝現在 git 裡的狀態）不算留住了快照。"""
    snap = _make_snapshot(tmp_path)
    os.remove(os.path.join(snap, "panel.pkl"))
    with pytest.raises(S.SnapshotStoreError, match="沒有貨可留"):
        S.pack(snap)


def test_missing_manifest_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(S.SnapshotStoreError):
        S.pack(str(d))


def test_unsafe_archive_paths_are_rejected(tmp_path):
    """下載來的壓縮檔是外部輸入，不可信任其路徑。"""
    bad = tmp_path / "bad.tar.gz"
    payload = tmp_path / "x.txt"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")
    with pytest.raises(S.SnapshotStoreError):
        S.unpack(str(bad), str(tmp_path / "out"))


def test_multi_top_level_archive_is_rejected(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    f1 = tmp_path / "a.txt"; f1.write_text("a", encoding="utf-8")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(f1, arcname="one/a.txt")
        tar.add(f1, arcname="two/a.txt")
    with pytest.raises(S.SnapshotStoreError):
        S.unpack(str(bad), str(tmp_path / "out"))


# ── 同一份資料跑到底：paper 頁不可自己另外下載 ──────────────────

def test_build_market_data_prefers_snapshot(tmp_path, monkeypatch):
    """★preflight 凍結了共享快照，四份報表共讀它，但 paper 頁走 build_market_data，
    原本一律重新下載 —— 同一次 CI 內兩邊可能拿到不同的歷史。

    歷史是真的會變的（回溯調整），所以「都在同一天跑」不構成同源保證。
    """
    from twstk.backtest import engine as E

    snap = _make_snapshot(tmp_path)
    monkeypatch.setenv("TWSTK_SNAPSHOT", os.path.join(snap, "panel.pkl"))

    def _boom(*a, **k):
        raise AssertionError("設了 TWSTK_SNAPSHOT 卻仍去下載價格")

    monkeypatch.setattr(E, "fetch_prices", _boom)
    monkeypatch.setattr(E, "fetch_benchmark", _boom)

    cfg = E.RunConfig(tickers=["0050", "2330", "2317"], days=100,
                      start_date="2026-01-01", end_date=None,
                      universe_size=0, initial_capital=200_000,
                      top_k=2, threshold=2.0, benchmark_ticker="0050")

    class _Bare:
        requires = frozenset()

    md = E.build_market_data(cfg, _Bare())
    assert md.close.shape[1] == 3
    assert md.market_close is not None, "benchmark 也該走快照（0050 驅動 regime）"


def test_falls_back_to_download_when_no_snapshot(tmp_path, monkeypatch):
    """沒設快照時行為不變——這個改動不能綁架本機研究流程。"""
    from twstk.backtest import engine as E

    monkeypatch.delenv("TWSTK_SNAPSHOT", raising=False)
    called = {"n": 0}

    def _fake(*a, **k):
        called["n"] += 1
        idx = pd.bdate_range("2026-01-01", periods=30)
        f = pd.DataFrame(100.0, index=idx, columns=["2330"])
        from twstk.data.prices import PricePanel
        return PricePanel(close=f, open=f, high=f, low=f, volume=f)

    monkeypatch.setattr(E, "fetch_prices", _fake)
    monkeypatch.setattr(E, "fetch_benchmark", lambda *a, **k: None)

    class _Bare:
        requires = frozenset()

    E.build_market_data(E.RunConfig(tickers=["2330"], days=30, universe_size=0), _Bare())
    assert called["n"] == 1


# ── CI 必須真的呼叫它 ───────────────────────────────────────

def test_workflow_publishes_the_snapshot():
    """★沒接上 CI 的保存機制等於沒有。"""
    wf = open(".github/workflows/update_ai_report.yml", encoding="utf-8").read()
    assert "snapshot_store.py publish" in wf, (
        "workflow 未在 preflight 之後保存快照——panel.pkl 仍然只存在於該次 runner，"
        "發布數字依舊事後無法重現")


def test_gitignore_still_excludes_panel_but_keeps_manifest():
    """panel.pkl 走 release，不進 git；manifest 留在 git 當索引。"""
    gi = open(".gitignore", encoding="utf-8").read()
    assert "!artifacts/snapshot_*/manifest.json" in gi
    assert "artifacts/snapshot_*/*" in gi
