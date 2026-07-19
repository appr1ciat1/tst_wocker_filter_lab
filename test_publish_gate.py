"""回歸測試：發布出處閘門（2026-07-19 稽核 B 項）。

設計要點（別把它改成「發布時重算 PBO」）：
    PBO/CSCV 與 Deflated Sharpe 需要**一組 trial**；每日發布只有一個 run，
    算不出來，每天跑完整 sweep 也不現實。所以閘門只回答
    「這組參數有沒有被正式驗收過、過期了沒」——出處，不是統計重算。

落地方式：
    先 observe（記錄不阻擋）→ 累積紀錄與通過率 → 再切 enforce。
    直接硬擋會讓每日發布立刻中斷（registry 現在一筆通過紀錄都沒有），
    逼人在壓力下臨時放行，反而破壞紀律。
"""

import os

import pytest

from research.experiment_registry import ExperimentRegistry, trial_record
from validation import publish_gate as G

CFG = {
    "sl_atr_mult": 3.5, "hold_days": 25, "position_size": 0.10,
    "slippage": 0.002, "artifact_label": "surge_pro",
    "initial_capital": 200_000, "start_date": "2019-01-01",
}


def _accepted_registry(tmp_path, cfg=CFG, decision="accept"):
    db = tmp_path / "experiments.sqlite"
    reg = ExperimentRegistry(db)
    reg.record_experiment(
        experiment_id="exp_ok", source="ablation", strategy_version="SURGE PRO",
        hypothesis="tiered add-on", number_of_trials=24,
        metrics={"sharpe": 1.2}, pbo=0.31, deflated_sharpe=0.97,
        decision=decision, config=cfg,
        trials=[trial_record("t0", parameters=cfg, metrics={"sharpe": 1.2})],
        data_paths=[],
    )
    return db


# ── 三種狀態 ────────────────────────────────────────────────

def test_missing_when_registry_absent(tmp_path):
    r = G.evaluate(CFG, registry_path=tmp_path / "nope.sqlite")
    assert r.status == "missing" and not r.ok


def test_valid_when_accepted_record_exists(tmp_path):
    r = G.evaluate(CFG, registry_path=_accepted_registry(tmp_path))
    assert r.status == "valid" and r.ok
    assert r.pbo == pytest.approx(0.31)
    assert r.deflated_sharpe == pytest.approx(0.97)


def test_expired_when_older_than_max_age(tmp_path):
    r = G.evaluate(CFG, registry_path=_accepted_registry(tmp_path), max_age_days=0)
    assert r.status == "expired" and not r.ok


def test_non_accepted_decision_does_not_count(tmp_path):
    """只有 accept/pass 類判定算通過；watchlist / reject 不算。"""
    db = _accepted_registry(tmp_path, decision="watchlist")
    assert G.evaluate(CFG, registry_path=db).status == "missing"


# ── 指紋語意：策略參數變動要失效，執行情境變動不該失效 ──────────

def test_any_strategy_param_change_invalidates(tmp_path):
    db = _accepted_registry(tmp_path)
    for key, val in (("sl_atr_mult", 3.0), ("hold_days", 20),
                     ("position_size", 0.15), ("slippage", 0.0)):
        r = G.evaluate({**CFG, key: val}, registry_path=db)
        assert r.status == "missing", f"改了 {key} 卻仍判定 {r.status}"


def test_execution_context_change_keeps_validity(tmp_path):
    """資金／日期／檔名標籤變動不代表策略變了，不該讓驗收失效。"""
    db = _accepted_registry(tmp_path)
    r = G.evaluate({**CFG, "initial_capital": 5_000_000,
                    "artifact_label": "whatever", "end_date": "2026-07-18",
                    "days": 900}, registry_path=db)
    assert r.status == "valid", r.notes


def test_new_unknown_param_invalidates_by_default(tmp_path):
    """★排除法而非白名單：新參數會自動改變指紋 → 標記為未驗收。

    寧可誤報（要求重新驗收），不可漏報（拿舊驗收替新策略背書）。
    """
    db = _accepted_registry(tmp_path)
    r = G.evaluate({**CFG, "some_new_knob_added_later": True}, registry_path=db)
    assert r.status == "missing"


# ── 模式 ───────────────────────────────────────────────────

def test_observe_mode_never_blocks(tmp_path):
    r = G.evaluate({**CFG, "sl_atr_mult": 9.9},
                   registry_path=tmp_path / "nope.sqlite", mode="observe")
    assert r.status == "missing"
    assert r.blocking is False, "觀察模式不得阻擋發布"


def test_enforce_mode_blocks_only_when_not_valid(tmp_path):
    db = _accepted_registry(tmp_path)
    assert G.evaluate(CFG, registry_path=db, mode="enforce").blocking is False
    bad = G.evaluate({**CFG, "hold_days": 1}, registry_path=db, mode="enforce")
    assert bad.blocking is True


def test_mode_defaults_to_observe_via_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TWSTK_PUBLISH_GATE", raising=False)
    assert G.evaluate(CFG, registry_path=tmp_path / "x").mode == "observe"
    monkeypatch.setenv("TWSTK_PUBLISH_GATE", "enforce")
    assert G.evaluate(CFG, registry_path=tmp_path / "x").mode == "enforce"
    monkeypatch.setenv("TWSTK_PUBLISH_GATE", "nonsense")
    assert G.evaluate(CFG, registry_path=tmp_path / "x").mode == "observe"


# ── 韌性：閘門自己壞掉不可拖垮發布 ──────────────────────────

def test_corrupt_registry_degrades_to_missing_not_crash(tmp_path):
    db = tmp_path / "experiments.sqlite"
    db.write_bytes(b"this is not a sqlite file")
    r = G.evaluate(CFG, registry_path=db)
    assert r.status == "missing"
    assert r.notes, "應留下失敗原因供追查"


def test_legacy_registry_without_fingerprint_column(tmp_path):
    """舊版 DB 沒有 config_fingerprint 欄位時，安靜回 missing 並說明。"""
    import sqlite3
    db = tmp_path / "experiments.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE experiments (experiment_id TEXT, created_at TEXT)")
    r = G.evaluate(CFG, registry_path=db)
    assert r.status == "missing"
    assert any("config_fingerprint" in n for n in r.notes)


def test_ai_report_wires_the_gate_without_hardcoding_mode():
    """ai_report 必須接閘門，且不得在程式碼寫死 mode。

    只看**實際程式碼行**，略過註解——註解裡本來就會提到 enforce 怎麼切。
    """
    code = [ln for ln in open("ai_report.py", encoding="utf-8").read().splitlines()
            if not ln.lstrip().startswith("#")]
    src = "\n".join(code)
    assert "publish_gate.evaluate(" in src, "ai_report 未接發布閘門"
    assert 'mode="enforce"' not in src and "mode='enforce'" not in src, (
        "不得在程式碼裡寫死 enforce；模式應由 TWSTK_PUBLISH_GATE 控制")
    assert "gate.blocking" in src, "缺少 enforce 時的阻擋分支"


def test_gate_writes_status_into_metadata():
    """狀態要落進 artifacts，之後才回看得到。"""
    code = open("ai_report.py", encoding="utf-8").read()
    assert "'publish_gate': gate_dict" in code, "metadata 未記錄閘門狀態"


def test_ai_report_fingerprint_uses_complete_cli_config():
    """regime/tier/gap 等核心旗標必須進 config，不能只 fingerprint 報表欄位。"""
    code = open("ai_report.py", encoding="utf-8").read()
    assert "**vars(args)" in code, (
        "ai_report 的驗收 config 未涵蓋完整 CLI；策略核心參數改動可能誤用舊驗收")


def test_environment_is_not_enforcing_during_tests():
    assert os.environ.get("TWSTK_PUBLISH_GATE") in (None, "", "observe")
