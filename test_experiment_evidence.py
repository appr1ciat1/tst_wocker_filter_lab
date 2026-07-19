"""回歸測試：試驗證據留存（2026-07-19 稽核的根因修復）。

背景：
    PBO/CSCV 要求「當初選出贏家時評估過的**全部**試驗集合」。這份證據一旦
    沒留就事後補不回來——不像沒跑 PBO 可以補跑。

    實際發生的事：sweep.py / ablation_study.py / factor_grid_search.py 早就
    會寫 artifacts/experiments.sqlite，但 CI 的 quarterly job 用
    `git add -u artifacts`，而 `-u` **只 stage 已追蹤的檔案**。該 DB 從未被
    追蹤，所以每季產生一次、每季隨 runner 銷毀。產生 SURGE PRO 那組
    strong_tiers 的搜尋，試驗集合已經永久遺失。

    修法是分兩層：純文字索引（不含日報酬）進 git 永久保存；完整 sqlite
    走 actions/upload-artifact 供重算 PBO。體積差 170x。
"""

import datetime as dt
import json
import random

import pytest

from research.experiment_registry import (
    ExperimentRegistry,
    export_index,
    trial_record,
)


def _make_registry(tmp_path, n_trials=6, n_days=250):
    db = tmp_path / "experiments.sqlite"
    reg = ExperimentRegistry(db)
    dates = [(dt.date(2024, 1, 2) + dt.timedelta(days=i)).isoformat()
             for i in range(n_days)]
    rng = random.Random(42)
    trials = [
        trial_record(
            f"trial_{k}",
            parameters={"sl_atr": 3.0 + k * 0.1, "hold_days": 20},
            metrics={"sharpe": 1.0 + k * 0.05,
                     "max_drawdown_pct": -0.3, "ann_return": 0.30},
            daily_returns=[{"date": d, "return": rng.gauss(0.0006, 0.015)}
                           for d in dates],
        )
        for k in range(n_trials)
    ]
    reg.record_experiment(
        experiment_id="exp_test", source="pytest", strategy_version="v8.5",
        hypothesis="evidence retention", parameter_space={"sl_atr": [3.0, 3.5]},
        number_of_trials=n_trials, metrics={"sharpe": 1.2},
        pbo=0.31, deflated_sharpe=0.97, decision="watchlist",
        trials=trials, data_paths=[],
    )
    return db


def test_index_has_one_line_per_trial(tmp_path):
    db = _make_registry(tmp_path, n_trials=6)
    idx = tmp_path / "experiments_index.jsonl"
    assert export_index(db, idx) == 6
    lines = idx.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    assert {json.loads(ln)["trial_id"] for ln in lines} == {
        f"trial_{k}" for k in range(6)}


def test_index_records_what_pbo_claims_need(tmp_path):
    """索引必須足以回答「試了幾組、參數為何、PBO/DSR 多少、哪個 commit」。"""
    db = _make_registry(tmp_path)
    idx = tmp_path / "i.jsonl"
    export_index(db, idx)
    row = json.loads(idx.read_text(encoding="utf-8").splitlines()[0])
    for key in ("experiment_id", "created_at", "git_commit", "number_of_trials",
                "pbo", "deflated_sharpe", "trial_id", "parameters", "sharpe"):
        assert key in row, f"索引缺少 {key}"
    assert row["number_of_trials"] == 6
    assert row["pbo"] == pytest.approx(0.31)
    assert row["parameters"]["hold_days"] == 20


def test_index_excludes_daily_returns(tmp_path):
    """★索引刻意不含日報酬——那是體積來源（170x）。

    含了就會走回「binary 進 git → repo 膨脹 → 拖垮 Pages」的老路。
    要重算 PBO 請取 artifact 裡的完整 sqlite。
    """
    db = _make_registry(tmp_path)
    idx = tmp_path / "i.jsonl"
    export_index(db, idx)
    text = idx.read_text(encoding="utf-8")
    assert "daily_returns" not in text
    row = json.loads(text.splitlines()[0])
    assert "daily_returns" not in row
    # 體積必須遠小於原始 DB
    assert idx.stat().st_size * 10 < db.stat().st_size, (
        f"索引 {idx.stat().st_size} 相對 DB {db.stat().st_size} 太大，"
        f"可能不慎把日報酬寫進去了"
    )


def test_export_is_idempotent(tmp_path):
    """重複匯出不得產生重複列（CI 每季會跑一次）。"""
    db = _make_registry(tmp_path, n_trials=4)
    idx = tmp_path / "i.jsonl"
    assert export_index(db, idx) == 4
    assert export_index(db, idx) == 4
    assert len(idx.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_export_on_missing_db_is_noop(tmp_path):
    """DB 不存在時安靜回 0，不得炸掉 CI。"""
    assert export_index(tmp_path / "nope.sqlite", tmp_path / "o.jsonl") == 0


def test_search_scripts_are_wired_to_registry():
    """搜尋腳本必須寫 registry，否則 trial set 又會遺失。"""
    import pathlib
    missing = []
    for name in ("sweep.py", "ablation_study.py", "factor_grid_search.py"):
        src = pathlib.Path(name).read_text(encoding="utf-8")
        if "experiment_registry" not in src:
            missing.append(name)
    assert not missing, f"這些搜尋腳本沒接 registry：{missing}"


def test_ci_persists_the_index_explicitly():
    """CI 必須「顯式」git add 索引。

    `git add -u` 只 stage 已追蹤檔案——這正是 experiments.sqlite
    每季消失的原因，不可重蹈。
    """
    import pathlib
    wf = pathlib.Path(".github/workflows/update_ai_report.yml").read_text(
        encoding="utf-8")
    assert "--export-index" in wf, "CI 未匯出試驗索引"
    assert "git add experiments_index.jsonl" in wf, (
        "CI 未顯式 add 索引（靠 `git add -u` 會漏掉未追蹤的新檔）")
    assert "artifacts/experiments.sqlite" in wf, "CI 未上傳完整 registry"
