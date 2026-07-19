"""
validation.publish_gate — 發布前的「出處閘門」（2026-07-19 稽核）

為什麼不是「發布時重算 PBO」
---------------------------
PBO/CSCV 與 Deflated Sharpe 本質上需要**一組 trial**（一次參數搜尋裡評估過的
全部配置）。每日發布只有一個 run＝一個 trial，算不出 PBO；每天跑完整 sweep
也不現實。

所以每日管線不該承擔搜尋成本，它只該回答一個問題：

    「這組參數有沒有被正式驗收過？那筆驗收過期了沒？」

這是出處閘門（provenance gate），不是統計重算。統計驗證在研究流程做一次，
發布流程只檢查紀錄是否存在且有效。

三種狀態
--------
    valid   ── registry 裡有這組配置的通過紀錄，且未過期
    expired ── 有過紀錄但超過 max_age_days
    missing ── 從來沒有被驗收過

觀察模式（預設）
----------------
閘門一上線，四大策略會全部 missing——因為 registry 裡一筆都沒有。
直接阻擋會讓每日發布立刻中斷，而且沒有任何基準能判斷「多嚴格才合理」，
逼人在壓力下臨時放行，反而破壞紀律。

所以預設 mode="observe"：計算、記錄、印出，**不阻擋**。
等 ablation/PBO 產生第一批紀錄、觀察一段時間知道實際通過率後，
再把 mode 切成 "enforce"。切換時 evaluate() 的邏輯完全不用改。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MAX_AGE_DAYS = 180

# fingerprint 不納入的鍵：它們每天/每次執行本來就會變，或屬於執行情境而非策略定義。
# ★採「排除法」而非「白名單」是刻意的：新增的策略參數會**自動**改變 fingerprint，
#   使舊驗收紀錄失效 → 狀態變 missing 被標記出來。寧可誤報，不可漏報。
EXCLUDED_KEYS = frozenset({
    "artifact_label",        # 只影響檔名
    "initial_capital",       # 執行情境，非策略定義
    "capital",
    "start_date", "end_date", "eval_start", "days",
    "report_date", "created_at",
})


@dataclass
class GateResult:
    status: str                       # valid | expired | missing
    fingerprint: str
    strategy: str | None = None
    experiment_id: str | None = None
    validated_at: str | None = None
    age_days: float | None = None
    pbo: float | None = None
    deflated_sharpe: float | None = None
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    mode: str = "observe"
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "valid"

    @property
    def blocking(self) -> bool:
        """enforce 模式下是否應該阻擋發布。"""
        return self.mode == "enforce" and not self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "fingerprint": self.fingerprint,
            "strategy": self.strategy,
            "experiment_id": self.experiment_id,
            "validated_at": self.validated_at,
            "age_days": self.age_days,
            "pbo": self.pbo,
            "deflated_sharpe": self.deflated_sharpe,
            "max_age_days": self.max_age_days,
            "notes": self.notes,
        }

    def describe(self) -> str:
        icon = {"valid": "✅", "expired": "🟠", "missing": "🔴"}.get(self.status, "❔")
        head = f"{icon} 發布閘門[{self.mode}]：{self.status}  fingerprint={self.fingerprint[:12]}"
        if self.status == "valid":
            return (f"{head}  ← 驗收於 {self.validated_at}"
                    f"（{self.age_days:.0f} 天前，PBO={self.pbo}, DSR={self.deflated_sharpe}）")
        if self.status == "expired":
            return (f"{head}  ← 最近一筆驗收 {self.validated_at}"
                    f" 已逾 {self.max_age_days} 天（{self.age_days:.0f} 天前）")
        return f"{head}  ← registry 中查無此配置的驗收紀錄"


def config_fingerprint(config: dict[str, Any]) -> str:
    """把「策略定義用的參數」壓成穩定指紋。

    排除執行情境與每日變動項（見 EXCLUDED_KEYS）。任何策略參數改動都會
    改變指紋，使舊驗收紀錄不再適用——這是刻意的。
    """
    clean = {k: v for k, v in sorted((config or {}).items())
             if k not in EXCLUDED_KEYS and not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 86400
    except Exception:
        return None


def evaluate(
    config: dict[str, Any],
    *,
    strategy: str | None = None,
    registry_path: str | os.PathLike[str] = "artifacts/experiments.sqlite",
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    mode: str | None = None,
) -> GateResult:
    """查 registry 判定這組配置的驗收狀態。**永不 raise**——查不到就回 missing。

    mode 未指定時讀環境變數 TWSTK_PUBLISH_GATE（observe|enforce），預設 observe。
    """
    mode = (mode or os.environ.get("TWSTK_PUBLISH_GATE") or "observe").lower()
    if mode not in ("observe", "enforce"):
        mode = "observe"

    fp = config_fingerprint(config)
    res = GateResult(status="missing", fingerprint=fp, strategy=strategy,
                     max_age_days=max_age_days, mode=mode)

    db = Path(registry_path)
    if not db.exists():
        res.notes.append(f"registry 不存在：{db}")
        return res

    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(experiments)")}
            if "config_fingerprint" not in cols:
                res.notes.append("registry schema 尚無 config_fingerprint 欄位"
                                 "（舊版 DB；跑一次搜尋腳本即會升級）")
                return res
            row = conn.execute(
                """
                SELECT experiment_id, created_at, pbo, deflated_sharpe, decision
                FROM experiments
                WHERE config_fingerprint = ?
                  AND decision IN ('accept', 'accepted', 'pass', 'promote')
                ORDER BY created_at DESC LIMIT 1
                """,
                (fp,),
            ).fetchone()
    except Exception as e:                       # registry 壞掉不該擋住發布
        res.notes.append(f"讀取 registry 失敗：{e}")
        return res

    if row is None:
        return res

    res.experiment_id = row["experiment_id"]
    res.validated_at = row["created_at"]
    res.pbo = row["pbo"]
    res.deflated_sharpe = row["deflated_sharpe"]
    res.age_days = _age_days(row["created_at"])
    res.status = ("expired" if (res.age_days is not None
                                and res.age_days > max_age_days) else "valid")
    return res


__all__ = ["GateResult", "config_fingerprint", "evaluate",
           "DEFAULT_MAX_AGE_DAYS", "EXCLUDED_KEYS"]
