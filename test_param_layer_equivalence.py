"""回歸測試：兩層參數組裝必須產生同一個引擎（2026-07-19 稽核 E 項）。

同一個策略有兩條組裝路徑：

    L3A  ai_report.py 的 argparse（97 個旗標）   → 發布報表走這條
    L3B  strategies/*_PARAMS + registry          → paper 頁與 ablation 走這條

兩層各有各的預設值，靠人記得對齊。已經漂過的實例：
    consec_loss_limit  引擎預設 3 / CLI 預設 99（README 列為 gotcha，靠人補旗標）
    corr_select_cap    CLI 2 / 插件 fallback 1
    regime_floor       CLI 0.1 / 插件 fallback 0.3

後兩者今天是死參數（corr_select_max=0、regime_graduated=False 時不生效），
所以不會表現成數字差異——但只要有人打開對應開關，同一個策略就會在兩條
路徑上分裂，而且沒有任何東西會叫。這個測試就是那個會叫的東西。

比對方式刻意用 validation.publish_gate.engine_fingerprint：
它取的是**引擎建構完成後的參數狀態**，與哪一層組裝無關；指紋相同 ⇔
兩條路徑會跑出同一個回測。這也正是發布閘門用來查驗收紀錄的鍵，
所以這個測試同時保證「ablation 驗收過的東西，發布端查得到」。

★CI 旗標一律從 .github/workflows/update_ai_report.yml 解析，不在測試裡
  抄一份——抄了就是第三份真相，改 workflow 忘了改插件時測試不會叫。
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pandas as pd
import pytest

import ai_report
import strategy.event_backtest as EB
from strategies.base import ExecConfig, MarketData
from strategies.registry import get_strategy
from validation.publish_gate import engine_fingerprint

WORKFLOW = Path(".github/workflows/update_ai_report.yml")

# workflow 的 --artifact-label → registry 註冊名
LABEL_TO_REGISTRY = {
    "v85": "momentum_v85",
    "guard": "mom_guard",
    "surge": "mom_surge",
    "surge_pro": "mom_surge_pro",
}

# 這些旗標描述「跑哪一段資料 / 多少錢」，不是策略定義，比對時剔除
CONTEXT_FLAGS = {"--start-date", "--eval-start", "--end-date", "--days",
                 "--capital", "--artifact-label"}


def parse_ci_flag_sets() -> dict[str, str]:
    """從 workflow 抽出四個 ai_report 指令的旗標（label → flags）。"""
    text = WORKFLOW.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"python ai_report\.py \\\n(.*?)(?=\n\s*(?:sed|#|-|cp|mv)\s)",
                         text, re.S):
        block = m.group(1)
        # 攤平續行、移除 GitHub Actions 表達式（保留其預設值）
        flat = block.replace("\\\n", " ")
        flat = re.sub(r"\$\{\{[^}]*?\|\|\s*'([^']*)'\s*\}\}", r"\1", flat)
        flat = re.sub(r"\$\{\{.*?\}\}", "", flat)
        flat = " ".join(flat.split())
        lab = re.search(r"--artifact-label\s+(\w+)", flat)
        if lab:
            out[lab.group(1)] = flat
    return out


def strip_context(flags: str) -> str:
    parts, keep, skip = shlex.split(flags), [], 0
    for i, p in enumerate(parts):
        if skip:
            skip -= 1
            continue
        if p in CONTEXT_FLAGS:
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            if nxt and not nxt.startswith("--"):
                skip = 1
            continue
        keep.append(p)
    return " ".join(shlex.quote(x) if (" " in x or ";" in x) else x for x in keep)


def cli_engine(flags: str, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["ai_report.py"] + shlex.split(f"{strip_context(flags)} --capital 200000"))
    return ai_report.build_backtester_from_args(ai_report.parse_args())


def _tiny_market_data(n: int = 70):
    idx = pd.bdate_range("2025-01-01", periods=n)
    cols = ["2330", "2317"]

    def frame(value):
        return pd.DataFrame(float(value), index=idx, columns=cols)

    return MarketData(
        close=frame(100), open=frame(100), high=frame(101), low=frame(99),
        volume=frame(1_000_000), market_close=pd.Series(100.0, index=idx),
        universe_mask=pd.DataFrame(True, index=idx, columns=cols),
    )


def plugin_engine(registry_name: str, monkeypatch, slippage: float):
    """跑真實 run_engine 並攔截引擎建構。

    刻意**不複製**任何一邊的建構程式碼——複製就是再造一份真相，
    而且會漏參數（稽核當下手抄 68 個 kwargs 漏了 27 個，憑空生出 7 個假差異）。
    """
    import sys as _sys

    from strategies.registry import list_strategies

    # ★必須先觸發插件的延遲匯入，否則下面掃不到持有引擎類別的模組
    #   （registry 在第一次查詢時才 import strategies/*）。
    list_strategies()

    captured: dict[str, object] = {}
    original = EB.EventDrivenBacktester

    class Spy(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.setdefault("engine", self)

    targets = [m for m in list(_sys.modules.values())
               if getattr(m, "EventDrivenBacktester", None) is original]
    for mod in targets:
        monkeypatch.setattr(mod, "EventDrivenBacktester", Spy, raising=False)

    cfg = ExecConfig(initial_capital=200_000, slippage=slippage,
                     top_k=7, threshold=2.0)
    try:
        get_strategy(registry_name).run_engine(_tiny_market_data(), cfg)
    except Exception:
        pass                      # 只要建構完成即可；合成資料跑不出交易無妨
    return captured.get("engine")


def test_workflow_exposes_all_four_strategies():
    flags = parse_ci_flag_sets()
    assert set(flags) == set(LABEL_TO_REGISTRY), (
        f"從 workflow 解析到的策略 {sorted(flags)} 與預期 "
        f"{sorted(LABEL_TO_REGISTRY)} 不符——workflow 可能改了，測試需同步")


@pytest.mark.parametrize("label", sorted(LABEL_TO_REGISTRY))
def test_cli_and_plugin_build_identical_engines(label, monkeypatch):
    flags = parse_ci_flag_sets()[label]
    slip = float(re.search(r"--slippage\s+([\d.]+)", flags).group(1))

    plugin = plugin_engine(LABEL_TO_REGISTRY[label], monkeypatch, slip)
    assert plugin is not None, f"{label}: 未攔截到插件路徑的引擎建構"
    cli = cli_engine(flags, monkeypatch)

    pf, cf = engine_fingerprint(plugin), engine_fingerprint(cli)
    if pf != cf:
        diffs = []
        for key in sorted(vars(cli)):
            if key.startswith("_"):
                continue
            a, b = getattr(cli, key, None), getattr(plugin, key, None)
            try:
                same = bool(a == b)
            except Exception:
                same = repr(a) == repr(b)
            if not same:
                diffs.append(f"{key}: CLI={a!r} plugin={b!r}")
        pytest.fail(f"{label} 兩層組裝出不同引擎：\n  " + "\n  ".join(diffs))


def test_fingerprint_ignores_execution_context():
    """資金規模不該改變指紋（分數股下報酬對規模不變）。"""
    a = EB.EventDrivenBacktester(initial_capital=200_000)
    b = EB.EventDrivenBacktester(initial_capital=9_000_000)
    assert engine_fingerprint(a) == engine_fingerprint(b)


def test_fingerprint_reacts_to_strategy_change():
    base = EB.EventDrivenBacktester()
    for kw in ({"sl_atr_mult": 9.9}, {"max_hold_days": 3},
               {"regime_sizing": True}, {"slippage": 0.05}):
        assert engine_fingerprint(EB.EventDrivenBacktester(**kw)) != \
            engine_fingerprint(base), f"{kw} 未改變指紋"


def test_fingerprint_is_type_stable():
    """0 與 0.0 必須視為同一個值。

    兩層組裝會讓同一旋鈕一邊 int、一邊 float（實例：corr_filter），
    值相同卻算出不同指紋 → 假的「配置已變更」。
    """
    assert engine_fingerprint(EB.EventDrivenBacktester(corr_filter=0)) == \
        engine_fingerprint(EB.EventDrivenBacktester(corr_filter=0.0))


def test_fingerprint_survives_a_run(monkeypatch):
    """跑過的引擎與剛建好的引擎，指紋必須相同。

    run() 會在引擎上寫 last_cash / last_positions 等**執行結果**；
    它們沒有底線前綴，不排除的話會污染指紋。
    """
    engine = EB.EventDrivenBacktester()
    before = engine_fingerprint(engine)
    engine.last_cash = 123456.0
    engine.last_positions = {"2330": {"shares": 1.0}}
    assert engine_fingerprint(engine) == before
