"""
twstk.data.contract — 每日資料契約 (Data Contract) 與共享快照 (Snapshot)

這一層解決三個同源漏洞：
  1. 交易日行事曆 fail-open  → 這裡提供「最近 session」計算，供 preflight 做 fail-closed。
  2. 四策略各自重抓、輸入不一致 → freeze_snapshot / load_snapshot 讓四策略共讀同一份凍結資料。
  3. 當日 bar 無驗證 → validate_panel 檢查最後一根 bar 的新鮮度 / 完整度 / 跳變。

設計原則：
  - 純資料層，只依賴 pandas / numpy（exchange_calendars 為選用，缺席時退化為近似工作日）。
  - **fail-closed**：契約不通過就讓呼叫端 (preflight) 以非零 exit code 中止，不發布任何新報表。
  - 快照用 pickle（與既有 *.pkl cache 一致，免 pyarrow 依賴）。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# 五個 OHLCV 欄位固定順序；快照與 fetch_panel_data 回傳一致。
FIELDS = ("Close", "Open", "High", "Low", "Volume")

# 台股單日漲跌幅上限 ±10%。單日 |報酬| 超過此門檻幾乎必為資料錯誤
# （未除權還原、拆併股、vendor 髒資料），故設一個寬鬆但能抓到明顯錯誤的門檻。
DEFAULT_MAX_DAILY_MOVE = 0.40


# ─────────────────────────── 交易日行事曆 ───────────────────────────

def _load_calendar():
    """回傳 XTAI 行事曆物件；未安裝 exchange_calendars 時回傳 None。"""
    try:
        import exchange_calendars as xcals
        return xcals.get_calendar("XTAI")
    except Exception:
        return None


def most_recent_session(as_of, calendar="auto") -> pd.Timestamp:
    """
    回傳「as_of（含）之前最近的一個台股交易 session」的日期（normalize 到 00:00）。

    calendar="auto" 會嘗試載入 XTAI；載入失敗時退化為「最近的工作日（一~五）」近似。
    近似僅用於環境缺套件的降級路徑，正式 CI 應安裝 exchange_calendars。
    """
    as_of = pd.Timestamp(as_of).normalize()
    cal = _load_calendar() if calendar == "auto" else calendar

    if cal is not None:
        # exchange_calendars: 找 <= as_of 的最後一個 session
        try:
            # is_session 對非 session 會是 False；往前回溯最多 10 天即可涵蓋長假
            probe = as_of
            for _ in range(15):
                if cal.is_session(probe):
                    return probe.normalize()
                probe = probe - timedelta(days=1)
        except Exception:
            pass  # 落到下方近似

    # 近似：往前找最近的工作日（無法排除國定假日，僅供降級）
    probe = as_of
    for _ in range(10):
        if probe.weekday() < 5:  # 一~五
            return probe.normalize()
        probe = probe - timedelta(days=1)
    return as_of


def is_trading_day(as_of, calendar="auto") -> bool | None:
    """
    as_of 是否為台股交易日。
    回傳 True/False；當「行事曆無法判定」時回傳 None（呼叫端據此決定 fail-closed）。
    """
    as_of = pd.Timestamp(as_of).normalize()
    cal = _load_calendar() if calendar == "auto" else calendar
    if cal is None:
        return None  # 無法判定 → 交由呼叫端 fail-closed，不擅自假設是交易日
    try:
        return bool(cal.is_session(as_of))
    except Exception:
        return None


# ─────────────────────────── 契約驗證 ───────────────────────────

@dataclass
class ContractResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = "✅ 資料契約通過" if self.ok else "❌ 資料契約未通過（fail-closed，停止發布）"
        lines = [head]
        for k, v in self.stats.items():
            lines.append(f"   · {k}: {v}")
        for r in self.reasons:
            lines.append(f"   ⚠️ {r}")
        return "\n".join(lines)


def _last_bar_finite_fraction(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    last = df.iloc[-1]
    return float(last.notna().mean())


def validate_panel(
    panel,
    as_of,
    *,
    scheduled: bool = True,
    calendar="auto",
    min_completeness: float = 0.90,
    min_column_coverage: float = 0.50,
    key_tickers=("0050",),
    max_daily_move: float = DEFAULT_MAX_DAILY_MOVE,
    require_volume: bool = True,
) -> ContractResult:
    """
    驗證一份 OHLCV 面板是否可用於「產生今日新訊號」。

    Parameters
    ----------
    panel : dict[str, pd.DataFrame] 或 5-tuple (close, open, high, low, vol)
        欄位皆為 (日期 x 代號)。
    as_of : 觸發當下的台灣日期（date/Timestamp）。
    scheduled : True=排程(嚴格，最後 bar 必須==預期 session)；False=手動 dispatch(寬鬆)。
    min_completeness : 最後一根 bar 上「有有效 Close 的代號比例」下限（橫斷面）。
    min_column_coverage : 每一檔在**整段期間**的 Close 覆蓋率下限（縱向）。
        擋的是「整檔股票靜默消失」——橫斷面完整度對此幾乎無感。
    key_tickers : 必須存在且最後一根 bar 有效的關鍵標的（預設 0050，regime 依賴它）。
    max_daily_move : 最後一根 bar 上單檔 |日報酬| 超過此值即視為資料異常。
    require_volume : 最後一根 bar 是否要求成交量欄位有效且 > 0。

    Returns
    -------
    ContractResult(ok, reasons, stats)
    """
    panel = _as_dict(panel)
    close = panel.get("Close")
    reasons: list[str] = []
    stats: dict = {}

    if close is None or close.empty:
        return ContractResult(False, ["Close 面板為空，無任何資料"], stats)

    close = close.sort_index()
    last_date = pd.Timestamp(close.index[-1]).normalize()
    expected = most_recent_session(as_of, calendar=calendar)
    stats["最後 bar 日期"] = last_date.strftime("%Y-%m-%d")
    stats["預期 session"] = expected.strftime("%Y-%m-%d")

    # 1) 新鮮度：排程模式最後一根 bar 必須就是預期 session（收盤後跑，應已有當日 bar）。
    #    dispatch（人工補跑）放寬：允許最後 bar <= as_of 且不早於預期 session 太多。
    if scheduled:
        if last_date != expected:
            reasons.append(
                f"最後 bar {last_date.date()} ≠ 預期 session {expected.date()}"
                f"（資料過期/未更新，不得冒充今日訊號）"
            )
    else:
        if last_date > pd.Timestamp(as_of).normalize():
            reasons.append(f"最後 bar {last_date.date()} 晚於 as_of {pd.Timestamp(as_of).date()}（未來資料？）")

    # 2) 完整度：最後一根 bar 上有效 Close 的比例。
    comp = _last_bar_finite_fraction(close)
    stats["最後 bar 完整度(Close)"] = f"{comp:.1%} ({int(close.iloc[-1].notna().sum())}/{close.shape[1]})"
    if comp < min_completeness:
        reasons.append(f"最後 bar 完整度 {comp:.1%} < 門檻 {min_completeness:.0%}")

    # 3) 關鍵標的必須就位（0050 撐 regime；缺它整套 regime sizing 失真）。
    for kt in key_tickers:
        col_ok = (kt in close.columns) and bool(np.isfinite(close.iloc[-1].get(kt, np.nan)))
        if not col_ok:
            reasons.append(f"關鍵標的 {kt} 在最後 bar 無有效值")

    # 4) OHLCV 完整性：最後一根 bar 的 Open/High/Low(/Volume) 也要在場。
    for f_ in ("Open", "High", "Low"):
        df = panel.get(f_)
        frac = _last_bar_finite_fraction(df)
        stats[f"最後 bar 完整度({f_})"] = f"{frac:.1%}"
        if frac < min_completeness:
            reasons.append(f"最後 bar {f_} 完整度 {frac:.1%} < 門檻 {min_completeness:.0%}")

    if require_volume:
        vol = panel.get("Volume")
        if vol is None or vol.empty:
            reasons.append("Volume 面板缺失")
        else:
            last_vol = vol.iloc[-1]
            pos_frac = float((last_vol > 0).mean())
            stats["最後 bar 有量比例"] = f"{pos_frac:.1%}"
            if pos_frac < min_completeness:
                reasons.append(f"最後 bar 有量比例 {pos_frac:.1%} < 門檻 {min_completeness:.0%}（疑似半根 bar）")

    # 5) 跳變偵測：最後一根 bar 的單日報酬異常大者列出（多半是未除權/髒資料）。
    if len(close) >= 2:
        prev = close.iloc[-2]
        curr = close.iloc[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (curr / prev - 1.0).abs()
        bad = ret[ret > max_daily_move].dropna()
        stats["異常跳變檔數"] = int(bad.shape[0])
        if not bad.empty:
            names = ", ".join(f"{t}={ret[t]:.0%}" for t in bad.index[:8])
            reasons.append(f"最後 bar 有 {bad.shape[0]} 檔單日 |報酬|>{max_daily_move:.0%}：{names}")

    # 6) 欄位級有效性（★橫斷面完整度看不到的盲區）。
    #    上面 1~5 全部只檢查「最後一根 bar」。一檔股票整段歷史都是 NaN，
    #    在最後 bar 完整度上只扣 1/N——實際踩過的坑：4 檔上櫃股因市場後綴
    #    剝除錯誤而整欄全空，116/120 = 96.7% 仍高於 90% 門檻，契約照樣放行。
    #    這裡改成逐欄檢查整段覆蓋率，讓「整檔靜默消失」必定 fail-closed。
    n_rows = len(close)
    if n_rows > 0:
        cover = close.notna().sum(axis=0) / n_rows
        dead = sorted(cover[cover <= 0.0].index.tolist())
        thin = sorted(cover[(cover > 0.0) & (cover < min_column_coverage)].index.tolist())
        stats["整欄全空檔數"] = len(dead)
        stats["覆蓋率不足檔數"] = f"{len(thin)}（門檻 {min_column_coverage:.0%}）"
        if dead:
            reasons.append(
                f"{len(dead)} 檔整段無任何有效 Close：{dead[:10]}"
                f"（常見成因：市場後綴剝除錯誤導致代號對不上）"
            )
        if thin:
            names = ", ".join(f"{t}={cover[t]:.0%}" for t in thin[:8])
            reasons.append(
                f"{len(thin)} 檔整段覆蓋率 < {min_column_coverage:.0%}：{names}"
            )

    return ContractResult(ok=(len(reasons) == 0), reasons=reasons, stats=stats)


# ─────────────────────────── 快照 (snapshot) ───────────────────────────

def _as_dict(panel) -> dict:
    """把 5-tuple 或 dict 統一成 {'Close':df,...}。"""
    if isinstance(panel, dict):
        return panel
    if isinstance(panel, (tuple, list)) and len(panel) == 5:
        return dict(zip(FIELDS, panel))
    raise TypeError("panel 需為 dict 或 (close, open, high, low, vol) 5-tuple")


def _panel_hash(panel_dict: dict) -> str:
    """對面板內容做穩定 hash，用於可重現性稽核。"""
    h = hashlib.sha256()
    for f_ in FIELDS:
        df = panel_dict.get(f_)
        if df is None or df.empty:
            h.update(f"{f_}:EMPTY".encode())
            continue
        df = df.sort_index()
        df = df.reindex(sorted(df.columns), axis=1)
        # hash_pandas_object(DataFrame) 只涵蓋 index + cell values，不涵蓋
        # column labels；曾可把 2330 欄改名 FAKE 而 panel_sha256 完全不變。
        # 欄名與 dtype 都是資料契約的一部分，必須顯式納入。
        column_schema = [
            {"label": str(col), "dtype": str(dtype)}
            for col, dtype in zip(df.columns, df.dtypes)
        ]
        col_hash = pd.util.hash_pandas_object(df, index=True).values
        h.update(f_.encode())
        h.update(json.dumps(
            column_schema, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8"))
        h.update(np.ascontiguousarray(col_hash).tobytes())
    return h.hexdigest()


def build_manifest(panel, as_of, *, provider: str, auto_adjust: bool,
                   contract: ContractResult | None = None) -> dict:
    """產出快照 manifest（provider / 還原方式 / hash / schema 版本 / 契約摘要）。"""
    panel_dict = _as_dict(panel)
    close = panel_dict.get("Close")
    last_date = pd.Timestamp(close.sort_index().index[-1]).strftime("%Y-%m-%d") if close is not None and not close.empty else None
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "last_session": last_date,
        "provider": provider,
        "auto_adjust": bool(auto_adjust),
        "n_tickers": int(close.shape[1]) if close is not None else 0,
        "n_sessions": int(close.shape[0]) if close is not None else 0,
        "panel_sha256": _panel_hash(panel_dict),
        "contract_ok": None if contract is None else bool(contract.ok),
        "contract_reasons": [] if contract is None else list(contract.reasons),
    }


AUX_SERIES_FILE = "aux_series.csv"


def _aux_file_hash(path: str) -> str:
    """對 aux CSV 的**檔案內容**取 hash。

    ★刻意雜湊檔案而非記憶體物件：CSV 往返會改變浮點的字串表示，
    對記憶體物件取 hash 會在「剛寫完就讀」時就對不上。
    要保護的是「載入時拿到的東西是否與凍結時相同」，那就直接對檔案取 hash。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_snapshot(panel, out_dir, manifest: dict, aux_series: dict | None = None) -> str:
    """
    凍結一份快照到 out_dir/：panel.pkl + manifest.json（+ aux_series.csv）。
    回傳 out_dir 路徑。四策略之後以 load_snapshot(out_dir) 共讀同一份資料。

    aux_series（2026-07-25 稽核 B4）
    -------------------------------
    panel 之外、但**同樣會改變回測結果**的外部序列，最重要的是 **VIX**。
    稽核查證：即使設了 TWSTK_SNAPSHOT，引擎仍會在 run() 內自行下載 VIX，
    ai_report 也會另外網路重抓 0050——兩者都繞過快照。實測 VIX 用凍結
    vs 即時，v8.5 年化差 5pp。

    ⇒ 只凍結 panel 不足以宣稱「當日 run 可重現」。這些序列必須一起凍結、
      一起 hash，出處才完整。
    """
    os.makedirs(out_dir, exist_ok=True)
    panel_dict = _as_dict(panel)
    actual_hash = _panel_hash(panel_dict)
    expected_hash = (manifest or {}).get("panel_sha256")
    if not expected_hash or expected_hash != actual_hash:
        raise SnapshotIntegrityError(
            "拒絕寫入 manifest 與 panel 不一致的快照："
            f"expected={expected_hash}, actual={actual_hash}"
        )
    with open(os.path.join(out_dir, "panel.pkl"), "wb") as f:
        pickle.dump({k: panel_dict.get(k) for k in FIELDS}, f, protocol=pickle.HIGHEST_PROTOCOL)

    if aux_series:
        df = pd.DataFrame({k: pd.Series(v) for k, v in aux_series.items()}).sort_index()
        df.index.name = "date"
        aux_path = os.path.join(out_dir, AUX_SERIES_FILE)
        # 固定換行符：否則同一份資料在 Windows/Linux 寫出的檔案 hash 不同
        df.to_csv(aux_path, encoding="utf-8", lineterminator="\n")
        manifest = {**(manifest or {}),
                    "aux_series": sorted(aux_series),
                    "aux_sha256": _aux_file_hash(aux_path)}

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out_dir


def load_aux_series(path, name=None, *, verify_manifest=True):
    """從快照讀外部序列（如 VIX）。path 可為快照目錄或其中的 panel.pkl。

    找不到就回 None（舊快照沒有這個檔），呼叫端據此退回即時下載並**出聲**。
    """
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    f = os.path.join(d, AUX_SERIES_FILE)
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    if verify_manifest:
        mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as mf:
                    expected = json.load(mf).get("aux_sha256")
            except Exception:
                expected = None
            if expected:
                actual = _aux_file_hash(f)
                if actual != expected:
                    raise SnapshotIntegrityError(
                        f"aux_series 與 manifest 不一致：expected={expected}, "
                        f"actual={actual}。拒絕以可能被竄改的資料回測。")
    if name is None:
        return df
    return df[name].dropna() if name in df.columns else None


class SnapshotTickerError(RuntimeError):
    """快照缺少所要求的 ticker，或該 ticker 整段無有效資料。"""


class SnapshotIntegrityError(RuntimeError):
    """panel.pkl 的實際內容與 manifest.json 所宣告的 hash 不一致。"""


def load_snapshot(path, tickers=None, start=None, end=None, strict=True,
                  verify_manifest=True):
    """
    載入凍結快照，回傳與 fetch_panel_data 相同形狀的 5-tuple
    (close, open, high, low, vol)。可選擇 subset 到指定 tickers / 日期區間。

    path 可為 snapshot 目錄或直接的 panel.pkl。

    strict=True（預設）時**fail-loud**：所要求的 ticker 若不存在於快照，
    或存在但 Close 整段全 NaN，直接拋 SnapshotTickerError。

    verify_manifest=True（預設）且同目錄有 manifest.json 時，會重算完整
    panel hash（含欄名）並核對。manifest 不存在時為相容舊快照而略過；
    存在但缺 hash／不相符則 fail-closed。

    ★這裡曾經是 `keep = [t for t in tickers if t in out.columns]`，
    缺股會被**靜默丟棄**。實際踩過的坑：ai_strategy 的市場後綴剝除把
    '5274.TWO' 寫成 '5274O'，於是快照同時有全空的 '5274' 與有資料的 '5274O'，
    這裡照樣「成功」載入，四檔上櫃股無聲消失，v8.5 年化憑空少 8.7pp
    而沒有任何警告。缺資料必須炸，不能默默算下去。
    """
    if os.path.isdir(path):
        pkl = os.path.join(path, "panel.pkl")
    else:
        pkl = path
    with open(pkl, "rb") as f:
        panel_dict = pickle.load(f)

    if verify_manifest:
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(pkl)), "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    expected_hash = json.load(f).get("panel_sha256")
            except Exception as exc:
                raise SnapshotIntegrityError(
                    f"無法讀取快照 manifest：{manifest_path}: {exc}"
                ) from exc
            actual_hash = _panel_hash(_as_dict(panel_dict))
            if not expected_hash:
                raise SnapshotIntegrityError(
                    f"快照 manifest 缺少 panel_sha256：{manifest_path}"
                )
            if actual_hash != expected_hash:
                raise SnapshotIntegrityError(
                    "快照內容與 manifest 不一致："
                    f"expected={expected_hash}, actual={actual_hash}。"
                    "拒絕以可能被竄改或配錯的資料回測。"
                )

    if tickers is not None and strict:
        close = panel_dict.get("Close")
        if close is None or close.empty:
            raise SnapshotTickerError(f"快照 {pkl} 沒有 Close 面板")
        missing = [t for t in tickers if t not in close.columns]
        empty = [t for t in tickers
                 if t in close.columns and close[t].notna().sum() == 0]
        if missing or empty:
            parts = []
            if missing:
                parts.append(f"快照缺少 {len(missing)} 檔：{missing[:10]}")
            if empty:
                parts.append(f"{len(empty)} 檔整段無有效 Close：{empty[:10]}")
            near = [c for c in close.columns
                    if any(str(c).startswith(str(t)) for t in (missing + empty))]
            if near:
                parts.append(f"（快照內疑似對應欄位：{near[:10]} — 檢查市場後綴剝除）")
            raise SnapshotTickerError(
                "; ".join(parts)
                + "。fail-loud：拒絕以殘缺股池繼續回測。"
                  "確認為刻意排除時，才傳 strict=False。"
            )

    def _slice(df):
        if df is None or df.empty:
            return df
        out = df.sort_index()
        if start is not None:
            out = out.loc[pd.Timestamp(start):]
        if end is not None:
            out = out.loc[:pd.Timestamp(end)]
        if tickers is not None:
            keep = [t for t in tickers if t in out.columns]
            out = out[keep]
        return out

    return tuple(_slice(panel_dict.get(f_)) for f_ in FIELDS)
