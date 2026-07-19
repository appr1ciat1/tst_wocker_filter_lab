"""
twstk.data.symbols — 代號 ↔ yfinance symbol ↔ 市場別 的**唯一**轉換點

為什麼要有這個模組（2026-07-19 稽核）：
  · 市場後綴曾被 `str.replace('.TW','').replace('.TWO','')` 剝壞，
    '5274.TWO' 變成 '5274O'，共享快照因此整整少掉四檔上櫃股，
    v8.5 年化憑空少 8.7pp 而沒有任何警告。
  · 「這檔是上市還是上櫃」原本有三個來源（字串後綴、法人快照 API、
    security_master.json），沒有一個是權威的。build_paper_page 用的是
    法人 API，而它包在 `except → {}` 裡，一失敗就全部退化成 .TW，
    上櫃股於是產生死連結——又是一次「靜默降級成看起來合理的錯誤答案」。
  · 「先試 .TW，失敗再試 .TWO」的探測邏輯在 5 個檔案各寫一次。

原則：
  1. **market 只有一個真相＝security_master.json**（3118 檔、已版控、離線）。
  2. **判不出市場時回 None，不要預設 .TW**——寧可不給連結，也不要說謊。
  3. 下載端的 .TW→.TWO 探測是「探測」不是「猜測」（抓不到才 fallback），
     本身安全，這裡只是把它收斂成一份。
"""

from __future__ import annotations

import re

# .TWO 必須排在 .TW 前面，且用 $ 錨定：
# 'x.replace(".TW","")' 會先吃掉 '.TWO' 的前綴，這正是 5274O 的成因。
_MARKET_SUFFIX_RE = re.compile(r"\.TWO$|\.TW$", re.IGNORECASE)

# security_master 的 market 值 → yfinance 後綴
_TPEX_TOKENS = {"tpex", "otc", "two", "櫃買", "上櫃"}
_TWSE_TOKENS = {"twse", "tse", "tw", "上市"}

# security master 允許多舊仍直接採用（顯示層 fail-soft，不為了它打網路）
_MAX_AGE_DAYS = 3650


def strip_market_suffix(symbol) -> str:
    """'2330.TW'→'2330'、'5274.TWO'→'5274'；無後綴則原樣回傳。"""
    return _MARKET_SUFFIX_RE.sub("", str(symbol))


def market_of(ticker) -> str | None:
    """代號 → 'twse' / 'tpex' / None（查不到就 None，**不猜**）。

    來源固定為 security_master.json；抓取失敗時 load_master 會 fail-soft
    沿用既有快取，再失敗才回 None。
    """
    code = strip_market_suffix(ticker)
    try:
        from twstk.data import security_master as sm
        rec = sm.describe(code, max_age_days=_MAX_AGE_DAYS)
    except Exception:
        return None
    raw = str((rec or {}).get("market") or "").strip().lower()
    if not raw:
        return None
    if raw in _TPEX_TOKENS:
        return "tpex"
    if raw in _TWSE_TOKENS:
        return "twse"
    return None


def yahoo_symbol(ticker) -> str | None:
    """代號 → yfinance / Yahoo Finance symbol；**市場未知時回 None**。

    呼叫端請據此決定「不給連結」，不要自行退回 '.TW'。
    若傳入的字串本身已帶後綴，直接尊重它（呼叫端已明示市場）。
    """
    raw = str(ticker)
    if _MARKET_SUFFIX_RE.search(raw):
        return raw
    mkt = market_of(raw)
    if mkt == "tpex":
        return f"{strip_market_suffix(raw)}.TWO"
    if mkt == "twse":
        return f"{strip_market_suffix(raw)}.TW"
    return None


def candidate_symbols(ticker) -> list[str]:
    """下載探測用：已知市場就只回那一個，未知才回 ['.TW', '.TWO'] 兩個都試。"""
    code = strip_market_suffix(ticker)
    known = yahoo_symbol(code)
    if known:
        return [known]
    return [f"{code}.TW", f"{code}.TWO"]


def probe_tw_then_two(tickers, fetch, *, warn_label: str | None = None) -> dict:
    """先試 .TW，缺的再試 .TWO，回傳 {ticker: value}。

    收斂原本散在 paper_tracker(×2) / paper_trade / s1_shadow_diff 的重複實作。

    Parameters
    ----------
    tickers : 純代號序列（不帶後綴）
    fetch : callable，接受 {ticker: yfinance_symbol}，回傳 {ticker: value}。
            抓不到的 ticker 就不要放進回傳 dict。
    warn_label : 給定時，兩輪都抓不到會印一行警告（fail-soft 但出聲）。

    這是**探測**不是猜測：以實際抓不抓得到為準，最後仍抓不到會出聲。
    """
    tickers = [strip_market_suffix(t) for t in tickers]
    out: dict = {}

    tw_map = {t: f"{t}.TW" for t in tickers}
    if tw_map:
        out.update(fetch(tw_map) or {})

    missing = [t for t in tickers if t not in out]
    if missing:
        two_map = {t: f"{t}.TWO" for t in missing}
        out.update(fetch(two_map) or {})

    still = [t for t in tickers if t not in out]
    if still and warn_label:
        print(f"   ⚠️ {warn_label}: {', '.join(still)}")
    return out


__all__ = [
    "strip_market_suffix",
    "market_of",
    "yahoo_symbol",
    "candidate_symbols",
    "probe_tw_then_two",
]
