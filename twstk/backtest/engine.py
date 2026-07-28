"""
twstk.backtest.engine —【套件 2】歷史回測引擎（與策略解耦）

依策略型態自動分派執行方式：
  - WeightStrategy → 共用權重成交核心（twstk.portfolio）
  - EngineStrategy → 策略自帶的事件引擎（忠實重現 v8.5 / SR v2 / v9）

資料層只抓策略真正需要的東西（看 strategy.requires，例如 us_signals）。
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from twstk.data.prices import PricePanel

from strategies.base import (
    Strategy, WeightStrategy, EngineStrategy, MarketData, ExecConfig,
)

from twstk.data import (
    fetch_prices, liquid_universe, fetch_benchmark,
    fetch_us_signals, align_us_to_tw, build_inst_flow_df, fetch_sbl_balances,
    fetch_global_context, fetch_chip_indicators,
)
from twstk.portfolio import (
    PortfolioConfig, simulate_weights, equity_dataframe, trades_dataframe,
)
from twstk.backtest.metrics import compute_risk_metrics


# 預設台股池（可由 CLI 覆寫）。
# 與既有 v8.5 / SR v2 / v9 回測一致，預設使用 ai_report 的 116 檔 EXTENDED_TICKERS，
# 動態流動性池再從中取 Top-N；如此 twstk 的回測數字才能對齊 compare_v85_v9 等既有腳本。
try:
    from ai_report import EXTENDED_TICKERS as DEFAULT_TICKERS
except Exception:  # noqa: BLE001 — 後備靜態小池
    DEFAULT_TICKERS = [
        "2330", "2317", "2454", "2308", "2382", "2412", "2881", "2882",
        "2891", "3008", "2303", "1301", "1303", "2002",
    ]


@dataclass
class RunConfig:
    """回測執行設定（資料區間 + 資金/成本/部位）。"""
    tickers: Optional[list] = None
    days: int = 1200
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    eval_start: Optional[str] = None
    universe_size: int = 60
    initial_capital: float = 1_000_000
    buy_cost: float = 0.001425
    sell_cost: float = 0.004425
    slippage: float = 0.0
    top_k: int = 7
    threshold: float = 2.0
    max_weight: float = 1.0                # WeightStrategy 用
    rebalance_threshold: float = 0.0       # WeightStrategy 用
    benchmark_ticker: str = "0050"
    use_inst_flow: bool = False            # ★載入新版三大法人因子
    refresh_latest: bool = False           # 即時 paper 才更新融資融券／借券快取


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict
    weights: pd.DataFrame = field(default=None)
    benchmark_equity: Optional[pd.Series] = None
    strategy_name: str = ""


def _benchmark_from_snapshot(ticker, cfg):
    """從凍結面板取 benchmark（0050）收盤。快照沒有這檔就回 None。

    ★不可從已 subset 的面板取：ai_report 曾犯過這個錯——close_df 早被
      EXTENDED_TICKERS 濾過，0050 不在其中，修正因而完全沒生效。
      這裡直接對快照指名 tickers=[ticker]。
    """
    snap = os.environ.get("TWSTK_SNAPSHOT")
    if not snap or not os.path.exists(snap):
        return None
    try:
        from twstk.data.contract import load_snapshot
        close, *_ = load_snapshot(snap, tickers=[str(ticker)],
                                  start=cfg.start_date, end=cfg.end_date)
        if close is None or close.empty or str(ticker) not in close.columns:
            return None
        return close[str(ticker)].dropna()
    except Exception:
        return None                      # 快照裡沒有這檔是正常情況，呼叫端會下載


def _panel_from_snapshot(tickers, cfg):
    """設了 TWSTK_SNAPSHOT 就改讀凍結面板（2026-07-26 稽核）。

    為什麼：preflight 每天凍結一份共享快照，四份報表（走 ai_report）確實共讀它，
    但 paper 頁走的是本函式，原本一律重新下載 —— 於是同一次 CI 內，報表與 paper 頁
    可能拿到**不同的歷史**。而歷史真的會變：資料源回溯調整價格，實測同程式同參數
    同視窗、只差下載日期一天，v8.5 年化 27.5% → 18.79%。

    抓不到就回 None 讓呼叫端照舊下載，但**要出聲**——這裡靜默降級等於把
    「兩份資料」偽裝成「一份資料」，正是最難查的那種錯。
    """
    snap = os.environ.get("TWSTK_SNAPSHOT")
    if not snap or not os.path.exists(snap):
        return None
    try:
        from twstk.data.contract import load_snapshot
        close, open_, high, low, volume = load_snapshot(
            snap, tickers=list(tickers), start=cfg.start_date, end=cfg.end_date)
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️ 快照載入失敗，改為即時下載（本次結果將與報表不同源）: {e}")
        return None
    print(f"   🧊 使用凍結快照：{snap}（{close.shape[0]} 日 × {close.shape[1]} 檔）")
    return PricePanel(close=close, open=open_, high=high, low=low, volume=volume)


def build_market_data(cfg: RunConfig, strategy: Strategy) -> MarketData:
    """依設定 + 策略需求組出乾淨的 MarketData（純資料層）。"""
    tickers = cfg.tickers or DEFAULT_TICKERS
    panel = _panel_from_snapshot(tickers, cfg)
    if panel is None:
        panel = fetch_prices(
            tickers, days=cfg.days, start_date=cfg.start_date, end_date=cfg.end_date,
        )

    universe_mask = None
    if cfg.universe_size and cfg.universe_size > 0:
        universe_mask = liquid_universe(panel.close, panel.volume, top_n=cfg.universe_size)

    market_close = None
    try:
        # ★benchmark 也要優先走快照：0050 驅動 regime filter，一旦它與面板不同源，
        #   「同一份資料跑到底」就破功了。快照裡沒有才回退即時下載。
        bench = _benchmark_from_snapshot(cfg.benchmark_ticker, cfg)
        if bench is None:
            bench = fetch_benchmark(
                cfg.benchmark_ticker,
                start_date=cfg.start_date, end_date=cfg.end_date, days=cfg.days,
            )
        market_close = bench
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️ benchmark 抓取失敗，停用 regime market_close: {e}")

    requires = getattr(strategy, "requires", frozenset())

    us_signals = None
    if "us_signals" in requires:
        try:
            raw = fetch_us_signals(
                start_date=(cfg.start_date or panel.close.index[0].strftime("%Y-%m-%d")),
                end_date=(cfg.end_date or panel.close.index[-1].strftime("%Y-%m-%d")),
            )
            us_signals = align_us_to_tw(raw, panel.close.index)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 美股訊號抓取失敗: {e}")

    global_context = None
    if "global_context" in requires:
        try:
            global_context = fetch_global_context(
                list(panel.close.columns), panel.close.index,
                start_date=panel.close.index[0], end_date=panel.close.index[-1],
            )
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 隔夜美股／全球龍頭資料抓取失敗，採中性降級: {e}")

    inst_flow_df = inst_ratio_df = None
    if cfg.use_inst_flow or "inst_flow" in requires:
        try:
            inst_flow_df, inst_ratio_df = build_inst_flow_df(
                list(panel.close.columns), panel.close, verbose=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 三大法人資料抓取失敗，跳過: {e}")

    short_sale_df = None
    margin_balance_df = margin_short_df = None
    if "chip_indicators" in requires:
        try:
            chips = fetch_chip_indicators(
                list(panel.close.columns), panel.close.index,
                start_date=panel.close.index[0], end_date=panel.close.index[-1],
                refresh_latest=cfg.refresh_latest,
            )
            margin_balance_df = chips.margin_balance
            margin_short_df = chips.margin_short_balance
            short_sale_df = chips.sbl_balance
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 融資／融券／借券資料載入失敗，採中性降級: {e}")
    elif "short_sale" in requires:
        try:
            sd = panel.close.index[0].strftime("%Y-%m-%d")
            ed = panel.close.index[-1].strftime("%Y-%m-%d")
            sbl = fetch_sbl_balances(list(panel.close.columns), start_date=sd, end_date=ed)
            short_sale_df = sbl.reindex(index=panel.close.index, columns=panel.close.columns)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 借券(SBL)資料抓取失敗，跳過: {e}")

    return MarketData(
        close=panel.close, open=panel.open, high=panel.high,
        low=panel.low, volume=panel.volume,
        market_close=market_close, universe_mask=universe_mask,
        us_signals=us_signals, global_context=global_context,
        inst_flow_df=inst_flow_df, inst_ratio_df=inst_ratio_df,
        short_sale_df=short_sale_df,
        margin_balance_df=margin_balance_df, margin_short_df=margin_short_df,
    )


def run_backtest(strategy: Strategy, cfg: RunConfig) -> BacktestResult:
    """以指定策略 + 設定跑一次歷史回測（依策略型態自動分派）。"""
    print(f"🚀 回測策略：{strategy.name} — {getattr(strategy, 'description', '')}")
    data = build_market_data(cfg, strategy)

    capital_cap = getattr(strategy, "capital_cap", None)
    effective_capital = min(cfg.initial_capital, capital_cap) if capital_cap else cfg.initial_capital
    if capital_cap and cfg.initial_capital > capital_cap:
        print(f"   💰 此版本投資上限 {capital_cap:,.0f}，回測資金由 {cfg.initial_capital:,.0f} 限為 {effective_capital:,.0f}")

    weights = None
    if isinstance(strategy, EngineStrategy):
        # 自帶引擎（忠實 v8.5 / SR v2 / v9）
        exec_cfg = ExecConfig(
            initial_capital=effective_capital,
            buy_cost=cfg.buy_cost, sell_cost=cfg.sell_cost, slippage=cfg.slippage,
            top_k=cfg.top_k, threshold=cfg.threshold,
        )
        trades_df, equity_df = strategy.run_engine(data, exec_cfg)
    elif isinstance(strategy, WeightStrategy):
        # 共用權重成交核心
        weights = strategy.target_weights(data)
        pcfg = PortfolioConfig(
            initial_capital=effective_capital,
            buy_cost=cfg.buy_cost, sell_cost=cfg.sell_cost, slippage=cfg.slippage,
            max_weight=cfg.max_weight, rebalance_threshold=cfg.rebalance_threshold,
        )
        state = simulate_weights(
            weights, data.open, data.close, pcfg,
            start=cfg.eval_start or cfg.start_date, end=cfg.end_date,
        )
        equity_df = equity_dataframe(state)
        trades_df = trades_dataframe(state)
    else:
        raise TypeError(f"未知策略型態：{type(strategy)}（需為 WeightStrategy 或 EngineStrategy）")

    eval_equity = equity_df
    if cfg.eval_start and not equity_df.empty:
        eval_equity = equity_df[equity_df.index >= pd.Timestamp(cfg.eval_start)]
    metrics = compute_risk_metrics(eval_equity, trades_df, effective_capital)

    benchmark_equity = None
    try:
        bstart = cfg.start_date
        if bstart is None and not equity_df.empty:
            bstart = str(equity_df.index[0].date())
        benchmark_equity = fetch_benchmark(
            cfg.benchmark_ticker, start_date=bstart, end_date=cfg.end_date,
        )
    except Exception:  # noqa: BLE001
        pass

    return BacktestResult(
        trades=trades_df, equity=equity_df, metrics=metrics,
        weights=weights, benchmark_equity=benchmark_equity,
        strategy_name=strategy.name,
    )
