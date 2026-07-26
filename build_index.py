#!/usr/bin/env python3
"""index.html 的四張核心策略卡片數字 → 直接從 report_*.html 產生。

為什麼需要這支：
    index.html 原本是手寫的，沒有任何腳本會更新它，但 deploy_pages.yml 把它
    列為必要檔案 = 它就是線上著陸頁。2026-07-19 稽核實測：首頁寫
    SURGE PRO 67.1%、SURGE 58.8%、GUARD 51.6%，而它連過去的報表分別是
    58.0% / 49.5% / 46.7%——著陸頁比報表高 5~9pp，且頁面還寫著
    「每日收盤後自動更新」，實際上從 2026-07-16 之後就沒動過。

作法：
    index.html 內以 data-auto="<key>" 標記需要自動填的元素，本腳本只改
    這些元素的內容，其餘版面完全不動（可重複執行、idempotent）。

用法：
    python build_index.py                # 就地更新 index.html
    python build_index.py --check        # 只檢查是否過期（CI 用，過期則非零退出）
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

# data-auto key → 來源報表檔
CARDS = {
    "surge_pro": "report_surge_pro.html",
    "surge": "report_surge.html",
    "guard": "report_guard.html",
    "v85": "report_v85.html",
}

METRICS = ("年化報酬率", "最大回撤", "Calmar Ratio")


def _text(html: str) -> str:
    t = re.sub(r"<[^>]+>", "|", html)
    return re.sub(r"\s+", " ", t)


def read_metrics(path: Path) -> dict[str, str]:
    """從報表 HTML 抓出摘要指標。抓不到就丟例外——不要靜默填舊值。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到報表 {path}；請先跑 ai_report.py 產生")
    t = _text(path.read_text(encoding="utf-8"))
    out = {}
    for key in METRICS:
        m = re.search(re.escape(key) + r"[|\s]+([+\-]?[\d.]+%?)", t)
        if not m:
            raise ValueError(f"{path.name} 內找不到「{key}」，報表格式可能已變更")
        out[key] = m.group(1)
    return out


def _source_stamp(report: Path) -> str:
    """報表自己標的資料日期；抓不到就退回檔案 mtime。"""
    t = _text(report.read_text(encoding="utf-8"))
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", t)
    if m:
        return m.group(1)
    return dt.date.fromtimestamp(report.stat().st_mtime).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _replace_auto(html: str, key: str, inner: str) -> str:
    """把 <tag data-auto="key">…</tag> 的內容換成 inner。"""
    pat = re.compile(
        r'(<(\w+)[^>]*data-auto="' + re.escape(key) + r'"[^>]*>)(.*?)(</\2>)',
        re.S,
    )
    new, n = pat.subn(lambda m: m.group(1) + inner + m.group(4), html)
    if n != 1:
        raise ValueError(f'index.html 內 data-auto="{key}" 出現 {n} 次，預期 1 次')
    return new


def build(root: Path) -> tuple[str, str]:
    index = root / "index.html"
    html = index.read_text(encoding="utf-8")

    stamps = set()
    for key, report in CARDS.items():
        rp = root / report
        m = read_metrics(rp)
        stamps.add(_source_stamp(rp))
        inner = (
            f"全期 年化 <b>{m['年化報酬率']}</b>"
            f" · MDD <b>{m['最大回撤']}</b>"
            f" · Calmar <b>{m['Calmar Ratio']}</b>"
        )
        html = _replace_auto(html, key, inner)

    data_day = sorted(stamps)[-1] if stamps else "未知"
    prov = (
        f"卡片數字由 build_index.py 直接讀自各策略報表 · "
        f"資料日 {data_day} · 產生於 {dt.date.today().isoformat()} · "
        f"commit {_git_commit()}。點進報表可見完整成本假設與逐筆交易。"
    )
    html = _replace_auto(html, "provenance", prov)
    return html, data_day


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只檢查是否與報表一致，不寫入（不一致則 exit 1）")
    ap.add_argument("--root", default=".", help="repo 根目錄")
    args = ap.parse_args()

    root = Path(args.root)
    index = root / "index.html"
    current = index.read_text(encoding="utf-8")
    new, data_day = build(root)

    # provenance 含「今天日期」，比對時忽略它，只比卡片數字
    def _cards_only(h: str) -> str:
        return _replace_auto(h, "provenance", "")

    if _cards_only(current) == _cards_only(new):
        print(f"✅ index.html 卡片數字與報表一致（資料日 {data_day}）")
        if not args.check:
            index.write_text(new, encoding="utf-8")   # 仍更新 provenance 戳記
        return 0

    if args.check:
        print("❌ index.html 卡片數字與 report_*.html 不一致 → 請跑 python build_index.py")
        return 1

    index.write_text(new, encoding="utf-8")
    print(f"✅ 已更新 index.html（資料日 {data_day}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
