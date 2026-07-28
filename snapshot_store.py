#!/usr/bin/env python3
"""snapshot_store.py — 讓「已發布的數字」事後真的能被重現

問題（2026-07-26 稽核）
----------------------
同一份程式、同一組參數、**同一個回測視窗**，只差下載資料的日期一天，
v8.5 年化就從 27.5% 變 18.79%（−8.7pp，大於任何策略對池的 α）。
成因是資料源會回溯調整歷史價格（七月台股除權息旺季，抽樣 40 檔有 14 檔
近月除息），整段歷史被重新縮放 → 動量排名改變 → 選股改變 → 路徑改變。

⇒ **任何發布數字都只在「那一次下載」成立。** 要事後驗證就必須留住那份面板。

preflight.py 早就會凍結 `artifacts/snapshot_<date>/`（panel.pkl + manifest.json
+ aux_series.csv），四策略也確實共讀它。但 `.gitignore` 只放行 manifest.json，
panel.pkl 沒有任何地方留著 —— **留了收據，沒留貨**。

為什麼不直接把 panel.pkl 進 git
-------------------------------
實測 8.54 MB／日（gzip 後 5.30 MB）。每年約 250 個交易日 ⇒ 1.3~2.1 GB，
而且是二進位、git 幾乎無法 delta。clone 會被拖垮。

作法
----
打包成單一 `.tar.gz` 上傳到 GitHub Release（不進 clone、永久保存），
git 裡仍只留 manifest.json，但 manifest 會記下 asset 名稱 ——
於是「版控的收據」直接指向「可下載的貨」，鏈條接起來。

    python snapshot_store.py publish artifacts/snapshot_20260728
    python snapshot_store.py fetch 2026-07-28 --dest artifacts/
    python snapshot_store.py verify artifacts/snapshot_20260728

打包／解包／驗證都不碰網路，可離線測試；只有 publish/fetch 會呼叫 `gh`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from twstk.data.contract import AUX_SERIES_FILE, load_aux_series, load_snapshot

# 所有快照掛在同一個 release 底下，避免每天生一個 release。
DEFAULT_TAG = "snapshots"
MANIFEST_FILE = "manifest.json"
PANEL_FILE = "panel.pkl"
ASSET_KEY = "release_asset"          # 寫回 manifest 的欄位


class SnapshotStoreError(RuntimeError):
    """打包／驗證／傳輸失敗。一律 fail-loud —— 靜默降級正是這套系統的老毛病。"""


def _read_manifest(snapshot_dir: str) -> dict:
    path = os.path.join(snapshot_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        raise SnapshotStoreError(f"{snapshot_dir} 內沒有 {MANIFEST_FILE}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def asset_name(manifest: dict) -> str:
    """`snapshot_<as_of>_<hash8>.tar.gz`。

    帶 hash 是刻意的：同一天若因故重跑並產生不同面板，兩份都留得住，
    不會其中一份被無聲覆蓋。
    """
    as_of = str(manifest.get("as_of") or "unknown").replace("/", "-")
    h = str(manifest.get("panel_sha256") or "")[:8] or "nohash"
    return f"snapshot_{as_of}_{h}.tar.gz"


def pack(snapshot_dir: str, out_path: str | None = None) -> str:
    """把快照目錄打成 .tar.gz。回傳壓縮檔路徑。"""
    snapshot_dir = os.path.abspath(snapshot_dir)
    manifest = _read_manifest(snapshot_dir)
    if not os.path.exists(os.path.join(snapshot_dir, PANEL_FILE)):
        raise SnapshotStoreError(f"{snapshot_dir} 內沒有 {PANEL_FILE}，沒有貨可留")

    out_path = out_path or os.path.join(tempfile.gettempdir(), asset_name(manifest))
    base = os.path.basename(snapshot_dir)
    with tarfile.open(out_path, "w:gz") as tar:
        for fn in (PANEL_FILE, MANIFEST_FILE, AUX_SERIES_FILE):
            p = os.path.join(snapshot_dir, fn)
            if os.path.exists(p):          # aux 可能不存在（舊快照）
                tar.add(p, arcname=f"{base}/{fn}")
    return out_path


def unpack(archive: str, dest_dir: str) -> str:
    """解開壓縮檔，回傳快照目錄路徑。"""
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        # 只允許單一頂層目錄、且不得逃出 dest（防惡意 tar 路徑穿越）
        tops = {n.split("/")[0] for n in names}
        if len(tops) != 1:
            raise SnapshotStoreError(f"壓縮檔應只含一個頂層目錄，實得 {sorted(tops)}")
        for n in names:
            if n.startswith("/") or ".." in n.split("/"):
                raise SnapshotStoreError(f"壓縮檔含不安全路徑：{n}")
        tar.extractall(dest_dir)
    return os.path.join(dest_dir, tops.pop())


def verify(snapshot_dir: str) -> dict:
    """驗證面板與 aux 的雜湊，回傳摘要。任何不符都 raise。

    直接複用 contract 的載入器 —— 它們本來就會比對 manifest 內的 hash，
    在這裡另寫一套比對邏輯只會是第二份真相。
    """
    manifest = _read_manifest(snapshot_dir)
    panel_path = os.path.join(snapshot_dir, PANEL_FILE)
    close, *_ = load_snapshot(panel_path)          # hash 不符會 raise
    summary = {
        "as_of": manifest.get("as_of"),
        "panel_sha256": manifest.get("panel_sha256"),
        "n_sessions": int(close.shape[0]),
        "n_tickers": int(close.shape[1]),
        "aux": [],
    }
    for name in (manifest.get("aux_series") or []):
        s = load_aux_series(panel_path, name)      # hash 不符會 raise
        if s is None:
            raise SnapshotStoreError(f"manifest 宣稱有 aux「{name}」，實際載入不到")
        summary["aux"].append({"name": name, "n": int(len(s))})
    return summary


def _gh(*args: str) -> str:
    if shutil.which("gh") is None:
        raise SnapshotStoreError(
            "找不到 gh CLI。GitHub Actions 內建；本機請安裝 GitHub CLI 後再試。")
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SnapshotStoreError(f"gh {' '.join(args)} 失敗：{r.stderr.strip()}")
    return r.stdout.strip()


def _ensure_release(tag: str) -> None:
    try:
        _gh("release", "view", tag)
    except SnapshotStoreError:
        _gh("release", "create", tag,
            "--title", "凍結資料快照",
            "--notes",
            "每個發布日的完整價格面板（panel.pkl + manifest + aux）。\n\n"
            "存在的理由：資料源會回溯調整歷史價格，同一組參數換一天下載，"
            "回測年化可以差 8.7pp。沒有這些快照，已發布的數字事後無法驗證。\n\n"
            "取用：`python snapshot_store.py fetch <YYYY-MM-DD>`")


def publish(snapshot_dir: str, tag: str = DEFAULT_TAG,
            *, update_manifest: bool = True) -> str:
    """打包並上傳到 release，同時把 asset 名稱寫回 manifest（git 追蹤得到）。"""
    manifest = _read_manifest(snapshot_dir)
    name = asset_name(manifest)
    archive = pack(snapshot_dir, os.path.join(tempfile.gettempdir(), name))

    _ensure_release(tag)
    _gh("release", "upload", tag, archive, "--clobber")

    if update_manifest:
        manifest[ASSET_KEY] = {"tag": tag, "asset": name}
        with open(os.path.join(snapshot_dir, MANIFEST_FILE), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
    size_mb = os.path.getsize(archive) / 1e6
    print(f"🧊 已上傳 {name}（{size_mb:.2f} MB）→ release「{tag}」")
    return name


def _resolve_asset(ref: str, tag: str) -> str:
    """ref 可以是日期（2026-07-28）、hash 前綴，或完整 asset 名稱。"""
    if ref.endswith(".tar.gz"):
        return ref
    listing = _gh("release", "view", tag, "--json", "assets")
    assets = [a["name"] for a in json.loads(listing).get("assets", [])]
    hits = [a for a in assets if ref in a]
    if not hits:
        raise SnapshotStoreError(
            f"release「{tag}」內找不到符合「{ref}」的快照；現有 {len(assets)} 份")
    if len(hits) > 1:
        raise SnapshotStoreError(f"「{ref}」對應多份快照，請指名：{sorted(hits)}")
    return hits[0]


def fetch(ref: str, dest_dir: str = "artifacts", tag: str = DEFAULT_TAG) -> str:
    """下載指定快照、解開並**驗證雜湊**。回傳快照目錄。"""
    name = _resolve_asset(ref, tag)
    with tempfile.TemporaryDirectory() as tmp:
        _gh("release", "download", tag, "--pattern", name, "--dir", tmp, "--clobber")
        snap = unpack(os.path.join(tmp, name), dest_dir)
    info = verify(snap)                    # 下載完立刻驗，不驗等於沒留
    print(f"✅ {name} → {snap}")
    print(f"   as_of={info['as_of']}  {info['n_sessions']} 個交易日 × "
          f"{info['n_tickers']} 檔  hash={str(info['panel_sha256'])[:16]}…")
    return snap


def _latest_snapshot_dir(root: str = "artifacts") -> str:
    pat = re.compile(r"^snapshot_\d{8}$")
    dirs = sorted(d for d in os.listdir(root) if pat.match(d)) if os.path.isdir(root) else []
    if not dirs:
        raise SnapshotStoreError(f"{root}/ 下沒有 snapshot_* 目錄；請先跑 preflight.py")
    return os.path.join(root, dirs[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description="凍結快照的保存與取回")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("publish", help="打包並上傳到 GitHub Release")
    p.add_argument("snapshot_dir", nargs="?", help="預設取 artifacts/ 下最新一份")
    p.add_argument("--tag", default=DEFAULT_TAG)

    f = sub.add_parser("fetch", help="下載並驗證某一份快照")
    f.add_argument("ref", help="日期 2026-07-28 / hash 前綴 / 完整 asset 名")
    f.add_argument("--dest", default="artifacts")
    f.add_argument("--tag", default=DEFAULT_TAG)

    v = sub.add_parser("verify", help="驗證本機快照的雜湊")
    v.add_argument("snapshot_dir", nargs="?")

    args = ap.parse_args()
    try:
        if args.cmd == "publish":
            publish(args.snapshot_dir or _latest_snapshot_dir(), args.tag)
        elif args.cmd == "fetch":
            fetch(args.ref, args.dest, args.tag)
        else:
            info = verify(args.snapshot_dir or _latest_snapshot_dir())
            print(json.dumps(info, ensure_ascii=False, indent=2))
    except SnapshotStoreError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
