# -*- coding: utf-8 -*-
"""準備人工標註工作包裡「要動手做」的那些檔案。

  A_薊馬葉害_框怎麼畫/待標註/    20 張既有的薊馬葉害影像（不附既有標註，兩人各標一次）
  B_潛葉蛾_補標150張/待標註/     150 張外部潛葉蛾影像（先去重，只取獨立場景）
  C_合成圖_看圖打勾/待檢查/      由 tools/copy_paste_clm.py --qa 產生

影像一律改名成 `01.jpg`、`02.jpg`……——組員面對的是最簡單的檔名，
原始檔名記在同一夾的 `_原始檔名對照.csv`（組員不用理它，是我們回收時對回去用的）。

用法：
    .venv/Scripts/python.exe tools/prepare_manual_packages.py
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P    # Datasets/ 的版面配置：單一真實來源

import numpy as np

import build_dataset_v5_6 as B
from check_dataset_leakage import dhash

REPO = B.REPO
EXTERNAL = (_P.EXTERNAL / "Datasets" / "外部資料集"
            / "Large-Scale Lemon Leaf Disease and Pest Image Data" / "Citrus_Pest")

A_N = 20
B_N = 150
SEED = 0


def _write_map(rows: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["新檔名", "原始檔名"])
        w.writerows(rows)


def prepare_a(dst: Path) -> int:
    """20 張薊馬葉害。刻意只取 train 的影像，不碰評估集。"""
    items, _ = B.collect(next(s for s in B.SOURCES if s["name"] == "Thrips_Damage"))
    v55map = B.v55_split_map()
    pool, _ = B.nested_split("Thrips_Damage", items, v55map)
    train = pool["train"]
    # 涵蓋不同受害程度：依框面積排序後等距取樣，避免 20 張全是同一種難度
    train.sort(key=lambda it: sum(b[3] * b[4] for b in it["boxes"]))
    idx = [round(i * (len(train) - 1) / (A_N - 1)) for i in range(A_N)]
    picks = [train[i] for i in sorted(set(idx))]

    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.glob("*.jpg"):
        p.unlink()
    rows = []
    for k, it in enumerate(picks, 1):
        shutil.copy2(it["img"], dst / f"{k:02d}.jpg")
        rows.append((f"{k:02d}.jpg", it["img"].name))
    _write_map(rows, dst.parent / "_原始檔名對照.csv")
    return len(rows)


def prepare_b(dst: Path) -> tuple[int, int]:
    """150 張外部潛葉蛾。先用 dHash 去重，同一個場景只留一張。"""
    files = sorted(p for p in EXTERNAL.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"找不到外部影像：{EXTERNAL}")

    H = np.stack([dhash(p) if dhash(p) is not None else np.zeros(64, bool) for p in files])
    parent = list(range(len(files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i0 in range(0, len(files), 200):
        a = H[i0:i0 + 200]
        d = (a[:, None, :] != H[None, :, :]).sum(-1)
        for bi in range(a.shape[0]):
            gi = i0 + bi
            for gj in range(gi + 1, len(files)):
                if int(d[bi, gj]) <= B.DUP_THRESHOLD:
                    ra, rb = find(gi), find(gj)
                    if ra != rb:
                        parent[rb] = ra

    seen, uniq = set(), []
    for i in range(len(files)):
        r = find(i)
        if r not in seen:                 # 每個場景只留一張代表
            seen.add(r)
            uniq.append(files[i])

    rng = random.Random(SEED)
    rng.shuffle(uniq)
    picks = sorted(uniq[:B_N], key=lambda p: p.name)

    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.glob("*.jpg"):
        p.unlink()
    rows = []
    for k, p in enumerate(picks, 1):
        shutil.copy2(p, dst / f"{k:03d}.jpg")
        rows.append((f"{k:03d}.jpg", p.name))
    _write_map(rows, dst.parent / "_原始檔名對照.csv")
    return len(rows), len(uniq)


def main() -> None:
    root = B.MANUAL_ROOT
    print("═" * 74)
    print("  準備人工標註工作包")
    print("═" * 74)

    na = prepare_a(root / "A_薊馬葉害_框怎麼畫" / "待標註")
    print(f"  A  {na} 張薊馬葉害影像（取自 train，依受害程度等距取樣）")

    nb, nuniq = prepare_b(root / "B_潛葉蛾_補標150張" / "待標註")
    print(f"  B  {nb} 張外部潛葉蛾影像（542 張去重後剩 {nuniq} 個獨立場景，隨機抽 {nb}）")

    for sub in ("A_薊馬葉害_框怎麼畫", "B_潛葉蛾_補標150張", "C_合成圖_看圖打勾"):
        (root / sub / "完成後放這裡").mkdir(parents=True, exist_ok=True)
    print("  三夾的『完成後放這裡』已建好")
    print("\n  C 的待檢查影像請跑：.venv/Scripts/python.exe tools/copy_paste_clm.py --qa 30")


if __name__ == "__main__":
    main()
