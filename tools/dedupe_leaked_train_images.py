"""移除與 valid/test 存在近重複的 train 影像（原始圖，不含增強圖）。

修法選擇：**只動 train，不動 valid/test。**

`docs/v5r_記錄_近重複影像跨split洩漏查驗.md` 找到的洩漏都是「train 裡有一張影像，
與 valid 或 test 裡的某張影像近乎相同」。有兩種修法：

  A. 重新分群切分（group-aware re-split）：把近重複的影像群組視為一個單位，
     整群分到同一個 split。這才是嚴謹的正確做法，但會讓 valid/test 的組成整組改變，
     所有已發表的 test 集數字（mAP50 0.8908 等）全部作廢，且必須重新訓練才能得到
     可比的新數字。

  B.（本檔採用）**只刪除 train 裡造成洩漏的那張原始圖及其標註。**
     valid/test 逐位元不變，因此本次 P3 診斷、最終評估報告的所有既有數字
     **完全不受影響、不需要重新量測**——只有「下一次訓練」會用到修正後的 train 集。

選 B 的理由：B 修的正是「模型訓練時看過評估集的畫面」這個實際危害，而且代價極小
（只從 train 移除幾十張），不需要牽動已發表的任何數字。A 更徹底，但成本
（全資料集重切分＋全部重訓＋所有既有數字作廢）目前不成比例。

**已知的殘留風險（本檔未處理）**：`build_dataset_v5r.py` 的增強圖（`{類別}_aug_*.jpg`）
是對某張 train 原始圖做 flip / 色彩擾動 / ±15° 仿射變換後的產物，本檔只刪除「原始圖」，
**不會**連帶刪除由它衍生出的增強圖。增強圖的 dHash 通常已經偏離原圖夠遠（尤其是水平
翻轉），不太可能再落在近重複閾值內，但這是推論、未逐一驗證——若要徹底排除，需要在
`build_dataset_v5r.py` 產生增強圖時就記錄它的來源檔名（目前的實作沒有存這個對應關係），
屬於 A 方案的範疇。

**安全性**：`Datasets/` 整個被 `.gitignore` 排除，不進版控；本檔的刪除操作可用
`build_dataset_v5r.py` 重新產生同一份輸出而完全復原（同 `SEED`）。預設是 dry-run，
只印出「會刪什麼」，不實際刪除；要真的刪除必須加 `--apply`。

用法：
    .venv/Scripts/python.exe tools/dedupe_leaked_train_images.py                # dry-run
    .venv/Scripts/python.exe tools/dedupe_leaked_train_images.py --apply        # 實際刪除
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import check_dataset_leakage as cdl          # noqa: E402

THRESHOLD = 6
AFFECTED = [(n, r) for n, r in cdl.SOURCES if n in ("Scale_Insect", "Thrips", "Aphid")]


def find_leaking_train_files(name: str, rel: str, out_root: Path, threshold: int) -> list[Path]:
    """回傳該類別 OutPut/train 裡「與某張 valid/test 影像近重複」的原始圖路徑（不含 _aug_）。"""
    src = cdl.SRC_ROOT / rel / "images"
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in cdl.IMG_EXT)
    hashes, src_md5 = {}, {}
    for p in files:
        h = cdl.dhash(p)
        if h is not None:
            hashes[p.name] = h
        src_md5[cdl.md5(p)] = p.name

    name_to_split: dict[str, str] = {}
    name_to_path: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        for p in (out_root / split / "images").glob(f"{name}_*.jpg"):
            if "_aug_" in p.stem:
                continue
            sn = src_md5.get(cdl.md5(p))
            if sn:
                name_to_split[sn] = split
                if split == "train":
                    name_to_path[sn] = p

    names = list(hashes.keys())
    H = np.stack([hashes[n] for n in names])
    N = len(names)
    leaking_train: set[str] = set()
    BLOCK = 200
    for i0 in range(0, N, BLOCK):
        a = H[i0:i0 + BLOCK]
        dist = (a[:, None, :] != H[None, :, :]).sum(-1)
        for bi in range(a.shape[0]):
            gi = i0 + bi
            for gj in range(gi + 1, N):
                if int(dist[bi, gj]) > threshold:
                    continue
                na, nb = names[gi], names[gj]
                sa, sb = name_to_split.get(na), name_to_split.get(nb)
                if sa == "train" and sb in ("valid", "test"):
                    leaking_train.add(na)
                elif sb == "train" and sa in ("valid", "test"):
                    leaking_train.add(nb)
    return [name_to_path[n] for n in leaking_train if n in name_to_path]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cdl.OUT_ROOT))
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--apply", action="store_true",
                    help="實際刪除檔案；不加此旗標只印出會刪什麼（dry-run）")
    args = ap.parse_args()

    out_root = Path(args.data)
    print(f"目標資料集：{out_root}")
    print("模式：" + ("**實際刪除**" if args.apply else "dry-run（不會動任何檔案，加 --apply 才會真的刪）"))
    print()

    total_files = total_boxes = 0
    for name, rel in AFFECTED:
        victims = find_leaking_train_files(name, rel, out_root, args.threshold)
        n_boxes = 0
        for img_path in sorted(victims):
            lbl_path = out_root / "train" / "labels" / f"{img_path.stem}.txt"
            n = len(lbl_path.read_text(encoding="utf-8-sig").splitlines()) if lbl_path.exists() else 0
            n_boxes += n
            print(f"  [{name}] {'刪除' if args.apply else '將刪除'} {img_path.name}"
                  f"（{n} 個框）{'' if lbl_path.exists() else '  ⚠ 找不到對應標註檔'}")
            if args.apply:
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
        print(f"{name:<16} 共 {len(victims)} 張、{n_boxes} 個框\n")
        total_files += len(victims)
        total_boxes += n_boxes

    print(f"合計：{total_files} 張影像、{total_boxes} 個框"
          f"{'已從 train 移除' if args.apply else '會從 train 移除（尚未執行，加 --apply）'}")
    if not args.apply:
        print("\n（未修改任何檔案。確認上面的清單無誤後，加 --apply 重跑即可實際刪除。）")


if __name__ == "__main__":
    main()
