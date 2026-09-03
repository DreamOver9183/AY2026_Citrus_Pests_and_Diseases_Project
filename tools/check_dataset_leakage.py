"""近重複影像跨 split 洩漏查驗。

`verify_dataset_v5r.py` 的洩漏檢查只比對「(檔案大小, 前 64KB md5)」，能抓到逐位元相同的
檔案，但抓不到：

  1. 同一張原始照片被匯出/上傳兩次，但兩次的 JPEG 壓縮參數不同（位元不同、像素幾乎相同）。
  2. 手機連拍（burst mode）在同一場景、同一秒內拍出的相鄰幾張——不同影格，但物件位置
     幾乎沒動，等同於同一個標註目標被切成兩張「不同」影像。

這兩種情況都會讓語意上相同的畫面同時出現在 train 與 valid/test，構成資料洩漏：
模型在 train 看過的畫面，換個檔名又出現在「未見過」的評估集裡。

方法：對每個類別的**來源池**（`collect()` 讀取的原始資料夾，而不是 OutPut，因為只有
來源池的檔名彼此獨立、可以做全兩兩比較）計算 8x8 差分雜湊（dHash，64-bit），
兩兩比較 Hamming 距離；再用 md5 把 OutPut 的 `{類別}_*.jpg`（排除 `_aug_`，
因為 `write_sample()` 是位元複製）對回來源檔名，取得每張來源影像的 split 歸屬。
距離 <= 閾值且分屬不同 split 的配對即為疑似洩漏。

閾值 6（64 bit 中）是近重複影像偵測的常見保守閾值；本檔案的 docstring 與
`docs/v5r_記錄_近重複影像跨split洩漏查驗.md` 附有以像素平均絕對誤差（MAE）驗證過的
範例，供交叉確認 dHash 的判斷不是巧合。

用法：
    .venv/Scripts/python.exe tools/check_dataset_leakage.py
    .venv/Scripts/python.exe tools/check_dataset_leakage.py --classes Scale_Insect,Thrips
    .venv/Scripts/python.exe tools/check_dataset_leakage.py --threshold 4 --out-dir other/leak_check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "Datasets" / "Datasets_YOLO26_v5r"
OUT_ROOT = SRC_ROOT / "OutPut"
IMG_EXT = (".jpg", ".jpeg", ".png")

# (輸出類別名, 來源目錄) —— 與 build_dataset_v5r.py 的 SOURCES 一致
SOURCES = [
    ("Oily_Spot", "Diseases/Greasy Spot"),
    ("Canker", "Diseases/canker/train"),
    ("Sooty_Mold", "Diseases/sooty mold"),
    ("Black_Spot", "Diseases/Melanose"),
    ("Scale_Insect", "Pests/Scale_Insect.yolo26/train"),
    ("Citrus_Leaf_Miner", "Pests/Citrus_Leaf_Miner.yolo26/train"),
    ("Thrips", "Pests/Thrips_v5r.yolo26/train"),
    ("Aphid", "Pests/Aphid.yolo26/train"),
]

DEFAULT_THRESHOLD = 6
HASH_SIZE = 8                      # 8x8 dHash -> 64 bit


def dhash(path: Path, hash_size: int = HASH_SIZE) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    except Exception:
        return None
    arr = np.asarray(im, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_class(name: str, rel: str, threshold: int, out_root: Path) -> list[tuple]:
    src = SRC_ROOT / rel / "images"
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        print(f"{name:<20} 來源資料夾找不到影像：{src}")
        return []

    hashes, src_md5 = {}, {}
    for p in files:
        h = dhash(p)
        if h is not None:
            hashes[p.name] = h
        src_md5[md5(p)] = p.name

    name_to_split: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for p in (out_root / split / "images").glob(f"{name}_*.jpg"):
            if "_aug_" in p.stem:
                continue                                # 增強圖是合成的，不參與洩漏比對
            sn = src_md5.get(md5(p))
            if sn:
                name_to_split[sn] = split

    names = list(hashes.keys())
    H = np.stack([hashes[n] for n in names]) if names else np.zeros((0, 64), dtype=bool)
    N = len(names)
    cross_pairs: list[tuple] = []
    BLOCK = 200
    for i0 in range(0, N, BLOCK):
        a = H[i0:i0 + BLOCK]
        dist = (a[:, None, :] != H[None, :, :]).sum(-1)
        for bi in range(a.shape[0]):
            gi = i0 + bi
            for gj in range(gi + 1, N):
                d = int(dist[bi, gj])
                if d <= threshold:
                    na, nb = names[gi], names[gj]
                    sa, sb = name_to_split.get(na), name_to_split.get(nb)
                    if sa and sb and sa != sb:
                        cross_pairs.append((na, nb, d, sa, sb))

    return cross_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default=None,
                    help="逗號分隔的類別子集，預設查全部 8 類")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                    help=f"dHash Hamming 距離閾值（64 bit 中），預設 {DEFAULT_THRESHOLD}")
    ap.add_argument("--data", default=str(OUT_ROOT), help="OutPut 資料集根目錄")
    ap.add_argument("--out-dir", default=None, help="逐配對明細 CSV 輸出目錄；不填則不寫檔")
    args = ap.parse_args()

    wanted = set(args.classes.split(",")) if args.classes else None
    out_root = Path(args.data)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'類別':<20}{'跨split配對':>12}{'受影響 valid':>13}{'受影響 test':>12}")
    total_affected = 0
    for name, rel in SOURCES:
        if wanted and name not in wanted:
            continue
        pairs = check_class(name, rel, args.threshold, out_root)

        av = {b if sa == "train" else a
              for a, b, d, sa, sb in pairs if "train" in (sa, sb) and "valid" in (sa, sb)}
        at = {b if sa == "train" else a
              for a, b, d, sa, sb in pairs if "train" in (sa, sb) and "test" in (sa, sb)}
        total_affected += len(av) + len(at)
        print(f"{name:<20}{len(pairs):>12}{len(av):>13}{len(at):>12}")

        if out_dir and pairs:
            with open(out_dir / f"{name}_leak_pairs.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["img_a", "img_b", "hamming", "split_a", "split_b"])
                for row in sorted(pairs, key=lambda x: x[2]):
                    w.writerow(row)

    print(f"\n合計受影響（valid+test）影像數：{total_affected}")
    if out_dir:
        print(f"逐配對明細已存到：{out_dir}")


if __name__ == "__main__":
    main()
