# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5r 產出驗收。

檢查項目：
  1. 影像／標註成對，無孤兒檔
  2. 類別 id 落在 0..nc-1，座標落在 [0,1] 且寬高 > 0
  3. 三個 split 之間無影像洩漏（以檔案內容簽章比對）
  4. 每類每 split 的影像數、框數
  5. 每類的邊長分位數與極差（letterbox 到 640 後），對照 v5 基線

用法：
    .venv/Scripts/python.exe tools/verify_dataset_v5r.py
    .venv/Scripts/python.exe tools/verify_dataset_v5r.py <資料集根目錄>
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "Datasets" / "Datasets_YOLO26_v5r" / "OutPut"
IMGSZ_REF = 640
SPLITS = ("train", "valid", "test")

CLASSES = ["Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
           "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid"]

# v5 基線（train split，letterbox 到 640 的面積等效邊長）
V5_BASELINE = {
    "Oily_Spot": (1.4, 0.978), "Canker": (6.9, 0.795), "Sooty_Mold": (1.4, 0.980),
    "Black_Spot": (1.5, 0.934), "Scale_Insect": (3.7, 0.855),
    "Citrus_Leaf_Miner": (10.8, 0.748), "Thrips": (24.7, 0.572), "Aphid": (26.0, 0.701),
}


def sig(p: Path) -> tuple[int, str]:
    with open(p, "rb") as f:
        head = f.read(65536)
    return (p.stat().st_size, hashlib.md5(head).hexdigest())


def pctl(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(len(s) * q), len(s) - 1)]


def main():
    errors: list[str] = []
    imgs_per_split: dict[str, list[Path]] = {}
    sides = defaultdict(list)                    # class -> [px]
    per = defaultdict(lambda: Counter())         # split -> Counter(class -> images)
    boxes = defaultdict(lambda: Counter())       # split -> Counter(class -> boxes)
    bg = Counter()

    allnames_all = sorted(CLASSES + ["Background"], key=len, reverse=True)

    print("═" * 78)
    print(f"  驗收：{ROOT}")
    print("═" * 78)

    for sp in SPLITS:
        idir, ldir = ROOT / sp / "images", ROOT / sp / "labels"
        if not idir.is_dir():
            errors.append(f"{sp}: 找不到 images 目錄")
            continue
        ims = sorted(idir.glob("*.jpg"))
        imgs_per_split[sp] = ims
        lbs = {p.stem for p in ldir.glob("*.txt")}
        for p in ims:
            if p.stem not in lbs:
                errors.append(f"{sp}/{p.name}: 缺對應標註")
        for stem in lbs - {p.stem for p in ims}:
            errors.append(f"{sp}/{stem}.txt: 標註無對應影像")

        for p in ims:
            head = p.stem.split("_aug_")[0]
            cls_name = next((c for c in allnames_all if head.startswith(c + "_") or head == c), head)
            lp = ldir / f"{p.stem}.txt"
            if not lp.exists():
                continue
            try:
                W, H = Image.open(p).size
            except Exception as e:
                errors.append(f"{sp}/{p.name}: 影像損毀 {type(e).__name__}")
                continue
            sc = IMGSZ_REF / max(W, H)
            n = 0
            for ln in lp.read_text(encoding="utf-8-sig").splitlines():
                parts = ln.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    errors.append(f"{sp}/{lp.name}: 欄位數 {len(parts)} != 5")
                    continue
                k = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:])
                if not 0 <= k < len(CLASSES):
                    errors.append(f"{sp}/{lp.name}: 類別 id {k} 越界")
                if not (0 <= cx <= 1 and 0 <= cy <= 1) or w <= 0 or h <= 0 \
                        or cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                        or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                    errors.append(f"{sp}/{lp.name}: 座標越界 ({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})")
                n += 1
                sides[CLASSES[k] if 0 <= k < len(CLASSES) else "?"].append(
                    (w * W * sc * h * H * sc) ** 0.5)
                boxes[sp][CLASSES[k] if 0 <= k < len(CLASSES) else "?"] += 1
            if n == 0:
                bg[sp] += 1
            per[sp][cls_name] += 1

    # ── 洩漏檢查 ────────────────────────────────────────────────────────
    print("\n【1】split 之間影像洩漏檢查")
    sigs = {sp: {} for sp in SPLITS}
    for sp, ims in imgs_per_split.items():
        for p in ims:
            if "_aug_" in p.stem:      # 增強圖只在 train，不參與比對
                continue
            sigs[sp].setdefault(sig(p), p.name)
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        dup = set(sigs.get(a, {})) & set(sigs.get(b, {}))
        mark = "✗" if dup else "✓"
        print(f"  {mark} {a} ∩ {b}: {len(dup)} 張重複")
        if dup:
            errors.append(f"{a}/{b} 有 {len(dup)} 張影像洩漏")

    # ── 組成 ────────────────────────────────────────────────────────────
    print(f"\n【2】影像數\n{'類別':<20}{'train':>9}{'valid':>8}{'test':>8}{'合計':>9}")
    allnames = CLASSES + ["Background"]
    for name in allnames:
        t, v, s = per["train"][name], per["valid"][name], per["test"][name]
        print(f"{name:<20}{t:>9,}{v:>8,}{s:>8,}{t+v+s:>9,}")
    tt = sum(per["train"].values()); vv = sum(per["valid"].values()); ss = sum(per["test"].values())
    print(f"{'合計':<20}{tt:>9,}{vv:>8,}{ss:>8,}{tt+vv+ss:>9,}")
    print(f"{'其中純背景':<20}{bg['train']:>9,}{bg['valid']:>8,}{bg['test']:>8,}{sum(bg.values()):>9,}")

    print(f"\n【3】標註框數\n{'類別':<20}{'train':>9}{'佔比':>8}{'valid':>8}{'test':>8}")
    tb = sum(boxes["train"].values())
    for name in CLASSES:
        print(f"{name:<20}{boxes['train'][name]:>9,}{boxes['train'][name]/max(tb,1)*100:>7.1f}%"
              f"{boxes['valid'][name]:>8,}{boxes['test'][name]:>8,}")
    print(f"{'合計':<20}{tb:>9,}{100.0:>7.1f}%{sum(boxes['valid'].values()):>8,}{sum(boxes['test'].values()):>8,}")

    # ── 尺度一致性 ──────────────────────────────────────────────────────
    print(f"\n【4】標註尺度一致性（全 split，letterbox 到 640 的面積等效邊長）")
    print(f"{'類別':<20}{'p10':>8}{'p50':>8}{'p99':>8}{'極差':>9}{'v5 極差':>10}{'變化':>10}{'v8 AP':>8}")
    for name in CLASSES:
        s = sides[name]
        if not s:
            continue
        p10, p50, p99 = pctl(s, .10), pctl(s, .50), pctl(s, .99)
        sp_ = p99 / max(p10, 0.01)
        v5sp, ap = V5_BASELINE[name]
        delta = f"{sp_/v5sp:.2f}x"
        print(f"{name:<20}{p10:>8.1f}{p50:>8.1f}{p99:>8.1f}{sp_:>8.1f}x{v5sp:>9.1f}x{delta:>10}{ap:>8.3f}")

    # ── 結論 ────────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    if errors:
        print(f"驗收未通過，{len(errors)} 項問題：")
        for e in errors[:25]:
            print("  ✗", e)
        if len(errors) > 25:
            print(f"  … 另有 {len(errors)-25} 項")
        sys.exit(1)
    print("驗收通過：成對性、座標、類別 id、split 洩漏檢查全數正常。")


if __name__ == "__main__":
    main()
