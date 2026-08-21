# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5r 建置腳本。

依《柑橘病蟲害 YOLO26 資料集建置技術摘要》(v5.1) 的方法論，套用到重新標註後的
v5r 來源，並修正 v5 的三處缺陷（詳見 docs/v5r_資料集建置說明.md）：

  1. Thrips 的兩個子類（Thysanoptera 蟲體 / thirps_leaf_damage 葉害）合併為單一
     Thrips 類。重標後兩者尺度已對齊，合併後極差 9.8x（v5 為 24.7x）。
  2. Aphid 丟棄面積等效邊長 < 20px（letterbox 到 640 後）的框。這些框全部來自
     41 張田間影像，中位數僅 7.6px，在 640 輸入下不可學也不可量。
  3. 檔名帶類別前綴，raw 與 aug 可直接由檔名區分（v5 實際輸出未實作此規範）。

用法：
    .venv/Scripts/python.exe tools/build_dataset_v5r.py
    .venv/Scripts/python.exe tools/build_dataset_v5r.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import albumentations as A
from PIL import Image as PILImage

# ══════════════════════════════════════════════════════════════════════════
# 組態
# ══════════════════════════════════════════════════════════════════════════

REPO = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO / "Datasets" / "Datasets_YOLO26_v5r"
OUT_ROOT = SRC_ROOT / "OutPut"

SEED = 0
SPLIT = (0.80, 0.10, 0.10)          # train / valid / test，先切分後增強
AUG_MULT = 4                         # 增強上限：原始訓練圖數的 4 倍
AUG_CAP = 1200                       # 每類訓練圖數上限（與 v5 的 Thrips/Aphid 實際上限一致）
SCALE_TARGET_BOXES = 5400            # Scale_Insect 降採樣目標框數（v5 為 5,388）
APHID_MIN_PX = 20.0                  # Aphid 專用：面積等效邊長下限（letterbox 到 640 後）
IMGSZ_REF = 640                      # 尺寸換算基準，與訓練解析度一致
JPEG_QUALITY = 92

CLASSES = [
    "Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
    "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid",
]

# (輸出類別 id, 來源目錄, 來源類別 id -> 是否採用；None 表示全部併入)
SOURCES = [
    dict(cid=0, name="Oily_Spot",         path="Diseases/Greasy Spot",                  keep=None),
    dict(cid=1, name="Canker",            path="Diseases/canker/train",                 keep=None),
    dict(cid=2, name="Sooty_Mold",        path="Diseases/sooty mold",                   keep=None),
    dict(cid=3, name="Black_Spot",        path="Diseases/Melanose",                      keep=None),
    dict(cid=4, name="Scale_Insect",      path="Pests/Scale_Insect.yolo26/train",       keep=None),
    dict(cid=5, name="Citrus_Leaf_Miner", path="Pests/Citrus_Leaf_Miner.yolo26/train",  keep=None),
    dict(cid=6, name="Thrips",            path="Pests/Thrips_v5r.yolo26/train",         keep=None),
    dict(cid=7, name="Aphid",             path="Pests/Aphid.yolo26/train",              keep=None),
]

BACKGROUND_SOURCES = [
    dict(name="Background", path="Healthy/Murcott"),
    dict(name="Background", path="Healthy/Ponkan"),
]

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 依技術摘要 §5.2 的 Albumentations 配置
AUG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=15, val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # 對應技術摘要的 ShiftScaleRotate(shift±5%, scale±10%, rotate±15°)。
        # Albumentations 2.x 已移除該運算元，改用等價的 Affine。
        # border_mode 沿用 Albumentations 1.x ShiftScaleRotate 的預設 REFLECT_101：
        # 若改用 BORDER_CONSTANT 會在旋轉後留下黑色楔形，而黑楔只出現在增強圖、
        # 增強圖又只存在於 7 個類別（Scale_Insect 與背景不增強），
        # 等於給模型一條「黑楔 ⇒ 不是介殼蟲/背景」的假捷徑。
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    seed=SEED,
)


# ══════════════════════════════════════════════════════════════════════════
# 讀取與正規化
# ══════════════════════════════════════════════════════════════════════════

def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXT:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """讀 YOLO 標註；多邊形轉為外接矩形（Detect 任務）。回傳 (cls, cx, cy, w, h)。"""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) > 4:                                  # polygon
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            cx, cy, w, h = vals[:4]
        # 夾在 [0,1] 內，避免來源溢出座標讓增強管線報錯
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
            continue
        rows.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return rows


def equiv_px(w: float, h: float, W: int, H: int) -> float:
    """框在 letterbox 到 IMGSZ_REF 之後的面積等效邊長（像素）。"""
    sc = IMGSZ_REF / max(W, H)
    return float(np.sqrt(w * W * sc * h * H * sc))


def collect(spec: dict) -> tuple[list[dict], Counter]:
    """收集單一來源的所有樣本，套用類別重映與 Aphid 過濾。"""
    base = SRC_ROOT / spec["path"]
    img_dir, lbl_dir = base / "images", base / "labels"
    items, stats = [], Counter()
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = find_image(img_dir, lbl.stem)
        if img is None:
            stats["缺影像"] += 1
            continue
        try:
            W, H = PILImage.open(img).size          # 只讀檔頭，不解碼整張
        except Exception:
            stats["讀取失敗"] += 1
            continue
        raw = parse_label(lbl)
        stats["原始框"] += len(raw)

        boxes = []
        for _cls, cx, cy, w, h in raw:
            if spec["cid"] == 7 and equiv_px(w, h, W, H) < APHID_MIN_PX:
                stats["Aphid_丟棄極小框"] += 1
                continue
            boxes.append((spec["cid"], cx, cy, w, h))

        if raw and not boxes:
            # 原本有標註、過濾後全空 → 丟棄整張，避免變成假陰性
            stats["整張丟棄"] += 1
            continue
        if not raw:
            stats["來源空標註"] += 1

        items.append(dict(img=img, boxes=boxes, W=W, H=H))
        stats["框"] += len(boxes)
    stats["影像"] = len(items)
    return items, stats


def collect_background(spec: dict) -> list[dict]:
    base = SRC_ROOT / spec["path"]
    items = []
    for img in sorted(base.glob("images/*")):
        if img.suffix not in IMG_EXT:
            continue
        try:
            W, H = PILImage.open(img).size
        except Exception:
            continue
        items.append(dict(img=img, boxes=[], W=W, H=H))
    return items


# ══════════════════════════════════════════════════════════════════════════
# 切分 / 降採樣 / 增強
# ══════════════════════════════════════════════════════════════════════════

def split_items(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(round(n * SPLIT[0]))
    n_va = int(round(n * SPLIT[1]))
    return {
        "train": [items[i] for i in idx[:n_tr]],
        "valid": [items[i] for i in idx[n_tr:n_tr + n_va]],
        "test":  [items[i] for i in idx[n_tr + n_va:]],
    }


def downsample_scale(train: list[dict], rng: random.Random) -> list[dict]:
    """Scale_Insect 智能降採樣：隨機抽取影像直到累計框數達標，不做增強。"""
    order = list(range(len(train)))
    rng.shuffle(order)
    kept, total = [], 0
    for i in order:
        if total >= SCALE_TARGET_BOXES:
            break
        kept.append(train[i])
        total += len(train[i]["boxes"])
    return kept


def write_sample(img_path: Path, boxes, out_img: Path, out_lbl: Path):
    shutil.copy2(img_path, out_img)
    out_lbl.write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes),
        encoding="utf-8",
    )


def write_augmented(item: dict, out_img: Path, out_lbl: Path) -> bool:
    im = cv2.imread(str(item["img"]))
    if im is None:
        return False
    bb = [[cx, cy, w, h] for _c, cx, cy, w, h in item["boxes"]]
    lb = [c for c, *_ in item["boxes"]]
    try:
        res = AUG(image=im, bboxes=bb, class_labels=lb)
    except Exception:
        return False
    out = [(int(c), *map(float, b)) for b, c in zip(res["bboxes"], res["class_labels"])]
    if item["boxes"] and not out:
        return False                                   # 增強後標註全失效，捨棄
    cv2.imwrite(str(out_img), res["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    out_lbl.write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in out),
        encoding="utf-8",
    )
    return True


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只統計，不寫檔")
    args = ap.parse_args()

    rng = random.Random(SEED)
    print("═" * 78)
    print("  Datasets_YOLO26_v5r 建置")
    print("═" * 78)

    # ── 1. 收集 ────────────────────────────────────────────────────────
    pools: dict[str, dict[str, list[dict]]] = {}
    print(f"\n【1】收集來源\n{'類別':<20}{'影像':>7}{'框':>8}{'備註':>28}")
    for spec in SOURCES:
        items, st = collect(spec)
        note = []
        if st["Aphid_丟棄極小框"]:
            note.append(f"丟棄極小框 {st['Aphid_丟棄極小框']}")
        if st["整張丟棄"]:
            note.append(f"整張丟棄 {st['整張丟棄']}")
        if st["來源空標註"]:
            note.append(f"空標註 {st['來源空標註']}")
        print(f"{spec['name']:<20}{st['影像']:>7,}{st['框']:>8,}{'  '.join(note):>28}")
        pools[spec["name"]] = split_items(items, rng)

    bg = []
    for spec in BACKGROUND_SOURCES:
        bg += collect_background(spec)
    print(f"{'Background(健康株)':<20}{len(bg):>7,}{0:>8,}{'標註清空為負樣本':>28}")
    pools["Background"] = split_items(bg, rng)

    # ── 2. Scale_Insect 降採樣 ─────────────────────────────────────────
    before = len(pools["Scale_Insect"]["train"])
    bbefore = sum(len(i["boxes"]) for i in pools["Scale_Insect"]["train"])
    pools["Scale_Insect"]["train"] = downsample_scale(pools["Scale_Insect"]["train"], rng)
    after = len(pools["Scale_Insect"]["train"])
    bafter = sum(len(i["boxes"]) for i in pools["Scale_Insect"]["train"])
    print(f"\n【2】Scale_Insect 降採樣  影像 {before:,}→{after:,} ({after/before*100:.1f}%)"
          f"   框 {bbefore:,}→{bafter:,} ({bafter/bbefore*100:.1f}%)   不做增強")

    # ── 3. 增強配額 ────────────────────────────────────────────────────
    print(f"\n【3】增強配額（target = min({AUG_MULT}× raw, {AUG_CAP})，Scale_Insect 與背景不增強）")
    print(f"{'類別':<20}{'raw train':>11}{'目標':>8}{'需增強':>8}")
    quota = {}
    for name in CLASSES + ["Background"]:
        raw = len(pools[name]["train"])
        if name in ("Scale_Insect", "Background"):
            tgt = raw
        else:
            tgt = min(AUG_MULT * raw, AUG_CAP)
        quota[name] = max(0, tgt - raw)
        print(f"{name:<20}{raw:>11,}{tgt:>8,}{quota[name]:>8,}")

    if args.dry_run:
        print("\n--dry-run：不寫檔，結束。")
        return

    # ── 4. 寫出 ────────────────────────────────────────────────────────
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for sp in ("train", "valid", "test"):
        (OUT_ROOT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / sp / "labels").mkdir(parents=True, exist_ok=True)

    print("\n【4】寫出")
    counters = defaultdict(Counter)
    for name in CLASSES + ["Background"]:
        for sp in ("train", "valid", "test"):
            items = pools[name][sp]
            for k, it in enumerate(items):
                stem = f"{name}_{k:05d}"
                write_sample(it["img"], it["boxes"],
                             OUT_ROOT / sp / "images" / f"{stem}.jpg",
                             OUT_ROOT / sp / "labels" / f"{stem}.txt")
                counters[sp][name] += 1
        # 增強只作用於 train
        need, made, guard = quota[name], 0, 0
        src = pools[name]["train"]
        while made < need and src and guard < need * 5:
            it = src[(made + guard) % len(src)]
            stem = f"{name}_aug_{made:05d}"
            ok = write_augmented(it,
                                 OUT_ROOT / "train" / "images" / f"{stem}.jpg",
                                 OUT_ROOT / "train" / "labels" / f"{stem}.txt")
            guard += 1
            if ok:
                made += 1
                counters["train"][name] += 1
        if need:
            print(f"  {name:<20} 增強 {made:,}/{need:,}")

    # ── 5. data.yaml ───────────────────────────────────────────────────
    yaml_text = (
        "# Datasets_YOLO26_v5r —— 由 tools/build_dataset_v5r.py 產生\n"
        f"# seed={SEED}  split={SPLIT}  先切分後增強\n"
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  - {c}\n" for c in CLASSES)
    )
    (OUT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # ── 6. 總結 ────────────────────────────────────────────────────────
    print(f"\n【5】完成  →  {OUT_ROOT}")
    print(f"\n{'類別':<20}{'train':>9}{'valid':>8}{'test':>8}")
    for name in CLASSES + ["Background"]:
        print(f"{name:<20}{counters['train'][name]:>9,}{counters['valid'][name]:>8,}{counters['test'][name]:>8,}")
    print(f"{'合計':<20}{sum(counters['train'].values()):>9,}"
          f"{sum(counters['valid'].values()):>8,}{sum(counters['test'].values()):>8,}")


if __name__ == "__main__":
    main()
