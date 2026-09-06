"""建置後的資料集分析：組成、框尺寸、分佈位移、增強比例。

`verify_dataset_v5r.py` 負責「有沒有壞掉」（成對性、座標越界、split 洩漏），
本檔負責「長什麼樣子」——寫報告要引用的統計值都從這裡出。

四組統計：

  1. **組成**：每類每 split 的影像數／框數、每張影像的平均框數、增強圖佔比。
  2. **框尺寸**：letterbox 到 640 後的面積等效邊長分位數與極差（p99/p10），
     以及 `<32px` 佔比——P3 用這個尺寸帶討論標註像素精度的地板。
  3. **分佈位移**：同一類在 train / valid / test 之間的中位框尺寸與每圖框數是否一致。
     v5 時代 Scale_Insect 在 valid/test 的佔比遠高於 train，是系統性偏差的來源，
     這組數字用來確認新版本沒有重蹈覆轍。
  4. **背景負樣本**：純背景影像（0 框）在各 split 的數量。

用法：
    .venv/Scripts/python.exe tools/analyze_dataset.py
    .venv/Scripts/python.exe tools/analyze_dataset.py Datasets/2_處理與切分/v5.6
    .venv/Scripts/python.exe tools/analyze_dataset.py <路徑> --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_paths as _P    # Datasets/ 的版面配置：單一真實來源

REPO = Path(__file__).resolve().parent.parent
IMGSZ_REF = 640
SPLITS = ("train", "valid", "test")


def load_classes(root: Path) -> list[str]:
    """由資料集自己的 data.yaml 讀類別清單（最小解析，不依賴 pyyaml）。"""
    p = root / "data.yaml"
    names, in_names = [], False
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if ln.startswith("  - "):
                names.append(ln[4:].strip())
            elif ln.strip() and not ln.startswith(" "):
                break
    return names


def analyze(root: Path) -> dict:
    classes = load_classes(root)
    # cls -> split -> list[等效邊長]
    sizes: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # cls -> split -> [影像數, 框數, 增強圖數]
    comp: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    per_img: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    bg = defaultdict(int)

    for sp in SPLITS:
        for lp in sorted((root / sp / "labels").glob("*.txt")):
            ip = next((root / sp / "images" / f"{lp.stem}{e}"
                       for e in (".jpg", ".jpeg", ".png")
                       if (root / sp / "images" / f"{lp.stem}{e}").exists()), None)
            if ip is None:
                continue
            try:
                W, H = Image.open(ip).size
            except Exception:
                continue
            sc = IMGSZ_REF / max(W, H)
            rows = [ln.split() for ln in lp.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            if not rows:
                bg[sp] += 1
            is_aug = "_aug_" in lp.stem
            # 檔名前綴即輸出類別名；依長度反向排序避免 Thrips 誤收 Thrips_Damage。
            # Background 不在 data.yaml 的 names 裡（它是負樣本，沒有類別 id），要另外納入。
            head = lp.stem.split("_aug_")[0]
            cls = next((c for c in sorted(classes + ["Background"], key=len, reverse=True)
                        if head.startswith(c + "_") or head == c), None)
            if cls is None:
                continue
            comp[cls][sp][0] += 1
            comp[cls][sp][1] += len(rows)
            comp[cls][sp][2] += 1 if is_aug else 0
            per_img[cls][sp].append(len(rows))
            for r in rows:
                if len(r) < 5:
                    continue
                w, h = float(r[3]), float(r[4])
                sizes[cls][sp].append(float(np.sqrt(w * W * sc * h * H * sc)))
    return dict(classes=classes, sizes=sizes, comp=comp, per_img=per_img, bg=bg)


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?",
                    default=str(_P.split("v5.6")))
    ap.add_argument("--csv", default=None, help="把逐類統計另存成 CSV")
    args = ap.parse_args()
    root = Path(args.root)

    r = analyze(root)
    classes, sizes, comp, per_img, bg = r["classes"], r["sizes"], r["comp"], r["per_img"], r["bg"]
    all_names = classes + ["Background"]

    print("═" * 96)
    print(f"  資料集分析：{root}")
    print("═" * 96)

    print(f"\n【1】組成\n{'類別':<20}{'train圖':>8}{'valid圖':>8}{'test圖':>7}"
          f"{'train框':>9}{'valid框':>8}{'test框':>7}{'增強圖':>8}{'增強佔比':>9}")
    tot = defaultdict(int)
    for name in all_names:
        c = comp[name]
        aug = c["train"][2]
        tr_img = c["train"][0]
        share = f"{100 * aug / tr_img:.0f}%" if tr_img else "—"
        print(f"{name:<20}{c['train'][0]:>8,}{c['valid'][0]:>8,}{c['test'][0]:>7,}"
              f"{c['train'][1]:>9,}{c['valid'][1]:>8,}{c['test'][1]:>7,}{aug:>8,}{share:>9}")
        for sp in SPLITS:
            tot[sp + "_img"] += c[sp][0]
            tot[sp + "_box"] += c[sp][1]
    print(f"{'合計':<20}{tot['train_img']:>8,}{tot['valid_img']:>8,}{tot['test_img']:>7,}"
          f"{tot['train_box']:>9,}{tot['valid_box']:>8,}{tot['test_box']:>7,}")
    print(f"{'其中純背景影像':<20}{bg['train']:>8,}{bg['valid']:>8,}{bg['test']:>7,}")

    print(f"\n【2】框尺寸（全 split 合併，letterbox 到 640 的面積等效邊長 px）")
    print(f"{'類別':<20}{'n框':>7}{'p10':>8}{'p50':>8}{'p90':>8}{'p99':>8}"
          f"{'極差':>8}{'<32px':>8}{'每圖框數':>9}")
    rows_csv = []
    for name in classes:
        a = np.array([v for sp in SPLITS for v in sizes[name][sp]])
        if not len(a):
            continue
        p10, p50, p90, p99 = pct(a, 10), pct(a, 50), pct(a, 90), pct(a, 99)
        ratio = p99 / max(p10, 1e-9)
        small = 100 * float((a < 32).mean())
        ipi = np.array([v for sp in SPLITS for v in per_img[name][sp]])
        print(f"{name:<20}{len(a):>7,}{p10:>8.1f}{p50:>8.1f}{p90:>8.1f}{p99:>8.1f}"
              f"{ratio:>7.1f}x{small:>7.1f}%{ipi.mean():>9.1f}")
        rows_csv.append(dict(cls=name, n_box=len(a), p10=round(p10, 1), p50=round(p50, 1),
                             p90=round(p90, 1), p99=round(p99, 1), ratio=round(ratio, 2),
                             lt32_pct=round(small, 1), boxes_per_img=round(float(ipi.mean()), 2)))

    print(f"\n【3】分佈位移檢查（同一類在三個 split 之間應該相近）")
    print(f"{'類別':<20}{'中位框尺寸 train/valid/test':>34}{'每圖框數 train/valid/test':>30}")
    for name in classes:
        med = [pct(np.array(sizes[name][sp]), 50) for sp in SPLITS]
        ipi = [float(np.mean(per_img[name][sp])) if per_img[name][sp] else float("nan")
               for sp in SPLITS]
        print(f"{name:<20}{f'{med[0]:.0f} / {med[1]:.0f} / {med[2]:.0f}':>34}"
              f"{f'{ipi[0]:.1f} / {ipi[1]:.1f} / {ipi[2]:.1f}':>30}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\n逐類統計已存到 {args.csv}")


if __name__ == "__main__":
    main()
