# -*- coding: utf-8 -*-
"""人工標註回收檢查（工作包 A / B 共用）。

發出去的那一半（說明書、範例圖、待標註影像）由 `prepare_manual_packages.py` 與
`make_annotation_guide.py` 產生；**這支是收回來的那一半**。

用途有兩個：

  1. **先交 2 張的格式預檢。** 組員做完前兩張就先回傳，跑這支確認格式對，
     再去做剩下的。150 張做完才發現存成 PascalVOC，就是白做一個半小時。
  2. **全部回收後的正式驗收**，B 還會順便把標註併進 v5.6 的來源樹
     （類別 id 由 0 重映成 5，見 docs/v5.6_說明_人工標註需求.md §1）。

輸出刻意用白話寫，因為要直接回覆給組員看。

用法：
    # 預檢（組員先交 2 張）
    .venv/Scripts/python.exe tools/check_annotation_return.py A --precheck

    # 正式回收
    .venv/Scripts/python.exe tools/check_annotation_return.py A
    .venv/Scripts/python.exe tools/check_annotation_return.py B --install

    # 指定別的資料夾（例如兩個人的標註分開放）
    .venv/Scripts/python.exe tools/check_annotation_return.py A --dir "....../甲"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image                     # noqa: E402

import build_dataset_v5_6 as B            # noqa: E402

PACKAGES = {
    "A": dict(
        folder="A_薊馬葉害_框怎麼畫",
        images="待標註",
        n_expect=20,
        cid=8,
        cls="Thrips_Damage",
        label="薊馬葉害",
        install=None,          # A 產出的是決策，不併進資料集
    ),
    "B": dict(
        folder="B_潛葉蛾_補標150張",
        images="待標註",
        n_expect=150,
        cid=B.EXT_CID,         # 5 = Citrus_Leaf_Miner
        cls="Citrus_Leaf_Miner",
        label="潛葉蛾",
        install=B.EXT_SOURCE,  # Pests/Citrus_Leaf_Miner_ext.yolo26/train
    ),
}

PRECHECK_MIN = 2


def read_label(p: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """讀一個回收的標註檔，回傳 (框, 問題清單)。"""
    boxes, bad = [], []
    raw = p.read_bytes()
    if raw[:1] == b"<" or b"<annotation>" in raw[:400]:
        return [], ["存成了 XML（PascalVOC）格式，不是 YOLO"]
    if raw[:1] == b"{" or raw[:1] == b"[":
        return [], ["存成了 JSON 格式，不是 YOLO"]
    for i, ln in enumerate(raw.decode("utf-8-sig", "replace").splitlines(), 1):
        q = ln.split()
        if not q:
            continue
        if len(q) != 5:
            bad.append(f"第 {i} 行有 {len(q)} 個數字，應該是 5 個")
            continue
        try:
            c = int(float(q[0]))
            cx, cy, w, h = (float(v) for v in q[1:])
        except ValueError:
            bad.append(f"第 {i} 行有不是數字的東西")
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy)) or not (0 < w <= 1 and 0 < h <= 1):
            bad.append(f"第 {i} 行的座標超出 0~1")
            continue
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            bad.append(f"第 {i} 行的框超出照片範圍")
            continue
        boxes.append((c, cx, cy, w, h))
    return boxes, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", choices=sorted(PACKAGES))
    ap.add_argument("--dir", default=None, help="回收資料夾；預設是該包的『完成後放這裡』")
    ap.add_argument("--precheck", action="store_true",
                    help="只檢查前幾張的格式，不要求交齊")
    ap.add_argument("--install", action="store_true",
                    help="通過後把標註併進 v5.6 的來源樹（只有 B 有這個動作）")
    args = ap.parse_args()

    spec = PACKAGES[args.package]
    root = B.MANUAL_ROOT / spec["folder"]
    src_dir = Path(args.dir) if args.dir else root / "完成後放這裡"
    img_dir = root / spec["images"]

    print("═" * 74)
    print(f"  工作包 {args.package}（{spec['label']}）"
          f"{'　格式預檢' if args.precheck else '　正式回收'}")
    print("═" * 74)
    print(f"  檢查資料夾：{src_dir}")

    if not src_dir.is_dir():
        raise SystemExit(f"\n找不到資料夾。組員應該把檔案放在：{src_dir}")

    txts = sorted(p for p in src_dir.glob("*.txt") if p.stem.lower() != "classes")
    imgs = {p.stem for p in img_dir.glob("*.jpg")} if img_dir.is_dir() else set()

    if not txts:
        raise SystemExit("\n這個資料夾裡沒有 .txt 標註檔。"
                         "\n請確認組員存檔時有選 YOLO 格式，而且存到了『完成後放這裡』。")

    problems: list[str] = []
    n_boxes = 0
    cls_seen: Counter = Counter()
    n_empty = 0
    ok_files: list[tuple[Path, list]] = []

    for t in txts:
        if imgs and t.stem not in imgs:
            problems.append(f"{t.name}：找不到同名的照片（檔名可能被改過）")
            continue
        boxes, bad = read_label(t)
        for b in bad:
            problems.append(f"{t.name}：{b}")
        if not boxes and not bad:
            n_empty += 1
        for c, *_ in boxes:
            cls_seen[c] += 1
        n_boxes += len(boxes)
        if not bad:
            ok_files.append((t, boxes))

    print(f"\n  交回標註檔  {len(txts)} 個"
          + (f"（待標註共 {len(imgs)} 張）" if imgs else ""))
    print(f"  標到的框    {n_boxes} 個"
          + (f"，平均每張 {n_boxes / max(len(ok_files), 1):.1f} 個" if ok_files else ""))
    if n_empty:
        print(f"  空的標註檔  {n_empty} 個（照片上沒有目標而跳過，正常）")
    if cls_seen:
        print(f"  類別編號    {dict(cls_seen)}"
              + ("　← 正確" if set(cls_seen) == {0} else "　← 應該全部是 0"))
    if set(cls_seen) - {0}:
        problems.append(f"類別編號不是全部都 0（看到 {sorted(cls_seen)}）。"
                        f"標註軟體裡應該只建一個類別。")

    missing = 0
    if not args.precheck and imgs:
        missing = len(imgs) - len({t.stem for t in txts})
        if missing > 0:
            print(f"  沒有標註檔的照片 {missing} 張"
                  f"（如果是『照片上沒有目標』而跳過就正常）")

    if args.precheck and len(txts) < PRECHECK_MIN:
        problems.append(f"預檢至少要交 {PRECHECK_MIN} 張，目前只有 {len(txts)} 張")

    print("\n" + "─" * 74)
    if problems:
        print("**格式有問題，請組員修正後重交：**\n")
        for m in problems[:15]:
            print(f"  ✗ {m}")
        if len(problems) > 15:
            print(f"  … 另有 {len(problems) - 15} 項類似問題")
        print("\n（把上面這幾行直接貼給組員就可以了，不用轉述）")
        sys.exit(1)

    print("**格式正確。**")
    if args.precheck:
        print(f"可以請組員繼續做剩下的 {max(0, spec['n_expect'] - len(txts))} 張。")
        return

    # ── 正式回收 ────────────────────────────────────────────────────────
    if args.package == "A":
        print("\n下一步：兩個人的標註都收齊之後，跑一致性評分——")
        print("  .venv/Scripts/python.exe tools/score_annotation_agreement.py 甲的資料夾 乙的資料夾")
        print("\n中位 IoU ≥ 0.85 才值得投入全類 208 張重標；低於就依 v11 §7 刪掉這個類別。")
        return

    if not args.install:
        print(f"\n加上 --install 就會把標註併進 v5.6 的來源樹"
              f"（類別 id {0} → {spec['cid']}）。")
        return

    dst = B.V56_ROOT / spec["install"]
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for t, boxes in ok_files:
        if not boxes:
            continue                       # 空標註不併入，避免變成假陰性
        img = img_dir / f"{t.stem}.jpg"
        if not img.is_file():
            continue
        shutil.copy2(img, dst / "images" / f"{t.stem}.jpg")
        (dst / "labels" / f"{t.stem}.txt").write_text(
            "".join(f"{spec['cid']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    for _c, cx, cy, w, h in boxes),
            encoding="utf-8")
        n += 1
    print(f"\n已併入 {n} 張 → {dst}")
    print(f"  類別 id 由 0 重映成 {spec['cid']}（{spec['cls']}）")
    print("\n下一步：")
    print("  .venv/Scripts/python.exe tools/build_dataset_v5_6.py --arm ext --dry-run")
    print("  .venv/Scripts/python.exe tools/build_dataset_v5_6.py --arm ext --out <另一個目錄>")
    print("  .venv/Scripts/python.exe tools/verify_dataset_v5_6.py")
    print("\n注意 ext 臂是**配額中性**的：外部影像佔用 CLM 既有的增強配額，")
    print("train 總張數與 base 臂相同，所以兩臂可以直接對比。")


if __name__ == "__main__":
    main()
