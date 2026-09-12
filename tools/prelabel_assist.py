# -*- coding: utf-8 -*-
r"""人工標註包 B 的「AI 預標 → 單人審圖」輔助工具。

2026-09-11 起 B 包（外部潛葉蛾 150 張）改為：先由 AI 代理粗標，再由一個人逐張審圖細修。
流程與理由見 docs/v5.6_說明_人工標註需求.md §7；執行代理的任務說明是
`B_潛葉蛾_補標150張/AGENTS.md`。那個資料夾裡的 `prelabel.cmd` 會呼叫本工具，
所以代理可以直接在那裡啟動、不必知道專案根目錄在哪。

本工具只做機械性的部分。**看圖判斷框在哪裡不在這裡**——那是執行代理
（Claude、Antigravity 皆可）與審圖者的工作。2026-09-11 試過用色彩門檻自動找框，
與人眼判斷的中位 IoU 只有 0.42、且會產生大面積誤框，所以不做自動框。

    grid   把影像疊上 0–10 的座標格線，給執行代理看圖讀座標
    write  把 _預標工作區/boxes.csv（格線座標）寫成 YOLO 標註到 預標註/
    sheet  把 預標註/ 的框畫回原圖、十張拼一張，給執行代理自我核對
    seed   把 預標註/ 補進 完成後放這裡/（只補缺的，絕不覆蓋已審過的）
    diff   審圖後比對 完成後放這裡/ 與 預標註/：哪些沒動過、增刪了多少框

用法（專案根目錄下）：
    .venv/Scripts/python.exe tools/prelabel_assist.py grid  --from 11 --to 20
    .venv/Scripts/python.exe tools/prelabel_assist.py write
    .venv/Scripts/python.exe tools/prelabel_assist.py sheet --from 11 --to 20
    .venv/Scripts/python.exe tools/prelabel_assist.py seed
    .venv/Scripts/python.exe tools/prelabel_assist.py diff

**影像必須已經轉正**（EXIF 方向寫進像素）。本工具一律不做 EXIF 處理地開圖，
這樣畫出來的就是任何標註軟體會看到的樣子；有影像仍帶旋轉標記就直接停下。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont   # noqa: E402

import build_dataset_v5_6 as B                # noqa: E402
import check_annotation_return as R           # noqa: E402  （順帶把 stdout 設成 UTF-8）

SPEC = R.PACKAGES["B"]
PKG = B.MANUAL_ROOT / SPEC["folder"]
IMAGES = PKG / SPEC["images"]
PRE = PKG / "預標註"
DONE = PKG / "完成後放這裡"
WORK = PKG / "_預標工作區"
CSV_PATH = WORK / "boxes.csv"
CLASS_NAME = "leafminer"      # 與說明書、組員標註軟體裡的類別名一致；建置時由 cid 重映
ORIENTATION = 0x0112
SAME_OBJECT_IOU = 0.1         # diff 配對用：低於這個就當成不同的框


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _all_keys() -> list[str]:
    return sorted(p.stem for p in IMAGES.glob("*.jpg"))


def _range(lo: int, hi: int) -> list[str]:
    keys = [f"{k:03d}" for k in range(lo, hi + 1)]
    if missing := [k for k in keys if not (IMAGES / f"{k}.jpg").is_file()]:
        raise SystemExit(f"✗ 找不到影像：{missing}")
    return keys


def _open_upright(key: str) -> Image.Image:
    """不做任何 EXIF 處理地開圖；帶旋轉標記就停——那代表還沒轉正，框會差 90°。"""
    im = Image.open(IMAGES / f"{key}.jpg")
    if im.getexif().get(ORIENTATION, 1) not in (None, 1):
        raise SystemExit(f"✗ {key}.jpg 還帶 EXIF 旋轉標記。先轉正"
                         "（docs/v5.6_說明_人工標註需求.md §7.1）再預標。")
    return im.convert("RGB")


def _read_yolo(p: Path) -> list[tuple[float, float, float, float]] | None:
    """讀 YOLO 標註，回傳 (x0, x1, y0, y1)（0–1）。檔案不存在回 None。"""
    if not p.is_file():
        return None
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        out.append((xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
def cmd_grid(lo: int, hi: int) -> None:
    out = WORK / "grid"
    out.mkdir(parents=True, exist_ok=True)
    f = _font(18)
    for key in _range(lo, hi):
        im = _open_upright(key)
        s = 900 / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
        w, h = im.size
        pad = 28
        canvas = Image.new("RGB", (w + pad + 24, h + pad + 12), "white")
        canvas.paste(im, (pad, pad))
        d = ImageDraw.Draw(canvas, "RGBA")
        for i in range(21):                       # 紅線每 1 格、黃線每 0.5 格
            x, y = pad + w * i / 20, pad + h * i / 20
            col = (255, 0, 0, 150) if i % 2 == 0 else (255, 255, 0, 90)
            d.line([(x, pad), (x, pad + h)], fill=col, width=1)
            d.line([(pad, y), (pad + w, y)], fill=col, width=1)
            if i % 2 == 0:
                d.text((x - 6, 4), str(i // 2), fill="black", font=f)
                d.text((4, y - 9), str(i // 2), fill="black", font=f)
        canvas.save(out / f"{key}.png")
    print(f"✓ 格線圖 {hi - lo + 1} 張 → {out}")
    print("  座標 0–10，左上角 (0, 0)。讀出的 x0,x1,y0,y1 直接填進 boxes.csv")


def _load_csv() -> dict[str, list[tuple[float, float, float, float] | None]]:
    """讀 boxes.csv。座標四個都空 = 無軌跡。任何一列有問題就整批停下並列出。"""
    if not CSV_PATH.is_file():
        raise SystemExit(f"✗ 找不到 {CSV_PATH}")
    rows: dict[str, list] = {}
    errs = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        for n, r in enumerate(csv.DictReader(fh), start=2):
            key = (r.get("image") or "").strip()
            if not key:
                continue
            key = key.removesuffix(".jpg").zfill(3)
            if not (IMAGES / f"{key}.jpg").is_file():
                errs.append(f"第 {n} 列：沒有 {key}.jpg")
                continue
            c = [(r.get(k) or "").strip() for k in ("x0", "x1", "y0", "y1")]
            if not any(c):
                rows.setdefault(key, []).append(None)
                continue
            if not all(c):
                errs.append(f"第 {n} 列（{key}）：座標要嘛四個都填，要嘛都留空（= 無軌跡）")
                continue
            try:
                x0, x1, y0, y1 = map(float, c)
            except ValueError:
                errs.append(f"第 {n} 列（{key}）：座標不是數字 {c}")
                continue
            if not (0 <= x0 < x1 <= 10 and 0 <= y0 < y1 <= 10):
                errs.append(f"第 {n} 列（{key}）：需 0 ≤ x0 < x1 ≤ 10、0 ≤ y0 < y1 ≤ 10，實際 {c}")
                continue
            rows.setdefault(key, []).append((x0, x1, y0, y1))
    for key, boxes in rows.items():
        if None in boxes and any(boxes):
            errs.append(f"{key}：同一張同時寫了「無軌跡」和框")
    if errs:
        raise SystemExit("✗ boxes.csv 有問題，未寫出任何標註：\n  " + "\n  ".join(errs))
    return rows


def cmd_write() -> None:
    rows = _load_csv()
    PRE.mkdir(parents=True, exist_ok=True)
    (PRE / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")
    # 預標註/ 一律與 csv 同步：csv 裡刪掉的，這裡也刪
    for p in PRE.glob("*.txt"):
        if p.name != "classes.txt" and p.stem not in rows:
            p.unlink()
    n_box = n_empty = 0
    for key, boxes in rows.items():
        lines = [f"0 {(x0 + x1) / 20:.6f} {(y0 + y1) / 20:.6f} {(x1 - x0) / 10:.6f} {(y1 - y0) / 10:.6f}"
                 for x0, x1, y0, y1 in (b for b in boxes if b)]
        (PRE / f"{key}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        n_box += len(lines)
        n_empty += not lines
    todo = [k for k in _all_keys() if k not in rows]
    print(f"✓ 預標 {len(rows)} 張、{n_box} 個框、無軌跡 {n_empty} 張 → {PRE}")
    print(f"  尚未預標：{len(todo)} 張" + (f"（{todo[0]}–{todo[-1]}）" if todo else ""))


def cmd_sheet(lo: int, hi: int) -> None:
    keys = _range(lo, hi)
    f = _font(22)
    for i in range(0, len(keys), 10):
        chunk = keys[i:i + 10]
        tiles = []
        for key in chunk:
            im = _open_upright(key)
            W, H = im.size
            d = ImageDraw.Draw(im)
            boxes = _read_yolo(PRE / f"{key}.txt")
            for x0, x1, y0, y1 in boxes or []:
                d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline=(255, 0, 0), width=7)
            tag = key + ("  未預標" if boxes is None else "  無軌跡" if not boxes else "")
            im = im.resize((360, round(360 * H / W)))
            ImageDraw.Draw(im).text((6, 6), tag, fill=(255, 255, 0), font=f)
            tiles.append(im)
        th = max(t.size[1] for t in tiles)
        sheet = Image.new("RGB", (360 * 5, th * 2), "black")
        for j, t in enumerate(tiles):
            sheet.paste(t, ((j % 5) * 360, (j // 5) * th))
        out = WORK / f"sheet_{chunk[0]}-{chunk[-1]}.png"
        sheet.save(out)
        print(f"✓ 核對拼圖 → {out}")


def cmd_seed() -> None:
    DONE.mkdir(parents=True, exist_ok=True)
    pre = [p for p in PRE.glob("*.txt") if p.name != "classes.txt"]
    if not pre:
        raise SystemExit(f"✗ {PRE} 還沒有預標。先跑 write。")
    added = kept = 0
    for p in pre:
        dst = DONE / p.name
        if dst.exists():                         # 已經在審的絕不覆蓋
            kept += 1
            continue
        shutil.copy2(p, dst)
        added += 1
    if not (DONE / "classes.txt").exists():
        shutil.copy2(PRE / "classes.txt", DONE / "classes.txt")
    todo = len(_all_keys()) - len(pre)
    print(f"✓ 補進 {added} 張、已存在未動 {kept} 張 → {DONE}")
    if todo:
        print(f"  還有 {todo} 張沒有預標；預標補完後再跑一次 seed，只會補缺的")


def cmd_diff() -> None:
    report = {"identical": [], "changed": [], "not_reviewed_missing": [], "no_prelabel": [],
              "boxes_added": 0, "boxes_removed": 0, "matched_iou": []}
    for key in _all_keys():
        pre, rev = _read_yolo(PRE / f"{key}.txt"), _read_yolo(DONE / f"{key}.txt")
        if pre is None:
            report["no_prelabel"].append(key)
            continue
        if rev is None:
            report["not_reviewed_missing"].append(key)
            continue
        if (PRE / f"{key}.txt").read_bytes() == (DONE / f"{key}.txt").read_bytes():
            report["identical"].append(key)
            continue
        report["changed"].append(key)
        pairs = sorted(((_iou(a, b), i, j) for i, a in enumerate(pre) for j, b in enumerate(rev)),
                       reverse=True)
        used_p, used_r = set(), set()
        for v, i, j in pairs:
            if v < SAME_OBJECT_IOU or i in used_p or j in used_r:
                continue
            used_p.add(i)
            used_r.add(j)
            report["matched_iou"].append(round(v, 4))
        report["boxes_removed"] += len(pre) - len(used_p)
        report["boxes_added"] += len(rev) - len(used_r)

    # LabelImg 會用它內建的 predefined_classes 覆寫 classes.txt，
    # 連帶把框的類別編號寫成別的數字（實測：001 被寫成 15）。這裡先擋一下。
    cls = DONE / "classes.txt"
    if cls.is_file() and cls.read_text(encoding="utf-8").split() != [CLASS_NAME]:
        print(f"⚠ {cls} 的內容不是單一的 {CLASS_NAME}，很可能被標註軟體覆寫了。")
        print(f"  修正：改回只有一行 {CLASS_NAME}，並確認標註檔的類別編號全部是 0")

    m = report["matched_iou"]
    print("═" * 70)
    print(f"  審過且有修改     {len(report['changed']):>4} 張")
    print(f"  與預標完全相同   {len(report['identical']):>4} 張   ← 可能沒審到，要再看一眼")
    print(f"  成品缺檔         {len(report['not_reviewed_missing']):>4} 張")
    print(f"  沒有預標         {len(report['no_prelabel']):>4} 張")
    print(f"  審圖新增的框     {report['boxes_added']:>4} 個   ← 預標漏掉的")
    print(f"  審圖刪掉的框     {report['boxes_removed']:>4} 個   ← 預標框錯的")
    if m:
        s = sorted(m)
        med = (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2
        print(f"  保留的框 預標 vs 審後 IoU：中位 {med:.3f}（共 {len(m)} 個）")
    print("═" * 70)
    if report["identical"]:
        print("  與預標完全相同：" + "、".join(report["identical"]))
    out = WORK / "diff_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  報告 {out}（ext 臂的 _provenance.json 要附上這份）")


def main() -> None:
    ap = argparse.ArgumentParser(description="B 包的 AI 預標 → 單人審圖輔助")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "sheet"):
        p = sub.add_parser(name)
        p.add_argument("--from", dest="lo", type=int, required=True)
        p.add_argument("--to", dest="hi", type=int, required=True)
    for name in ("write", "seed", "diff"):
        sub.add_parser(name)
    a = ap.parse_args()
    if a.cmd == "grid":
        cmd_grid(a.lo, a.hi)
    elif a.cmd == "sheet":
        cmd_sheet(a.lo, a.hi)
    else:
        {"write": cmd_write, "seed": cmd_seed, "diff": cmd_diff}[a.cmd]()


if __name__ == "__main__":
    main()
