# -*- coding: utf-8 -*-
"""產生人工標註工作包裡的「正例／反例對照圖」。

標註約定用文字講不清楚。這支腳本從**既有的標註**渲染出對照圖：
綠框是照約定該畫的樣子，紅框是常見的錯誤畫法（由既有框依規則變形而來，
所以每一張反例都對應一張真實影像，不是憑空畫的示意圖）。

產出：
    Datasets/Datasets_YOLO26_v5.6/人工標註/A_薊馬葉害_框怎麼畫/範例圖/
    Datasets/Datasets_YOLO26_v5.6/人工標註/B_潛葉蛾_補標150張/範例圖/

用法：
    .venv/Scripts/python.exe tools/make_annotation_guide.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_dataset_v5_6 as B

FONT_PATH = Path("C:/Windows/Fonts/msjh.ttc")     # 微軟正黑體，Windows 內建
LONG_SIDE = 1100
GREEN = (60, 200, 90)
RED = (235, 70, 70)
BLACK = (25, 25, 25)


def _font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def _open(p: Path) -> Image.Image:
    im = Image.open(p).convert("RGB")
    k = LONG_SIDE / max(im.size)
    return im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)


def _draw(im: Image.Image, boxes, color, width_k: float = 1.0) -> None:
    d = ImageDraw.Draw(im)
    W, H = im.size
    w = max(3, int(W / 260 * width_k))
    for cx, cy, bw, bh in boxes:
        d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H,
                     (cx + bw / 2) * W, (cy + bh / 2) * H], outline=color, width=w)


def _caption(im: Image.Image, title: str, note: str, color) -> Image.Image:
    """在圖片上方加一條說明帶。"""
    pad, fs = 16, 30
    ft, fn = _font(fs), _font(int(fs * 0.75))
    lines = [note[i:i + 34] for i in range(0, len(note), 34)] or [""]
    bar = fs + pad * 2 + len(lines) * int(fs * 0.95)
    out = Image.new("RGB", (im.width, im.height + bar), (255, 255, 255))
    out.paste(im, (0, bar))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, im.width, bar], fill=(248, 248, 248))
    d.rectangle([0, 0, 10, bar], fill=color)
    d.text((pad + 12, pad - 4), title, font=ft, fill=color)
    for i, ln in enumerate(lines):
        d.text((pad + 12, pad + fs + 2 + i * int(fs * 0.95)), ln, font=fn, fill=BLACK)
    return out


def _boxes_of(item: dict) -> list[tuple[float, float, float, float]]:
    return [(cx, cy, w, h) for _c, cx, cy, w, h in item["boxes"]]


def _scale_box(b, k: float):
    cx, cy, w, h = b
    w, h = min(w * k, 0.999), min(h * k, 0.999)
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return (cx, cy, w, h)


def _union(bs):
    x0 = min(cx - w / 2 for cx, cy, w, h in bs)
    x1 = max(cx + w / 2 for cx, cy, w, h in bs)
    y0 = min(cy - h / 2 for cx, cy, w, h in bs)
    y1 = max(cy + h / 2 for cx, cy, w, h in bs)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)



def _max_overlap(boxes) -> float:
    """一組框裡兩兩交集面積佔較小框的最大比例。0 = 完全不重疊。"""
    best = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[i], boxes[j]
            ix = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
            iy = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
            best = max(best, ix * iy / max(min(aw * ah, bw * bh), 1e-9))
    return best

def _save(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)


# ══════════════════════════════════════════════════════════════════════

def build_a(out_dir: Path) -> int:
    """薊馬葉害：示範 3 張 + 反例 3 張。

    **選圖刻意避開框最大的那些。** 這個工作包的前提就是「既有的框不一致」，
    所以不能把既有標註當成標準答案展示——尤其框佔葉面很大的那幾張，
    本來就是最可疑的。改取框面積分佈的中段（框相對貼合的那些），
    而且標題寫「現在的畫法」而不是「這樣畫是對的」，避免循環論證。
    真正在教的是三條規則，反例才是重點。
    """
    items, _ = B.collect(next(s for s in B.SOURCES if s["name"] == "Thrips_Damage"))
    single = sorted([it for it in items if len(it["boxes"]) == 1],
                    key=lambda it: it["boxes"][0][3] * it["boxes"][0][4])
    multi = sorted([it for it in items if 2 <= len(it["boxes"]) <= 3],
                   key=lambda it: _max_overlap(_boxes_of(it)))
    n = 0

    mid = len(single) // 2
    picks = [single[mid], single[max(0, mid - len(single) // 4)]]
    picks += multi[:1]
    for i, it in enumerate(picks, 1):
        im = _open(it["img"])
        _draw(im, _boxes_of(it), GREEN)
        nb = len(it["boxes"])
        note = ("一片連續的銀白食痕 = 一個框，框貼著食痕外緣，不要留白邊。"
                if nb == 1 else
                f"這張圖上有 {nb} 處分開的食痕，中間隔了健康葉肉，所以畫成 {nb} 個框。")
        _save(_caption(im, f"示範 {i}：現在的畫法", note, GREEN),
              out_dir / f"示範{i}.jpg")
        n += 1

    it = single[mid]
    im = _open(it["img"])
    _draw(im, [_scale_box(b, 1.35) for b in _boxes_of(it)], RED)
    _draw(im, _boxes_of(it), GREEN)
    _save(_caption(im, "反例 1：框太鬆", "紅框把周圍健康的葉肉也框進去了。"
                                        "綠框是比較基準——請像綠框那樣貼著食痕的外緣畫。", RED),
          out_dir / "反例1_框太鬆.jpg")
    n += 1

    im = _open(it["img"])
    _draw(im, [_scale_box(b, 0.7) for b in _boxes_of(it)], RED)
    _draw(im, _boxes_of(it), GREEN)
    _save(_caption(im, "反例 2：框太緊", "紅框切掉了食痕的邊緣。"
                                        "食痕的最外側也要被框住，不要只框中間最白的部分。", RED),
          out_dir / "反例2_框太緊.jpg")
    n += 1

    if multi:
        it2 = multi[0]   # 已依框之間的重疊度排序，取分得最開的那張
        im = _open(it2["img"])
        _draw(im, [_union(_boxes_of(it2))], RED)
        _draw(im, _boxes_of(it2), GREEN)
        _save(_caption(im, "反例 3：該分開卻合併成一個大框",
                       "兩處食痕中間隔了一大片健康葉肉，就要分成兩個框（綠框）。"
                       "合併成一個紅框會把健康的部分也算進去。", RED),
              out_dir / "反例3_該分開卻合併.jpg")
        n += 1
    return n


def build_b(out_dir: Path) -> int:
    """潛葉蛾：正例 6 張 + 反例 3 張。"""
    items, _ = B.collect(next(s for s in B.SOURCES if s["name"] == "Citrus_Leaf_Miner"))
    ok = [it for it in items if it["boxes"]]
    # 說明文字一律由**影像實際內容**推導，不能按位置硬派——
    # 圖上明明只有一個框、字卻寫「兩個框」，對組員是負面效果。
    single = sorted([it for it in ok if len(it["boxes"]) == 1],
                    key=lambda it: -it["boxes"][0][3] * it["boxes"][0][4])
    # 多框範例挑 2-3 個框、且**框之間幾乎不重疊**的：教「一條軌跡一個框」
    # 要讓人一眼看出哪個框對應哪條軌跡。十幾個框擠在一起、或框框互相疊住，
    # 都會讓範例失去教學效果。
    multi = sorted([it for it in ok if 2 <= len(it["boxes"]) <= 3],
                   key=lambda it: (_max_overlap(_boxes_of(it)),
                                   -sum(b[3] * b[4] for b in it["boxes"])))
    picks = []
    for idx in (0, len(single) // 3, len(single) * 2 // 3, len(single) - 2):
        if 0 <= idx < len(single) and single[idx] not in picks:
            picks.append(single[idx])
    picks += multi[:2]

    def note_for(it: dict) -> str:
        nb = len(it["boxes"])
        if nb >= 2:
            return (f"這片葉子上有 {nb} 條分開的食痕，所以畫 {nb} 個框——"
                    f"一條軌跡一個框，不要合併成一個大框。")
        w, h = it["boxes"][0][3], it["boxes"][0][4]
        area = w * h
        if area < 0.02:
            return "食痕比較小、顏色比較淡的也要標，只要看得出是蜿蜒的痕跡就標。"
        if max(w, h) / max(min(w, h), 1e-6) > 2.0:
            return "軌跡是細長的，框住它的最外圍就好，不要沿著彎曲的線去描。"
        return "整條蜿蜒的食痕框成一個框，從頭框到尾，框貼著痕跡的最外緣。"

    n = 0
    for i, it in enumerate(picks, 1):
        im = _open(it["img"])
        _draw(im, _boxes_of(it), GREEN)
        _save(_caption(im, f"正例 {i}：這樣畫是對的", note_for(it), GREEN),
              out_dir / f"正例{i}.jpg")
        n += 1

    it = ok[len(ok) // 4]
    im = _open(it["img"])
    _draw(im, [(0.5, 0.5, 0.96, 0.96)], RED)
    _draw(im, _boxes_of(it), GREEN)
    _save(_caption(im, "反例 1：框了整片葉子", "我們要的是食痕（綠框），不是葉子。"
                                              "框整片葉子會讓模型學錯東西。", RED),
          out_dir / "反例1_框了整片葉子.jpg")
    n += 1

    im = _open(it["img"])
    _draw(im, [_scale_box(b, 1.4) for b in _boxes_of(it)], RED)
    _draw(im, _boxes_of(it), GREEN)
    _save(_caption(im, "反例 2：框太鬆", "紅框外圍留了太多空白。請貼著食痕的最外緣畫。", RED),
          out_dir / "反例2_框太鬆.jpg")
    n += 1

    im = _open(it["img"])
    _draw(im, [_scale_box(b, 0.55) for b in _boxes_of(it)], RED)
    _draw(im, _boxes_of(it), GREEN)
    _save(_caption(im, "反例 3：只框了一半", "軌跡的頭尾都要包進去。"
                                            "紅框只框到中間一段，漏掉了兩端。", RED),
          out_dir / "反例3_只框一半.jpg")
    n += 1
    return n


def main() -> None:
    root = B.MANUAL_ROOT
    print("═" * 70)
    print("  產生標註範例圖")
    print("═" * 70)
    na = build_a(root / "A_薊馬葉害_框怎麼畫" / "範例圖")
    print(f"  A 薊馬葉害   {na} 張  ->  A_薊馬葉害_框怎麼畫/範例圖/")
    nb = build_b(root / "B_潛葉蛾_補標150張" / "範例圖")
    print(f"  B 潛葉蛾     {nb} 張  ->  B_潛葉蛾_補標150張/範例圖/")
    print("\n（C 的對照圖由 tools/copy_paste_clm.py --qa 產生）")


if __name__ == "__main__":
    main()
