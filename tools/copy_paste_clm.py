# -*- coding: utf-8 -*-
"""潛葉蛾食痕的 copy-paste 合成。

══════════════════════════════════════════════════════════════════════════
【結論：實測不可行，`--arm cp` 不啟用。本檔保留供日後有輪廓標註時再用。】

2026-09-05 產了 30 張送進視覺驗收前，先自己看過——**只有約 10 張堪用**，
遠低於 28/30 的門檻。證據：
`Datasets/Datasets_YOLO26_v5.6/人工標註/C_合成圖_看圖打勾/證據/`

兩種失敗：

  1. **顏色發散成紫色**：`MIXED_CLONE` 逐通道取較強的梯度，當來源貼片與
     目標葉面的亮度差很大時，Poisson 解在三個通道得到不同的 DC 位移，
     結果整塊變成紫／藍紫（總覽圖的 01、03、18、29、30）。
  2. **框裡面是空的**（更嚴重）：融合把貼片洗淡到看不見，只剩一個框畫在
     乾淨的葉面上（04、07、11、17、23、24、26、27）。
     這等於**製造沒有物件的標註**，直接教模型認錯東西。

**根因不是融合方式，是標註形態。** 換過四種做法對照
（`MIXED` / `NORMAL` / `NORMAL`+色彩對齊 / 直接貼+色彩對齊，見證據圖）都救不了：
本專案是**方框**標註，「剪下食痕」實際上剪下的是一個**方形區域**，
裡面除了食痕還有葉緣、樹枝、天空。貼到另一片葉子中間就會出現
「葉子中央長出一段葉緣」這種不可能的畫面；融合程式為了讓邊界自然而洗淡整塊，
洗到最後食痕消失，只剩空框。

要做對需要**食痕本身的輪廓遮罩**（polygon/mask），而不是外接矩形。
取得那個的成本 = 用輪廓重標一次 CLM，那個工時拿去標真實照片顯然更划算。

（順帶一提，ultralytics 內建的 `copy_paste` 超參數對本專案也是 no-op：
`CopyPaste.__call__` 在 `len(labels["instances"].segments) == 0` 時直接 return，
原因同樣是我們沒有輪廓標註。）
══════════════════════════════════════════════════════════════════════════

以下為原始設計說明，保留備查。

**為什麼只對這一類做，而且只貼到健康株影像上**

`Citrus_Leaf_Miner` 的弱點是漏檢（Precision > Recall），而它只有 111 張
train 原圖——場景多樣性是瓶頸。健康株影像（`Healthy/Murcott`、`Healthy/Ponkan`
共 400 張，其中 320 張在 train）與 CLM 完全同域：都是 2048x1536 的田間手持照。
把食痕貼到這些葉面上，CLM 的背景數從 111 擴到 431，而且不需要新拍任何照片。

其他類別不做的理由：
  * `Thrips`：它缺的是**小尺寸樣本**，而縮放增強（`aug_profiles.py` 的
    `Affine(scale=(0.35,1.00))`）能直接控制尺寸，比剪接乾淨得多——
    剪接還要處理接縫，卻換不到縮放給不了的東西。
  * `Thrips_Damage`：它的問題是**框畫不準**（R@0.75 只有 0.29），不是樣本少。
    剪接不會讓框的定義變一致。
  * `Scale_Insect` / `Canker`：框中位僅 27/22px，貼上去的小塊在 JPEG
    壓縮後幾乎只剩雜訊。

**為什麼用 `seamlessClone` 而不是直接貼**

直接貼矩形會在葉面留下亮度/色溫不連續的邊。那條邊對模型來說是一個
比食痕本身更好認的特徵——模型會學會「找矩形接縫」而不是「找食痕」，
在真實照片上完全失效。`cv2.seamlessClone(MIXED_CLONE)` 解 Poisson 方程式，
讓貼上區域的**梯度**與周圍融合，邊界因此消失。

**MIXED_CLONE 的取捨（必須人工驗收的原因）**

`MIXED_CLONE` 會在來源與目標的梯度之間取較強者，所以貼上區域會保留食痕的
紋理、同時吸收目標葉面的光照。代價是**低對比的食痕可能被洗淡**。
這無法用數值判定，只能看圖——所以 `--arm cp` 一定要先過
`人工標註/C_合成圖_看圖打勾/` 的 30 張視覺抽驗（門檻 28/30）。

用法：
    # 產生 QA 包（不建整份資料集，給組員看圖打勾用）
    .venv/Scripts/python.exe tools/copy_paste_clm.py --qa 30
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from build_dataset_v5r import imread_any   # noqa: E402  非 ASCII 路徑安全的讀圖

IMGSZ_REF = 640
JPEG_QUALITY = 92

# 貼上後的目標尺寸，取自真實 CLM 框的分佈（letterbox 到 640 的面積等效邊長）：
# p10 39 / p50 165 / p90 278。刻意避開兩端，貼太小會被壓縮吃掉、貼太大會蓋住整片葉。
PASTE_EQUIV_PX = (60, 260)
MARGIN = 0.08          # 裁切時在框外多留一圈，讓 seamlessClone 有梯度可融
EDGE_KEEP = 0.10       # 貼上位置距離影像邊緣至少要留這個比例，避免 ROI 越界
MAX_TRY = 12



def imwrite_any(path: Path, img: np.ndarray, quality: int = JPEG_QUALITY) -> bool:
    """cv2.imwrite 在 Windows 遇到非 ASCII 路徑會**靜默失敗**（回傳 False 不丟例外）。

    本專案的人工標註工作區路徑帶中文，所以一律改走 imencode + write_bytes。
    """
    ok, buf = cv2.imencode(path.suffix or ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True

def _equiv_px(w: float, h: float, W: int, H: int) -> float:
    sc = IMGSZ_REF / max(W, H)
    return float(np.sqrt(w * W * sc * h * H * sc))


def _crop_patch(img: np.ndarray, box: tuple[float, float, float, float]
                ) -> tuple[np.ndarray, float, float] | None:
    """依 YOLO 相對座標裁出帶邊界的貼片。回傳 (貼片, 框在貼片內的寬佔比, 高佔比)。"""
    H, W = img.shape[:2]
    _, cx, cy, w, h = box
    bw, bh = w * W, h * H
    x0 = int(round((cx - w / 2) * W - bw * MARGIN))
    x1 = int(round((cx + w / 2) * W + bw * MARGIN))
    y0 = int(round((cy - h / 2) * H - bh * MARGIN))
    y1 = int(round((cy + h / 2) * H + bh * MARGIN))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    patch = img[y0:y1, x0:x1].copy()
    # 框在貼片座標系裡的相對大小（貼片可能因觸邊而被裁掉一部分邊界）
    fw = bw / (x1 - x0)
    fh = bh / (y1 - y0)
    return patch, min(fw, 1.0), min(fh, 1.0)


def _paste_one(target: np.ndarray, patch: np.ndarray, fw: float, fh: float,
               rng: random.Random) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """把 patch 融進 target，回傳 (新影像, 貼上後的 YOLO 框)。"""
    TH, TW = target.shape[:2]
    want = rng.uniform(*PASTE_EQUIV_PX)               # 目標等效邊長（640 尺度）
    scale_to_640 = IMGSZ_REF / max(TW, TH)
    # 貼片裡「框」的面積等效邊長要等於 want
    ph, pw = patch.shape[:2]
    cur = np.sqrt(pw * fw * scale_to_640 * ph * fh * scale_to_640)
    if cur <= 1e-6:
        return None
    k = want / cur
    nw, nh = int(round(pw * k)), int(round(ph * k))
    if nw < 8 or nh < 8 or nw > TW * 0.8 or nh > TH * 0.8:
        return None
    patch = cv2.resize(patch, (nw, nh),
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        patch = cv2.flip(patch, 1)
    if rng.random() < 0.5:
        rot = rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                          cv2.ROTATE_90_COUNTERCLOCKWISE])
        patch = cv2.rotate(patch, rot)
        nh, nw = patch.shape[:2]
        if rot != cv2.ROTATE_180:
            fw, fh = fh, fw          # 轉 90 度，貼片的長寬互換，框的佔比跟著換

    # 遮罩：整片為 255，只留 1px 黑邊（seamlessClone 要求非零區嚴格在內部）
    mask = np.full((nh, nw), 255, np.uint8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

    lo_x, hi_x = int(TW * EDGE_KEEP) + nw // 2, int(TW * (1 - EDGE_KEEP)) - nw // 2
    lo_y, hi_y = int(TH * EDGE_KEEP) + nh // 2, int(TH * (1 - EDGE_KEEP)) - nh // 2
    if lo_x >= hi_x or lo_y >= hi_y:
        return None
    cx_px, cy_px = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
    try:
        out = cv2.seamlessClone(patch, target, mask, (cx_px, cy_px), cv2.MIXED_CLONE)
    except cv2.error:
        return None

    bw_px, bh_px = nw * fw, nh * fh
    return out, (cx_px / TW, cy_px / TH, bw_px / TW, bh_px / TH)


def synthesize(sources: list[dict], targets: list[dict], n: int,
               out_img_dir: Path, out_lbl_dir: Path, prefix: str,
               seed: int = 0, prov: list | None = None,
               cid: int = 5) -> int:
    """產生 n 張合成圖。`sources`/`targets` 是 build script 的 item dict。

    **`targets` 必須全部來自 train**——貼到評估集的背景上會讓模型在訓練時
    看過評估影像的畫面，構成洩漏（驗收 Gate 3 會查）。
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    src_with_box = [s for s in sources if s["boxes"]]
    if not src_with_box or not targets:
        return 0

    rng = random.Random(seed)
    made = guard = 0
    while made < n and guard < n * MAX_TRY:
        guard += 1
        s = src_with_box[rng.randrange(len(src_with_box))]
        t = targets[rng.randrange(len(targets))]
        simg = imread_any(s["img"])
        timg = imread_any(t["img"])
        if simg is None or timg is None:
            continue
        cropped = _crop_patch(simg, s["boxes"][rng.randrange(len(s["boxes"]))])
        if cropped is None:
            continue
        patch, fw, fh = cropped
        res = _paste_one(timg, patch, fw, fh, rng)
        if res is None:
            continue
        img, (cx, cy, bw, bh) = res
        # 再貼第二個食痕（真實 CLM 平均每張 1.4 個框）
        boxes = [(cid, cx, cy, bw, bh)]
        if rng.random() < 0.4:
            s2 = src_with_box[rng.randrange(len(src_with_box))]
            simg2 = imread_any(s2["img"])
            c2 = _crop_patch(simg2, s2["boxes"][rng.randrange(len(s2["boxes"]))]) \
                if simg2 is not None else None
            if c2 is not None:
                r2 = _paste_one(img, c2[0], c2[1], c2[2], rng)
                if r2 is not None:
                    img, b2 = r2
                    boxes.append((cid, *b2))

        stem = f"{prefix}_{made:05d}"
        imwrite_any(out_img_dir / f"{stem}.jpg", img)
        (out_lbl_dir / f"{stem}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, a, b, w, h in boxes),
            encoding="utf-8")
        if prov is not None:
            import hashlib

            def _md5(p: Path) -> str:
                hh = hashlib.md5()
                with open(p, "rb") as f:
                    for c in iter(lambda: f.read(1 << 20), b""):
                        hh.update(c)
                return hh.hexdigest()

            prov.append(dict(out=f"train/images/{stem}.jpg", kind="cp", cls="Citrus_Leaf_Miner",
                             src=str(s["img"].name), src_md5=s.get("md5") or _md5(s["img"]),
                             paste=str(t["img"].name), paste_md5=t.get("md5") or _md5(t["img"]),
                             v55_split=s.get("v55", "?"), split="train"))
        made += 1
    return made


# ══════════════════════════════════════════════════════════════════════
# QA 包：只產合成圖與編號對照，不建整份資料集
# ══════════════════════════════════════════════════════════════════════

def _draw_box(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for _c, cx, cy, w, h in boxes:
        x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), max(2, W // 400))
    return out


def make_qa_package(n: int, out_dir: Path) -> int:
    """給 `人工標註/C_合成圖_看圖打勾/待檢查/` 用的編號合成圖。"""
    import build_dataset_v5_6 as B

    v55map = B.v55_split_map()
    clm, _ = B.collect(next(s for s in B.SOURCES if s["name"] == "Citrus_Leaf_Miner"))
    clm_pool, _ = B.nested_split("Citrus_Leaf_Miner", clm, v55map)
    bg = []
    for spec in B.BACKGROUND_SOURCES:
        bg += B.collect_background(spec)
    bg_pool, _ = B.nested_split("Background", bg, v55map)

    tmp_i, tmp_l = out_dir / "_tmp_img", out_dir / "_tmp_lbl"
    made = synthesize(clm_pool["train"], bg_pool["train"], n, tmp_i, tmp_l,
                      prefix="qa", seed=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(made):
        img = cv2.imdecode(np.frombuffer(
            (tmp_i / f"qa_{k:05d}.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        rows = (tmp_l / f"qa_{k:05d}.txt").read_text(encoding="utf-8").splitlines()
        boxes = [tuple(float(v) for v in r.split()) for r in rows if r.strip()]
        vis = _draw_box(img, boxes)
        h, w = vis.shape[:2]
        k2 = 1280 / max(h, w)
        vis = cv2.resize(vis, (int(w * k2), int(h * k2)), interpolation=cv2.INTER_AREA)
        cv2.putText(vis, f"{k + 1:02d}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 0, 0), 8)
        cv2.putText(vis, f"{k + 1:02d}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 220, 255), 3)
        imwrite_any(out_dir / f"{k + 1:02d}.jpg", vis, quality=90)
    for d in (tmp_i, tmp_l):
        for p in d.glob("*"):
            p.unlink()
        d.rmdir()
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=int, default=30, help="產生幾張供人工視覺驗收")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import build_dataset_v5_6 as B
    out = Path(args.out) if args.out else (
        B.MANUAL_ROOT / "C_合成圖_看圖打勾" / "待檢查")
    n = make_qa_package(args.qa, out)
    print(f"產生 {n} 張合成圖（已畫上標註框、左上角有編號） -> {out}")
    print("請組員照 檢查表.md 逐張打勾；28/30 以上才可以啟用 --arm cp。")


if __name__ == "__main__":
    main()
