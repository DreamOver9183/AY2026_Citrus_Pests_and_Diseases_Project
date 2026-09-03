"""P3 定位精度診斷：把 AP50-95 的損失拆成「找不到」與「框不準」，並歸因到子域與尺寸帶。

背景：Thrips / Aphid 的 AP50-95 / AP50 僅 0.55，是八類最低；但它們的框中位數
（198.8 / 165.2 px @640）遠大於 Scale_Insect（27.0 px，比值 0.70）。
「小目標」解釋不了這個排序，本工具用來檢定真正的成因。

四個量測：

  1. provenance 還原   valid/test 不做增強，`build_dataset_v5r.write_sample()` 是位元複製，
                       因此 md5 可精確對回來源影像，取得每個框的來源子類
                       （Thysanoptera / thirps_leaf_damage、Aphid / Aphid_Leaf_Damage）。
  2. 分層 AP           以「其他子域 = ignore」的方式計算子域 AP50 / AP50-95，
                       避開背景 FP 無法歸屬子域的問題（COCO crowd 的標準作法）。
  3. 殘差分解          配對成功的框，把 IoU 損失拆成中心偏移與尺寸誤差，
                       並用兩個 oracle（把中心換成 GT／把尺寸換成 GT）量各自的貢獻。
  4. 尺寸帶分層        <32 / 32-96 / 96-256 / >256 px，逐帶算 Recall@0.5、Recall@0.75、
                       中位 IoU —— 回答「<32px 的 Scale_Insect / Canker 到底輸在哪」。

判讀規則（先寫死，免得事後各說各話）：

  * Recall@0.5 高但 Recall@0.75 低        → 找得到、框不準 → 定位問題
  * 殘差有系統性偏移（bias_w 明顯非 0）   → 框定義不一致 → 標註端可修
  * 殘差無偏移但散布大（spread_w 大）     → 標註噪聲 → 已達上限，模型端無事可做
  * oracle_size < oracle_center           → 尺寸畫不準（邊界約定）而非位置畫不準

用法：
    .venv/Scripts/python.exe tools/diag_localization.py
    .venv/Scripts/python.exe tools/diag_localization.py --split test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "Datasets" / "Datasets_YOLO26_v5r"
DATA_ROOT = SRC_ROOT / "OutPut"
WEIGHTS = ROOT / "Train Code/v9/Train_output/Phase4/best.pt"
OUT_DIR = ROOT / "Train Code/v9/Train_output/Phase4/P3_diag"

NAMES = ["Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
         "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid"]
IMGSZ = 640
IOU_THRS = np.round(np.arange(0.50, 0.96, 0.05), 2)
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 合併類 -> (來源目錄, 子類名稱依來源 label 的 class id 排列)。
# v5.5 起 Thrips 已在資料集層級拆成兩類，屆時這一項會由 `configure()` 自動移除
# ——已經拆好的類別不需要（也無法）再做 provenance 還原。
MERGED_ALL = {
    6: ("Pests/Thrips_v5r.yolo26/train", ["Insect", "Damage"]),
    7: ("Pests/Aphid.yolo26/train", ["Insect", "Damage"]),
}
MERGED = dict(MERGED_ALL)
SIZE_BINS = [(0, 32, "<32px"), (32, 96, "32-96px"), (96, 256, "96-256px"), (256, 1e9, ">256px")]


def configure(root: Path) -> None:
    """依資料集自己的 data.yaml 設定類別清單與需要 provenance 還原的合併類。

    v5r 是 8 類、Thrips 與 Aphid 各自合併了蟲體與葉害兩個子域，需要靠 md5 還原子域；
    v5.5 起 Thrips 已拆成獨立類別（`Thrips_Damage`），該類就不再需要還原。
    只做最小 yaml 解析，維持本檔不依賴 pyyaml。
    """
    global DATA_ROOT, NAMES, MERGED
    DATA_ROOT = root
    p = root / "data.yaml"
    if p.is_file():
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
        if names:
            NAMES = names
    MERGED = {cid: v for cid, v in MERGED_ALL.items()
              if cid < len(NAMES) and f"{NAMES[cid]}_Damage" not in NAMES}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """讀 YOLO 標註，多邊形轉外接矩形——與 `build_dataset_v5r.parse_label()` 同行為。

    來源的 `Aphid_Leaf_Damage` 全部是多邊形標註（13–41 欄），若照 5 欄格式硬取前四個值，
    provenance 會全數對不上（實測 98 個框無法歸屬子域）。
    """
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        vals = [float(v) for v in p[1:]]
        if len(vals) > 4:                                   # polygon
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        else:
            box = tuple(vals[:4])
        out.append((int(float(p[0])), *box))
    return out


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXT:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clip01(cx: float, cy: float, w: float, h: float) -> tuple | None:
    """複製 `build_dataset_v5r.parse_label()` 的夾取行為，否則溢出框的座標對不上。"""
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def build_provenance() -> dict[str, dict[tuple, int]]:
    """md5(來源影像) -> {(cx,cy,w,h) 四捨五入到 6 位: 來源子類 id}。"""
    idx: dict[str, dict[tuple, int]] = {}
    for _cid, (rel, _sub) in MERGED.items():
        base = SRC_ROOT / rel
        for lbl in sorted((base / "labels").glob("*.txt")):
            img = find_image(base / "images", lbl.stem)
            if img is None:
                continue
            m = {}
            for c, *box in parse_label(lbl):
                cb = clip01(*box)
                if cb is not None:
                    m[tuple(round(v, 6) for v in cb)] = c
            idx[md5(img)] = m
    return idx


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) -> (N,M)"""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ara = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    arb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(ara[:, None] + arb[None, :] - inter, 1e-9, None)


def coco_ap(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    """101 點內插 AP。scores / tps 等長，tps 為 0/1。"""
    if n_gt == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    o = np.argsort(-scores)
    tp = np.cumsum(tps[o])
    fp = np.cumsum(1 - tps[o])
    rec = tp / n_gt
    prec = tp / np.clip(tp + fp, 1e-9, None)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.interp(np.linspace(0, 1, 101), rec, prec, left=prec[0], right=0).mean())


def collect(split: str, prov: dict) -> list[dict]:
    """讀出一個 split 的 GT（正規化座標）並附上子域標籤。

    這裡**不**換算像素座標：部分田間影像帶 EXIF 方向旗標，PIL 的 `.size` 回傳檔案原始
    長寬，而 ultralytics 走 cv2 讀圖會套用 EXIF 旋轉——兩者長寬相反，直接用 PIL 的尺寸
    換算會讓 GT 與預測不在同一個座標系（實測 Oily_Spot 的 recall 因此虛低 29 個百分點）。
    像素座標一律等推論拿到 `r.orig_shape` 之後再算，見 `attach_pixels()`。
    """
    items, hit, miss = [], 0, 0
    # 檔名前綴即輸出類別名（`{類別}_{序號}` / `{類別}_aug_{序號}`），只有仍需還原的合併類
    # 要算 md5。v5.5 拆類後 Thrips 不在 MERGED 裡，就不會白算一次雜湊。
    merged_prefixes = tuple(NAMES[c] + "_" for c in MERGED)
    for lbl in sorted((DATA_ROOT / split / "labels").glob("*.txt")):
        img = find_image(DATA_ROOT / split / "images", lbl.stem)
        if img is None:
            continue
        sub = prov.get(md5(img)) if lbl.stem.startswith(merged_prefixes) else None
        boxes = []
        for c, cx, cy, w, h in parse_label(lbl):
            sid = sub.get(tuple(round(v, 6) for v in (cx, cy, w, h))) if sub else None
            if c in MERGED:
                hit += sid is not None
                miss += sid is None
            boxes.append(dict(
                cls=c, nb=(cx, cy, w, h),
                stratum=(f"{NAMES[c]}/{MERGED[c][1][sid]}"
                         if c in MERGED and sid is not None else NAMES[c]),
            ))
        items.append(dict(img=img, boxes=boxes))
    if hit + miss:
        print(f"  provenance 還原：{hit}/{hit + miss} 框（未命中 {miss}）")
    return items


def attach_pixels(it: dict, orig_shape: tuple[int, int]) -> None:
    """用 ultralytics 實際讀到的影像尺寸把 GT 換算成像素座標與 640 等效邊長。"""
    H, W = orig_shape
    s = IMGSZ / max(W, H)
    it["W"], it["H"] = W, H
    for b in it["boxes"]:
        cx, cy, w, h = b["nb"]
        b["xyxy"] = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        b["px"] = float(np.sqrt((w * W * s) * (h * H * s)))


def match_stratum(items: list[dict], st: str, cid: int, t: float, want_resid: bool):
    """單一 IoU 閾值下的配對。其他子域的 GT 視為 ignore。

    回傳 (scores, tps, resid, per_img)；`per_img` 是逐影像的 (scores, tps, n_gt)，
    供 bootstrap 以影像為單位重抽樣——子域的 n 很小（Thrips/Damage 只有 20 框），
    沒有變異量就沒有判準，而重訓量 σ 太貴。
    """
    sc, tp, resid, per_img = [], [], defaultdict(list), []
    for img_i, it in enumerate(items):
        gp = [b for b in it["boxes"] if b["stratum"] == st]
        gi = [b for b in it["boxes"] if b["cls"] == cid and b["stratum"] != st]
        P = it["pred"]
        m = P["cls"] == cid
        px, pc = P["xyxy"][m], P["conf"][m]
        o = np.argsort(-pc)
        px, pc = px[o], pc[o]
        Ip = iou_xyxy(px, np.array([g["xyxy"] for g in gp])) if gp else None
        Ii = iou_xyxy(px, np.array([g["xyxy"] for g in gi])) if gi else None
        used: set[int] = set()
        isc, itp = [], []
        for k in range(len(px)):
            j, best = -1, t
            if Ip is not None:
                for q in range(len(gp)):
                    if q not in used and Ip[k, q] >= best:
                        j, best = q, Ip[k, q]
            if j >= 0:
                used.add(j)
                isc.append(pc[k])
                itp.append(1)
                if want_resid:
                    g, p = gp[j], px[k]
                    gx0, gy0, gx1, gy1 = g["xyxy"]
                    gw, gh = gx1 - gx0, gy1 - gy0
                    pw, ph = p[2] - p[0], p[3] - p[1]
                    gcx, gcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                    pcx, pcy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                    s = float(np.sqrt(gw * gh))
                    gt1 = np.array([g["xyxy"]])
                    oc = np.array([[gcx - pw / 2, gcy - ph / 2, gcx + pw / 2, gcy + ph / 2]])
                    os_ = np.array([[pcx - gw / 2, pcy - gh / 2, pcx + gw / 2, pcy + gh / 2]])
                    resid["iou"].append(float(Ip[k, j]))
                    resid["oracle_center"].append(float(iou_xyxy(oc, gt1)[0, 0]))
                    resid["oracle_size"].append(float(iou_xyxy(os_, gt1)[0, 0]))
                    resid["dcx"].append((pcx - gcx) / s)
                    resid["dcy"].append((pcy - gcy) / s)
                    resid["lw"].append(float(np.log2(pw / gw)))
                    resid["lh"].append(float(np.log2(ph / gh)))
                    resid["px"].append(g["px"])
                    resid["img"].append(img_i)
            elif Ii is not None and len(gi) and Ii[k].max() >= t:
                pass                                    # 命中其他子域 → ignore，不計 FP
            else:
                isc.append(pc[k])
                itp.append(0)
        sc.extend(isc)
        tp.extend(itp)
        per_img.append((np.array(isc), np.array(itp, dtype=float), len(gp)))
    return np.array(sc), np.array(tp, dtype=float), resid, per_img


def bootstrap_se(per_img_by_t: dict[float, list], n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """以影像為單位重抽樣，回傳 AP50-95 的 (SE, CI 下界, CI 上界)。

    量的是「評估集抽樣」的不確定度，**不是** run 間變異——後者只能靠重訓量，
    通常更大。判準因此要求「超過 2×SE 且 valid / test 同號」，見 §3.1。
    """
    rng = np.random.default_rng(seed)
    thrs = sorted(per_img_by_t)
    n = len(per_img_by_t[thrs[0]])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aps = []
        for t in thrs:
            pi = per_img_by_t[t]
            n_gt = sum(pi[i][2] for i in idx)
            if not n_gt:
                continue
            sc = np.concatenate([pi[i][0] for i in idx]) if n else np.array([])
            tp = np.concatenate([pi[i][1] for i in idx]) if n else np.array([])
            aps.append(coco_ap(sc, tp, n_gt))
        if aps:
            vals.append(float(np.mean(aps)))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.array(vals)
    return float(v.std(ddof=1)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap_box_stats(resid: dict, per_img75: list, n_img: int, n_boot: int,
                        seed: int = 0) -> dict[str, float]:
    """medIoU 與 R@0.75 的 bootstrap SE（同樣以影像為重抽樣單位）。

    這兩個是逐框統計，解析度遠高於子域 AP（AP 還受 PR 曲線形狀影響），
    子域樣本少的時候應該用它們當主判準，見 §3.1。
    """
    if not resid.get("iou"):
        return {}
    rng = np.random.default_rng(seed)
    by_img: dict[int, list[float]] = defaultdict(list)
    for i, v in zip(resid["img"], resid["iou"]):
        by_img[i].append(v)
    med, r75 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        vals = [v for i in idx for v in by_img.get(i, ())]
        if vals:
            med.append(float(np.median(vals)))
        n_gt = sum(per_img75[i][2] for i in idx)
        if n_gt:
            r75.append(sum(per_img75[i][1].sum() for i in idx) / n_gt)
    out = {}
    if med:
        out["med_iou_se"] = float(np.std(med, ddof=1))
    if r75:
        out["recall75_se"] = float(np.std(r75, ddof=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid,test")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(DATA_ROOT),
                    help="資料集根目錄（含 train/valid/test 與 data.yaml）")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="以影像為單位重抽樣的次數（建議 500）；0 = 不做")
    args = ap.parse_args()

    configure(Path(args.data))
    print(f"資料集：{DATA_ROOT}\n類別（{len(NAMES)}）：{', '.join(NAMES)}")
    print(f"需 provenance 還原的合併類：{[NAMES[c] for c in MERGED] or '無'}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("建立 provenance 索引 …")
    prov = build_provenance()

    for split in args.split.split(","):
        print(f"\n{'=' * 96}\n{split}\n{'=' * 96}")
        items = collect(split, prov)

        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            res = model.predict([str(it["img"]) for it in chunk], imgsz=IMGSZ, conf=0.001,
                                iou=0.7, max_det=300, verbose=False)
            for it, r in zip(chunk, res):
                b = r.boxes
                it["pred"] = dict(xyxy=b.xyxy.cpu().numpy(),
                                  conf=b.conf.cpu().numpy(),
                                  cls=b.cls.cpu().numpy().astype(int))
                attach_pixels(it, r.orig_shape)
            print(f"\r  推論 {min(i + 8, len(items))}/{len(items)}", end="")
        print()

        strata = sorted({b["stratum"] for it in items for b in it["boxes"]})
        rows, resid_rows = [], []

        for st in strata:
            cid = NAMES.index(st.split("/")[0])
            n_gt = sum(sum(b["stratum"] == st for b in it["boxes"]) for it in items)
            aps, rec, resid, per_t = {}, {}, None, {}
            for t in IOU_THRS:
                sc, tp, rd, pi = match_stratum(items, st, cid, float(t), want_resid=(t == 0.5))
                aps[float(t)] = coco_ap(sc, tp, n_gt)
                rec[float(t)] = float(tp.sum()) / n_gt if n_gt else float("nan")
                per_t[float(t)] = pi
                if t == 0.5:
                    resid = rd

            r = dict(split=split, stratum=st, n_gt=n_gt,
                     ap50=aps[0.5], ap75=aps[0.75],
                     ap50_95=float(np.nanmean(list(aps.values()))),
                     recall50=rec[0.5], recall75=rec[0.75],
                     med_px=float(np.median([b["px"] for it in items
                                             for b in it["boxes"] if b["stratum"] == st])))
            r["ratio"] = r["ap50_95"] / r["ap50"] if r["ap50"] else float("nan")
            if args.bootstrap:
                se, lo, hi = bootstrap_se(per_t, args.bootstrap)
                r.update(boot_se=se, ci_lo=lo, ci_hi=hi)
                r.update(bootstrap_box_stats(resid, per_t[0.75], len(items), args.bootstrap))
            if resid and resid["iou"]:
                r.update(med_iou=float(np.median(resid["iou"])),
                         oracle_center=float(np.median(resid["oracle_center"])),
                         oracle_size=float(np.median(resid["oracle_size"])),
                         bias_w=float(np.median(resid["lw"])),
                         bias_h=float(np.median(resid["lh"])),
                         spread_w=float(np.percentile(np.abs(resid["lw"]), 75)),
                         off_center=float(np.median(np.hypot(resid["dcx"], resid["dcy"]))))
                for k in range(len(resid["iou"])):
                    resid_rows.append(dict(split=split, stratum=st, px=resid["px"][k],
                                           iou=resid["iou"][k], lw=resid["lw"][k],
                                           lh=resid["lh"][k], dcx=resid["dcx"][k],
                                           dcy=resid["dcy"][k]))
            rows.append(r)

        band_rows = []
        for cid, nm in enumerate(NAMES):
            for lo, hi, tag in SIZE_BINS:
                gts = [b for it in items for b in it["boxes"]
                       if b["cls"] == cid and lo <= b["px"] < hi]
                if len(gts) < 10:
                    continue
                sub = [x for x in resid_rows
                       if x["stratum"].split("/")[0] == nm and lo <= x["px"] < hi]
                lw = np.abs([x["lw"] for x in sub]) if sub else np.array([np.nan])
                band_rows.append(dict(
                    split=split, cls=nm, band=tag, n_gt=len(gts),
                    recall50=sum(x["iou"] >= 0.5 for x in sub) / len(gts),
                    recall75=sum(x["iou"] >= 0.75 for x in sub) / len(gts),
                    med_iou=float(np.median([x["iou"] for x in sub])) if sub else float("nan"),
                    bias_w=float(np.median([x["lw"] for x in sub])) if sub else float("nan"),
                    spread_w=float(np.percentile(lw, 75))))

        for name, data in (("strata", rows), ("bands", band_rows), ("residuals", resid_rows)):
            if not data:
                continue
            keys = sorted({k for d in data for k in d})
            with open(out_dir / f"{split}_{name}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)

        nan = float("nan")
        hdr = (f"{'±2SE(AP)':>10}{'±2SE(R75)':>11}{'±2SE(IoU)':>11}") if args.bootstrap else ""
        print(f"\n{'分層':<22}{'n':>5}{'p50px':>7}{'AP50':>8}{'AP50-95':>9}{'比值':>7}"
              f"{'R@.5':>7}{'R@.75':>7}{'medIoU':>8}{'bias_w':>8}{'orc_c':>7}{'orc_s':>7}{hdr}")
        for r in sorted(rows, key=lambda x: x["ratio"]):
            line = (f"{r['stratum']:<22}{r['n_gt']:>5}{r['med_px']:>7.0f}{r['ap50']:>8.3f}"
                    f"{r['ap50_95']:>9.3f}{r['ratio']:>7.2f}{r['recall50']:>7.3f}"
                    f"{r['recall75']:>7.3f}{r.get('med_iou', nan):>8.3f}"
                    f"{r.get('bias_w', nan):>8.2f}{r.get('oracle_center', nan):>7.3f}"
                    f"{r.get('oracle_size', nan):>7.3f}")
            if args.bootstrap:
                line += (f"{2 * r['boot_se']:>10.3f}{2 * r.get('recall75_se', nan):>11.3f}"
                         f"{2 * r.get('med_iou_se', nan):>11.3f}")
            print(line)

        print(f"\n{'尺寸帶':<30}{'n':>6}{'R@.5':>8}{'R@.75':>8}{'medIoU':>8}"
              f"{'bias_w':>8}{'|Δw|p75':>9}")
        for b in band_rows:
            print(f"{b['cls'] + ' ' + b['band']:<30}{b['n_gt']:>6}{b['recall50']:>8.3f}"
                  f"{b['recall75']:>8.3f}{b['med_iou']:>8.3f}{b['bias_w']:>8.2f}"
                  f"{b['spread_w']:>9.2f}")

    print(f"\n輸出：{out_dir}")


if __name__ == "__main__":
    main()
