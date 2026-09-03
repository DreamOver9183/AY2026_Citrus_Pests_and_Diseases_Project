"""量化近重複洩漏對 P3 分層指標的實際影響。

`docs/v5r_記錄_近重複影像跨split洩漏查驗.md` 已確認 Scale_Insect / Thrips / Aphid 的
valid/test 有 13.4% 影像與 train 存在近重複，但當時沒有量化這對
`docs/v9_說明_P3定位精度改善與測試流程.md` §1.3 的分層指標（中位 IoU、AP50-95、R@0.75）
造成多大偏差。

做法：把每個 valid/test 影像標記成「洩漏」（與某張 train 影像 dHash 距離 ≤ 6）或
「乾淨」，同一個分層（例如 Thrips/Insect）內分開算指標。若模型在 train 見過近乎相同的
畫面，「洩漏」子集的數字應該系統性優於「乾淨」子集——差距大小就是量化結果。

本檔完全重用 `tools/diag_localization.py` 與 `tools/check_dataset_leakage.py` 的既有函式
（import 而非複製），不引入新的比對邏輯，確保與既有報告的數字定義一致。

用法：
    .venv/Scripts/python.exe tools/quantify_leak_impact.py
    .venv/Scripts/python.exe tools/quantify_leak_impact.py --bootstrap 500
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

import diag_localization as dl              # noqa: E402
import check_dataset_leakage as cdl          # noqa: E402

THRESHOLD = 6
AFFECTED = [(n, r) for n, r in cdl.SOURCES if n in ("Scale_Insect", "Thrips", "Aphid")]

# 要拆「洩漏 / 乾淨」比較的分層：(stratum 名稱, 類別 id)
STRATA = [
    ("Scale_Insect", 4),
    ("Thrips/Insect", 6), ("Thrips/Damage", 6),
    ("Aphid/Insect", 7), ("Aphid/Damage", 7),
]


def leaked_output_paths(name: str, rel: str, out_root: Path) -> set[str]:
    """回傳該類別在 valid/test 裡「與某張 train 影像近重複」的 OutPut 檔案路徑集合。"""
    src = cdl.SRC_ROOT / rel / "images"
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in cdl.IMG_EXT)
    hashes, src_md5 = {}, {}
    for p in files:
        h = cdl.dhash(p)
        if h is not None:
            hashes[p.name] = h
        src_md5[cdl.md5(p)] = p.name

    # source 檔名 -> (split, OutPut 路徑)；只需要 valid/test 的路徑
    name_to_split: dict[str, str] = {}
    name_to_path: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        for p in (out_root / split / "images").glob(f"{name}_*.jpg"):
            if "_aug_" in p.stem:
                continue
            sn = src_md5.get(cdl.md5(p))
            if sn:
                name_to_split[sn] = split
                if split != "train":
                    name_to_path[sn] = p

    names = list(hashes.keys())
    H = np.stack([hashes[n] for n in names])
    N = len(names)
    leaked: set[str] = set()
    BLOCK = 200
    for i0 in range(0, N, BLOCK):
        a = H[i0:i0 + BLOCK]
        dist = (a[:, None, :] != H[None, :, :]).sum(-1)
        for bi in range(a.shape[0]):
            gi = i0 + bi
            for gj in range(gi + 1, N):
                if int(dist[bi, gj]) > THRESHOLD:
                    continue
                na, nb = names[gi], names[gj]
                sa, sb = name_to_split.get(na), name_to_split.get(nb)
                if sa == "train" and sb in ("valid", "test"):
                    leaked.add(str(name_to_path[nb]))
                elif sb == "train" and sa in ("valid", "test"):
                    leaked.add(str(name_to_path[na]))
    return leaked


def stats_for(items: list[dict], st: str, cid: int, n_boot: int) -> dict:
    n_gt = sum(sum(b["stratum"] == st for b in it["boxes"]) for it in items)
    aps, rec, resid, per_t = {}, {}, None, {}
    for t in dl.IOU_THRS:
        sc, tp, rd, pi = dl.match_stratum(items, st, cid, float(t), want_resid=(t == 0.5))
        aps[float(t)] = dl.coco_ap(sc, tp, n_gt)
        rec[float(t)] = float(tp.sum()) / n_gt if n_gt else float("nan")
        per_t[float(t)] = pi
        if t == 0.5:
            resid = rd
    out = dict(n_gt=n_gt, n_img=len(items), ap50=aps[0.5], ap50_95=float(np.nanmean(list(aps.values()))),
               recall50=rec[0.5], recall75=rec[0.75],
               med_iou=float(np.median(resid["iou"])) if resid["iou"] else float("nan"))
    if n_boot and resid["iou"]:
        se, _, _ = dl.bootstrap_se(per_t, n_boot)
        out["ap_se"] = se
        box = dl.bootstrap_box_stats(resid, per_t[0.75], len(items), n_boot)
        out.update(box)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--weights", default=str(dl.WEIGHTS))
    args = ap.parse_args()

    dl.configure(dl.DATA_ROOT)
    print("建立 provenance 索引 …")
    prov = dl.build_provenance()

    print("計算各類別的洩漏影像清單 …")
    leaked_paths: set[str] = set()
    for name, rel in AFFECTED:
        s = leaked_output_paths(name, rel, dl.DATA_ROOT)
        print(f"  {name:<16} {len(s)} 張（valid+test）")
        leaked_paths |= s

    from ultralytics import YOLO
    model = YOLO(args.weights)

    for split in ("valid", "test"):
        items = dl.collect(split, prov)
        for it in items:
            it["leaked"] = str(it["img"]) in leaked_paths
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            res = model.predict([str(it["img"]) for it in chunk], imgsz=dl.IMGSZ, conf=0.001,
                                iou=0.7, max_det=300, verbose=False)
            for it, r in zip(chunk, res):
                b = r.boxes
                it["pred"] = dict(xyxy=b.xyxy.cpu().numpy(), conf=b.conf.cpu().numpy(),
                                  cls=b.cls.cpu().numpy().astype(int))
                dl.attach_pixels(it, r.orig_shape)
            print(f"\r  {split} 推論 {min(i + 8, len(items))}/{len(items)}", end="")
        print()

        n_leaked = sum(it["leaked"] for it in items)
        print(f"\n{'=' * 100}\n{split}（洩漏影像 {n_leaked} / {len(items)}）\n{'=' * 100}")
        print(f"{'分層':<16}{'子集':<6}{'n影像':>6}{'n框':>6}{'AP50':>8}{'AP50-95':>9}"
              f"{'R@.5':>7}{'R@.75':>7}{'中位IoU':>8}{'±2SE':>7}{'ΔmedIoU':>9}{'顯著?':>7}")
        for st, cid in STRATA:
            leaked_items = [it for it in items if it["leaked"]]
            clean_items = [it for it in items if not it["leaked"]]
            sl = stats_for(leaked_items, st, cid, args.bootstrap)
            sc = stats_for(clean_items, st, cid, args.bootstrap)
            delta = sl["med_iou"] - sc["med_iou"] if sl["n_gt"] and sc["n_gt"] else float("nan")
            se_l, se_c = sl.get("med_iou_se", float("nan")), sc.get("med_iou_se", float("nan"))
            combined_se = float(np.hypot(se_l, se_c)) if se_l == se_l and se_c == se_c else float("nan")
            sig = "是" if delta == delta and combined_se == combined_se and abs(delta) > 2 * combined_se else ""
            for tag, s in (("洩漏", sl), ("乾淨", sc)):
                se = s.get("med_iou_se", float("nan"))
                d = f"{delta:+.3f}" if tag == "洩漏" else ""
                sg = sig if tag == "洩漏" else ""
                print(f"{st:<16}{tag:<6}{s['n_img']:>6}{s['n_gt']:>6}{s['ap50']:>8.3f}"
                      f"{s['ap50_95']:>9.3f}{s['recall50']:>7.3f}{s['recall75']:>7.3f}"
                      f"{s['med_iou']:>8.3f}{2*se:>7.3f}{d:>9}{sg:>7}")


if __name__ == "__main__":
    main()
