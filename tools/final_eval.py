"""階段 4：在本機對交付權重跑 valid / test 的完整評估。

擷取邏輯移植自 `Train Code/v9/RESUME.ipynb` 的 Step.7（cell 12），該版本已在 Kaggle 上
驗證過六個消融臂與三次長跑。此處只做三處改動：路徑改本機、split 參數化、加上
四象限與 Macro/Micro 統計（公式見 docs/archive/v8_報告_模型訓練評估.md §4.2）。

只用標準庫 ＋ ultralytics，不依賴 pandas（本機 venv 沒裝）。

兩個 conf 工作點的意義不同，不可混用：

  val()               mAP 用 conf=0.001，混淆矩陣用 conf=0.25
                      （detect/val.py:54 `confusion_matrix_conf = 0.25 if conf is None else conf`）
                      → 主表用這個，與 v8 報告的取法一致

  val(conf=0.4985)    整條管線都在 0.4985 之上
                      → 只取混淆矩陣與 P/R/F1。**mAP 欄位不可引用**，
                        高閾值必然壓低 mAP，那是取法造成的，不是模型變差

用法：
    .venv/Scripts/python.exe tools/final_eval.py                    # valid + test，兩個閾值
    .venv/Scripts/python.exe tools/final_eval.py --split test
    .venv/Scripts/python.exe tools/final_eval.py --conf 0.4985
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "Datasets" / "Datasets_YOLO26_v5r" / "OutPut"
RUNS_ZIP = ROOT / "Train Code/v9/Train_output/Epoch_160_VER/RESUME2/runs_A0.zip"
WEIGHT_IN_ZIP = "detect/v5r_A0_160e/weights/best.pt"
OUT_ROOT = ROOT / "Train Code/v9/Train_output/Phase4"

NAMES = ["Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
         "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid"]
IMGSZ = 640
BATCH = 8
DEPLOY_CONF = 0.4985          # 本模型的最佳 F1 截斷點（v8 時代是 0.285）


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """只用標準庫寫 CSV —— ultralytics 本身也沒有強制要求 pandas。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def ensure_weights() -> Path:
    """從 runs zip 抽出 best.pt（已存在就沿用）。"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dst = OUT_ROOT / "best.pt"
    if dst.exists():
        return dst
    if not RUNS_ZIP.exists():
        raise FileNotFoundError(f"找不到 {RUNS_ZIP}")
    with zipfile.ZipFile(RUNS_ZIP) as z, z.open(WEIGHT_IN_ZIP) as src, open(dst, "wb") as out:
        out.write(src.read())
    print(f"▷ 已從 {RUNS_ZIP.name} 抽出 best.pt → {dst}")
    return dst


def ensure_data_yaml() -> Path:
    """產生指向本機絕對路徑的 data.yaml。

    v5r 原本的 data.yaml 寫 `path: .`，是相對於該檔案所在目錄的；ultralytics 解析
    相對路徑時會以 settings 的 datasets_dir 為基準，容易指到別的地方。這裡直接寫絕對路徑。
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    p = OUT_ROOT / "v5r_local.yaml"
    p.write_text(
        f"path: {DATA_ROOT.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: [{', '.join(NAMES)}]\n",
        encoding="utf-8",
    )
    return p


def quadrants(cm) -> tuple[list[dict], dict]:
    """由混淆矩陣算每類四象限與全域指標。

    cm 的形狀是 (nc+1, nc+1)，列=預測、欄=真實，最後一列/欄為背景。
    TN 沿用 v8 報告的約定：N_total − TP − FP − FN。這個定義在物件偵測裡本來就是
    人為的（沒有真正的「真陰性」樣本），但兩份報告用同一個約定表格才能並排看。
    """
    nc = len(NAMES)
    n_total = float(sum(sum(row) for row in cm))
    rows, sum_tp, sum_fp, sum_fn = [], 0, 0, 0
    for i in range(nc):
        tp = int(cm[i][i])
        fp = int(sum(cm[i])) - tp                       # 該類預測總數 − 正確的
        fn = int(sum(cm[j][i] for j in range(nc + 1))) - tp   # 該類真實總數 − 抓到的
        tn = int(n_total) - tp - fp - fn
        sum_tp, sum_fp, sum_fn = sum_tp + tp, sum_fp + fp, sum_fn + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "class_id": i, "class": NAMES[i],
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round((tp + tn) / n_total, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "fpr": round(fp / (fp + tn), 5) if fp + tn else 0.0,
            "f1": round(2 * prec * rec / (prec + rec), 5) if prec + rec else 0.0,
        })
    g = {
        "n_total": int(n_total),
        "total_TP": sum_tp, "total_FP": sum_fp, "total_FN": sum_fn,
        "macro_accuracy": round(sum(r["accuracy"] for r in rows) / nc, 5),
        "macro_precision": round(sum(r["precision"] for r in rows) / nc, 5),
        "macro_recall": round(sum(r["recall"] for r in rows) / nc, 5),
        "macro_f1": round(sum(r["f1"] for r in rows) / nc, 5),
        "micro_precision": round(sum_tp / (sum_tp + sum_fp), 5) if sum_tp + sum_fp else 0.0,
        "micro_recall": round(sum_tp / (sum_tp + sum_fn), 5) if sum_tp + sum_fn else 0.0,
        "detection_jaccard": round(sum_tp / (sum_tp + sum_fp + sum_fn), 5)
        if sum_tp + sum_fp + sum_fn else 0.0,
    }
    mp, mr = g["micro_precision"], g["micro_recall"]
    g["micro_f1"] = round(2 * mp * mr / (mp + mr), 5) if mp + mr else 0.0
    return rows, g


def evaluate(weights: Path, data_yaml: Path, split: str, conf: float | None) -> dict:
    from ultralytics import YOLO

    tag = f"{split}_conf{'default' if conf is None else conf}"
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n▷ 評估 split={split}  conf={'預設' if conf is None else conf}\n{'=' * 72}")

    kw = dict(data=str(data_yaml), split=split, imgsz=IMGSZ, batch=BATCH,
              device="cpu", plots=True, verbose=False,
              project=str(out), name="val", exist_ok=True)
    if conf is not None:
        kw["conf"] = conf
    # plots=True 是必要條件，不是為了畫圖：ultralytics 把 confusion_matrix.process_batch
    # 包在 `if self.args.plots` 裡（detect/val.py:196），plots=False 會讓混淆矩陣全零。
    m = YOLO(str(weights)).val(**kw)

    names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}

    # ── per-class ────────────────────────────────────────────────────
    per_class = []
    for i, ci in enumerate(m.box.ap_class_index):
        ci = int(ci)
        per_class.append({
            "class_id": ci, "class": names.get(ci, str(ci)),
            "precision": round(float(m.box.p[i]), 5),
            "recall": round(float(m.box.r[i]), 5),
            "f1": round(float(m.box.f1[i]), 5),
            "ap50": round(float(m.box.ap50[i]), 5),
            "ap50_95": round(float(m.box.ap[i]), 5),
        })
    per_class.sort(key=lambda r: r["class_id"])
    write_csv(out / "per_class.csv",
              ["class_id", "class", "precision", "recall", "f1", "ap50", "ap50_95"], per_class)

    # ── 混淆矩陣（列=預測，欄=真實，最後一列/欄為背景）────────────────
    cm = m.confusion_matrix.matrix
    nc = len(NAMES)
    labels = [names.get(i, str(i)) for i in range(nc)] + ["background"]
    with open(out / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"true_{l}" for l in labels])
        for i, lab in enumerate(labels):
            w.writerow([f"pred_{lab}"] + [int(cm[i][j]) for j in range(len(labels))])
    fp_bg = {labels[i]: int(cm[i][nc]) for i in range(nc)}
    fn_bg = {labels[i]: int(cm[nc][i]) for i in range(nc)}

    # ── 四象限與全域指標 ──────────────────────────────────────────────
    quad, glob = quadrants(cm)
    write_csv(out / "quadrants.csv",
              ["class_id", "class", "TP", "FP", "FN", "TN",
               "accuracy", "precision", "recall", "fpr", "f1"], quad)

    # ── F1-conf 曲線 ─────────────────────────────────────────────────
    # 只有在未過濾（conf=None）時才有意義：一旦在 conf=c 之上過濾，曲線就只剩 [c, 1]，
    # 極大值必然落在左端點，回報出來的「最佳截斷點」是取樣範圍造成的假象。
    best_conf = None
    try:
        if conf is not None:
            raise RuntimeError("已過濾，F1-conf 曲線不具意義")
        x, y, _xl, _yl = m.curves_results[1]        # F1-Confidence(B)
        x = [float(v) for v in x]
        mean_f1 = ([sum(col) / len(col) for col in zip(*y)] if hasattr(y[0], "__len__")
                   else [float(v) for v in y])
        with open(out / "f1_conf.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conf", "mean_f1"])
            w.writerows(zip(x, mean_f1))
        best_conf = round(x[mean_f1.index(max(mean_f1))], 4)
    except Exception as e:
        print(f"▷ F1-conf 曲線取用失敗（不影響主判準）：{type(e).__name__}: {e}")

    summary = {
        "split": split,
        "conf": "default (mAP@0.001 / confusion_matrix@0.25)" if conf is None else conf,
        "weights": str(weights.relative_to(ROOT)),
        "checkpoint_epoch": 69,
        "imgsz": IMGSZ,
        "device": "cpu (fp32)",
        "overall": {"mAP50": round(float(m.box.map50), 5),
                    "mAP50_95": round(float(m.box.map), 5),
                    "precision": round(float(m.box.mp), 5),
                    "recall": round(float(m.box.mr), 5)},
        "global_from_confusion_matrix": glob,
        "best_f1_conf": best_conf,
        "fp_from_background": fp_bg,
        "fn_to_background": fn_bg,
        # CPU 延遲不代表部署延遲，僅記錄
        "speed_ms_cpu": {k: round(float(v), 3) for k, v in m.speed.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    # ── 主控台輸出 ────────────────────────────────────────────────────
    o = summary["overall"]
    print(f"\n  mAP50={o['mAP50']:.5f}  mAP50-95={o['mAP50_95']:.5f}  "
          f"P={o['precision']:.5f}  R={o['recall']:.5f}")
    if conf is not None:
        print("  ⚠ 此輪的 mAP 因高閾值而失真，不可引用；只取混淆矩陣與 P/R/F1")
    print(f"  最佳 F1 截斷點 conf = {best_conf}")
    print(f"\n  {'類別':<20}{'P':>9}{'R':>9}{'F1':>9}{'AP50':>9}{'AP50-95':>10}{'FP_bg':>8}{'FN_bg':>8}")
    for r in per_class:
        print(f"  {r['class']:<20}{r['precision']:>9.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}"
              f"{r['ap50']:>9.4f}{r['ap50_95']:>10.4f}"
              f"{fp_bg.get(r['class'], 0):>8}{fn_bg.get(r['class'], 0):>8}")
    print(f"\n  Macro  P={glob['macro_precision']:.4f} R={glob['macro_recall']:.4f} "
          f"F1={glob['macro_f1']:.4f} Acc={glob['macro_accuracy']:.4f}")
    print(f"  Micro  P={glob['micro_precision']:.4f} R={glob['micro_recall']:.4f} "
          f"F1={glob['micro_f1']:.4f}   Jaccard={glob['detection_jaccard']:.4f}")
    print(f"  ΣTP={glob['total_TP']}  ΣFP={glob['total_FP']}  ΣFN={glob['total_FN']}"
          f"  N_total={glob['n_total']}")
    print(f"\n  → {out.relative_to(ROOT)}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="階段 4 最終評估（本機 CPU）")
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--conf", type=float, default=None,
                    help=f"只跑這個閾值；預設同時跑「預設」與部署閾值 {DEPLOY_CONF}")
    args = ap.parse_args()

    weights = ensure_weights()
    data_yaml = ensure_data_yaml()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    confs = [args.conf] if args.conf is not None else [None, DEPLOY_CONF]

    results = {}
    for sp in splits:
        for cf in confs:
            results[f"{sp}_{'default' if cf is None else cf}"] = evaluate(weights, data_yaml, sp, cf)

    # ── 總表 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}\n▷ 總表（mAP 只看 conf=預設 的列）\n{'=' * 72}")
    print(f"  {'split / conf':<24}{'mAP50':>10}{'mAP50-95':>11}{'P':>9}{'R':>9}")
    for k, v in results.items():
        o = v["overall"]
        note = "" if v["conf"] != DEPLOY_CONF else "   ← mAP 失真"
        print(f"  {k:<24}{o['mAP50']:>10.5f}{o['mAP50_95']:>11.5f}"
              f"{o['precision']:>9.5f}{o['recall']:>9.5f}{note}")
    (OUT_ROOT / "all_summaries.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n▷ 全部結果：{OUT_ROOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
