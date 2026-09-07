# -*- coding: utf-8 -*-
r"""把 YOLO26 權重匯出成 LiteRT 以外的格式，並在桌機上量精度與延遲。

**與 `export_tflite.py` 分開，刻意不合併** —— 那支已經通過完整驗收
（fp32 mAP50 0.81554 / w8a32 0.81936），是部署路線的主線，不該為了加格式而動它。

用法（容器內）：

    docker run --rm -v "%cd%":/work citrus-tflite-export \
        python tools/export_other_formats.py --weights "<.pt>" \
            --formats onnx,torchscript,mnn,ncnn --imgsz-list 640 --val

──────────────────────────────────────────────────────────────────────
這支腳本回答什麼、不回答什麼
──────────────────────────────────────────────────────────────────────
**回答**：
  - 這個格式在 YOLO26（end2end topk）上轉不轉得出來
  - 轉出來的檔案多大、桌機 CPU 上多快、完整 401 張 test 的 mAP 掉多少

**不回答**：**手機上多快。** benchmark APK 只吃 `.tflite`。
要量 NCNN / MNN 的手機延遲必須用 Android NDK 交叉編譯各自的 benchmark 執行檔，
那是獨立的一大塊工作。

> **桌機延遲不能外推到手機。** x86 與 ARM 的向量化、記憶體頻寬、
> 各框架的後端最佳化程度都不同。這裡的延遲只能用來**排序**，不能當部署數字。

──────────────────────────────────────────────────────────────────────
各格式的已知特性（取自 ultralytics 8.4.121）
──────────────────────────────────────────────────────────────────────
| 格式 | 支援的參數 | end2end | 備註 |
| --- | --- | --- | --- |
| `onnx` | batch/data/dynamic/quantize/opset/simplify/nms/fraction | 保留 | 參數最多的格式 |
| `torchscript` | batch/quantize/nms/dynamic | 保留 | — |
| `mnn` | batch/dynamic/quantize/opset/simplify/nms | 保留 | 經 ONNX 中轉 |
| `ncnn` | batch/quantize | **強制關閉** | `exporter.py:665` 把 ncnn 列入不支援 topk 的清單 |

**NCNN 的 end2end 被強制關掉**，所以它的輸出是 raw tensor、延遲不含 NMS，
與保留 end2end 的格式**不可直接比較**。
（實測 ncnn 的 mAP 與 `fp32__e2e0` 完全相同 —— 因為它們就是同一個模型。）

──────────────────────────────────────────────────────────────────────
兩個實測踩到的坑
──────────────────────────────────────────────────────────────────────
**1. ultralytics 用「路徑字尾」判斷格式**

AutoBackend 是靠 `_ncnn_model`、`_saved_model` 這種字尾認格式的。
把產出目錄改名時若沒保留字尾，`YOLO(<path>)` 會直接

    TypeError: model='...' is not a supported model format

模型檔本身完全正常（`model.ncnn.param` / `model.ncnn.bin` 都在），
純粹是名字認不出來。所以 `export_one` 改名時會把字尾接回去。

**2. MNN 在本容器裡載入失敗（環境問題，不是模型問題）**

    ImportError: _mnncengine.cpython-312-x86_64-linux-gnu.so:
                 cannot enable executable stack as shared object requires: Invalid argument

MNN 的原生擴充要求可執行堆疊（executable stack），而現代核心／容器預設拒絕。
這與 YOLO26 或 end2end 無關，是 MNN wheel 自身的建置方式。
要解需要 `execstack -c` 或 `patchelf` 清掉那個旗標並重建映像 —— **本輪未做**。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dataset_paths as P  # noqa: E402

OUT_DIR = P.REPO / "Benchmark" / "Model" / "other"
DEFAULT_DATASET = "v5.6"

# 產物是目錄而不是單一檔案的格式
DIR_ARTIFACTS = {"ncnn"}
# exporter.py:665 —— 這些格式不支援 topk，end2end 會被強制關掉
E2E_FORCED_OFF = {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu", "qnn"}


def local_data_yaml(version: str, workdir: Path) -> Path:
    """data.yaml 的 path 是相對的 `.`，容器裡 cwd 不是資料集目錄，要改絕對路徑。"""
    import yaml

    src = P.split(version) / "data.yaml"
    if not src.is_file():
        raise SystemExit(f"✗ 找不到 {src}")
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    d["path"] = str(P.split(version).resolve())
    out = workdir / f"data_{version}_abs.yaml"
    out.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def run_val(model_path: Path, data_yaml: Path, imgsz: int) -> dict:
    """完整 test 的 mAP ＋ ultralytics 自己量的逐張推論時間。

    `speed["inference"]` 是**桌機**的每張推論毫秒數，只能用來排序格式，
    不能當手機的部署數字。
    """
    from ultralytics import YOLO

    m = YOLO(str(model_path)).val(data=str(data_yaml), split="test", imgsz=imgsz,
                                  batch=1, plots=False, verbose=False)
    nm = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    return {
        "mAP50": round(float(m.box.map50), 5),
        "mAP50_95": round(float(m.box.map), 5),
        "precision": round(float(m.box.mp), 5),
        "recall": round(float(m.box.mr), 5),
        "desktop_ms": {k: round(float(v), 3) for k, v in (m.speed or {}).items()},
        "per_class_ap50": {nm[int(c)]: round(float(m.box.ap50[i]), 5)
                           for i, c in enumerate(m.box.ap_class_index)},
    }


def dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1048576
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1048576


def export_one(pt: Path, fmt: str, imgsz: int, workdir: Path) -> Path:
    from ultralytics import YOLO

    staged = workdir / f"{fmt}_{imgsz}" / pt.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pt, staged)

    print(f"\n{'─' * 70}")
    print(f"  {fmt}  @ imgsz={imgsz}"
          + ("   （end2end 會被 exporter 強制關閉）" if fmt in E2E_FORCED_OFF else ""))
    print(f"{'─' * 70}")

    t0 = time.time()
    produced = Path(YOLO(str(staged)).export(format=fmt, imgsz=imgsz,
                                             device="cpu", verbose=False))
    dt = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ⚠ ultralytics 是**用路徑字尾判斷格式**的（AutoBackend 找 "_ncnn_model"、
    #   "_saved_model" 等），所以改名時必須把那個字尾保留下來，
    #   否則 YOLO(<path>) 會直接 TypeError: not a supported model format。
    suffix = produced.suffix or ""
    tail = "_ncnn_model" if produced.name.endswith("_ncnn_model") else ""
    dest = OUT_DIR / f"{pt.stem}__{fmt}__i{imgsz}{tail}{suffix}"
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.move(str(produced), dest)
    print(f"  ✓ {dest.name}   {dir_size_mb(dest):.2f} MB   耗時 {dt:.0f} s")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX / TorchScript / MNN / NCNN 匯出與桌機評估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--formats", default="onnx,torchscript,mnn,ncnn")
    ap.add_argument("--imgsz-list", default="640")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--val", action="store_true", help="跑完整 test 的 mAP 與桌機延遲")
    ap.add_argument("--workdir", default="/tmp/other_export")
    a = ap.parse_args()

    from ultralytics.engine.exporter import export_formats
    tbl = export_formats()
    supported = {tbl["Argument"][i]: tbl["Arguments"][i] for i in range(len(tbl["Argument"]))}

    fmts = [f.strip() for f in a.formats.split(",") if f.strip()]
    if bad := [f for f in fmts if f not in supported]:
        raise SystemExit(f"✗ 未知的格式：{bad}")
    imgszs = [int(x) for x in a.imgsz_list.split(",") if x.strip()]

    pt = Path(a.weights)
    if not pt.is_absolute():
        pt = P.REPO / a.weights
    if not pt.is_file():
        raise SystemExit(f"✗ 找不到權重 {pt}")

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    data_yaml = local_data_yaml(a.dataset, workdir)

    print("═" * 70)
    print(f"  來源   {pt.relative_to(P.REPO) if pt.is_relative_to(P.REPO) else pt}")
    print(f"  格式   {'、'.join(fmts)}   imgsz {imgszs}")
    print("═" * 70)

    report = {"weights": str(pt), "dataset": a.dataset, "results": {}}

    if a.val:
        report["pytorch_baseline"] = {}
        for s in sorted(set(imgszs)):
            print(f"\n▷ PyTorch 基準 @ imgsz={s} …")
            r = run_val(pt, data_yaml, s)
            report["pytorch_baseline"][str(s)] = r
            print(f"    mAP50 {r['mAP50']:.5f}   桌機 {r['desktop_ms'].get('inference', 0):.1f} ms/張")

    for fmt in fmts:
        for s in imgszs:
            key = f"{fmt}__i{s}"
            entry = {"format": fmt, "imgsz": s,
                     "supported_args": supported[fmt],
                     "end2end_forced_off": fmt in E2E_FORCED_OFF}
            try:
                dest = export_one(pt, fmt, s, workdir)
                entry.update(exported=True, file=dest.name,
                             size_mb=round(dir_size_mb(dest), 2))
            except Exception as e:                               # noqa: BLE001
                print(f"  ✗ {fmt} @ {s} 匯出失敗：{type(e).__name__}: {e}")
                entry.update(exported=False, error=f"{type(e).__name__}: {e}")
                report["results"][key] = entry
                continue

            if a.val:
                print(f"  完整 mAP（401 張 @ imgsz={s}）…")
                try:
                    r = run_val(dest, data_yaml, s)
                    entry["val"] = r
                    base = (report.get("pytorch_baseline", {}) or {}).get(str(s))
                    d = f"   Δ {r['mAP50'] - base['mAP50']:+.5f}" if base else ""
                    print(f"    mAP50 {r['mAP50']:.5f}{d}   "
                          f"桌機 {r['desktop_ms'].get('inference', 0):.1f} ms/張")
                except Exception as e:                           # noqa: BLE001
                    print(f"    ✗ val 失敗：{type(e).__name__}: {e}")
                    entry["val"] = {"error": f"{type(e).__name__}: {e}"}
            report["results"][key] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rp = OUT_DIR / f"{pt.stem}__other_formats_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "═" * 92)
    print(f"  {'格式':<20}{'MB':>9}{'mAP50':>10}{'Δ':>10}{'桌機ms':>9}{'end2end':>10}")
    print("═" * 92)
    for k, e in report["results"].items():
        if not e.get("exported"):
            print(f"  {k:<20}  ✗ {e.get('error', '')[:56]}")
            continue
        v = e.get("val", {})
        m = v.get("mAP50")
        base = (report.get("pytorch_baseline", {}) or {}).get(str(e["imgsz"]), {}).get("mAP50")
        d = f"{m - base:+.5f}" if (m is not None and base is not None) else "—"
        ms = v.get("desktop_ms", {}).get("inference")
        print(f"  {k:<20}{e['size_mb']:>9.2f}"
              f"{(f'{m:.5f}' if m is not None else '—'):>10}{d:>10}"
              f"{(f'{ms:.1f}' if ms else '—'):>9}"
              f"{('強制關' if e['end2end_forced_off'] else '保留'):>10}")
    print("═" * 92)
    print("  ⚠ 桌機 ms 只能用來排序格式，**不能外推到手機**。")
    print(f"  報告 {rp.relative_to(P.REPO)}")


if __name__ == "__main__":
    main()
