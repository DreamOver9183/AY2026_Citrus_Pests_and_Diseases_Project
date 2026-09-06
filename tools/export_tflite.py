# -*- coding: utf-8 -*-
r"""把 YOLO26 的 `.pt` 權重轉成手機端要用的 `.tflite`，並驗證轉出來的東西還是同一個模型。

**這支腳本只能在 Docker 容器裡跑。** ultralytics 的 LiteRT 匯出第一行就是

    assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"

本專案的開發機是 Windows。容器定義在 `Benchmark/export/Dockerfile`，
「為什麼不走 ONNX」的完整理由在 `Benchmark/export/requirements.txt` 的註解。

用法（在專案根目錄執行）:

    docker build -t citrus-tflite-export -f Benchmark/export/Dockerfile .
    docker run --rm -v "%cd%":/work citrus-tflite-export \
        python tools/export_tflite.py --weights "<某個 .pt>" --verify

──────────────────────────────────────────────────────────────────────
三種變體，以及一個**會影響 benchmark 解讀**的差異
──────────────────────────────────────────────────────────────────────
ultralytics/engine/exporter.py 有這一段：

    if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
        # Static activation quantization collapses the end2end class-index output
        model.end2end = False

也就是說：

| 變體 | quantize | 校正資料 | `end2end` | 量到的延遲包含 NMS 嗎 |
| --- | --- | --- | --- | --- |
| `fp32`  | None     | 不需要 | **保留** | **包含**（topk 在圖裡） |
| `w8a32` | `w8a32`  | 不需要 | **保留** | **包含** |
| `int8`  | `8`      | **需要** | **被關掉** | **不包含**——輸出是原始張量，NMS 要在 App 端另外做 |

所以 `int8` 那一列的 FPS **不能直接**跟另外兩列比：它少做了一段工作。
真正的 apples-to-apples INT8 選項是 **`w8a32`**（權重 INT8、啟動值 FP32，
不需校正，且保留 end2end）。這件事在計畫階段被當成「退路」，實際上它是主力。

輸出檔名一律 `<stem>__<variant>.tflite`，放進 `Benchmark/Model/`。
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dataset_paths as P  # noqa: E402

# variant → (quantize 參數, 是否需要校正資料, end2end 是否保留)
VARIANTS = {
    "fp32": (None, False, True),
    "w8a32": ("w8a32", False, True),
    "int8": (8, True, False),
    "w8a16": ("w8a16", True, False),
}

DEFAULT_DATASET = "v5.6"
OUT_DIR = P.REPO / "Benchmark" / "Model"


# ══════════════════════════════════════════════════════════════════════
#  前置閘：與其讓 ultralytics 在轉了三分鐘之後才 assert，不如一開始就說清楚
# ══════════════════════════════════════════════════════════════════════
def platform_gate() -> None:
    """複製 exporter 的平台斷言，在使用者等待之前就給出可行動的訊息。"""
    system, machine = platform.system(), platform.machine().lower()
    is_arm64 = machine in {"arm64", "aarch64"}
    if system == "Darwin" or (system == "Linux" and not is_arm64):
        return
    print(f"\n✗ 這個環境（{system} {machine}）轉不出 LiteRT。")
    print("  ultralytics 的 export_litert 第一行是")
    print('      assert MACOS or (LINUX and not ARM64)')
    print("  請改用容器：")
    print("      docker build -t citrus-tflite-export -f Benchmark/export/Dockerfile .")
    print('      docker run --rm -v "%cd%":/work citrus-tflite-export \\')
    print('          python tools/export_tflite.py --weights "<.pt>" --verify')
    sys.exit(2)


def dependency_gate(need_quantizer: bool) -> None:
    """用 find_spec 探測，不 import——import litert 要數秒與數百 MB。"""
    import importlib.util

    need = ["litert_torch", "ai_edge_litert"] + (["ai_edge_quantizer"] if need_quantizer else [])
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        print(f"\n✗ 缺少匯出相依：{'、'.join(missing)}")
        print("  這代表你不在匯出容器裡，或映像建置沒有成功。")
        sys.exit(2)


# ══════════════════════════════════════════════════════════════════════
#  資料集：容器內要一份絕對路徑的 data.yaml
# ══════════════════════════════════════════════════════════════════════
def local_data_yaml(version: str, workdir: Path) -> Path:
    """產生一份 path 為絕對路徑的 data.yaml。

    專案的 data.yaml 寫的是 `path: .`，而 ultralytics 的 check_det_dataset 是
    「相對於 cwd」解析的（`Path(".")` 存在，所以不會走 DATASETS_DIR 那條分支）。
    容器裡 cwd 是 /work，不是資料集目錄，直接餵原檔會找不到影像。
    """
    import yaml

    src = P.split(version) / "data.yaml"
    if not src.is_file():
        raise SystemExit(f"✗ 找不到 {src}")
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    d["path"] = str(P.split(version).resolve())
    out = workdir / f"data_{version}_abs.yaml"
    out.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


# ══════════════════════════════════════════════════════════════════════
#  匯出
# ══════════════════════════════════════════════════════════════════════
def export_one(pt: Path, variant: str, workdir: Path, data_yaml: Path | None,
               fraction: float, imgsz: int | None) -> Path:
    """轉一個變體，回傳最終的 .tflite 路徑。"""
    from ultralytics import YOLO

    quantize, needs_calib, keeps_e2e = VARIANTS[variant]

    # exporter 把產物寫在**來源 .pt 旁邊**，所以先複製到專屬目錄，
    # 免得在 Train Output/ 底下留下一堆 .tflite。
    staged = workdir / variant / pt.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pt, staged)

    kw = dict(format="litert", device="cpu", verbose=False)
    if quantize is not None:
        kw["quantize"] = quantize
    if needs_calib:
        kw["data"] = str(data_yaml)
        kw["fraction"] = fraction
    if imgsz is not None:
        kw["imgsz"] = imgsz
    # 不傳 imgsz 時 export() 會從 model.args["imgsz"] 取訓練解析度——那才是對的預設。

    print(f"\n{'─' * 66}")
    print(f"  變體 {variant}   quantize={quantize!r}   end2end={'保留' if keeps_e2e else '會被關掉'}")
    if needs_calib:
        print(f"  校正資料 {data_yaml.name}  fraction={fraction}")
    print(f"{'─' * 66}")

    t0 = time.time()
    produced = Path(YOLO(str(staged)).export(**kw))
    dt = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f"{pt.stem}__{variant}.tflite"
    shutil.move(str(produced), final)
    mb = final.stat().st_size / 1024 / 1024
    print(f"  ✓ {final.name}   {mb:.1f} MB   耗時 {dt:.0f} s")
    return final


# ══════════════════════════════════════════════════════════════════════
#  驗證：轉出來的還是同一個模型嗎
# ══════════════════════════════════════════════════════════════════════
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _predict(model, images, conf: float):
    """回傳每張影像的 [(x1,y1,x2,y2,score,cls), ...]，依分數遞減。

    **一次只餵一張。** litert-torch 轉出來的圖 batch 維度是**固定的 1**
    （追蹤時就寫死了），把整份清單丟給 predict() 會讓 ultralytics 湊成一個
    batch=N 的輸入，然後在 set_tensor 直接爆掉：

        ValueError: Cannot set tensor: Dimension mismatch.
                    Got 20 but expected 1 for dimension 0 of input 0.

    PyTorch 端其實可以一次吃多張，但這裡刻意也走同一條路——兩邊的前處理
    完全一致，比對出來的差異才只來自序列化本身。
    """
    out = []
    for p in images:
        r = model.predict(p, conf=conf, verbose=False, device="cpu")[0]
        b = r.boxes
        rows = [] if b is None or len(b) == 0 else [
            (*map(float, xy), float(c), int(k))
            for xy, c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())
        ]
        out.append(sorted(rows, key=lambda t: -t[4]))
    return out


def _greedy_match(ref, got):
    """把兩份框做貪婪配對（IoU 高的先配），回傳 (配對, 未配到的 ref, 未配到的 got)。

    **不要只比最高分框。** 第一版就是那樣寫的，然後 w8a32 出現一張 top-1 IoU = 0
    的影像——查下去發現只是兩個分數幾乎一樣的框換了名次（0.8166/0.7906 →
    0.8345/0.8192），兩個物件其實都被找到，IoU 分別是 0.978 與 0.967。
    「同一個模型嗎」這個問題要用全體框回答，最高分框那個指標太脆弱。
    """
    cand = sorted(((_iou(a[:4], b[:4]), i, j)
                   for i, a in enumerate(ref) for j, b in enumerate(got)),
                  key=lambda t: -t[0])
    used_r, used_g, pairs = set(), set(), []
    for v, i, j in cand:
        if v <= 0 or i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        pairs.append((i, j, v))
    return (pairs,
            [i for i in range(len(ref)) if i not in used_r],
            [j for j in range(len(got)) if j not in used_g])


def verify(pt: Path, tflite: Path, version: str, n: int, conf: float) -> dict:
    """同一批影像跑 PyTorch 與 tflite，把所有框貪婪配對後比對幾何與類別。

    這一關的用途是**擋住壞掉的產物**，不是量精度。量化本來就會掉一點，
    但「框都對不上」就代表轉換過程出了事，那種東西拿去 benchmark 只是在量垃圾。
    """
    from ultralytics import YOLO

    # **等距取樣，不是取前 n 張。** 檔名帶類別前綴，排序後同類別會連在一起——
    # 取前 20 張會全部落在 Aphid（實測就是這樣），等於只驗了九分之一的類別。
    # 等距抽樣讓這 20 張橫跨全部九類。
    all_imgs = sorted((P.split(version) / "test" / "images").iterdir())
    if not all_imgs:
        raise SystemExit(f"✗ {version} 的 test/images 是空的")
    step = max(1, len(all_imgs) // n)
    test_imgs = all_imgs[::step][:n]
    paths = [str(p) for p in test_imgs]
    covered = sorted({p.name.rsplit("_", 1)[0] for p in test_imgs})
    print(f"    取樣 {len(paths)}/{len(all_imgs)} 張，涵蓋 {len(covered)} 類：{'、'.join(covered)}")

    ref = _predict(YOLO(str(pt)), paths, conf)
    got = _predict(YOLO(str(tflite)), paths, conf)

    ious, cls_ok, miss, extra, worst_img = [], 0, 0, 0, None
    for p, r, g in zip(paths, ref, got):
        pairs, ur, ug = _greedy_match(r, g)
        miss += len(ur)
        extra += len(ug)
        for i, j, v in pairs:
            ious.append(v)
            cls_ok += int(r[i][5] == g[j][5])
            if worst_img is None or v < worst_img[1]:
                worst_img = (Path(p).name, round(v, 4))

    n_ref = sum(len(r) for r in ref)
    n_got = sum(len(g) for g in got)
    res = dict(
        images=len(paths),
        ref_boxes=n_ref,
        tflite_boxes=n_got,
        matched=len(ious),
        matched_iou_mean=round(sum(ious) / len(ious), 4) if ious else None,
        matched_iou_min=round(min(ious), 4) if ious else None,
        worst_image=worst_img,
        class_match=f"{cls_ok}/{len(ious)}" if ious else "0/0",
        unmatched_ref=miss,      # PyTorch 有、tflite 沒有（通常是重複框被合併掉）
        unmatched_tflite=extra,  # tflite 多出來的
    )

    # 判準三條，全部要過：
    #   1. 配對框的平均 IoU ≥ 0.90。比偵測常用的 0.50 嚴得多，因為這裡比的是
    #      **同一個模型的兩種序列化**，不是兩個模型。fp32 實測 0.9996、
    #      w8a32 實測全部 ≥ 0.967——掉到 0.90 以下就不是量化雜訊了。
    #   2. 配對框的類別 100% 一致。量化不該改變分類結果。
    #   3. 未配對框 ≤ 10%。留這個餘裕是因為 end2end 偶爾會對同一物件吐兩個框
    #      （實測 Aphid_00008 就是），量化把它們合併掉反而是好事，不該因此擋掉。
    unmatched_rate = (miss + extra) / max(n_ref + n_got, 1)
    res["unmatched_rate"] = round(unmatched_rate, 4)
    ok = (res["matched_iou_mean"] is not None
          and res["matched_iou_mean"] >= 0.90
          and cls_ok == len(ious) and len(ious) > 0
          and unmatched_rate <= 0.10)
    res["verdict"] = "通過" if ok else "不通過"
    return res


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO26 .pt → .tflite（LiteRT），含與 PyTorch 的比對驗證",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="來源 .pt（相對於專案根目錄或絕對路徑）")
    ap.add_argument("--variants", default="fp32",
                    help="逗號分隔，可用 " + "、".join(VARIANTS) + "（預設只轉 fp32）")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="校正與驗證用的資料集版本")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="靜態量化的校正取樣比例（v5.6 train 7,264 張，0.05 ≈ 363 張）")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="不指定則沿用訓練時的解析度（建議不要指定）")
    ap.add_argument("--verify", action="store_true", help="轉完後與 PyTorch 比對")
    ap.add_argument("--verify-only", action="store_true",
                    help="跳過轉換，只驗 Benchmark/Model/ 裡已存在的檔案（隱含 --verify）")
    ap.add_argument("--verify-n", type=int, default=20, help="比對用幾張 test 影像")
    ap.add_argument("--conf", type=float, default=0.25, help="比對時的信心門檻")
    ap.add_argument("--workdir", default="/tmp/tflite_export")
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"✗ 未知的變體：{bad}。可用：{list(VARIANTS)}")

    if a.verify_only:
        a.verify = True
    # --verify-only 不轉換，所以不需要平台閘（但仍要 litert 才讀得回 .tflite）
    if not a.verify_only:
        platform_gate()
    dependency_gate(any(VARIANTS[v][1] for v in variants) and not a.verify_only)

    pt = Path(a.weights)
    if not pt.is_absolute():
        pt = P.REPO / a.weights
    if not pt.is_file():
        raise SystemExit(f"✗ 找不到權重 {pt}")

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    data_yaml = (local_data_yaml(a.dataset, workdir)
                 if any(VARIANTS[v][1] for v in variants) else None)

    print("═" * 66)
    print(f"  來源   {pt.relative_to(P.REPO) if pt.is_relative_to(P.REPO) else pt}")
    print(f"  變體   {'、'.join(variants)}")
    print(f"  輸出   {OUT_DIR.relative_to(P.REPO)}/")
    print("═" * 66)

    report = {"weights": str(pt), "dataset": a.dataset, "results": {}}
    failed = []

    for v in variants:
        if a.verify_only:
            out = OUT_DIR / f"{pt.stem}__{v}.tflite"
            if not out.is_file():
                print(f"  ✗ {v}：找不到 {out.name}，--verify-only 需要檔案已存在")
                report["results"][v] = {"exported": False, "error": "檔案不存在"}
                failed.append(v)
                continue
            print(f"\n{'─' * 66}")
            print(f"  變體 {v}（--verify-only，沿用既有檔案）")
            print(f"{'─' * 66}")
        else:
            try:
                out = export_one(pt, v, workdir, data_yaml, a.fraction, a.imgsz)
            except Exception as e:                               # noqa: BLE001
                print(f"  ✗ {v} 匯出失敗：{type(e).__name__}: {e}")
                report["results"][v] = {"exported": False, "error": f"{type(e).__name__}: {e}"}
                failed.append(v)
                continue

        entry = {"exported": True, "file": out.name,
                 "size_mb": round(out.stat().st_size / 1024 / 1024, 2),
                 "end2end_kept": VARIANTS[v][2]}
        if a.verify:
            print(f"  驗證中（{a.verify_n} 張 {a.dataset} test 影像）…")
            try:
                entry["verify"] = verify(pt, out, a.dataset, a.verify_n, a.conf)
                r = entry["verify"]
                print(f"    PyTorch {r['ref_boxes']} 框 / tflite {r['tflite_boxes']} 框"
                      f"   配對 {r['matched']} 對")
                print(f"    配對框 IoU 平均 {r['matched_iou_mean']}"
                      f"（最低 {r['matched_iou_min']} @ {r['worst_image'][0] if r['worst_image'] else '—'}）")
                print(f"    類別一致 {r['class_match']}"
                      f"   未配對 ref {r['unmatched_ref']} / tflite {r['unmatched_tflite']}"
                      f"（{r['unmatched_rate']:.1%}）")
                print(f"    → {r['verdict']}")
                if r["verdict"] != "通過":
                    failed.append(f"{v}(驗證)")
            except Exception as e:                               # noqa: BLE001
                print(f"    ✗ 驗證失敗：{type(e).__name__}: {e}")
                entry["verify"] = {"error": f"{type(e).__name__}: {e}"}
                failed.append(f"{v}(驗證)")
        report["results"][v] = entry

    rp = OUT_DIR / f"{pt.stem}__export_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "═" * 66)
    print(f"  報告   {rp.relative_to(P.REPO)}")
    if failed:
        print(f"  ✗ 有問題的變體：{'、'.join(failed)}")
        print("    **不要**把沒通過驗證的檔案拿去 benchmark——那只是在量壞掉的圖。")
        sys.exit(1)
    print("  ✓ 全部通過。可以進 Benchmark 流程了。")
    print("═" * 66)


if __name__ == "__main__":
    main()
