# -*- coding: utf-8 -*-
r"""把 YOLO26 的 `.pt` 權重轉成手機端要用的 `.tflite`，並驗證轉出來的東西還是同一個模型。

**這支腳本只能在 Docker 容器裡跑。** ultralytics 的 LiteRT 匯出第一行就是

    assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"

本專案的開發機是 Windows。容器定義在 `Benchmark/export/Dockerfile`，
「為什麼不走 ONNX」的完整理由在 `Benchmark/export/requirements.txt` 的註解。

用法（在專案根目錄執行）:

    docker build -t citrus-tflite-export -f Benchmark/export/Dockerfile .
    docker run --rm -v "%cd%":/work citrus-tflite-export \
        python tools/export_tflite.py --weights "<某個 .pt>" \
            --variants fp32,w8a32 --imgsz-list 640,416 --verify --val

──────────────────────────────────────────────────────────────────────
四個可掃的軸，以及它們的交互作用
──────────────────────────────────────────────────────────────────────
**1. `--variants`（量化方式）**

ultralytics/engine/exporter.py 有這一段：

    if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
        # Static activation quantization collapses the end2end class-index output
        model.end2end = False

| variant | quantize | 校正資料 | `end2end` |
| --- | --- | --- | --- |
| `fp32`  | None    | 不需要 | 可自選 |
| `w8a32` | `w8a32` | 不需要 | 可自選 |
| `int8`  | `8`     | **需要** | **一定關掉** |
| `w8a16` | `w8a16` | **需要** | **一定關掉** |

`quantize=16`（FP16）**對 litert 不支援** —— `litert` 不在 `FP16_FORMATS`，
`validate_args` 會 assert。LiteRT 的 FP16 是**執行期**行為（GPU delegate 預設 FP16，
或 XNNPACK 的 `FORCE_FP16` 旗標），不是另外匯出一個檔。

**2. `--imgsz-list`（輸入解析度）**

`imgsz` 不在任何格式的 Arguments 清單裡，所以 `validate_args` 不檢查它，是通用參數。
單獨降解析度**補不上 6.8 倍的缺口**（實測 640 是 272 ms，依 GFLOPs 外推 320 也還要 67 ms），
必須與量化相乘。

**3. `--end2end-list`（要不要保留 NMS-free 的 topk 頭）**

`end2end` **不在官方文件的參數表裡**，但它存在：`DEFAULT_CFG_DICT["end2end"]` 預設 None，
`exporter.py:663` 會 `model.end2end = self.args.end2end`。因為它不在任何格式的
Arguments 清單裡，`validate_args` 不會擋。

這很重要：在此之前只能靠 `quantize=8` **間接**關掉 end2end，於是
「INT8 的效果」與「拿掉 topk 的效果」永遠綁在一起分不開。

**為什麼要拿掉 topk**：平台 1 實測 GPU delegate 只吃得下 54/559 個節點（9.7%），
不支援的算子是 `GATHER_ND` / `FLOOR_MOD` / `CAST INT64` / `SELECT_V2` / `LESS` / `NOT_EQUAL`
—— 正是 end2end 頭的實作。

> ⚠ **`end2end=False` 的圖不含 NMS**，App 端必須自己做。
> 它的延遲與保留 end2end 的變體**不可直接比較**。

**4. `--max-det-list`（end2end 圖裡 TopK 的 k）**

`max_det` 在官方文件的參數表裡是「輸出保留的最大偵測數」，看起來只跟 NMS 有關，
但對 end2end 模型它**直接決定圖裡烘進去的 TopK k**：

    exporter.py:866   m.max_det = min(self.args.max_det, available)   ← 無條件執行
    exporter.py:1291  "end2end graphs bake TopK k=max_det"

預設 300 對本資料集是純浪費 —— v5.6 單張最多只有 **58** 個框（train）/ 44（valid）/
32（test），而 imgsz=640 時要從 **34,000 個 anchor** 裡挑 300。

> ⚠ **這是精度換延遲，不是免費的。** `val()` 算 mAP 時用 `conf=0.001`，
> 會產生大量低分偵測；把 k 砍小會截掉低分尾巴而壓低召回。
> **每個 max_det 都必須量 mAP**，不能只看延遲就下結論。

輸出檔名 `<stem>__<variant>__i<imgsz>__e2e{1,0}[__md<max_det>].tflite`。
`max_det` 只在非預設時進檔名，既有產物的名字因此不受影響。
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

# variant → (quantize 參數, 是否需要校正資料, end2end 能不能自選)
VARIANTS = {
    "fp32": (None, False, True),
    "w8a32": ("w8a32", False, True),
    "int8": (8, True, False),
    "w8a16": ("w8a16", True, False),
}

DEFAULT_DATASET = "v5.6"
OUT_DIR = P.REPO / "Benchmark" / "Model"


DEFAULT_MAX_DET = 300           # ultralytics 的預設


class Cfg:
    """一個待匯出的組合。`tag` 是它在所有產出裡的唯一識別。"""

    def __init__(self, variant: str, imgsz: int, e2e_req: str, max_det: int = DEFAULT_MAX_DET):
        self.variant = variant
        self.imgsz = imgsz
        self.max_det = max_det
        quantize, needs_calib, e2e_selectable = VARIANTS[variant]
        self.quantize = quantize
        self.needs_calib = needs_calib
        # 靜態量化的變體不論要求什麼，exporter 都會強制關掉 end2end
        if not e2e_selectable:
            self.end2end = False
            self.e2e_forced = True
        else:
            self.end2end = {"auto": True, "true": True, "false": False}[e2e_req]
            self.e2e_forced = False
        # auto 就不要顯式傳 end2end，讓 exporter 走它自己的預設路徑
        self.pass_end2end = (not self.e2e_forced) and e2e_req != "auto"

    @property
    def tag(self) -> str:
        t = f"{self.variant}__i{self.imgsz}__e2e{1 if self.end2end else 0}"
        # max_det 只在非預設時進檔名，這樣既有產物的名字不會變
        if self.max_det != DEFAULT_MAX_DET:
            t += f"__md{self.max_det}"
        return t

    def __repr__(self) -> str:
        return self.tag


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
def export_one(pt: Path, cfg: Cfg, workdir: Path, data_yaml: Path,
               fraction: float, split: str) -> Path:
    """轉一個組合，回傳最終的 .tflite 路徑。"""
    from ultralytics import YOLO

    # exporter 把產物寫在**來源 .pt 旁邊**，所以先複製到專屬目錄，
    # 免得在 Train Output/ 底下留下一堆 .tflite。
    staged = workdir / cfg.tag / pt.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pt, staged)

    kw = dict(format="litert", device="cpu", verbose=False, imgsz=cfg.imgsz)
    # max_det 直接決定 end2end 圖裡烘進去的 TopK k：
    #   exporter.py:866   m.max_det = min(self.args.max_det, available)
    #   exporter.py:1291  "end2end graphs bake TopK k=max_det"
    # 預設 300 對本資料集是浪費——v5.6 單張最多才 58 個框（train）/ 32（test）。
    # 但**這是精度換延遲**：val 的 mAP 用 conf=0.001，截斷會砍掉低分尾巴而影響召回，
    # 所以每個 max_det 都要量 mAP，不能只看延遲。
    if cfg.max_det != DEFAULT_MAX_DET:
        kw["max_det"] = cfg.max_det
    if cfg.quantize is not None:
        kw["quantize"] = cfg.quantize
    if cfg.needs_calib:
        # split 必須是 train：v5.6 的 valid/test 就是我們的評估集，
        # 拿評估集當校正資料會洩漏。ultralytics 的預設是 'val'，所以一定要顯式指定。
        kw["data"] = str(data_yaml)
        kw["fraction"] = fraction
        kw["split"] = split
    if cfg.pass_end2end:
        kw["end2end"] = cfg.end2end

    e2e_note = ("被 exporter 強制關閉" if cfg.e2e_forced
                else ("保留" if cfg.end2end else "以 end2end=False 顯式關閉"))
    print(f"\n{'─' * 70}")
    print(f"  {cfg.tag}")
    print(f"    quantize={cfg.quantize!r}  imgsz={cfg.imgsz}  end2end={e2e_note}")
    if cfg.needs_calib:
        print(f"    校正 {data_yaml.name}  split={split}  fraction={fraction}")
    print(f"{'─' * 70}")

    t0 = time.time()
    produced = Path(YOLO(str(staged)).export(**kw))
    dt = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f"{pt.stem}__{cfg.tag}.tflite"
    shutil.move(str(produced), final)
    mb = final.stat().st_size / 1024 / 1024
    print(f"  ✓ {final.name}   {mb:.2f} MB   耗時 {dt:.0f} s")
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


def _predict(model, images, conf: float, imgsz: int):
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
        r = model.predict(p, conf=conf, imgsz=imgsz, verbose=False, device="cpu")[0]
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


def verify(pt: Path, tflite: Path, version: str, n: int, conf: float, imgsz: int) -> dict:
    """同一批影像跑 PyTorch 與 tflite，把所有框貪婪配對後比對幾何與類別。

    這一關的用途是**擋住壞掉的產物**，不是量精度——量化本來就會掉一點。
    要回答「這對部署有沒有影響」必須跑完整的 val()（見 --val）。

    **PyTorch 端也用同一個 imgsz**，否則比的是兩個不同解析度的模型。
    """
    from ultralytics import YOLO

    # **等距取樣，不是取前 n 張。** 檔名帶類別前綴，排序後同類別會連在一起——
    # 取前 20 張會全部落在 Aphid（實測就是這樣），等於只驗了九分之一的類別。
    all_imgs = sorted((P.split(version) / "test" / "images").iterdir())
    if not all_imgs:
        raise SystemExit(f"✗ {version} 的 test/images 是空的")
    step = max(1, len(all_imgs) // n)
    test_imgs = all_imgs[::step][:n]
    paths = [str(p) for p in test_imgs]
    covered = sorted({p.name.rsplit("_", 1)[0] for p in test_imgs})
    print(f"    取樣 {len(paths)}/{len(all_imgs)} 張，涵蓋 {len(covered)} 類")

    ref = _predict(YOLO(str(pt)), paths, conf, imgsz)
    got = _predict(YOLO(str(tflite)), paths, conf, imgsz)

    ious, cls_ok, miss, extra, worst = [], 0, 0, 0, None
    for p, r, g in zip(paths, ref, got):
        pairs, ur, ug = _greedy_match(r, g)
        miss += len(ur)
        extra += len(ug)
        for i, j, v in pairs:
            ious.append(v)
            cls_ok += int(r[i][5] == g[j][5])
            if worst is None or v < worst[1]:
                worst = (Path(p).name, round(v, 4))

    n_ref, n_got = sum(len(r) for r in ref), sum(len(g) for g in got)
    unmatched_rate = (miss + extra) / max(n_ref + n_got, 1)
    res = dict(
        images=len(paths), ref_boxes=n_ref, tflite_boxes=n_got, matched=len(ious),
        matched_iou_mean=round(sum(ious) / len(ious), 4) if ious else None,
        matched_iou_min=round(min(ious), 4) if ious else None,
        worst_image=worst,
        class_match=f"{cls_ok}/{len(ious)}" if ious else "0/0",
        unmatched_ref=miss, unmatched_tflite=extra,
        unmatched_rate=round(unmatched_rate, 4),
    )

    # 判準三條，全部要過：
    #   1. 配對框的平均 IoU ≥ 0.90。比偵測常用的 0.50 嚴得多，因為這裡比的是
    #      **同一個模型的兩種序列化**，不是兩個模型。
    #   2. 配對框的類別 100% 一致。量化不該改變分類結果。
    #   3. 未配對框 ≤ 10%。end2end 偶爾會對同一物件吐兩個框，量化把它們合併掉
    #      反而是好事，不該因此擋掉。
    ok = (res["matched_iou_mean"] is not None and res["matched_iou_mean"] >= 0.90
          and cls_ok == len(ious) and len(ious) > 0 and unmatched_rate <= 0.10)
    res["verdict"] = "通過" if ok else "不通過"
    return res


# ══════════════════════════════════════════════════════════════════════
#  完整 mAP：唯一能回答「這對部署有沒有影響」的東西
# ══════════════════════════════════════════════════════════════════════
def run_val(model_path: Path, data_yaml: Path, imgsz: int) -> dict:
    """在完整的 test split 上跑 val()。

    `batch=1` 是必要的，不是保守：litert 追蹤時把 batch 維度寫死成 1。
    PyTorch 基準也要用 batch=1 才是同一條量測路徑。
    """
    from ultralytics import YOLO

    t0 = time.time()
    m = YOLO(str(model_path)).val(data=str(data_yaml), split="test", imgsz=imgsz,
                                  batch=1, plots=False, verbose=False)
    nm = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    return {
        "mAP50": round(float(m.box.map50), 5),
        "mAP50_95": round(float(m.box.map), 5),
        "precision": round(float(m.box.mp), 5),
        "recall": round(float(m.box.mr), 5),
        "per_class_ap50": {nm[int(c)]: round(float(m.box.ap50[i]), 5)
                           for i, c in enumerate(m.box.ap_class_index)},
        "seconds": round(time.time() - t0, 1),
    }


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO26 .pt → .tflite（LiteRT）參數掃描，含驗收與完整 mAP",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="來源 .pt（相對於專案根目錄或絕對路徑）")
    ap.add_argument("--variants", default="fp32",
                    help="逗號分隔，可用 " + "、".join(VARIANTS))
    ap.add_argument("--imgsz-list", default="640", help="逗號分隔的輸入解析度")
    ap.add_argument("--end2end-list", default="auto",
                    help="逗號分隔的 auto/true/false。靜態量化的變體一律被強制 false")
    ap.add_argument("--max-det-list", default=str(DEFAULT_MAX_DET),
                    help="逗號分隔。直接決定 end2end 圖裡的 TopK k（exporter.py:1291）。"
                         "v5.6 單張最多 58 個框，預設 300 是浪費——但截斷會影響 mAP，要量")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="校正與評估用的資料集版本")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="靜態量化的校正取樣比例（v5.6 train 7,264 張，0.05 ≈ 363 張）")
    ap.add_argument("--split", default="train",
                    help="校正資料的 split。**預設刻意是 train**——ultralytics 預設 val，"
                         "但 v5.6 的 valid/test 是評估集，拿來校正會洩漏")
    ap.add_argument("--verify", action="store_true", help="轉完後與 PyTorch 逐框比對")
    ap.add_argument("--val", action="store_true", help="跑完整 test 的 mAP（每組約 40 s）")
    ap.add_argument("--verify-only", action="store_true",
                    help="跳過轉換，只驗 Benchmark/Model/ 裡已存在的檔案")
    ap.add_argument("--verify-n", type=int, default=20, help="逐框比對用幾張 test 影像")
    ap.add_argument("--conf", type=float, default=0.25, help="逐框比對的信心門檻")
    ap.add_argument("--workdir", default="/tmp/tflite_export")
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if bad := [v for v in variants if v not in VARIANTS]:
        raise SystemExit(f"✗ 未知的變體：{bad}。可用：{list(VARIANTS)}")
    imgszs = [int(x) for x in a.imgsz_list.split(",") if x.strip()]
    e2es = [x.strip() for x in a.end2end_list.split(",") if x.strip()]
    if bad := [x for x in e2es if x not in {"auto", "true", "false"}]:
        raise SystemExit(f"✗ --end2end-list 只接受 auto/true/false，收到 {bad}")

    # 交叉相乘後去重：int8/w8a16 的 auto 與 false 會產生同一個 tag
    maxdets = [int(x) for x in a.max_det_list.split(",") if x.strip()]
    cfgs, seen = [], set()
    for v in variants:
        for s in imgszs:
            for e in e2es:
                for md in maxdets:
                    c = Cfg(v, s, e, md)
                    if c.tag not in seen:
                        seen.add(c.tag)
                        cfgs.append(c)

    if a.verify_only:
        a.verify = True
    if not a.verify_only:
        platform_gate()
    dependency_gate(any(c.needs_calib for c in cfgs) and not a.verify_only)

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
    print(f"  組合   {len(cfgs)} 個：{'、'.join(c.tag for c in cfgs)}")
    print(f"  輸出   {OUT_DIR.relative_to(P.REPO)}/")
    print("═" * 70)

    report = {"weights": str(pt), "dataset": a.dataset,
              "calibration_split": a.split, "fraction": a.fraction, "results": {}}
    failed = []

    # PyTorch 基準：每個用到的 imgsz 各量一次，才能算出「量化掉了多少」
    if a.val:
        report["pytorch_baseline"] = {}
        for s in sorted(set(imgszs)):
            print(f"\n▷ PyTorch 基準 @ imgsz={s} …")
            r = run_val(pt, data_yaml, s)
            report["pytorch_baseline"][str(s)] = r
            print(f"    mAP50 {r['mAP50']:.5f}   mAP50-95 {r['mAP50_95']:.5f}   ({r['seconds']:.0f} s)")

    for cfg in cfgs:
        if a.verify_only:
            out = OUT_DIR / f"{pt.stem}__{cfg.tag}.tflite"
            if not out.is_file():
                print(f"  ✗ {cfg.tag}：找不到 {out.name}")
                report["results"][cfg.tag] = {"exported": False, "error": "檔案不存在"}
                failed.append(cfg.tag)
                continue
            print(f"\n{'─' * 70}\n  {cfg.tag}（--verify-only，沿用既有檔案）\n{'─' * 70}")
        else:
            try:
                out = export_one(pt, cfg, workdir, data_yaml, a.fraction, a.split)
            except Exception as e:                               # noqa: BLE001
                print(f"  ✗ {cfg.tag} 匯出失敗：{type(e).__name__}: {e}")
                report["results"][cfg.tag] = {"exported": False,
                                              "error": f"{type(e).__name__}: {e}"}
                failed.append(cfg.tag)
                continue

        entry = {"exported": True, "file": out.name, "variant": cfg.variant,
                 "imgsz": cfg.imgsz, "end2end": cfg.end2end, "max_det": cfg.max_det,
                 "end2end_forced_off": cfg.e2e_forced,
                 "size_mb": round(out.stat().st_size / 1024 / 1024, 2)}

        if a.verify:
            print(f"  逐框驗收（{a.verify_n} 張 {a.dataset} test 影像 @ imgsz={cfg.imgsz}）…")
            try:
                r = verify(pt, out, a.dataset, a.verify_n, a.conf, cfg.imgsz)
                entry["verify"] = r
                print(f"    PyTorch {r['ref_boxes']} 框 / tflite {r['tflite_boxes']} 框"
                      f"   配對 {r['matched']} 對")
                print(f"    配對 IoU 平均 {r['matched_iou_mean']}（最低 {r['matched_iou_min']}）"
                      f"   類別一致 {r['class_match']}   未配對 {r['unmatched_rate']:.1%}")
                print(f"    → {r['verdict']}")
                if r["verdict"] != "通過":
                    failed.append(f"{cfg.tag}(驗收)")
            except Exception as e:                               # noqa: BLE001
                print(f"    ✗ 驗收失敗：{type(e).__name__}: {e}")
                entry["verify"] = {"error": f"{type(e).__name__}: {e}"}
                failed.append(f"{cfg.tag}(驗收)")

        if a.val:
            print(f"  完整 mAP（401 張 @ imgsz={cfg.imgsz}）…")
            try:
                r = run_val(out, data_yaml, cfg.imgsz)
                entry["val"] = r
                base = (report.get("pytorch_baseline", {}) or {}).get(str(cfg.imgsz))
                d = f"   Δ {r['mAP50'] - base['mAP50']:+.5f}" if base else ""
                print(f"    mAP50 {r['mAP50']:.5f}{d}   mAP50-95 {r['mAP50_95']:.5f}"
                      f"   ({r['seconds']:.0f} s)")
            except Exception as e:                               # noqa: BLE001
                print(f"    ✗ val 失敗：{type(e).__name__}: {e}")
                entry["val"] = {"error": f"{type(e).__name__}: {e}"}

        report["results"][cfg.tag] = entry

    rp = OUT_DIR / f"{pt.stem}__export_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 總表 ──────────────────────────────────────────────────────────
    print("\n" + "═" * 88)
    print(f"  {'組合':<28}{'MB':>7}{'mAP50':>9}{'Δ':>9}{'驗收':>8}{'end2end':>10}")
    print("═" * 88)
    for tag, e in report["results"].items():
        if not e.get("exported"):
            print(f"  {tag:<28}  ✗ {e.get('error', '')[:44]}")
            continue
        v = e.get("val", {})
        m = v.get("mAP50")
        base = (report.get("pytorch_baseline", {}) or {}).get(str(e["imgsz"]), {}).get("mAP50")
        d = f"{m - base:+.5f}" if (m is not None and base is not None) else "—"
        print(f"  {tag:<28}{e['size_mb']:>7.2f}"
              f"{(f'{m:.5f}' if m is not None else '—'):>9}{d:>9}"
              f"{e.get('verify', {}).get('verdict', '—'):>8}"
              f"{('保留' if e['end2end'] else ('強制關' if e['end2end_forced_off'] else '關')):>10}")
    print("═" * 88)
    print(f"  報告   {rp.relative_to(P.REPO)}")
    if failed:
        print(f"  ✗ 有問題：{'、'.join(failed)}")
        print("    **不要**把沒通過驗收的檔案拿去 benchmark。")
        sys.exit(1)
    print("  ✓ 全部通過。可以進 Benchmark 流程了。")


if __name__ == "__main__":
    main()
