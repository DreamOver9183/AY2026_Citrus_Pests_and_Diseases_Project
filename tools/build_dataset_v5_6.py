# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5.6 建置腳本。

在 v5.5 的基礎上做三項修正，**都不需要新拍任何一張照片**：

  1. **切分改成逐類絕對評估量，且與 v5.5 巢狀相容**（見 `EVAL_TARGET`）。
     v5.5 的 valid/test 成員**全部保留**，新增的評估影像只從 v5.5 的 train 抽，
     所以 `v5.6 eval ⊇ v5.5 eval`——v10 的數字可以在「v5.5 子集」上重算並直接對照。
     四個類別的評估集擴大之後，全部九類的 pooled ±2SE 都會 ≤ 0.10
     （v11 計畫書 §2 訂的 B 級目標，原本以為要新增 353 張田間照才做得到）。

  2. **逐類增強 profile**（`tools/aug_profiles.py`），取代 v5.5 的單一全域管線。
     其中最重要的一項是**拿掉任意角度旋轉**：實測 v5.5 的增強圖框面積比原始圖
     大 1.075–1.300 倍，`Thrips_Damage` 最嚴重，而它正是定位最差的類別。
     詳細機制與實測數字寫在 `aug_profiles.py` 的檔頭。

  3. **自足的來源樹**。v5.5 的建置腳本有 6 個類別指回 `Datasets_YOLO26_v5r`，
     即使 v5.5 自己的樹裡就有位元相同的副本。v5.6 **只讀 `V56_ROOT`**，
     不跨版本依賴，重現時不必同時保留兩個版本的資料夾。

**實驗臂**（`--arm`）：

  * `base`（預設）——只有上述三項，零人工成本，隨時可建。
  * `cp`  ——額外把 CLM 食痕以 `cv2.seamlessClone` 貼到 train 的健康株影像上。
            需先通過 `人工標註/C_合成圖_看圖打勾/` 的視覺驗收。
  * `ext` ——額外納入 `Pests/Citrus_Leaf_Miner_ext.yolo26` 的外部影像（**只進 train**）。
            需先完成 `人工標註/B_潛葉蛾_補標150張/` 的標註。
  * `all` ——`cp` + `ext`。

`cp` 與 `ext` 都是**配額中性**的：合成圖／外部圖佔用既有的
`min(4×raw, 1200)` 增強配額，不增加 train 總張數，所以與 `base` 是乾淨的 A/B。

用法：
    .venv/Scripts/python.exe tools/build_dataset_v5_6.py --dry-run
    .venv/Scripts/python.exe tools/build_dataset_v5_6.py
    .venv/Scripts/python.exe tools/build_dataset_v5_6.py --arm cp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P    # Datasets/ 的版面配置：單一真實來源

import cv2                                # noqa: E402
import numpy as np                        # noqa: E402
from PIL import Image as PILImage         # noqa: E402

from build_dataset_v5r import (           # noqa: E402  重用 v5r 的解析/寫出邏輯
    IMG_EXT, JPEG_QUALITY, equiv_px, find_image, parse_label, write_sample,
)
from check_dataset_leakage import dhash   # noqa: E402  與洩漏查驗共用同一個感知雜湊
import aug_profiles                       # noqa: E402  逐類增強 profile

REPO = Path(__file__).resolve().parent.parent
V55_OUT = _P.split("v5.5")
V56_ROOT = _P.raw("v5.6")
OUT_ROOT = _P.split("v5.6")
MANUAL_ROOT = _P.MANUAL

SEED = 0
AUG_MULT = 4
AUG_CAP = 1200
APHID_MIN_PX = 20.0
DUP_THRESHOLD = 6           # dHash（64-bit）的 Hamming 距離閾值，與洩漏查驗一致

CLASSES = list(aug_profiles.CLASSES)

# ── 逐類絕對評估量（valid, test）─────────────────────────────────────────
# 只列有變動的類別；沒列到的**完全沿用 v5.5 的切分**，一張都不動。
# 數字由各類自己實測的 ±2SE 依 1/sqrt(n) 回推，目標是 pooled(valid+test) <= 0.10：
#   CLM    test ±2SE 0.210 @20 圖 -> 90 圖 -> 0.099
#   TD     test ±2SE 0.161 @21 圖 -> 80 圖 -> 0.082
#   Thrips test ±2SE 0.170 @39 圖 -> 120 圖 -> 0.097
#   SI     test ±2SE 0.082 @25 圖 -> 70 圖 -> 0.049
EVAL_TARGET: dict[str, tuple[int, int]] = {
    "Citrus_Leaf_Miner": (45, 45),
    "Thrips_Damage":     (40, 40),
    "Thrips":            (60, 60),
    "Scale_Insect":      (35, 35),
}

# v10 實測的 test ±2SE 與當時的 test 影像數，用來投影新切分的 ±2SE（Gate 4 會查）
SE_BASELINE: dict[str, tuple[float, int]] = {
    "Oily_Spot": (0.022, 24), "Canker": (0.061, 24), "Sooty_Mold": (0.030, 32),
    "Black_Spot": (0.043, 25), "Scale_Insect": (0.082, 25),
    "Citrus_Leaf_Miner": (0.210, 20), "Thrips": (0.170, 39),
    "Aphid": (0.048, 76), "Thrips_Damage": (0.161, 21),
}

# root 一律是 V56_ROOT——v5.6 的來源樹是自足的。
# sub_keep: 只保留來源標註裡這個子類 id 的框，None = 全收
SOURCES = [
    dict(cid=0, name="Oily_Spot",         path="Diseases/Greasy Spot",                  sub_keep=None),
    dict(cid=1, name="Canker",            path="Diseases/Canker/train",                 sub_keep=None),
    dict(cid=2, name="Sooty_Mold",        path="Diseases/sooty mold",                   sub_keep=None),
    dict(cid=3, name="Black_Spot",        path="Diseases/Melanose",                     sub_keep=None),
    dict(cid=4, name="Scale_Insect",      path="Pests/Scale_Insect_v5.5.yolo26/train",  sub_keep=None),
    dict(cid=5, name="Citrus_Leaf_Miner", path="Pests/Citrus_Leaf_Miner.yolo26/train",  sub_keep=None),
    dict(cid=6, name="Thrips",            path="Pests/Thrips_v5r.yolo26/train",         sub_keep=0),
    dict(cid=7, name="Aphid",             path="Pests/Aphid.yolo26/train",              sub_keep=0),
    dict(cid=8, name="Thrips_Damage",     path="Pests/Thrips_v5r.yolo26/train",         sub_keep=1),
]

BACKGROUND_SOURCES = [
    dict(name="Background", path="Healthy/Murcott"),
    dict(name="Background", path="Healthy/Ponkan"),
]

# Thrips 與 Thrips_Damage 來自同一批來源檔案、只是依子類分流，增強配額必須合併計算
# 再依 raw 張數等比分配（沿用 v5.5 的作法，見 build_dataset_v5_5.py）。
LINKED_QUOTA_GROUPS = [("Thrips", "Thrips_Damage")]

# cp / ext 臂佔用 CLM 增強配額的張數（配額中性，不增加 train 總量）
CP_QUOTA = 120
EXT_SOURCE = "Pests/Citrus_Leaf_Miner_ext.yolo26/train"
EXT_CID = 5


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_of(stem: str) -> str:
    """由輸出檔名還原類別。長名優先，避免 Thrips_ 誤吃 Thrips_Damage_。"""
    head = stem
    for tag in ("_aug_", "_cp_", "_ext_"):
        head = head.split(tag)[0]
    for c in sorted(CLASSES + ["Background"], key=len, reverse=True):
        if head.startswith(c + "_") or head == c:
            return c
    return "?"


def v55_split_map() -> dict[tuple[str, str], str]:
    """掃 v5.5 的 OutPut，建立 (類別, 影像 md5) -> v5.5 所屬 split 的對照表。

    v5.5 的 raw 樣本是用 `shutil.copy2` 位元複製過去的，所以 md5 與來源檔一致，
    可以直接拿來還原 provenance（手法同 tools/diag_localization.py）。
    """
    out: dict[tuple[str, str], str] = {}
    for sp in ("train", "valid", "test"):
        for p in sorted((V55_OUT / sp / "images").glob("*.jpg")):
            if "_aug_" in p.stem:          # 增強圖不是任何來源檔的副本
                continue
            out[(class_of(p.stem), md5(p))] = sp
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def singleton_flags(items: list[dict], threshold: int = DUP_THRESHOLD) -> list[bool]:
    """標出哪些影像**不屬於任何近重複群組**（群組大小 == 1）。

    只有這些影像可以被提拔進評估集：把近重複群組的成員放進 valid/test，
    模型就有機會在 train 看到幾乎一樣的圖，等於抄解答
    （見 docs/v5r_記錄_近重複影像跨split洩漏查驗.md）。
    """
    if len(items) < 2:
        return [True] * len(items)
    hashes = []
    for it in items:
        h = dhash(it["img"])
        hashes.append(np.zeros(64, dtype=bool) if h is None else h)
    H = np.stack(hashes)
    uf = _UnionFind(len(items))
    BLOCK = 200
    for i0 in range(0, len(items), BLOCK):
        a = H[i0:i0 + BLOCK]
        dist = (a[:, None, :] != H[None, :, :]).sum(-1)
        for bi in range(a.shape[0]):
            gi = i0 + bi
            for gj in range(gi + 1, len(items)):
                if int(dist[bi, gj]) <= threshold:
                    uf.union(gi, gj)
    sizes = Counter(uf.find(i) for i in range(len(items)))
    return [sizes[uf.find(i)] == 1 for i in range(len(items))]


# ══════════════════════════════════════════════════════════════════════
# 收集與切分
# ══════════════════════════════════════════════════════════════════════

def collect(spec: dict, cid: int | None = None) -> tuple[list[dict], Counter]:
    """收集單一來源，套用子類過濾（sub_keep）與 Aphid 極小框過濾。"""
    base = V56_ROOT / spec["path"]
    img_dir, lbl_dir = base / "images", base / "labels"
    items, stats = [], Counter()
    use_cid = spec.get("cid") if cid is None else cid
    if not lbl_dir.is_dir():
        return items, stats
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = find_image(img_dir, lbl.stem)
        if img is None:
            stats["缺影像"] += 1
            continue
        try:
            W, H = PILImage.open(img).size
        except Exception:
            stats["讀取失敗"] += 1
            continue
        raw = parse_label(lbl)
        stats["原始框"] += len(raw)

        boxes = []
        for _cls, cx, cy, w, h in raw:
            if spec.get("sub_keep") is not None and _cls != spec["sub_keep"]:
                continue
            if use_cid == 7 and equiv_px(w, h, W, H) < APHID_MIN_PX:
                stats["Aphid_丟棄極小框"] += 1
                continue
            boxes.append((use_cid, cx, cy, w, h))

        if raw and not boxes:
            stats["整張丟棄"] += 1
            continue
        if not raw:
            stats["來源空標註"] += 1

        items.append(dict(img=img, boxes=boxes, W=W, H=H))
        stats["框"] += len(boxes)
    stats["影像"] = len(items)
    return items, stats


def collect_background(spec: dict) -> list[dict]:
    base = V56_ROOT / spec["path"]
    items = []
    for img in sorted(base.glob("images/*")):
        if img.suffix not in IMG_EXT:
            continue
        try:
            W, H = PILImage.open(img).size
        except Exception:
            continue
        items.append(dict(img=img, boxes=[], W=W, H=H))
    return items


def nested_split(name: str, items: list[dict],
                 v55map: dict[tuple[str, str], str]) -> tuple[dict, dict]:
    """巢狀切分：鎖住 v5.5 的 valid/test，缺額只從 v5.5 的 train 補。

    回傳 (pool, info)。info 帶著補了幾張、候選單張有幾張，供 --dry-run 列印。
    """
    for it in items:
        it["md5"] = md5(it["img"])
        it["v55"] = v55map.get((name, it["md5"]), "未知")

    locked = {sp: [it for it in items if it["v55"] == sp] for sp in ("valid", "test")}
    rest = [it for it in items if it["v55"] not in ("valid", "test")]
    unknown = [it for it in rest if it["v55"] == "未知"]

    target = EVAL_TARGET.get(name)
    if target is None:
        # 沒有指定目標 -> 完全沿用 v5.5 的切分
        pool = {"train": rest, "valid": locked["valid"], "test": locked["test"]}
        return pool, dict(name=name, changed=False, add=(0, 0),
                          singles=0, unknown=len(unknown),
                          locked=(len(locked["valid"]), len(locked["test"])))

    need = {sp: target[i] - len(locked[sp]) for i, sp in enumerate(("valid", "test"))}
    for sp in ("valid", "test"):
        assert need[sp] >= 0, (
            f"{name}: 目標 {sp}={target} 小於 v5.5 已有的 {len(locked[sp])} 張，"
            f"會破壞巢狀相容性")

    flags = singleton_flags(rest)
    cands = [it for it, ok in zip(rest, flags) if ok]
    assert len(cands) >= need["valid"] + need["test"], (
        f"{name}: 需要再補 {need['valid'] + need['test']} 張評估影像，"
        f"但 train 裡只有 {len(cands)} 張沒有近重複的單張影像")

    rng = random.Random(SEED)             # 每類獨立取種子，沿用 v5.5 的作法
    order = list(range(len(cands)))
    rng.shuffle(order)
    picked_v = {id(cands[i]) for i in order[:need["valid"]]}
    picked_t = {id(cands[i]) for i in order[need["valid"]:need["valid"] + need["test"]]}

    pool = {
        "valid": locked["valid"] + [it for it in cands if id(it) in picked_v],
        "test":  locked["test"] + [it for it in cands if id(it) in picked_t],
        "train": [it for it in rest
                  if id(it) not in picked_v and id(it) not in picked_t],
    }
    return pool, dict(name=name, changed=True, add=(need["valid"], need["test"]),
                      singles=len(cands), unknown=len(unknown),
                      locked=(len(locked["valid"]), len(locked["test"])))


# ══════════════════════════════════════════════════════════════════════
# 寫出
# ══════════════════════════════════════════════════════════════════════

def write_augmented(item: dict, compose, out_img: Path, out_lbl: Path) -> bool:
    """套用該類別專屬的 profile 寫出一張增強圖。"""
    im = cv2.imread(str(item["img"]))
    if im is None:
        return False
    bb = [[cx, cy, w, h] for _c, cx, cy, w, h in item["boxes"]]
    lb = [c for c, *_ in item["boxes"]]
    try:
        res = compose(image=im, bboxes=bb, class_labels=lb)
    except Exception:
        return False
    out = [(int(c), *map(float, b)) for b, c in zip(res["bboxes"], res["class_labels"])]
    if item["boxes"] and not out:
        return False                      # 增強後標註全失效，捨棄
    cv2.imwrite(str(out_img), res["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    out_lbl.write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in out),
        encoding="utf-8",
    )
    return True


def project_se(name: str, n_valid: int, n_test: int) -> float:
    """把 v10 實測的 test ±2SE 依 1/sqrt(n) 投影到新的 pooled 評估集大小。"""
    se, n0 = SE_BASELINE[name]
    n = max(n_valid + n_test, 1)
    return se * (n0 / n) ** 0.5


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只統計，不寫檔")
    ap.add_argument("--arm", default="base", choices=("base", "cp", "ext", "all"))
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out)
    use_cp = args.arm in ("cp", "all")
    use_ext = args.arm in ("ext", "all")

    print("═" * 82)
    print(f"  Datasets_YOLO26_v5.6 建置   arm={args.arm}   "
          f"ROTATE_MODE={aug_profiles.ROTATE_MODE}")
    print("═" * 82)

    assert V56_ROOT.is_dir(), f"找不到來源樹 {V56_ROOT}"
    assert V55_OUT.is_dir(), f"找不到 v5.5 產出 {V55_OUT}（巢狀切分需要它當基準）"

    # ── 1. 讀 v5.5 的切分當基準 ─────────────────────────────────────────
    print("\n【1】讀 v5.5 的切分當巢狀基準")
    v55map = v55_split_map()
    n55 = Counter(v55map.values())
    print(f"  v5.5 raw 影像 {len(v55map):,} 張"
          f"（train {n55['train']:,} / valid {n55['valid']:,} / test {n55['test']:,}）")

    # ── 2. 收集 + 巢狀切分 ──────────────────────────────────────────────
    print(f"\n【2】收集來源並巢狀切分")
    print(f"{'類別':<20}{'來源圖':>7}{'框':>7}{'v5.5 v/t':>11}"
          f"{'v5.6 v/t':>11}{'補進eval':>9}{'單張候選':>9}")
    pools: dict[str, dict] = {}
    infos: dict[str, dict] = {}
    for spec in SOURCES:
        items, st = collect(spec)
        assert items, f"{spec['name']}: 來源 {spec['path']} 沒收到任何樣本"
        pool, info = nested_split(spec["name"], items, v55map)
        pools[spec["name"]] = pool
        infos[spec["name"]] = info
        lv, lt = info["locked"]
        nv, nt = len(pool["valid"]), len(pool["test"])
        add = f"+{info['add'][0]}/+{info['add'][1]}" if info["changed"] else "—"
        sg = f"{info['singles']:,}" if info["changed"] else "—"
        print(f"{spec['name']:<20}{st['影像']:>7,}{st['框']:>7,}"
              f"{f'{lv}/{lt}':>11}{f'{nv}/{nt}':>11}{add:>9}{sg:>9}")
        if info["unknown"]:
            print(f"{'':<20}⚠ 有 {info['unknown']} 張在 v5.5 找不到對應（會歸入 train）")

    bg = []
    for spec in BACKGROUND_SOURCES:
        bg += collect_background(spec)
    bg_pool, bg_info = nested_split("Background", bg, v55map)
    pools["Background"] = bg_pool
    infos["Background"] = bg_info
    bg_old = "{}/{}".format(*bg_info["locked"])
    bg_new = "{}/{}".format(len(bg_pool["valid"]), len(bg_pool["test"]))
    print(f"{'Background(健康株)':<20}{len(bg):>7,}{0:>7,}"
          f"{bg_old:>11}{bg_new:>11}{'—':>9}{'—':>9}")

    # ── 3. 巢狀相容性自檢 ───────────────────────────────────────────────
    print("\n【3】巢狀相容性：v5.5 的評估影像是否全部仍在 v5.6 的評估集裡")
    bad = []
    for name, pool in pools.items():
        keep = {it["md5"] for it in pool["valid"] + pool["test"]}
        for it in pool["train"]:
            if it["v55"] in ("valid", "test") and it["md5"] not in keep:
                bad.append((name, it["img"].name, it["v55"]))
    if bad:
        for n, f, sp in bad[:10]:
            print(f"  ✗ {n}/{f} 原本在 v5.5 的 {sp}，現在掉到 train")
        raise SystemExit(f"巢狀相容性檢查失敗，{len(bad)} 張")
    print("  ✓ 全部通過——v5.6 的評估集是 v5.5 評估集的超集")

    # ── 4. ±2SE 投影 ────────────────────────────────────────────────────
    print(f"\n【4】pooled ±2SE 投影（v10 實測值依 1/sqrt(n) 外推）")
    print(f"{'類別':<20}{'v5.5 eval':>10}{'v5.6 eval':>10}"
          f"{'v5.5 ±2SE':>11}{'v5.6 ±2SE':>11}{'':>4}")
    worst = 0.0
    for name in CLASSES:
        info, pool = infos[name], pools[name]
        o_v, o_t = info["locked"]
        n_v, n_t = len(pool["valid"]), len(pool["test"])
        before, after = project_se(name, o_v, o_t), project_se(name, n_v, n_t)
        worst = max(worst, after)
        mark = "✓" if after <= 0.10 else "✗"
        print(f"{name:<20}{o_v + o_t:>10,}{n_v + n_t:>10,}"
              f"{before:>11.3f}{after:>11.3f}{mark:>4}")
    print(f"{'最差':<20}{'':>10}{'':>10}{'':>11}{worst:>11.3f}"
          f"{'✓' if worst <= 0.10 else '✗':>4}")

    # ── 5. 外部來源（ext 臂）────────────────────────────────────────────
    ext_items: list[dict] = []
    if use_ext:
        ext_spec = dict(name="Citrus_Leaf_Miner_ext", path=EXT_SOURCE, sub_keep=None)
        ext_items, ext_st = collect(ext_spec, cid=EXT_CID)
        print(f"\n【5】外部試點（--arm {args.arm}）：{len(ext_items):,} 張 / "
              f"{ext_st['框']:,} 框，**只進 train**")
        if not ext_items:
            raise SystemExit(
                f"--arm {args.arm} 需要 {V56_ROOT / EXT_SOURCE} 有已標註的影像，"
                f"目前是空的。請先完成 人工標註/B_潛葉蛾_補標150張/ 的工作。")

    # ── 6. 增強配額 ─────────────────────────────────────────────────────
    print(f"\n【6】增強配額（target = min({AUG_MULT}x raw, {AUG_CAP})）")
    print(f"{'類別':<20}{'raw train':>11}{'目標':>8}{'合成':>7}{'外部':>7}{'一般增強':>10}")
    quota: dict[str, int] = {}
    cp_n: dict[str, int] = defaultdict(int)
    ext_n: dict[str, int] = defaultdict(int)
    linked = {m for grp in LINKED_QUOTA_GROUPS for m in grp}
    for a, b in LINKED_QUOTA_GROUPS:
        ra, rb = len(pools[a]["train"]), len(pools[b]["train"])
        tot = min(AUG_MULT * (ra + rb), AUG_CAP) if (ra + rb) else 0
        ta = round(tot * ra / (ra + rb)) if (ra + rb) else 0
        quota[a], quota[b] = max(0, ta - ra), max(0, tot - ta - rb)
    for name in CLASSES + ["Background"]:
        if name in linked:
            continue
        raw = len(pools[name]["train"])
        tgt = raw if name == "Background" else min(AUG_MULT * raw, AUG_CAP)
        quota[name] = max(0, tgt - raw)
    # cp / ext 從 CLM 的配額裡切，不額外增加張數
    if use_cp:
        cp_n["Citrus_Leaf_Miner"] = min(CP_QUOTA, quota["Citrus_Leaf_Miner"])
        quota["Citrus_Leaf_Miner"] -= cp_n["Citrus_Leaf_Miner"]
    if use_ext:
        ext_n["Citrus_Leaf_Miner"] = min(len(ext_items), quota["Citrus_Leaf_Miner"])
        quota["Citrus_Leaf_Miner"] -= ext_n["Citrus_Leaf_Miner"]
    for name in CLASSES + ["Background"]:
        raw = len(pools[name]["train"])
        tgt = raw + quota[name] + cp_n[name] + ext_n[name]
        print(f"{name:<20}{raw:>11,}{tgt:>8,}{cp_n[name]:>7,}"
              f"{ext_n[name]:>7,}{quota[name]:>10,}")

    if args.dry_run:
        print("\n--dry-run：不寫檔，結束。")
        return

    # ── 7. 寫出 ─────────────────────────────────────────────────────────
    if out_root.exists():
        shutil.rmtree(out_root)
    for sp in ("train", "valid", "test"):
        (out_root / sp / "images").mkdir(parents=True, exist_ok=True)
        (out_root / sp / "labels").mkdir(parents=True, exist_ok=True)

    print("\n【7】寫出")
    counters: dict[str, Counter] = defaultdict(Counter)
    prov: list[dict] = []

    def rel(p: Path) -> str:
        try:
            return p.relative_to(V56_ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    for name in CLASSES + ["Background"]:
        compose = aug_profiles.build_profile(name)
        for sp in ("train", "valid", "test"):
            for k, it in enumerate(pools[name][sp]):
                stem = f"{name}_{k:05d}"
                write_sample(it["img"], it["boxes"],
                             out_root / sp / "images" / f"{stem}.jpg",
                             out_root / sp / "labels" / f"{stem}.txt")
                counters[sp][name] += 1
                prov.append(dict(out=f"{sp}/images/{stem}.jpg", kind="raw", cls=name,
                                 src=rel(it["img"]), src_md5=it["md5"],
                                 v55_split=it["v55"], split=sp))

        # 外部影像：只進 train
        for k, it in enumerate(ext_items if name == "Citrus_Leaf_Miner" else []):
            if k >= ext_n[name]:
                break
            stem = f"{name}_ext_{k:05d}"
            write_sample(it["img"], it["boxes"],
                         out_root / "train" / "images" / f"{stem}.jpg",
                         out_root / "train" / "labels" / f"{stem}.txt")
            counters["train"][name] += 1
            prov.append(dict(out=f"train/images/{stem}.jpg", kind="external", cls=name,
                             src=rel(it["img"]), src_md5=md5(it["img"]),
                             v55_split="外部", split="train"))

        # 一般增強：來源一律取自 train
        need, made, guard = quota[name], 0, 0
        src = pools[name]["train"]
        while made < need and src and guard < need * 5:
            it = src[(made + guard) % len(src)]
            stem = f"{name}_aug_{made:05d}"
            ok = write_augmented(it, compose,
                                 out_root / "train" / "images" / f"{stem}.jpg",
                                 out_root / "train" / "labels" / f"{stem}.txt")
            guard += 1
            if ok:
                prov.append(dict(out=f"train/images/{stem}.jpg", kind="aug", cls=name,
                                 src=rel(it["img"]), src_md5=it["md5"],
                                 v55_split=it["v55"], split="train",
                                 profile=name if name in aug_profiles.PROFILE_SPECS
                                 else "_default"))
                made += 1
                counters["train"][name] += 1
        if need:
            print(f"  {name:<20} 增強 {made:,}/{need:,}")

    # copy-paste 合成（cp 臂）
    if use_cp and cp_n["Citrus_Leaf_Miner"]:
        from copy_paste_clm import synthesize          # noqa: E402  只在需要時才載入
        made = synthesize(
            sources=pools["Citrus_Leaf_Miner"]["train"],
            targets=pools["Background"]["train"],
            n=cp_n["Citrus_Leaf_Miner"],
            out_img_dir=out_root / "train" / "images",
            out_lbl_dir=out_root / "train" / "labels",
            prefix="Citrus_Leaf_Miner_cp", seed=SEED, prov=prov,
        )
        counters["train"]["Citrus_Leaf_Miner"] += made
        print(f"  {'Citrus_Leaf_Miner':<20} 合成 {made:,}/{cp_n['Citrus_Leaf_Miner']:,}")

    # ── 8. data.yaml 與 provenance ──────────────────────────────────────
    yaml_text = (
        "# Datasets_YOLO26_v5.6 —— 由 tools/build_dataset_v5_6.py 產生\n"
        f"# seed={SEED}  arm={args.arm}  rotate_mode={aug_profiles.ROTATE_MODE}\n"
        f"# 切分：逐類絕對評估量，且與 v5.5 巢狀相容（v5.6 eval 是 v5.5 eval 的超集）\n"
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  - {c}\n" for c in CLASSES)
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    (out_root / "_provenance.json").write_text(json.dumps(dict(
        meta=dict(
            built=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            arm=args.arm, seed=SEED, rotate_mode=aug_profiles.ROTATE_MODE,
            aug_mult=AUG_MULT, aug_cap=AUG_CAP, dup_threshold=DUP_THRESHOLD,
            eval_target={k: list(v) for k, v in EVAL_TARGET.items()},
            classes=CLASSES,
        ),
        records=prov,
    ), ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 9. 總結 ─────────────────────────────────────────────────────────
    print(f"\n【8】完成  ->  {out_root}")
    print(f"\n{'類別':<20}{'train':>9}{'valid':>8}{'test':>8}")
    for name in CLASSES + ["Background"]:
        print(f"{name:<20}{counters['train'][name]:>9,}"
              f"{counters['valid'][name]:>8,}{counters['test'][name]:>8,}")
    print(f"{'合計':<20}{sum(counters['train'].values()):>9,}"
          f"{sum(counters['valid'].values()):>8,}{sum(counters['test'].values()):>8,}")
    print(f"\nprovenance：{len(prov):,} 筆  ->  {out_root / '_provenance.json'}")


if __name__ == "__main__":
    main()
