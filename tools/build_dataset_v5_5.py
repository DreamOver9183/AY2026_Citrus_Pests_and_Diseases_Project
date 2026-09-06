# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5.5 建置腳本。

在 v5r 的基礎上做三項資料端修正：

  1. **Scale_Insect 全面重標**（來源改到 `Datasets_YOLO26_v5.5/Pests/Scale_Insect_v5.5.yolo26`，
     1,119→251 張、15,379→1,179 框）。池內近重複已查驗為 0 組
     （`docs/v5r_記錄_近重複影像跨split洩漏查驗.md`），洩漏問題從根源解決。
     連帶移除 v5r 的 `downsample_scale()` 特例——重標後這一類不再獨佔 raw pool
     （71.2%→15.5%），沒有理由再保留特殊處理，改走與其他類別相同的標準增強路徑。
  2. **Canker 全面重標**（來源改到 `Datasets_YOLO26_v5.5/Diseases/Canker`）。
     標註格式由幾乎全多邊形改為 100% 矩形，對應
     `docs/v9_說明_P3定位精度改善與測試流程.md` §2.4 的建議
     （「重標時直接畫矩形，不要走多邊形，否則沿黃暈畫的外接矩形會系統性偏大」）。
  3. **Thrips 依來源子類拆成兩個輸出類別**：`Thrips`（僅 `Thysanoptera` 蟲體）／
     `Thrips_Damage`（僅 `thirps_leaf_damage` 葉害，新增類別 id 8）。
     **Aphid 丟棄 `Aphid_Leaf_Damage` 子類**（96 框／34 圖，整張影像移除）。
     兩者的來源檔案與 v5r 完全相同，只是輸出時依標註子類分流／過濾。
  4. **近重複去洩漏：群組整群移進 train，不刪任何影像**（見 `move_duplicate_clusters_to_train`）。
     切分與影像處理方式完全照舊，只在切分之後多一道搬移：dHash 距離 ≤ 6 的
     近重複群組（≥2 張）全部集中到 train，valid/test 只留沒有近重複的影像。
     模型因此不可能在 train 看到評估集的答案，而資料一張都沒少。

**與 `build_dataset_v5r.py`的關鍵差異——每個類別各自獨立切分：**

v5r 用單一 `random.Random(SEED)` 依序穿過所有類別的 `split_items()` 呼叫，
任何一個類別的來源張數改變，都會連帶重排後面所有類別的切分結果。
本次 Scale_Insect（1,119→251）與 Canker（244→241）的張數已經改變，若沿用同一顆
共用的 rng，即使 Thrips／Aphid 的來源檔案完全沒動，它們在 v5.5 的切分也會被迫
偏離 v5r——而且是以一種難以追蹤、純粹取決於處理順序的方式偏離。

因此本檔改為**每個類別各自用獨立的 `random.Random(SEED)`**：一個類別的資料異動
不會再連帶影響任何其他類別的 train/valid/test 歸屬，往後只改一個類別時更安全。

**代價（無法避免，需明確承擔）**：即使如此，Thrips／Aphid 在 v5.5 的切分
**仍然不等於**在 v5r 的切分——因為 v5r 是「單一共用 rng 依序穿過全部類別」，
而任何獨立取種子的方案，重新播放到 Thrips／Aphid 時的 rng 狀態都不可能與 v5r
原始那次跑法一致。**P3 診斷用的「固定評估子集」（valid 299/test 298 框）在
v5.5 上不成立**，需要在建置完成後對 v5.5 重新跑 `tools/diag_localization.py`
與 `tools/quantify_leak_impact.py` 建立新的基準與 bootstrap 門檻。

`build_dataset_v5r.py` 本身完全未改動，v5r 的既有產出與可重現性不受影響。

用法：
    .venv/Scripts/python.exe tools/build_dataset_v5_5.py
    .venv/Scripts/python.exe tools/build_dataset_v5_5.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P    # Datasets/ 的版面配置：單一真實來源
from build_dataset_v5r import (          # noqa: E402  重用 v5r 的解析/寫出邏輯，方法論保持一致
    AUG, IMG_EXT, equiv_px, find_image, parse_label, write_augmented, write_sample,
)
from check_dataset_leakage import dhash  # noqa: E402  與洩漏查驗共用同一個感知雜湊定義
from PIL import Image as PILImage        # noqa: E402

import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
V5R_ROOT = _P.raw("v5r")
V55_ROOT = _P.raw("v5.5")
OUT_ROOT = _P.split("v5.5")

SEED = 0
SPLIT = (0.80, 0.10, 0.10)          # 與 v5r 相同的切分比例，但每類獨立取種子
AUG_MULT = 4
AUG_CAP = 1200
APHID_MIN_PX = 20.0
DUP_THRESHOLD = 6                   # dHash（64-bit）的 Hamming 距離閾值，與洩漏查驗一致

CLASSES = [
    "Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
    "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid", "Thrips_Damage",
]

# root: 來源根目錄；sub_keep: 只保留來源標註裡這個子類 id 的框，None = 全收
SOURCES = [
    dict(cid=0, name="Oily_Spot",         root=V5R_ROOT, path="Diseases/Greasy Spot",                  sub_keep=None),
    dict(cid=1, name="Canker",            root=V55_ROOT, path="Diseases/Canker/train",                 sub_keep=None),
    dict(cid=2, name="Sooty_Mold",        root=V5R_ROOT, path="Diseases/sooty mold",                   sub_keep=None),
    dict(cid=3, name="Black_Spot",        root=V5R_ROOT, path="Diseases/Melanose",                     sub_keep=None),
    dict(cid=4, name="Scale_Insect",      root=V55_ROOT, path="Pests/Scale_Insect_v5.5.yolo26/train",  sub_keep=None),
    dict(cid=5, name="Citrus_Leaf_Miner", root=V5R_ROOT, path="Pests/Citrus_Leaf_Miner.yolo26/train",  sub_keep=None),
    dict(cid=6, name="Thrips",            root=V5R_ROOT, path="Pests/Thrips_v5r.yolo26/train",         sub_keep=0),
    dict(cid=7, name="Aphid",             root=V5R_ROOT, path="Pests/Aphid.yolo26/train",              sub_keep=0),
    dict(cid=8, name="Thrips_Damage",     root=V5R_ROOT, path="Pests/Thrips_v5r.yolo26/train",         sub_keep=1),
]

BACKGROUND_SOURCES = [
    dict(name="Background", root=V5R_ROOT, path="Healthy/Murcott"),
    dict(name="Background", root=V5R_ROOT, path="Healthy/Ponkan"),
]

# Thrips 與 Thrips_Damage 來自同一批來源檔案、只是依子類分流，增強配額必須合併計算
# 再依 raw 張數等比分配。各自獨立套 min(4×raw,1200) 會讓葉害子域的樣本權重從
# 1200 漲到 1884，等於同時改了類別定義與子域比重，無法歸因
#（見 docs/v9_說明_P3定位精度改善與測試流程.md §2.3 約束 B）。
LINKED_QUOTA_GROUPS = [("Thrips", "Thrips_Damage")]


def collect(spec: dict) -> tuple[list[dict], Counter]:
    """收集單一來源的所有樣本，套用子類過濾（sub_keep）與 Aphid 極小框過濾。"""
    base = spec["root"] / spec["path"]
    img_dir, lbl_dir = base / "images", base / "labels"
    items, stats = [], Counter()
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
            if spec["cid"] == 7 and equiv_px(w, h, W, H) < APHID_MIN_PX:
                stats["Aphid_丟棄極小框"] += 1
                continue
            boxes.append((spec["cid"], cx, cy, w, h))

        if raw and not boxes:
            # 子類不符（例如 Thrips 只收蟲體、這張是葉害）或過濾後全空 → 整張丟棄
            stats["整張丟棄"] += 1
            continue
        if not raw:
            stats["來源空標註"] += 1

        items.append(dict(img=img, boxes=boxes, W=W, H=H))
        stats["框"] += len(boxes)
    stats["影像"] = len(items)
    return items, stats


def collect_background(spec: dict) -> list[dict]:
    base = spec["root"] / spec["path"]
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


def split_items(items: list[dict], seed: int) -> dict[str, list[dict]]:
    """每個類別各自獨立取種子切分——與 build_dataset_v5r.py 的差異，見檔頭說明。"""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(round(n * SPLIT[0]))
    n_va = int(round(n * SPLIT[1]))
    return {
        "train": [items[i] for i in idx[:n_tr]],
        "valid": [items[i] for i in idx[n_tr:n_tr + n_va]],
        "test":  [items[i] for i in idx[n_tr + n_va:]],
    }


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


def move_duplicate_clusters_to_train(pool: dict[str, list[dict]],
                                     threshold: int = DUP_THRESHOLD) -> int:
    """把近重複影像群組整群移進 train，就地修改 `pool`，回傳搬移張數。

    `docs/v5r_記錄_近重複影像跨split洩漏查驗.md` 查到的洩漏，全部來自同一場景的
    連拍相鄰影格或同一張原圖的重複匯出被隨機切分到不同 split。這裡不刪除任何影像，
    只是把**同一個近重複群組（≥2 張）的成員全部集中到 train**：

      * train 裡出現重複只是輕微冗餘，模型不會因此看到評估集的答案；
      * valid/test 只保留在該類別中沒有任何近重複的影像，模型不可能「抄解答」；
      * 順帶消除 valid↔test 之間的重複——本專案用「valid 與 test 同號」當顯著性
        的實務判準（見 docs/v9_說明_P3定位精度改善與測試流程.md §3.1），
        兩個評估集若共用近重複影像，一致性會被虛假墊高。

    以群組（連通分量）而非單一配對為單位，是為了處理連拍序列這種 3 張以上的情況：
    只搬其中一張仍會在剩下的成員之間留下跨 split 的重複。
    """
    items = pool["train"] + pool["valid"] + pool["test"]
    if len(items) < 2:
        return 0

    hashes = []
    for it in items:
        h = dhash(it["img"])
        if h is None:                     # 讀不到就當作獨立影像，不參與分群
            h = np.zeros(64, dtype=bool)
        hashes.append(h)
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

    sizes: Counter = Counter(uf.find(i) for i in range(len(items)))
    in_cluster = {id(items[i]) for i in range(len(items)) if sizes[uf.find(i)] > 1}

    moved = 0
    for sp in ("valid", "test"):
        keep, move = [], []
        for it in pool[sp]:
            (move if id(it) in in_cluster else keep).append(it)
        pool[sp] = keep
        pool["train"].extend(move)
        moved += len(move)
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只統計，不寫檔")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out)

    print("═" * 78)
    print("  Datasets_YOLO26_v5.5 建置")
    print("═" * 78)

    # ── 1. 收集（每類獨立切分）────────────────────────────────────────────
    pools: dict[str, dict[str, list[dict]]] = {}
    print(f"\n【1】收集來源\n{'類別':<20}{'影像':>7}{'框':>8}{'來源':>6}  備註")
    for spec in SOURCES:
        items, st = collect(spec)
        note = []
        if st["Aphid_丟棄極小框"]:
            note.append(f"丟棄極小框 {st['Aphid_丟棄極小框']}")
        if st["整張丟棄"]:
            note.append(f"整張丟棄 {st['整張丟棄']}")
        if st["來源空標註"]:
            note.append(f"空標註 {st['來源空標註']}")
        root_tag = "v5.5" if spec["root"] == V55_ROOT else "v5r"
        print(f"{spec['name']:<20}{st['影像']:>7,}{st['框']:>8,}{root_tag:>6}  {'  '.join(note)}")
        pools[spec["name"]] = split_items(items, SEED)

    bg = []
    for spec in BACKGROUND_SOURCES:
        bg += collect_background(spec)
    print(f"{'Background(健康株)':<20}{len(bg):>7,}{0:>8,}{'v5r':>6}  標註清空為負樣本")
    pools["Background"] = split_items(bg, SEED)

    # ── 2. 近重複群組整群移進 train（不刪任何影像）──────────────────────
    print(f"\n【2】近重複去洩漏（dHash 距離 ≤ {DUP_THRESHOLD} 的群組整群移進 train，不刪圖）")
    print(f"{'類別':<20}{'移進train':>10}{'valid':>8}{'test':>7}")
    total_moved = 0
    for name in CLASSES:
        before_v, before_t = len(pools[name]["valid"]), len(pools[name]["test"])
        moved = move_duplicate_clusters_to_train(pools[name])
        total_moved += moved
        if moved:
            after_v, after_t = len(pools[name]["valid"]), len(pools[name]["test"])
            print(f"{name:<20}{moved:>10,}{f'{before_v}→{after_v}':>8}{f'{before_t}→{after_t}':>7}")
    print(f"{'合計':<20}{total_moved:>10,}")

    # ── 4. 增強配額（Scale_Insect 改走標準路徑；Thrips/Thrips_Damage 合併配額）──
    print(f"\n【3】增強配額（target = min({AUG_MULT}× raw, {AUG_CAP})，"
          f"Scale_Insect 這次也走標準增強，不再降採樣）")
    print(f"{'類別':<20}{'raw train':>11}{'目標':>8}{'需增強':>8}")
    quota: dict[str, int] = {}
    linked_members = {m for grp in LINKED_QUOTA_GROUPS for m in grp}
    for a, b in LINKED_QUOTA_GROUPS:
        ra, rb = len(pools[a]["train"]), len(pools[b]["train"])
        tgt_total = min(AUG_MULT * (ra + rb), AUG_CAP) if (ra + rb) else 0
        ta = round(tgt_total * ra / (ra + rb)) if (ra + rb) else 0
        tb = tgt_total - ta
        quota[a] = max(0, ta - ra)
        quota[b] = max(0, tb - rb)
        print(f"{a:<20}{ra:>11,}{ta:>8,}{quota[a]:>8,}   （與 {b} 共用配額）")
        print(f"{b:<20}{rb:>11,}{tb:>8,}{quota[b]:>8,}   （與 {a} 共用配額）")
    for name in CLASSES + ["Background"]:
        if name in linked_members:
            continue
        raw = len(pools[name]["train"])
        tgt = raw if name == "Background" else min(AUG_MULT * raw, AUG_CAP)
        quota[name] = max(0, tgt - raw)
        print(f"{name:<20}{raw:>11,}{tgt:>8,}{quota[name]:>8,}")

    if args.dry_run:
        print("\n--dry-run：不寫檔，結束。")
        return

    # ── 5. 寫出 ────────────────────────────────────────────────────────
    if out_root.exists():
        shutil.rmtree(out_root)
    for sp in ("train", "valid", "test"):
        (out_root / sp / "images").mkdir(parents=True, exist_ok=True)
        (out_root / sp / "labels").mkdir(parents=True, exist_ok=True)

    print("\n【4】寫出")
    counters: dict[str, Counter] = defaultdict(Counter)
    for name in CLASSES + ["Background"]:
        for sp in ("train", "valid", "test"):
            items = pools[name][sp]
            for k, it in enumerate(items):
                stem = f"{name}_{k:05d}"
                write_sample(it["img"], it["boxes"],
                             out_root / sp / "images" / f"{stem}.jpg",
                             out_root / sp / "labels" / f"{stem}.txt")
                counters[sp][name] += 1
        need, made, guard = quota[name], 0, 0
        src = pools[name]["train"]
        while made < need and src and guard < need * 5:
            it = src[(made + guard) % len(src)]
            stem = f"{name}_aug_{made:05d}"
            ok = write_augmented(it,
                                 out_root / "train" / "images" / f"{stem}.jpg",
                                 out_root / "train" / "labels" / f"{stem}.txt")
            guard += 1
            if ok:
                made += 1
                counters["train"][name] += 1
        if need:
            print(f"  {name:<20} 增強 {made:,}/{need:,}")

    # ── 6. data.yaml ───────────────────────────────────────────────────
    yaml_text = (
        "# Datasets_YOLO26_v5.5 —— 由 tools/build_dataset_v5_5.py 產生\n"
        f"# seed={SEED}（每類別各自獨立切分，不共用單一 rng）  split={SPLIT}  先切分後增強\n"
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  - {c}\n" for c in CLASSES)
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # ── 7. 總結 ────────────────────────────────────────────────────────
    print(f"\n【5】完成  →  {out_root}")
    print(f"\n{'類別':<20}{'train':>9}{'valid':>8}{'test':>8}")
    for name in CLASSES + ["Background"]:
        print(f"{name:<20}{counters['train'][name]:>9,}{counters['valid'][name]:>8,}{counters['test'][name]:>8,}")
    print(f"{'合計':<20}{sum(counters['train'].values()):>9,}"
          f"{sum(counters['valid'].values()):>8,}{sum(counters['test'].values()):>8,}")


if __name__ == "__main__":
    main()
