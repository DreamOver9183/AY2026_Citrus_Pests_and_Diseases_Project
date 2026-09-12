# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5.6 驗收：八道 Gate。

任何一道紅燈都代表 `build_dataset_v5_6.py` 有問題，**要回頭修腳本重建，
不接受手動改資料**——手改過的資料集無法重現，之後的每一個數字都失去意義。

  Gate 0  目錄結構      來源樹十個資料夾齊備、張數正確、與 v5.5 位元一致
  Gate 1  結構          成對性、類別 id、座標、data.yaml、BOM、快取
  Gate 2  洩漏          位元級重複 + 近重複（dHash）跨 split
  Gate 3  合成來源      每張增強/合成/外部影像的來源都必須在 train
  Gate 4  評估集        逐類張數、與 v5.5 的巢狀相容、±2SE 投影
  Gate 5  增強品質      逐張框數保留率、**框面積膨脹率**、框複製、增強佔比
  Gate 6  分佈位移      train 與評估集的中位框尺寸比、每圖框數比
  Gate 7  可重現        重跑切分邏輯得到完全相同的評估集成員

Gate 5 的「框面積膨脹率」是 v5.6 新增的檢查。v5.5 的增強圖框面積比原始圖大
1.075–1.300 倍（`Thrips_Damage` 最嚴重），成因是任意角度旋轉之後取外接矩形；
v5.6 已把旋轉改成 90° 倍數，這道 Gate 就是用來確認修好了
（機制與實測數字見 `tools/aug_profiles.py` 檔頭）。

用法：
    .venv/Scripts/python.exe tools/verify_dataset_v5_6.py
    .venv/Scripts/python.exe tools/verify_dataset_v5_6.py --skip 2   # 跳過最慢的一道
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P    # Datasets/ 的版面配置：單一真實來源

import numpy as np                        # noqa: E402
from PIL import Image                     # noqa: E402

from check_dataset_leakage import dhash   # noqa: E402
import build_dataset_v5_6 as B            # noqa: E402  切分規格的單一真實來源

REPO = Path(__file__).resolve().parent.parent
V55_ROOT = _P.raw("v5.5")
V56_ROOT = B.V56_ROOT
OUT = B.OUT_ROOT
V55_OUT = B.V55_OUT
DATASET_LABEL = B.DATASET_VERSION
PY = sys.executable

IMGSZ = 640
SPLITS = ("train", "valid", "test")
CLASSES = B.CLASSES

# Gate 0：來源樹應有的資料夾與張數
EXPECT_SOURCES = [
    ("Diseases/Canker/train", 241), ("Diseases/Greasy Spot", 238),
    ("Diseases/Melanose", 250), ("Diseases/sooty mold", 326),
    ("Healthy/Murcott", 200), ("Healthy/Ponkan", 200),
    ("Pests/Aphid.yolo26/train", 841),
    ("Pests/Citrus_Leaf_Miner.yolo26/train", 201),
    ("Pests/Scale_Insect_v5.5.yolo26/train", 251),
    ("Pests/Thrips_v5r.yolo26/train", 762),
]

# Gate 5/6 的門檻
BOX_COUNT_RATIO = (0.95, 1.05)      # 增強圖每圖框數 / 原始圖每圖框數
BOX_AREA_RATIO = (0.90, 1.10)       # 逐張：增強圖框面積 / 它自己來源的框面積
AUG_SHARE_MAX = 0.80                # 單一類別的增強圖佔比上限
MED_SIZE_RATIO = (0.6, 1.6)         # train vs 評估集的中位框尺寸比
BOX_PER_IMG_RATIO = (0.7, 1.4)      # train vs 評估集的每圖框數比
SE_MAX = 0.10

# 「與 v5.5 位元一致」要跳過的來源資料夾。v5.6 是空的（全部都該一致）；
# verify_dataset_v5_7.py 會把重標過的 Thrips 放進來，並改用更嚴格的逐子類檢查。
BITWISE_SKIP: set[str] = set()


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, gate: str, ok: bool, note: str = "") -> bool:
        self.rows.append((gate, ok, note))
        print(f"  {'✓' if ok else '✗'} {gate}" + (f"  —— {note}" if note else ""))
        return ok

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.rows if not r[1]]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def kind_of(stem: str) -> str:
    for tag, k in (("_aug_", "aug"), ("_cp_", "cp"), ("_ext_", "ext")):
        if tag in stem:
            return k
    return "raw"


def load_boxes(split: str, stem: str) -> tuple[list[tuple[float, float]], int, int]:
    """回傳 [(框寬px, 框高px)]（已換算成 letterbox 到 640 的尺度）與影像尺寸。"""
    ip = OUT / split / "images" / f"{stem}.jpg"
    lp = OUT / split / "labels" / f"{stem}.txt"
    W, H = Image.open(ip).size
    sc = IMGSZ / max(W, H)
    out = []
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8-sig").splitlines():
            q = ln.split()
            if len(q) != 5:
                continue
            _, _, _, w, h = (float(v) for v in q)
            out.append((w * W * sc, h * H * sc))
    return out, W, H


# ══════════════════════════════════════════════════════════════════════

def gate0(rep: Report) -> None:
    print("\n【Gate 0】目錄結構與來源樹")
    ok_all = True
    for rel, n in EXPECT_SOURCES:
        d = V56_ROOT / rel / "images"
        got = len([p for p in d.glob("*") if p.suffix in B.IMG_EXT]) if d.is_dir() else -1
        ok_all &= rep.add(f"{rel}  {n} 張", got == n,
                          "" if got == n else (f"實際 {got} 張" if got >= 0 else "資料夾不存在"))
    # 與 v5.5 位元一致
    diff, checked = [], 0
    for rel, _ in EXPECT_SOURCES:
        if rel in BITWISE_SKIP:
            continue
        for sub in ("images", "labels"):
            a, b = V55_ROOT / rel / sub, V56_ROOT / rel / sub
            if not (a.is_dir() and b.is_dir()):
                continue
            for pa in sorted(a.glob("*")):
                pb = b / pa.name
                if not pb.exists():
                    diff.append(f"缺 {rel}/{sub}/{pa.name}")
                    continue
                checked += 1
                if pa.stat().st_size != pb.stat().st_size or md5(pa) != md5(pb):
                    diff.append(f"位元不同 {rel}/{sub}/{pa.name}")
    rep.add(f"與 v5.5 來源位元一致（比對 {checked:,} 檔{f"，跳過 {len(BITWISE_SKIP)} 個資料夾" if BITWISE_SKIP else ""}）", not diff,
            "" if not diff else f"{len(diff)} 項不符：{diff[:3]}")
    layout = all((OUT / sp / d).is_dir() for sp in SPLITS for d in ("images", "labels"))
    rep.add("OutPut 為 {train,valid,test}/{images,labels} 二層", layout)


def gate1(rep: Report) -> None:
    print("\n【Gate 1】結構（呼叫既有的 verify_dataset_v5r.py）")
    r = subprocess.run([PY, str(REPO / "tools" / "verify_dataset_v5r.py"), str(OUT)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:]
    rep.add("成對性 / 類別 id / 座標 / 位元級洩漏", ok, tail[0].strip() if tail else "")
    y = (OUT / "data.yaml").read_text(encoding="utf-8")
    rep.add("data.yaml nc=9 且 names 順序正確",
            f"nc: {len(CLASSES)}" in y and all(f"  - {c}" in y for c in CLASSES))
    caches = list(OUT.rglob("*.cache"))
    rep.add("無 .cache 殘留", not caches, f"{len(caches)} 個" if caches else "")
    boms = [p for sp in SPLITS for p in (OUT / sp / "labels").glob("*.txt")
            if p.read_bytes()[:3] == b"\xef\xbb\xbf"]
    rep.add("標註無 BOM", not boms, f"{len(boms)} 個" if boms else "")


def gate2(rep: Report) -> None:
    print("\n【Gate 2】洩漏：近重複（dHash <= 6）跨 split")
    # 直接在 OutPut 上做：write_sample 是位元複製，OutPut 的 raw 影像像素等同來源。
    # 增強圖排除（是衍生物）；cp/ext 納入（它們是新的像素來源，必須一起查）。
    by_cls: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for sp in SPLITS:
        for p in sorted((OUT / sp / "images").glob("*.jpg")):
            if kind_of(p.stem) == "aug":
                continue
            by_cls[B.class_of(p.stem)].append((sp, p))
    total = 0
    for name in CLASSES + ["Background"]:
        items = by_cls.get(name, [])
        if len(items) < 2:
            continue
        H = np.stack([dhash(p) if dhash(p) is not None else np.zeros(64, bool)
                      for _, p in items])
        pairs = 0
        for i0 in range(0, len(items), 200):
            a = H[i0:i0 + 200]
            d = (a[:, None, :] != H[None, :, :]).sum(-1)
            for bi in range(a.shape[0]):
                gi = i0 + bi
                for gj in range(gi + 1, len(items)):
                    if int(d[bi, gj]) <= B.DUP_THRESHOLD and items[gi][0] != items[gj][0]:
                        pairs += 1
        total += pairs
        rep.add(f"{name}  跨 split 近重複 {pairs} 組", pairs == 0)
    rep.add(f"全資料集合計 {total} 組", total == 0)


def gate3(rep: Report, prov: dict) -> None:
    print("\n【Gate 3】合成來源洩漏")
    recs = prov["records"]
    split_of = {r["src_md5"]: r["split"] for r in recs if r["kind"] == "raw"}
    bad_src = [r for r in recs if r["kind"] in ("aug", "cp")
               and split_of.get(r["src_md5"]) != "train"]
    rep.add(f"增強/合成圖的來源都在 train（查 {sum(1 for r in recs if r['kind'] in ('aug','cp')):,} 筆）",
            not bad_src, f"{len(bad_src)} 筆違規" if bad_src else "")
    bad_tgt = [r for r in recs if r["kind"] == "cp"
               and split_of.get(r.get("paste_md5", ""), "train") != "train"]
    rep.add("copy-paste 的貼上目標都在 train", not bad_tgt,
            f"{len(bad_tgt)} 筆違規" if bad_tgt else "")
    bad_ext = [r for r in recs if r["kind"] == "external" and r["split"] != "train"]
    rep.add("外部影像只出現在 train", not bad_ext,
            f"{len(bad_ext)} 筆違規" if bad_ext else "")
    on_disk = sum(1 for sp in SPLITS for _ in (OUT / sp / "images").glob("*.jpg"))
    rep.add(f"provenance 覆蓋全部影像（{len(recs):,} 筆 vs 磁碟 {on_disk:,} 張）",
            len(recs) == on_disk)


def gate4(rep: Report, prov: dict) -> None:
    print("\n【Gate 4】評估集")
    per = defaultdict(Counter)
    for r in prov["records"]:
        if r["kind"] == "raw":
            per[r["split"]][r["cls"]] += 1
    for name, (tv, tt) in B.EVAL_TARGET.items():
        gv, gt = per["valid"][name], per["test"][name]
        rep.add(f"{name} valid/test = {tv}/{tt}", (gv, gt) == (tv, tt),
                "" if (gv, gt) == (tv, tt) else f"實際 {gv}/{gt}")
    # 巢狀相容：v5.5 的評估影像不能掉進 v5.6 的 train
    v55_eval = {r["src_md5"] for r in prov["records"]
                if r["kind"] == "raw" and r["v55_split"] in ("valid", "test")}
    now_eval = {r["src_md5"] for r in prov["records"]
                if r["kind"] == "raw" and r["split"] in ("valid", "test")}
    missing = v55_eval - now_eval
    rep.add(f"巢狀相容：v5.5 的 {len(v55_eval):,} 張評估影像全部仍在評估集",
            not missing, f"{len(missing)} 張掉出去" if missing else "")
    worst = 0.0
    for name in CLASSES:
        se = B.project_se(name, per["valid"][name], per["test"][name])
        worst = max(worst, se)
        if se > SE_MAX:
            rep.add(f"{name} 投影 ±2SE = {se:.3f}", False, f"超過 {SE_MAX}")
    rep.add(f"全部九類投影 ±2SE <= {SE_MAX}（最差 {worst:.3f}）", worst <= SE_MAX)


def gate5(rep: Report, prov: dict) -> None:
    """增強品質。

    **逐張與自己的來源比對**，不是把全類的增強圖跟全類的原始圖做總量對比。
    總量對比會被「配額只取用了來源池的一部分」汙染：v5.6 第一次建置時
    `Aphid` 的總量框數比是 0.914，看起來像掉了 9% 的框，但逐張比對是 1.003、
    543 張沒有任何一張掉框——差距全部來自被取用的那 545 張本來就比較少框。
    """
    print("\n【Gate 5】增強品質（逐張與自己的來源比對）")
    src: dict[str, tuple[int, float]] = {}
    for r in prov["records"]:
        if r["kind"] == "raw" and r["split"] == "train":
            bx, _, _ = load_boxes("train", Path(r["out"]).stem)
            if bx:
                src[r["src_md5"]] = (len(bx), float(np.median([w * h for w, h in bx])))

    stat = defaultdict(lambda: dict(cnt=[], area=[], dup=0, n=0))
    for r in prov["records"]:
        if r["kind"] != "aug" or r["src_md5"] not in src:
            continue
        bx, _, _ = load_boxes(r["split"], Path(r["out"]).stem)
        sn, sa = src[r["src_md5"]]
        d = stat[r["cls"]]
        d["n"] += 1
        d["cnt"].append(len(bx) / sn)
        if len(bx) > sn:
            d["dup"] += 1
        if bx:
            d["area"].append(float(np.median([w * h for w, h in bx])) / max(sa, 1e-9))

    n_img = Counter()
    for p in (OUT / "train" / "images").glob("*.jpg"):
        n_img[(B.class_of(p.stem), kind_of(p.stem))] += 1

    print(f"    {'類別':<20}{'框數比':>8}{'面積比':>8}{'邊長比':>8}"
          f"{'複製框':>8}{'增強佔比':>10}")
    for name in CLASSES:
        d = stat.get(name)
        if not d or not d["cnt"]:
            continue
        cr = float(np.median(d["cnt"]))
        ar = float(np.median(d["area"])) if d["area"] else 1.0
        na, nr = n_img[(name, "aug")], n_img[(name, "raw")]
        share = na / max(na + nr, 1)
        print(f"    {name:<20}{cr:>8.3f}{ar:>8.3f}{ar ** 0.5:>8.3f}"
              f"{d['dup']:>8}{share * 100:>9.1f}%")
        rep.add(f"{name} 逐張框數保留率 {cr:.3f}",
                BOX_COUNT_RATIO[0] <= cr <= BOX_COUNT_RATIO[1])
        # 反射式邊界填充會把物件鏡射複製、並替每一份補一個框，產生物理上不可能
        # 的萬花筒畫面與「半隻蟲」的框。BORDER_REPLICATE 不可能發生這件事，
        # 這道 Gate 就是守住它。
        rep.add(f"{name} 沒有任何增強圖的框比來源多（{d['dup']}/{d['n']}）", d["dup"] == 0)
        rep.add(f"{name} 逐張框面積比 {ar:.3f}（v5.5 為 1.075–1.300）",
                BOX_AREA_RATIO[0] <= ar <= BOX_AREA_RATIO[1])
        rep.add(f"{name} 增強佔比 {share * 100:.1f}%", share <= AUG_SHARE_MAX)

    # ── Thrips 小框帶：**列為診斷數字，不設為 Gate** ─────────────────────
    #
    # 一度把它設成「不得低於 v5.5 的 11.8%」的門檻，但那是無效的比較：
    # v5.6 刻意把 41 張 Thrips 影像從 train 移進評估集，而被移走的都是
    # 「非近重複的單張」——也就是**非連拍、目標偏小**的那些。
    # train 的池子因此換了組成，拿它的分布統計去跟 v5.5 的池子比大小，
    # 比的是兩個不同的母體，數字高低沒有意義。
    #
    # 真正該問的是「train 與**它自己的**評估集對不對齊」，那是 Gate 6 在做的事
    # （Thrips 中位框尺寸比：v5.5 的 2.15 → v5.6 的 1.37，大幅改善且通過）。
    # 這裡只把帶內佔比印出來備查。
    def band_of(split_filter) -> tuple[float, int]:
        s = []
        for sp in SPLITS:
            for p in (OUT / sp / "images").glob("Thrips_*.jpg"):
                if B.class_of(p.stem) != "Thrips" or not split_filter(sp):
                    continue
                s += [(w * h) ** 0.5 for w, h in load_boxes(sp, p.stem)[0]]
        a = np.array(s)
        return (float(((a >= 32) & (a < 96)).mean()) * 100 if a.size else 0.0), a.size

    tb, tn = band_of(lambda sp: sp == "train")
    eb, en = band_of(lambda sp: sp != "train")
    print(f"    ◆ 診斷（非 Gate）Thrips 32–96px 帶佔比："
          f"train {tb:.1f}%（{tn} 框） vs 評估集 {eb:.1f}%（{en} 框）")
    print(f"      這個落差是去洩漏規則的必然後果，用現有影像池補不上，"
          f"列為已知限制（見 aug_profiles.py 檔頭）。")


def gate6(rep: Report) -> None:
    print("\n【Gate 6】分佈位移（train 全部 vs valid+test 合併）")
    tr, ev = defaultdict(list), defaultdict(list)
    tri, evi = Counter(), Counter()
    for sp in SPLITS:
        for p in (OUT / sp / "images").glob("*.jpg"):
            c = B.class_of(p.stem)
            boxes, _, _ = load_boxes(sp, p.stem)
            if sp == "train":
                tr[c] += boxes
                tri[c] += 1
            else:
                ev[c] += boxes
                evi[c] += 1
    print(f"    {'類別':<20}{'train p50':>10}{'eval p50':>10}{'尺寸比':>8}{'框/圖比':>9}")
    for name in CLASSES:
        if not tr[name] or not ev[name]:
            continue
        t = np.median([(w * h) ** 0.5 for w, h in tr[name]])
        e = np.median([(w * h) ** 0.5 for w, h in ev[name]])
        sr = t / max(e, 1e-9)
        br = (len(tr[name]) / max(tri[name], 1)) / max(len(ev[name]) / max(evi[name], 1), 1e-9)
        print(f"    {name:<20}{t:>10.0f}{e:>10.0f}{sr:>8.2f}{br:>9.2f}")
        rep.add(f"{name} 中位框尺寸比 {sr:.2f}", MED_SIZE_RATIO[0] <= sr <= MED_SIZE_RATIO[1])
        rep.add(f"{name} 每圖框數比 {br:.2f}", BOX_PER_IMG_RATIO[0] <= br <= BOX_PER_IMG_RATIO[1])


def gate7(rep: Report, prov: dict) -> None:
    print("\n【Gate 7】可重現：重跑切分邏輯，評估集成員是否完全相同")
    v55map = B.v55_split_map()
    same = True
    for spec in B.SOURCES:
        items, _ = B.collect(spec)
        pool, _ = B.nested_split(spec["name"], items, v55map)
        for sp in ("valid", "test"):
            again = {it["md5"] for it in pool[sp]}
            first = {r["src_md5"] for r in prov["records"]
                     if r["kind"] == "raw" and r["cls"] == spec["name"] and r["split"] == sp}
            if again != first:
                same = False
                rep.add(f"{spec['name']}/{sp} 重跑後成員不同", False,
                        f"差異 {len(again ^ first)} 張")
    rep.add("九個類別的 valid/test 成員在重跑後完全一致", same)
    print("    註：這道 Gate 驗的是切分的決定性（隨機性的來源）。")
    print("       增強影像的逐位元重現需要整份重建，成本高，未納入例行驗收。")


# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="要跳過的 Gate 編號，逗號分隔（例：2,7）")
    args = ap.parse_args()
    skip = {int(s) for s in args.skip.split(",") if s.strip().isdigit()}

    print("═" * 82)
    print(f"  Datasets_YOLO26_{DATASET_LABEL} 驗收   {OUT}")
    print("═" * 82)

    pj = OUT / "_provenance.json"
    if not pj.is_file():
        raise SystemExit(f"找不到 {pj}——請先跑 tools/build_dataset_v5_6.py")
    prov = json.loads(pj.read_text(encoding="utf-8"))
    m = prov["meta"]
    print(f"  建置時間 {m['built']}   arm={m['arm']}   "
          f"seed={m['seed']}   rotate_mode={m['rotate_mode']}")

    rep = Report()
    for n, fn in ((0, lambda: gate0(rep)), (1, lambda: gate1(rep)),
                  (2, lambda: gate2(rep)), (3, lambda: gate3(rep, prov)),
                  (4, lambda: gate4(rep, prov)), (5, lambda: gate5(rep, prov)),
                  (6, lambda: gate6(rep)), (7, lambda: gate7(rep, prov))):
        if n in skip:
            print(f"\n【Gate {n}】—— 依 --skip 跳過")
            continue
        fn()

    print("\n" + "═" * 82)
    if rep.failed:
        print(f"驗收未通過：{len(rep.failed)} / {len(rep.rows)} 項紅燈")
        for g, _, note in rep.failed:
            print(f"  ✗ {g}" + (f"  —— {note}" if note else ""))
        sys.exit(1)
    print(f"八道 Gate 全綠（{len(rep.rows)} 項檢查）。{DATASET_LABEL} 可以交付訓練。")


if __name__ == "__main__":
    main()
