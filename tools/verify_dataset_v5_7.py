# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5.7 驗收（八道 Gate）。

沿用 `verify_dataset_v5_6.py` 的全部 Gate，只換掉兩件事：

1. **來源樹與產出目錄指向 v5.7**（由 `build_dataset_v5_7.retarget()` 做掉）。
2. **Gate 0 的「與 v5.5 位元一致」對 Thrips 來源另外處理。**
   v5.7 就是為了重標 `Thrips_Damage` 才存在的，那個資料夾**本來就該不一樣**；
   原樣套用 v5.6 的 Gate 0 只會得到一個必然失敗的紅字，久了就會被習慣性忽略。
   這支改成問一個更嚴格的問題——**它是不是「只」按預期變了**：

   * 影像集合（以內容 md5 為準）與 v5.5 完全相同，一張不多、一張不少
   * `Thysanoptera`（子類 0）的框**逐字未動**
   * 只有 `thirps_leaf_damage`（子類 1）的框有變動

   任何一項不成立就是紅字。這樣既允許預期中的改動，又擋得住「不小心動到別的東西」。

用法：
    .venv/Scripts/python.exe tools/verify_dataset_v5_7.py
    .venv/Scripts/python.exe tools/verify_dataset_v5_7.py --data "Datasets/2_處理與切分/v5.7_ext"
    .venv/Scripts/python.exe tools/verify_dataset_v5_7.py --skip 2,7
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P              # noqa: E402
import build_dataset_v5_7 as B7         # noqa: E402

THRIPS_REL = "Pests/Thrips_v5r.yolo26/train"
RF_HASH = re.compile(r"_jpg\.rf\.[0-9A-Za-z]+")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _boxes(p: Path, sub: int) -> list[tuple]:
    out = []
    for ln in p.read_text(encoding="utf-8-sig").splitlines():
        q = ln.split()
        if len(q) >= 5 and int(float(q[0])) == sub:
            out.append(tuple(round(float(v), 6) for v in q[1:5]))
    return sorted(out)


def check_thrips_source(rep, v55_root: Path, v57_root: Path) -> None:
    """Thrips 來源：只准 `thirps_leaf_damage` 變，其餘都不准。"""
    a, b = v55_root / THRIPS_REL, v57_root / THRIPS_REL
    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}     # 來源樹裡 jpg 之外還有 1 jpeg + 2 png
    ma = {_md5(p): p.name for p in sorted((a / "images").glob("*")) if p.suffix.lower() in ext}
    mb = {_md5(p): p.name for p in sorted((b / "images").glob("*")) if p.suffix.lower() in ext}
    rep.add(f"Thrips 影像集合與 v5.5 相同（各 {len(ma)} / {len(mb)} 張，以內容比對）",
            set(ma) == set(mb),
            "" if set(ma) == set(mb) else f"只在 v5.5 {len(set(ma)-set(mb))} 張、只在 v5.7 {len(set(mb)-set(ma))} 張")

    renamed = {ma[h]: mb[h] for h in set(ma) & set(mb) if ma[h] != mb[h]}
    if renamed:
        print(f"      （{len(renamed)} 張被 Roboflow 改了檔名，內容相同：" +
              "、".join(f"{k[:-4]}→{v[:-4]}" for k, v in sorted(renamed.items())[:3]) + " …）")

    stem_of = lambda n: RF_HASH.sub("", n)[:-4]                       # noqa: E731
    la = {stem_of(p.name): p for p in (a / "labels").glob("*.txt")}
    lb = {stem_of(p.name): p for p in (b / "labels").glob("*.txt")}
    ren_stem = {stem_of(k): stem_of(v) for k, v in renamed.items()}
    bad0, chg1, n0a, n0b, n1a, n1b = [], 0, 0, 0, 0, 0
    for k, pa in la.items():
        pb = lb.get(ren_stem.get(k, k))
        if pb is None:
            bad0.append(f"缺 {k}")
            continue
        a0, b0 = _boxes(pa, 0), _boxes(pb, 0)
        n0a += len(a0); n0b += len(b0)
        if a0 != b0:
            bad0.append(k)
        a1, b1 = _boxes(pa, 1), _boxes(pb, 1)
        n1a += len(a1); n1b += len(b1)
        if a1 != b1:
            chg1 += 1
    rep.add(f"Thysanoptera（子類 0）逐字未動（{n0a} → {n0b} 框）", not bad0,
            "" if not bad0 else f"{len(bad0)} 個檔案有變：{bad0[:3]}")
    rep.add(f"只有 thirps_leaf_damage（子類 1）變動（{n1a} → {n1b} 框，{chg1} 個檔案）",
            chg1 > 0 and n1b > 0,
            "" if chg1 else "一個檔案都沒變——重標的內容沒有裝進來？")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="要驗收的產出目錄；預設是 v5.7 的 base 臂")
    ap.add_argument("--skip", default="", help="要跳過的 Gate 編號，逗號分隔")
    args = ap.parse_args()

    # 一定要在 import verify_dataset_v5_6 之前改，它在 module 層就把常數綁定了
    B7.retarget()
    if args.data:
        B7.B.OUT_ROOT = Path(args.data)

    import verify_dataset_v5_6 as V      # noqa: E402  retarget 之後才 import

    V.BITWISE_SKIP = {THRIPS_REL}        # 改用下面的逐子類檢查，比位元比對更嚴格

    print("═" * 82)
    print(f"  Datasets_YOLO26_v5.7 驗收   {V.OUT}")
    print(f"  來源樹 {V.V56_ROOT}")
    print("═" * 82)

    rep = V.Report()
    print("\n【Gate 0+】Thrips 來源的變動是否只限於 thirps_leaf_damage")
    check_thrips_source(rep, V.V55_ROOT, V.V56_ROOT)
    extra_failed = rep.failed

    sys.argv = [sys.argv[0]] + (["--skip", args.skip] if args.skip else [])
    try:
        V.main()
    except SystemExit as e:
        raise SystemExit(e.code or (1 if extra_failed else 0))
    if extra_failed:
        print(f"\n八道 Gate 雖然全綠，但 Gate 0+ 有 {len(extra_failed)} 項紅燈。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
