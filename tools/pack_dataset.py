# -*- coding: utf-8 -*-
"""把切分好的資料集壓成可以上傳 Kaggle 的 zip。

壓縮層級刻意用 **STORED（不壓縮）**：資料集九成以上是 JPEG，
v5.6 實測 DEFLATE 只從 4.51 GB 壓到 4.44 GB（**1.5%**），
卻要多花數倍的時間。上傳的瓶頸是頻寬不是這 1.5%。

zip 的頂層與 `Datasets_YOLO26_v5.6.zip` 一致——`data.yaml`、`train/`、
`valid/`、`test/`、`_provenance.json` 直接放在根，解壓後就是 Kaggle
`/kaggle/input/<slug>/` 底下的樣子。

用法：
    .venv/Scripts/python.exe tools/pack_dataset.py v5.7
    .venv/Scripts/python.exe tools/pack_dataset.py v5.7_ext --name Datasets_YOLO26_v5.7_ext
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="2_處理與切分 底下的資料夾名，例 v5.7")
    ap.add_argument("--name", default=None, help="輸出檔名（不含 .zip）；預設 Datasets_YOLO26_<version>")
    ap.add_argument("--force", action="store_true", help="輸出已存在時覆蓋")
    args = ap.parse_args()

    src = _P.split(args.version)
    if not src.is_dir():
        raise SystemExit(f"找不到 {src}")
    dst = _P.final(f"{args.name or f'Datasets_YOLO26_{args.version}'}.zip")
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst.name} 已存在。要覆蓋請加 --force")
    dst.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"來源 {src}")
    print(f"  {len(files):,} 個檔案、{total / 1e9:.2f} GB  ->  {dst}")

    tmp = dst.with_suffix(".zip.part")
    done = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(src).as_posix())
            done += p.stat().st_size
            if i % 500 == 0 or i == len(files):
                bar = int(24 * done / max(total, 1))
                print(f"\r  [{'█' * bar}{'░' * (24 - bar)}] "
                      f"{done / max(total, 1) * 100:5.1f}%  {i:,}/{len(files):,}", end="")
    print()
    tmp.replace(dst)
    print(f"完成：{dst}  {dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
