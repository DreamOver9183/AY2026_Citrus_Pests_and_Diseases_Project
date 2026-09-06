# -*- coding: utf-8 -*-
"""`Datasets/` 的版面配置——**所有腳本的單一真實來源**。

以前每支腳本各自寫死 `REPO / "Datasets" / "Datasets_YOLO26_v5.6"` 這種路徑，
散在九個檔案裡。搬一次資料夾就要改九個地方，而且漏改不會馬上報錯——
只會在某支很久沒跑的腳本上炸掉。現在全部改成從這裡取。

──────────────────────────────────────────────────────────────────────
`Datasets/` 依**處理階段**分三層，而不是依版本：

    1_原始影像/     人工標註過的來源影像，還沒切分、沒增強
      ├── v5r/            {Diseases, Healthy, Pests}
      ├── v5.5/           同上（Scale_Insect 與 Canker 已重標）
      ├── v5.6/           同上（與 v5.5 位元一致，另加外部試點的位置）
      ├── _人工標註待辦/    發給組員的工作包，回收後併進上面的來源樹
      └── _外部資料集/      論文引用的公開資料集，全部沒有標註

    2_處理與切分/   切分 + 增強之後，可以直接餵給 YOLO 的形態
      └── v5 / v5r / v5.5 / v5.6      各自 {train, valid, test} + data.yaml

    3_最終輸出/     打包好、可以上傳 Kaggle 或交付的壓縮檔

**為什麼這樣分**：這三層是資料的三個生命階段，而且**只有第一層是不可重生的**。
第二層可以用 `build_dataset_*.py` 從第一層重建，第三層可以從第二層壓出來。
按階段分，「哪些東西丟了會真的救不回來」一眼就看得出來。
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "Datasets"

RAW = DATASETS / "1_原始影像"
SPLIT = DATASETS / "2_處理與切分"
FINAL = DATASETS / "3_最終輸出"

MANUAL = RAW / "_人工標註待辦"
EXTERNAL = RAW / "_外部資料集"


def raw(version: str) -> Path:
    """來源樹（未切分）。version 例：'v5r'、'v5.5'、'v5.6'。"""
    return RAW / version


def split(version: str) -> Path:
    """切分＋增強後的產出。version 例：'v5'、'v5r'、'v5.5'、'v5.6'。"""
    return SPLIT / version


def final(name: str) -> Path:
    """打包好的壓縮檔。"""
    return FINAL / name


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("═" * 66)
    print(f"  Datasets 版面   {DATASETS}")
    print("═" * 66)
    for stage, p in (("1 原始影像", RAW), ("2 處理與切分", SPLIT), ("3 最終輸出", FINAL)):
        print(f"\n{stage}   {'✓' if p.is_dir() else '✗ 不存在'}")
        if p.is_dir():
            for child in sorted(p.iterdir()):
                mark = "／" if child.is_dir() else "  "
                print(f"    {child.name}{mark}")
