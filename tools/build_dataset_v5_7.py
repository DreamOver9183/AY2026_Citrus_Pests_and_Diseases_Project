# -*- coding: utf-8 -*-
"""Datasets_YOLO26_v5.7 建置腳本。

**與 v5.6 的唯一差別是來源樹**：`Thrips_Damage` 的 208 張依新的標註約定重標過。

    舊約定：看兩塊食痕中間夾的健康葉肉有多寬，決定要不要分成兩個框
    新約定：**一張葉子上的葉害，不論分成幾塊，一律用一個框整片包住**

改約定的理由是量出來的，不是拍腦袋：工作包 A 讓兩個人照同一份書面約定各標 20 張，
舊約定的中位 IoU 只有 **0.746**（判準 0.85，未通過）；改成一葉一框之後，
同樣兩個人、同樣 20 張，升到 **0.899**。完整紀錄見
`docs/v5.6_結果_工作包A標註一致性.md`。

重標後的來源：`Thrips_Damage` 245 框 → **215 框**（仍是 208 張，其中 6 張是兩片葉、
各一個框），框中位面積 11.30% → 15.40%。`Thysanoptera`（837 框 / 554 張）完全沒動。

> **v5.6 與 v5.7 的 `Thrips_Damage` 數字不可直接比。** 框的定義換了，
> 不是模型變好或變壞。其餘八類的來源與 v5.6 位元相同，可以比。

建置邏輯**完全沿用 `build_dataset_v5_6.py`**（切分、增強 profile、八道 Gate、
`--arm` 的配額中性規則都一樣），這支只把來源樹與預設產出目錄換掉，
避免同一套 600 行邏輯出現第二份而漂移。

用法（與 v5.6 相同，多一個 `--root` 可覆寫來源樹）：
    .venv/Scripts/python.exe tools/build_dataset_v5_7.py --dry-run
    .venv/Scripts/python.exe tools/build_dataset_v5_7.py
    .venv/Scripts/python.exe tools/build_dataset_v5_7.py --arm ext --out <另一個目錄>

⚠ `--arm ext` 一定要給不同的 `--out`，否則會 `shutil.rmtree` 掉 base 臂的產出。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P          # noqa: E402
import build_dataset_v5_6 as B      # noqa: E402

VERSION = "v5.7"


def retarget(version: str = VERSION) -> None:
    """把 v5.6 建置模組的來源樹與預設產出指向指定版本。

    模組裡的 `V56_ROOT` / `OUT_ROOT` 都是在函式內部才解參考的全域變數，
    所以改這兩個名字就足以整支換版本，不必複製任何邏輯。
    """
    B.DATASET_VERSION = version
    B.V56_ROOT = _P.raw(version)
    B.OUT_ROOT = _P.split(version)


def main() -> None:
    retarget()
    assert B.V56_ROOT.is_dir(), f"找不到來源樹 {B.V56_ROOT}"
    print(f"（build_dataset_v5_7：來源樹 {B.V56_ROOT.name}、"
          f"預設產出 {B.OUT_ROOT.name}，其餘邏輯與 v5.6 相同）")
    B.main()


if __name__ == "__main__":
    main()
