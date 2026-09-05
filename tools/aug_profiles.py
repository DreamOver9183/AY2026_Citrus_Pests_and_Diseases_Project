# -*- coding: utf-8 -*-
"""v5.6 逐類增強 profile —— 建置與驗收的單一真實來源。

v5.5 以前所有類別共用一條全域管線（見 `build_dataset_v5r.py:AUG`）。
v5.6 改成逐類 profile，理由是三個弱點各不相同，而全域管線對它們全部無效：

  * `Thrips`：`32–96px` 帶 R@0.5 只有 0.429/0.529，`>256px` 帶是 1.000，
    跨 split 一致（`docs/v10_結果_v5.5首次訓練評估.md` §3.2）。而現行的
    `Affine(scale=(0.9,1.1))` 對這件事**完全無效**——模擬顯示帶內佔比
    只從 11.8% 動到 12.6%。改成 `scale=(0.35,1.00)` 才推得到 23.2%。
  * `Citrus_Leaf_Miner`：Precision > Recall（漏檢型），
    `docs/v11_計畫_資料擴充與切分重新設計.md` §3.1 明列缺「受光照與背景干擾」的樣本。
  * `Thrips_Damage`：medIoU 0.725、R@0.75 僅 0.290–0.375（找得到但框不準）。

──────────────────────────────────────────────────────────────────────────
【v5.6 最重要的一項修正：拿掉任意角度旋轉】

Albumentations 旋轉 bbox 的做法是「把原框的四個角轉一轉，再取新的外接矩形」。
對**任何**框（連正方形也一樣）這都是高估：正方形框轉 15° 之後，
外接矩形的面積是原本的 1.50 倍。物件本身並沒有變大，只有標註變鬆了。

v5.5 的全域管線帶著 `Affine(rotate=(-15,15), p=0.5)`，於是增強圖的框
系統性偏大。實測 v5.5 的 train（增強圖 vs 原始圖的框面積中位數比）：

    Sooty_Mold        1.075x      Aphid              1.100x
    Scale_Insect      1.108x      Black_Spot         1.115x
    Canker            1.116x      Oily_Spot          1.135x
    Citrus_Leaf_Miner 1.136x      Thrips             1.184x
    Thrips_Damage     1.300x   ← 全表最差

理論預期（±15°、p=0.5、平均 |7.5°|）約 1.13x，**實測與預期吻合**。
膨脹幅度由框的長寬比決定，而長寬比最大的 `Thrips_Damage`（p50 2.05、p90 3.68）
膨脹最嚴重——它同時也是定位最差的類別。這條關係在九個類別上單調成立。

由於增強圖佔 train 的 45–75%，模型有相當比例的監督訊號是「偏大的框」，
這會特別傷 mAP@0.5:0.95 這種高 IoU 門檻的指標
（v10 實測 mAP50 0.809 但 mAP50-95 只有 0.606）。

**修正方式：旋轉一律只用 90° 的整數倍**（`RandomRotate90`）。
90° 旋轉把一個緊貼的外接矩形映射成另一個緊貼的外接矩形，**零膨脹**，
而且提供 4 個朝向、比 ±15° 的角度多樣性更高。田間手持照本來就沒有固定朝向。

`ROTATE_MODE = "legacy15"` 可以切回 v5.5 的行為，供 A/B 對照用。

──────────────────────────────────────────────────────────────────────────
【第二項修正：邊界填充改用 BORDER_REPLICATE】

`Affine` 在平移／縮放之後要填補空出來的邊界。v5r 起用的是 `BORDER_REFLECT_101`
（鏡射），理由是避開 `BORDER_CONSTANT` 的黑色楔形——黑楔只出現在增強圖上，
會變成模型的假捷徑。這個理由本身仍然成立。

但鏡射有一個沒被注意到的副作用：**它會把物件複製出好幾份**。Albumentations 會
（正確地）替鏡射出來的每一份都補上一個框，於是一張原本 1 個框的影像，
增強後變成 5 個框——而且那 5 隻蟲排成鏡像對稱的萬花筒，是物理上不可能的畫面，
邊緣被接縫切掉的還會產生「半隻蟲」的框。

改用 `BORDER_REPLICATE`（邊緣像素向外延伸）之後：邊界會是一道拉長的糊影，
**不可能複製出第二個物件**，也不會有黑楔。這是三者中唯一沒有副作用的選項。

──────────────────────────────────────────────────────────────────────────
【為什麼沒有用縮放去補 Thrips 的小框】

`Thrips` 在 `32–96px` 帶的召回只有 0.429/0.529，而 `>256px` 帶是 1.000
（v10 §3.2），train 帶內佔比 11.5%、評估集 29.7%——這個落差是真的，
而且不是抽樣雜訊（v5.6 把評估集擴大到 120 張之後仍然存在）。

成因是**去洩漏規則的必然後果**：連拍的近重複群組一律整群移進 train，
而連拍的田間照本來就傾向拍較大的目標。train 因此被系統性地灌滿大框。

試過兩條路，都不可行，實測數字如下（quota=493）：

  * **加大縮放範圍**：`scale` 從 (0.90,1.10) 一路開到 (0.45,1.05)，
    帶內佔比只從 10.9% 動到 14.4%。原因是 Thrips 的框主體落在 200–450px，
    乘 0.45 還有 90–200px，多數仍然跨不進這個很窄的帶。
  * **偏向小框來源加權抽樣**：帶內佔比可以拉到 38–43%，但 434 張 train 裡
    **只有 34 張含有帶內的框**，配額會讓這 34 張各被重複 14–19 次。
    用犧牲場景多樣性換帶內佔比，代價明顯更大。

**結論：用現有影像池無法在不造假的前提下補上這個落差。**
因此 `Thrips` 只走 `_default` 的幾何設定，這個落差列為 v5.6 的已知限制，
在報告裡據實寫明，而不是用會產生假影的增強去湊一個數字。
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import albumentations as A

SEED = 0

CLASSES = [
    "Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot",
    "Scale_Insect", "Citrus_Leaf_Miner", "Thrips", "Aphid", "Thrips_Damage",
]

# 四個病害類別的色相偏移上限鎖死 ±3°：放寬會把綠葉推成病斑黃，
# 等於製造假標籤（來源：《資料集建立流程 SOP》§5.1）。
DISEASE_CLASSES = frozenset({"Oily_Spot", "Canker", "Sooty_Mold", "Black_Spot"})
HUE_LIMIT_DISEASE = 3

# "90only" = v5.6 預設（零框膨脹）；"legacy15" = v5.5 行為，只在需要 A/B 時用
ROTATE_MODE = "90only"

_MIN_VISIBILITY = 0.2


def _photometric(hue: int = 3, bright: float = 0.15, contrast: float = 0.15) -> list:
    """光度類算子。全部不動框座標，零標註語意風險。"""
    return [
        A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=15,
                             val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=bright,
                                   contrast_limit=contrast, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]


def _geometric(scale: tuple[float, float] = (0.9, 1.1),
               rot90: bool = True, vflip: bool = True) -> list:
    """幾何類算子。旋轉的處理見檔頭說明。"""
    ops: list = [A.HorizontalFlip(p=0.5)]
    if vflip:
        ops.append(A.VerticalFlip(p=0.5))
    if rot90 and ROTATE_MODE == "90only":
        ops.append(A.RandomRotate90(p=0.5))
    ops.append(A.Affine(
        translate_percent=(-0.05, 0.05),
        scale=scale,
        # legacy15 才給任意角度；預設 0 度，旋轉全交給 RandomRotate90
        rotate=(-15, 15) if ROTATE_MODE == "legacy15" else (0, 0),
        border_mode=cv2.BORDER_REPLICATE,
        p=0.5,
    ))
    return ops


# ── 逐類規格 ────────────────────────────────────────────────────────────
# 每一條「新增算子」都要能指回一項已量到的事實，否則就是雜訊。
PROFILE_SPECS: dict[str, dict] = {
    "_default": dict(
        ops=lambda: _photometric() + _geometric(),
        why="v5.5 的全域管線，僅把任意角度旋轉換成 90° 倍數（見檔頭）",
    ),
    "Thrips": dict(
        ops=lambda: _photometric() + _geometric(),
        why="只加 RandomRotate90 / VerticalFlip（零框膨脹的朝向多樣性）。"
            "**不做大幅縮放**——理由見檔頭「為什麼沒有用縮放去補小框」",
    ),
    "Citrus_Leaf_Miner": dict(
        ops=lambda: _photometric(bright=0.25, contrast=0.25) + [
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
                           shadow_intensity_range=(0.2, 0.45), p=0.2),
            A.RandomToneCurve(scale=0.15, p=0.3),
            A.PlanckianJitter(mode="blackbody", p=0.3),
        ] + _geometric(),
        why="漏檢型（P > R）且 v11 §3.1 明列缺『受光照與背景干擾』的樣本；"
            "三個新增算子都是光度類，不動框座標",
    ),
    "Thrips_Damage": dict(
        ops=lambda: _photometric() + [A.RandomToneCurve(scale=0.15, p=0.3)]
                    + _geometric(),
        why="R@0.75 僅 0.290–0.375（框不準而非漏檢），"
            "所以只補光度多樣性、不加任何會動到框的算子；"
            "本類長寬比 p90 = 3.68，是拿掉任意角度旋轉的最大受益者",
    ),
}


def build_profile(class_name: str) -> A.Compose:
    """取得某個類別的增強管線。未列名的類別走 `_default`。"""
    spec = PROFILE_SPECS.get(class_name, PROFILE_SPECS["_default"])
    ops = spec["ops"]()
    _assert_ops(class_name, ops)
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                 min_visibility=_MIN_VISIBILITY),
        seed=SEED,
    )


# ── 三條硬性約束 ────────────────────────────────────────────────────────
def _assert_ops(class_name: str, ops: list) -> None:
    for op in ops:
        # 約束 1：旋轉只能是 90° 的整數倍（legacy15 是刻意的例外）
        if isinstance(op, A.Affine) and ROTATE_MODE == "90only":
            rot = getattr(op, "rotate", (0, 0))
            lo, hi = (rot if isinstance(rot, (tuple, list)) else (rot, rot))
            assert lo == 0 and hi == 0, (
                f"{class_name}: Affine 帶了任意角度旋轉 {rot}。"
                f"轉 15° 會讓外接矩形面積脹到 1.50 倍（正方形框），"
                f"旋轉請一律走 RandomRotate90。"
            )

        # 約束 2：病害類別的色相偏移鎖死 ±3°
        if isinstance(op, A.HueSaturationValue) and class_name in DISEASE_CLASSES:
            lim = op.hue_shift_limit
            hi = max(abs(v) for v in lim) if isinstance(lim, (tuple, list)) else abs(lim)
            assert hi <= HUE_LIMIT_DISEASE, (
                f"{class_name}: 色相偏移 ±{hi}° 超過病害類別上限 "
                f"±{HUE_LIMIT_DISEASE}°，會把綠葉推成病斑黃。"
            )

        # 約束 3：不使用 CoarseDropout
        assert not isinstance(op, A.CoarseDropout), (
            f"{class_name}: 不要用 CoarseDropout。它會讓 GT 變成 amodal"
            f"（框住被遮擋的完整目標），與人工標註的 modal 約定不一致。"
        )


def profile_summary() -> list[dict]:
    """給驗收腳本與報告用的攤平描述。"""
    out = []
    for name, spec in PROFILE_SPECS.items():
        ops = spec["ops"]()
        _assert_ops(name, ops)
        out.append(dict(
            profile=name,
            n_ops=len(ops),
            ops=[type(o).__name__ for o in ops],
            why=spec["why"],
        ))
    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("═" * 78)
    print(f"  逐類增強 profile   ROTATE_MODE = {ROTATE_MODE}")
    print("═" * 78)
    for row in profile_summary():
        print(f"\n▸ {row['profile']}  （{row['n_ops']} 個算子）")
        print(f"  {' → '.join(row['ops'])}")
        print(f"  依據：{row['why']}")
    covered = [c for c in CLASSES if c in PROFILE_SPECS]
    print(f"\n專屬 profile：{', '.join(covered)}")
    print(f"走 _default：{', '.join(c for c in CLASSES if c not in PROFILE_SPECS)}")
    for c in CLASSES:
        build_profile(c)
    print("\n九個類別的 profile 全部通過三條硬性約束。")
