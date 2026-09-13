# -*- coding: utf-8 -*-
"""由 `Train Code/v11.5/train_v11_5.ipynb` 生出 v5.7 的兩本訓練 notebook。

**為什麼用生成的，而不是手抄兩份**：base 與 ext 是一組 A/B，
「兩臂的訓練程式完全相同」是這個比較能成立的前提。手抄兩份 16 個 cell 的
notebook，遲早會有一邊被順手改到而沒人發現。這支腳本只改四個 cell——
標題（markdown）、第一個設定 cell、Step.2 的資料集指紋、
以及 Step.7 寫 `summary.json` 時那個寫死的 `dataset` 欄——
其餘 12 個 cell **逐字複製**，並在最後印出「有幾個 cell 沒被動過」當證據。

產出：
    Train Code/v11.5/train_v5_7_base.ipynb    RUN=v5.7_v11_5
    Train Code/v11.5/train_v5_7_ext.ipynb     RUN=v5.7ext_v11_5

用法：
    .venv/Scripts/python.exe tools/make_train_notebooks_v5_7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as _P     # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC_NB = REPO / "Train Code" / "v11.5" / "train_v11_5.ipynb"
OUT_DIR = REPO / "Train Code" / "v11.5"

ARMS = {
    "base": dict(
        run="v5.7_v11_5", slug="datasets-yolo26-v5-7", out="train_v5_7_base.ipynb",
        split_dir="v5.7", arm="base",
        one_liner="v5.7 的對照臂：只換資料集（`Thrips_Damage` 重標），其餘與 v11.5 逐字相同。",
    ),
    "ext": dict(
        run="v5.7ext_v11_5", slug="datasets-yolo26-v5-7-ext", out="train_v5_7_ext.ipynb",
        split_dir="v5.7_ext", arm="ext",
        one_liner="v5.7 的實驗臂：train 多了 124 張外部潛葉蛾影像，**配額中性**（佔用 CLM 既有的增強配額，train 總張數與 base 相同）。",
    ),
}


def counts(split_dir: str) -> dict[str, int]:
    root = _P.split(split_dir)
    return {sp: len(list((root / sp / "images").glob("*.jpg"))) for sp in ("train", "valid", "test")}


def header(cfg: dict, n: dict) -> str:
    return f"""# {cfg['run']} 訓練 — Datasets_YOLO26_{cfg['split_dir']}（`arm={cfg['arm']}`）

{cfg['one_liner']}

---

## 這一輪在問什麼

兩件事一起驗，而且**互不干擾**（動到的是不同類別）：

| | 改了什麼 | 要看哪一個類別 |
| --- | --- | --- |
| **v5.6 → v5.7** | `Thrips_Damage` 的 208 張依新約定重標（一張葉子一個框，245 框 → 215 框） | `Thrips_Damage` |
| **base → ext** | train 多 124 張外部潛葉蛾影像（配額中性） | `Citrus_Leaf_Miner` |

所以要跑**兩本**：`train_v5_7_base.ipynb` 與 `train_v5_7_ext.ipynb`。
兩本的訓練程式逐字相同，**只有資料集不同**，由 `tools/make_train_notebooks_v5_7.py` 生成。

---

## 判準（先寫死，再看結果）

| # | 問題 | 判準 |
| --- | --- | --- |
| 1 | 外部影像對 CLM 有沒有幫助 | ext 的 `Citrus_Leaf_Miner` AP50 減 base，要**超過 2σ**（平台期 σ ≈ 0.002，逐類雜訊地板 **±0.04**）才算數 |
| 2 | 重標有沒有修好 `Thrips_Damage` 的矛盾 | valid − test 的差距要掉到**顯著門檻 0.165 以內**（v11.5 是 0.202，三輪都沒收斂） |
| 3 | 其餘七類 | **不預期任何變化**；超過 ±0.04 才需要解釋 |

> `Thrips_Damage` 的 **AP 絕對值不能跟 v5.6 比**——框的定義換了，
> 框中位面積從 11.30% 變成 15.40%。能比的是「valid/test 還矛不矛盾」。

---

## 所需資料

| 項目 | 內容 |
| --- | --- |
| 資料集 | **`Datasets_YOLO26_{cfg['split_dir']}`**（本機 `Datasets/2_處理與切分/{cfg['split_dir']}/`）上傳成 Kaggle Dataset，slug `{cfg['slug']}` |
| 規模 | 9 類，`train {n['train']:,} / valid {n['valid']} / test {n['test']}` 影像 |
| 內含 | `data.yaml`、`train/`、`valid/`、`test/`、`_provenance.json` |
| 網路 | **Internet 必須開啟**（要抓 `yolo26n.pt`） |

> Step.2 除了原本的張數指紋，另外**直接讀 `_provenance.json` 的 `meta.version` 與 `meta.arm`**。
> 張數在 base 與 ext 是一樣的（配額中性就是這個意思），光靠張數分不出兩臂——
> 上傳錯資料集會直接停在 Step.2。

---

## 參數

**與 v11.5 逐字相同**（`patience=0`、70 輪、`MuSGD`、`imgsz=640`、`batch=20`、`seed=0`），
一個都沒動。這一輪唯一的變因是資料集。

主要結果一律報 **`last.pt`**：固定輪數跑完的自然終點，valid 沒參與任何決策，
所以 valid 與 test 可以合法併計。`best.pt` 只作對照。

---

## 所需時間

約 **3.4 小時**（訓練 70 輪 3.1 h ＋ 資料集複製與四次評估 0.3 h）。
Kaggle 單場上限 12 h，不需要拆場。兩臂合計約 6.8 h。

建議 **Save & Run All**。
"""


def patch_cell1(src: str, cfg: dict) -> str:
    out = src.replace('RUN      = "v5.6_v11_5"', f'RUN      = "{cfg["run"]}"')
    out = out.replace(
        '# Datasets_YOLO26_v5.6（本機 Datasets/2_處理與切分/v5.6）上傳成 Kaggle Dataset 後的路徑\n'
        'DATASET_SRC = "/kaggle/input/datasets-yolo26-v5-6"',
        f'# Datasets_YOLO26_{cfg["split_dir"]}（本機 Datasets/2_處理與切分/{cfg["split_dir"]}）'
        f'上傳成 Kaggle Dataset 後的路徑\n'
        f'DATASET_SRC = "/kaggle/input/{cfg["slug"]}"\n'
        f'EXPECT_ARM  = "{cfg["arm"]}"        # Step.2 會拿 _provenance.json 對這個值\n'
        f'EXPECT_VER  = "{cfg["split_dir"].replace("_ext", "")}"')
    assert out != src, "cell 1 沒有被改到"
    return out


def patch_cell5(src: str, cfg: dict, n: dict) -> str:
    out = src.replace('DST = "/kaggle/working/datasets-yolo26-v5-6"',
                      f'DST = "/kaggle/working/{cfg["slug"]}"')
    out = out.replace(
        'assert NC == 9, f"nc={NC}，v5.6 應為 9（8 = 舊的 v5r，跑錯資料集了）"',
        'assert NC == 9, f"nc={NC}，應為 9（8 = 舊的 v5r，跑錯資料集了）"')
    old_fp = out[out.index('_n = {sp: len('):out.index('# BOM 與舊快取')]
    new_fp = f'''_n = {{sp: len(os.listdir(os.path.join(DST, sp, "images"))) for sp in ("train", "valid", "test")}}
print(f"▷ 影像張數 {{_n}}")
assert _n["valid"] == {n['valid']} and _n["test"] == {n['test']}, (
    f"valid/test 應為 {n['valid']}/{n['test']}，實際 {{_n['valid']}}/{{_n['test']}}。"
    f"327/326 代表上傳的是 v5.5。")   # v5.6 與 v5.7 同為 401/401，靠下面的 provenance 分辨
assert _n["train"] == {n['train']}, f"train 應為 {n['train']:,}，實際 {{_n['train']}}"

# ══════════════════════════════════════════════════════════════════════
# 這一項是 v5.7 新加的，而且比張數重要。
#
# base 與 ext 的張數**完全一樣**（配額中性：外部影像佔用 CLM 既有的增強配額），
# 所以上一段的指紋分不出兩臂——拿錯資料集會一路跑完 3.4 小時才發現。
# _provenance.json 的 meta 直接寫著版本與臂別，拿它當真正的指紋。
# ══════════════════════════════════════════════════════════════════════
with open(os.path.join(DST, "_provenance.json"), encoding="utf-8") as f:
    _meta = json.load(f)["meta"]
print(f"▷ provenance meta: version={{_meta.get('version')}}  arm={{_meta.get('arm')}}  "
      f"seed={{_meta.get('seed')}}  rotate_mode={{_meta.get('rotate_mode')}}")
assert _meta.get("version") == EXPECT_VER, (
    f"資料集版本是 {{_meta.get('version')}}，這本 notebook 要的是 {{EXPECT_VER}}")
assert _meta.get("arm") == EXPECT_ARM, (
    f"資料集臂別是 {{_meta.get('arm')}}，這本 notebook 要的是 {{EXPECT_ARM}}。"
    f"base 與 ext 的張數相同，只有這裡分得出來。")

'''
    out = out.replace(old_fp, new_fp)
    out = out.replace("import os, shutil, sys, yaml", "import json, os, shutil, sys, yaml")
    assert "EXPECT_ARM" in out, "cell 5 的指紋沒有換成功"
    return out


def patch_cell15(src: str, cfg: dict) -> str:
    """`summary.json` 的 `dataset` 欄在 v11.5 是寫死的 "v5.6"。

    v11 有一半的分析時間花在事後確認「到底跑了哪一版」，就是因為這種寫死的字串；
    v11.5 為此加了張數指紋，卻漏掉這一行本身。改成用 Step.2 讀到的 provenance，
    並補上 `arm`——base 與 ext 的張數相同，`arm` 是唯一分得出來的欄位。
    """
    old = '    "dataset": "v5.6",                      # 由 Step.2 的張數指紋確認過\n'
    new = ('    "dataset": _meta.get("version", EXPECT_VER),   # 來自 _provenance.json，不是寫死的\n'
           '    "arm": _meta.get("arm", EXPECT_ARM),\n')
    if src.count(old) != 1:
        raise SystemExit("cell 15 的 dataset 欄找不到，train_v11_5.ipynb 可能改過")
    return src.replace(old, new)


def main() -> None:
    nb0 = json.loads(SRC_NB.read_text(encoding="utf-8"))
    print(f"來源 {SRC_NB.relative_to(REPO)}（{len(nb0['cells'])} cells）")

    for key, cfg in ARMS.items():
        n = counts(cfg["split_dir"])
        if n["train"] == 0:
            raise SystemExit(f"{cfg['split_dir']} 還沒建好（train 0 張），先跑 build_dataset_v5_7.py")
        nb = json.loads(json.dumps(nb0))          # deep copy
        untouched = 0
        for i, c in enumerate(nb["cells"]):
            src = "".join(c["source"])
            if i == 0:
                new = header(cfg, n)
            elif i == 1:
                new = patch_cell1(src, cfg)
            elif i == 5:
                new = patch_cell5(src, cfg, n)
            elif i == 15:
                new = patch_cell15(src, cfg)
            else:
                untouched += 1
                continue
            c["source"] = new.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
        dst = OUT_DIR / cfg["out"]
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✓ {dst.relative_to(REPO)}   RUN={cfg['run']}   "
              f"train/valid/test = {n['train']:,}/{n['valid']}/{n['test']}   "
              f"逐字沿用 {untouched} 個 cell")


if __name__ == "__main__":
    main()
