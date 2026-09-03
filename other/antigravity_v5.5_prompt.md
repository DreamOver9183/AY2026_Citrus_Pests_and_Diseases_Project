# 任務：把 `tools/build_dataset_v5r.py` 擴充成可建置 `Datasets_YOLO26_v5.5`

你要修改**一個檔案**：`tools/build_dataset_v5r.py`，然後用它建出一個新的資料集版本 v5.5。

這個腳本目前產生的 `Datasets_YOLO26_v5r` 已經被用來訓練出交付模型，並且有一整套已發表的
評估數字建立在它的 train/valid/test 切分之上。**v5.5 必須沿用完全相同的切分**，否則所有既有
數字都不可比，整份工作作廢。下面 §2 的三條約束就是為了保證這件事，請逐條照做。

---

## 1. 要做的兩件事

| # | 動作 | 對象 |
| --- | --- | --- |
| 1 | **刪除**（整張影像移除） | 來源 `Pests/Aphid.yolo26/train` 的 class 1 = `Aphid_Leaf_Damage` |
| 2 | **拆成新類別** | 來源 `Pests/Thrips_v5r.yolo26/train` 的 class 1 = `thirps_leaf_damage` → 新輸出類別 id **8**，名稱 `Thrips_Damage` |

### 類別對應（只能依資料夾名稱與來源 `data.yaml` 判斷，已在此寫死，不要自行推測）

| 來源目錄 | 該目錄 `data.yaml` 的 `names` | 本次處置 |
| --- | --- | --- |
| `Datasets/Datasets_YOLO26_v5r/Pests/Thrips_v5r.yolo26/train` | `['Thysanoptera', 'thirps_leaf_damage']` | class **1** → 拆為新類別 `Thrips_Damage` |
| `Datasets/Datasets_YOLO26_v5r/Pests/Aphid.yolo26/train` | `['Aphid', 'Aphid_Leaf_Damage']` | class **1** → 刪除整張影像 |

輸出類別 **id 0–7 必須維持原順序不變**：

```
0 Oily_Spot  1 Canker  2 Sooty_Mold  3 Black_Spot
4 Scale_Insect  5 Citrus_Leaf_Miner  6 Thrips  7 Aphid  8 Thrips_Damage  ← 新增
```

已經驗證過的前提：**兩個子域從不共存於同一張影像**（171/171、16/16、21/21、29/29、1/1、3/3
都是「只含葉害框」的影像）。所以刪除是整張移除、改標是整個標註檔改寫，不需要編輯單一個框。
請在程式裡加一個 assert 檢查這件事，若出現混合影像就中止並報錯。

---

## 2. 三條硬約束（違反任何一條，產出即作廢）

### 約束 A：`collect()` 絕對不可過濾，必須「先切分、再刪除／改標」

`main()` 用**單一** `rng = random.Random(SEED)` 依序穿過所有 `split_items()` 呼叫：

```python
rng = random.Random(SEED)
for spec in SOURCES:
    items, st = collect(spec)
    pools[spec["name"]] = split_items(items, rng)   # ← rng 狀態逐個累積
pools["Background"] = split_items(bg, rng)
```

`split_items()` 內部是 `rng.shuffle(idx)`。只要任何一個來源丟給它的項目數改變，**該來源的切分
結果與其後所有來源的切分全部改變**。Aphid 是 `SOURCES` 的最後一個，但它自己的 valid/test 會
整組換掉——而 Aphid 的 valid/test 正是評估基準的一部分。

**正確順序：**

1. `collect()` 原樣收集，不做任何刪除；只**額外保留**每個框的來源子類 id。
2. `split_items()` 照跑 → 切分與 v5r 逐張相同。
3. **切分完成之後**才套用刪除與改標。
4. `downsample_scale()` 與增強配額都在修改後的 pool 上重算。

保留子類 id 的建議作法（不要改 `write_sample()` 的簽章）：在 `collect()` 裡建一個與
`boxes` 等長的平行清單，存進 `item["subs"]`：

```python
boxes, subs = [], []
for _cls, cx, cy, w, h in raw:
    if spec["cid"] == 7 and equiv_px(w, h, W, H) < APHID_MIN_PX:
        stats["Aphid_丟棄極小框"] += 1
        continue
    boxes.append((spec["cid"], cx, cy, w, h))
    subs.append(_cls)                      # ← 新增：保留來源子類
...
items.append(dict(img=img, boxes=boxes, subs=subs, W=W, H=H))
```

### 約束 B：拆出來的兩類共用原本的增強配額，依 raw 張數等比分配

現行配額是 `tgt = min(AUG_MULT × raw, AUG_CAP)`，即 `min(4×raw, 1200)`。

若讓 `Thrips` 與 `Thrips_Damage` **各自**套這條公式，會變成 1200 + 684 = 1884 張，
比原本的 1200 張多出 684——**葉害子域的樣本權重憑空翻倍**。那樣這個資料集就同時改變了
「類別定義」與「子域比重」兩件事，後續實驗無法歸因，等於白做。

**正確作法**：拆出來的兩類視為同一個配額群組：

```
total_raw = 439 (Thrips 蟲體) + 171 (Thrips_Damage) = 610
tgt_total = min(4 × 610, 1200) = 1200            # 與現行 Thrips 完全相同
Thrips        → round(1200 × 439 / 610) = 864    # 需增強 864 − 439 = 425
Thrips_Damage → 1200 − 864            = 336      # 需增強 336 − 171 = 165
```

最後一個類別取餘數，確保群組總和精確等於 `tgt_total`。
Aphid 不屬於任何群組，照原公式走：raw 673 − 29 = 644 → `min(4×644, 1200) = 1200`，不變。

### 約束 C：不得改動下列任何東西

| 不可改動 | 原因 |
| --- | --- |
| `split_items()` | 動了就換切分 |
| `downsample_scale()` | Scale_Insect 的降採樣結果會變 |
| `write_sample()`、`write_augmented()` | 前者是位元複製，驗收靠它；後者從 `item["boxes"]` 取類別標籤，改標會自動傳遞，不需要動 |
| `AUG`（Albumentations Compose）的任何參數、`seed`、`border_mode` | `border_mode` 若改成 `BORDER_CONSTANT`，旋轉後的黑楔會變成假捷徑 |
| `SEED`、`SPLIT`、`AUG_MULT`、`AUG_CAP`、`SCALE_TARGET_BOXES`、`APHID_MIN_PX`、`JPEG_QUALITY` | 全部是既有結果的一部分 |
| `parse_label()` 的多邊形轉外接矩形與 `[0,1]` 夾取邏輯 | 來源 `Aphid_Leaf_Damage` 全是多邊形標註（13–41 欄），改了會靜默壞掉 |
| 輸出檔名規則 `{類別名}_{序號:05d}.jpg` / `{類別名}_aug_{序號:05d}.jpg` | 驗收腳本用檔名前綴判類別 |

新類別的檔名因此會是 `Thrips_Damage_00000.jpg`。

---

## 3. 介面

新增三個 CLI 參數，預設值維持現行行為（不帶參數時產出與 v5r 完全相同）：

```
--drop-subclass   Aphid:1                          # 可重複；格式 <SOURCES的name>:<來源子類id>
--split-subclass  Thrips:1=Thrips_Damage           # 可重複；格式 <name>:<子類id>=<新類別名>
--out             Datasets/Datasets_YOLO26_v5.5/OutPut
```

本次要執行的指令：

```bash
.venv/Scripts/python.exe tools/build_dataset_v5r.py \
  --drop-subclass Aphid:1 \
  --split-subclass Thrips:1=Thrips_Damage \
  --out Datasets/Datasets_YOLO26_v5.5/OutPut
```

`OUT_ROOT` 目前是模組層級常數，被 `main()` 直接使用，請改為可由 `--out` 覆寫。

模組層級的 `CLASSES` 是 8 個類別的清單，被增強配額迴圈、寫出迴圈、`data.yaml` 與總結表四處
使用。拆類會讓類別數變成 9，請在 `main()` 內用一個區域變數承接（例如 `classes = list(CLASSES)`
再 append），不要就地修改模組常數。

產出的 `data.yaml`：

```yaml
nc: 9
names:
  - Oily_Spot
  - Canker
  - Sooty_Mold
  - Black_Spot
  - Scale_Insect
  - Citrus_Leaf_Miner
  - Thrips
  - Aphid
  - Thrips_Damage
```

第一行註解請改成 `# Datasets_YOLO26_v5.5 —— 由 tools/build_dataset_v5r.py 產生`。

---

## 4. 期望數字（請自行核對，對不上就不要交）

| 項目 | v5r（現況） | v5.5（應得） |
| --- | ---: | ---: |
| train 影像總數 | 7,145 | **7,145** |
| valid 影像總數 | 438 | **437** |
| test 影像總數 | 438 | **435** |
| Thrips train 影像 | 1,200 | **864** |
| Thrips_Damage train 影像 | — | **336** |
| Aphid train 影像 | 1,200 | **1,200** |
| Thrips raw train（增強前） | 610 | **439** |
| Thrips_Damage raw train | — | **171** |
| Aphid raw train | 673 | **644** |
| valid 框：Thrips / Thrips_Damage / Aphid | 125 / — / 197 | **105 / 20 / 194** |
| test 框：Thrips / Thrips_Damage / Aphid | 106 / — / 220 | **85 / 21 / 213** |
| valid 影像：Thrips / Thrips_Damage / Aphid | 76 / — / 84 | **60 / 16 / 83** |
| test 影像：Thrips / Thrips_Damage / Aphid | 76 / — / 84 | **55 / 21 / 81** |

其餘六個類別（Oily_Spot、Canker、Sooty_Mold、Black_Spot、Scale_Insect、
Citrus_Leaf_Miner）與 Background 的所有數字**必須與 v5r 完全相同**。任何一個不同，
就代表約束 A 被違反了。

---

## 5. 自我驗收（交件前請自己跑一次）

### 5.1 切分是否被動到（最關鍵）

```bash
.venv/Scripts/python.exe -c "import hashlib,pathlib; f=lambda d:{hashlib.md5(p.read_bytes()).hexdigest() for p in pathlib.Path(d).glob('*.jpg')}; a=f('Datasets/Datasets_YOLO26_v5r/OutPut/valid/images'); b=f('Datasets/Datasets_YOLO26_v5.5/OutPut/valid/images'); print('v5r',len(a),'v5.5',len(b),'新增',len(b-a),'減少',len(a-b))"
```

通過條件：**新增 = 0、減少 = 1**（valid 只少掉 1 張 Aphid 葉害影像）。
test 的對應數字是 **新增 = 0、減少 = 3**。

> 注意：valid/test 的影像**檔名會改變**（序號是位置性的，pool 縮小後會重新編號），
> 所以只能比對 md5，不能比對檔名。
> **只要「新增」不是 0，就代表資料被重新切分或重新編碼，產出作廢。**

### 5.2 資料集驗收

```bash
.venv/Scripts/python.exe tools/verify_dataset_v5r.py Datasets/Datasets_YOLO26_v5.5/OutPut
```

這支腳本目前只認得 8 個類別，**第 9 類會被報成「類別 id 越界」——這是預期的，不用修它**
（驗收腳本由專案這邊另外處理）。除了這一項以外不應有其他 error，且「三個 split 之間無影像
洩漏」必須通過。

### 5.3 建置 console 輸出

完整保留，尤其是【1】收集、【2】降採樣、【3】增強配額、【5】完成 這四段表格。

---

## 6. 交付物

1. 改動後的 `tools/build_dataset_v5r.py`（unified diff 或完整檔案皆可）。
2. 建置指令的完整 console 輸出。
3. `Datasets/Datasets_YOLO26_v5.5/OutPut/`（`train`/`valid`/`test` + `data.yaml`）。
4. §5.1 兩個 md5 比對指令的輸出。

不要修改 `tools/build_dataset_v5r.py` 以外的任何檔案。不要重構、不要「順手優化」現有函式，
不要調整未列在 §3 的任何常數——這個腳本的每一個既有行為都綁著已發表的數字。
