# 柑橘病蟲害目標偵測（AY2026 專題）

以 **YOLO26-n-P2** 偵測柑橘葉片上的九種病蟲害，部署目標是手機／邊緣裝置。

| | |
| --- | --- |
| **類別** | 九類：油斑病 · 潰瘍病 · 煤煙病 · 黑點病 · 介殼蟲 · 潛葉蛾 · 薊馬 · 蚜蟲 · 薊馬葉害 |
| **架構** | 官方 `yolo26-p2`（`end2end=True`、`reg_max=1`，NMS-free），未做任何自訂修改 |
| **執行環境** | `.venv`（Python）＋ **`ultralytics==8.4.121` 釘死** |
| **訓練** | Kaggle GPU（notebook）；建置、驗收、診斷都在本機 CPU |

> **為什麼版本釘死**：v9 之後的所有數字都在 8.4.121 上取得，換版本會失去可對照性。

---

## 現在做到哪裡

| 版本 | 資料集 | 狀態 | test mAP50 / mAP50-95 |
| --- | --- | --- | --- |
| **v9 / A0** | v5r（8 類） | **目前的交付權重**（`best.pt` @ ep69） | **0.891 / 0.678** |
| v10 | v5.5（9 類） | 已訓練完畢 | 0.809 / 0.606（8 類等效 0.864 / 0.658） |
| **v5.6** | — | **資料集已建置並通過驗收，尚未訓練** | — |

**下一步是拿 v5.6 訓練。** 它把全部九類的評估誤差壓到 ±0.10 以內，
而且沒有新拍任何一張照片——完整說明見
[docs/v5.6_計畫_無新拍攝的資料強化.md](docs/v5.6_計畫_無新拍攝的資料強化.md)。

> 訓練時記得配套改成 `patience=0` ＋ 固定輪數。v5.6 的誤差估計建立在
> 「valid 沒有參與任何決策、可以與 test 併計」這個前提上。

---

## 目錄

```
.
├── README.md            ← 你在這裡
├── docs/                所有文件。索引在 docs/README.md
│   └── archive/         已失效但仍有用途的舊文件（附封存理由）
├── tools/               建置 / 驗收 / 診斷 / 標註流程的腳本
├── Train Code/          訓練用的 notebook
│   ├── yolo26-p2.yaml   官方架構快照（訓練時實際讀的是已安裝的 ultralytics）
│   ├── v9/              六臂消融與長跑 —— 交付權重的來源
│   └── v10/             v5.5 的首次訓練
├── Train Records/       v8 時代的訓練產出（已封存，數字與現行不可比）
└── Datasets/            未進版控。v5 / v5r / v5.5 / v5.6 與外部資料集
```

**`Datasets/` 不進版控**（`.gitignore` 排除），因為它有數十 GB。
每一版都能用 `tools/` 裡對應的建置腳本從來源樹重現。

---

### `docs/` —— 從哪裡開始讀

| 想知道什麼 | 讀這份 |
| --- | --- |
| **專案現況與下一步** | [v5.6_計畫_無新拍攝的資料強化.md](docs/v5.6_計畫_無新拍攝的資料強化.md) |
| **正式交付成果** | [v9_報告_最終技術評估.md](docs/v9_報告_最終技術評估.md)（八章完整報告） |
| 要對外簡報 | [v9_報告_階段性進度週報.md](docs/v9_報告_階段性進度週報.md) |
| 要動資料集之前 | [v5.6_結果_資料集分析與驗收.md](docs/v5.6_結果_資料集分析與驗收.md) |
| 要發派標註工作 | [v5.6_說明_人工標註需求.md](docs/v5.6_說明_人工標註需求.md) |
| 寫報告要列公式 | [通用_參考_評估指標與公式.md](docs/通用_參考_評估指標與公式.md) |

完整清單與命名規則見 **[docs/README.md](docs/README.md)**。
檔名格式是 `{範疇}_{類型}_{主題}.md`，不帶日期（日期寫在內文表頭）。

---

### `tools/` —— 依用途分組

**建置資料集**

| 檔案 | 用途 |
| --- | --- |
| `build_dataset_v5_6.py` | **現行版本。** 巢狀切分（v5.6 eval ⊇ v5.5 eval）＋ 逐類增強 profile ＋ `--arm` ＋ provenance |
| `aug_profiles.py` | 逐類增強設定的單一真實來源，含三條硬性約束 |
| `build_dataset_v5_5.py` · `build_dataset_v5r.py` | 前兩版，**未改動以保可重現性** |

**驗收**

| 檔案 | 用途 |
| --- | --- |
| `verify_dataset_v5_6.py` | 八道 Gate：結構 / 洩漏 / 合成來源 / 評估集 / 增強品質 / 分佈位移 / 可重現 |
| `verify_dataset_v5r.py` | 成對性、類別 id、座標、位元級洩漏（被上面那支呼叫） |
| `check_dataset_leakage.py` | 近重複影像跨 split 洩漏（dHash；位元級檢查抓不到的重複） |
| `analyze_dataset.py` | 組成、框尺寸分位數、分佈位移、增強比例 |

**診斷與評估**

| 檔案 | 用途 |
| --- | --- |
| `diag_localization.py` | 子域分層 AP、殘差分解、尺寸帶、bootstrap（CPU 約 12 分鐘） |
| `final_eval.py` | 本機跑 valid/test 最終評估（CPU 約 6 分鐘） |
| `compare_arms.py` | 消融結果比較，自動套用 2σ 判準 |
| `quantify_leak_impact.py` | 把洩漏對分層指標的影響量化成「洩漏 vs 乾淨」對比 |

**人工標註流程**

| 檔案 | 用途 |
| --- | --- |
| `prepare_manual_packages.py` | 產生待標註影像（含 dHash 去重與等距取樣） |
| `make_annotation_guide.py` | 由既有標註渲染正例／反例對照圖 |
| `check_annotation_return.py` | **收件端**：格式預檢 ＋ 正式驗收 ＋ 併回來源樹（類別 id 重映） |
| `score_annotation_agreement.py` | 兩人標註一致性（貪婪配對 ＋ 中位 IoU） |

**保留但不再使用**

| 檔案 | 為什麼保留 |
| --- | --- |
| `copy_paste_clm.py` | 影像剪接**實測後否決**；檔頭記錄完整失敗分析，日後若有輪廓標註可再用 |
| `dedupe_leaked_train_images.py` | 被 v5.5 的「近重複群組整群移進 train，不刪圖」取代 |

---

## 常用指令

一律在專案根目錄下執行。

```bash
.venv/Scripts/python.exe tools/build_dataset_v5_6.py --dry-run
```

```bash
.venv/Scripts/python.exe tools/build_dataset_v5_6.py
```

```bash
.venv/Scripts/python.exe tools/verify_dataset_v5_6.py
```

```bash
.venv/Scripts/python.exe tools/analyze_dataset.py Datasets/Datasets_YOLO26_v5.6/OutPut
```

`--dry-run` 會先印出切分數字、巢狀相容性與預測誤差——**數字對不上就不要寫檔**。
八道 Gate 全綠才算完成；任何一道紅燈都要回頭修腳本重建，**不接受手動改資料**。

---

## 三條專案層級的規則

1. **v8 的歷史數字與現行完全不可比。** 不只量尺不同——v5r 的 valid 有 44.5%、
   test 有 47.5% 的影像落在 v5 的 train 裡，**不存在對 v8 乾淨的評估集**。
   唯一合法的基準是重訓的 A0。理由見 [docs/archive/README.md](docs/archive/README.md)。

2. **紅燈只有兩種處置：要嘛資料真的有問題（就重建），要嘛量法有問題（就改量法）。**
   不因為看起來刺眼就調寬門檻放行。v5.6 的驗收過程兩種都發生過，
   記錄在 [v5.6_結果](docs/v5.6_結果_資料集分析與驗收.md) §4。

3. **對外報告一律同時列 `best.pt` 與平台期平均。** `best.pt` 是在數十個
   檢查點裡挑最大值，帶有選擇偏誤——v10 實測高出平台期 2.8σ。
