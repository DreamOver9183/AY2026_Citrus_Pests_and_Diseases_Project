# 文件索引

## 命名格式

```
{範疇}_{類型}_{主題}.md
```

| 欄位 | 可用值 |
| --- | --- |
| 範疇 | `v5` · `v5r` · `v5.5` · `v8` · `v9` · `v10` · `通用` |
| 類型 | `報告`（對外呈現） · `結果`（實驗數據） · `記錄`（工程過程） · `說明`（建置方法） · `參考`（不隨版本變動的定義） |

檔名不帶日期——文件會持續更新，日期寫在內文的表頭。

---

## 現行文件

| 檔案 | 內容 | 何時該讀 |
| --- | --- | --- |
| [v9_報告_最終技術評估.md](v9_報告_最終技術評估.md) | **正式交付文件。** 八章節完整技術報告：超參數、資料分佈、收斂動態、混淆矩陣與四象限、誤差診斷、消融實驗、改進建議、研究限制 | 要正式引用或交付 |
| [v9_報告_階段性進度週報.md](v9_報告_階段性進度週報.md) | 同樣的成果，但濃縮成可在會議現場展示的版本，含四張圖表 | 要向外簡報專案現況 |
| [v9_說明_P3定位精度改善與測試流程.md](v9_說明_P3定位精度改善與測試流程.md) | **P3 的重新定義。** 子域分層診斷推翻「小目標」歸因，給出刪除／拆類／標註規則三套方案與四臂測試流程、bootstrap 判準 | 要動 Thrips / Aphid / Scale_Insect 的標註之前 |
| [v9_結果_最終評估_valid與test.md](v9_結果_最終評估_valid與test.md) | 交付權重 `best.pt` @ ep69 在兩個 split 上的完整數據：per-class、混淆矩陣、四象限、部署閾值 | 要查最終評估的原始數字 |
| [v9_記錄_實驗重整與六臂消融.md](v9_記錄_實驗重整與六臂消融.md) | 從程式審查、六臂設計到逐臂結果與三次長跑的完整工程紀錄（1100+ 行） | 要查某個決定的來龍去脈 |
| [v10_結果_v5.5首次訓練評估.md](v10_結果_v5.5首次訓練評估.md) | **v10 @ v5.5 的第一份結果。** test mAP50 0.809 / mAP50-95 0.606；關鍵發現是評估集太小——Scale_Insect 的 valid/test 差 0.308（超過 2σ），per-class 結論多半不成立 | 要引用 v10 數字或決定下一步之前 |
| [v5.5_結果_資料集分析.md](v5.5_結果_資料集分析.md) | **v5.5 的組成、框尺寸、分佈位移與洩漏驗收**；含兩個解讀指標時必須考慮的風險（Thrips 評估集偏小、Canker 佔 train 框數 34.3%） | 要用 v5.5 訓練或解讀其指標之前 |
| [v5r_說明_資料集建置與標註重整.md](v5r_說明_資料集建置與標註重整.md) | `Datasets_YOLO26_v5r` 怎麼從 v5 重建：Thrips 合併、Aphid 過濾、極差從 24.7x 降到 9.1x | 要動資料集之前 |
| [v5r_記錄_近重複影像跨split洩漏查驗.md](v5r_記錄_近重複影像跨split洩漏查驗.md) | 洩漏的查驗、量化與修法全紀錄；**§9：v5.5 已全類別零洩漏**（近重複群組整群移進 train，不刪圖）。v5r 本身仍有洩漏且不打算修 | 評估既有 v9 數字的可信度、要動任何資料集之前 |
| [通用_參考_評估指標與公式.md](通用_參考_評估指標與公式.md) | 混淆矩陣四象限、Accuracy/Precision/Recall/FPR/F1 的定義與 LaTeX 公式 | 寫報告要列公式 |

## 已封存

見 [archive/](archive/)——舊版本的報告與說明，數字已不適用於現行專案，但仍保留作為歷史紀錄與格式範本。

---

## 相關程式

| 路徑 | 用途 |
| --- | --- |
| `tools/final_eval.py` | 本機執行 valid/test 最終評估（CPU 約 6 分鐘） |
| `tools/diag_localization.py` | P3 定位精度診斷：provenance 還原、子域分層 AP、殘差分解、尺寸帶、bootstrap（CPU 約 12 分鐘） |
| `tools/compare_arms.py` | 六臂消融結果比較，自動套用 2σ 判準 |
| `tools/build_dataset_v5r.py` | 由來源資料集建置 v5r（未改動，維持可重現） |
| `tools/build_dataset_v5_5.py` | 由 v5r + 新 Scale_Insect/Canker 重標 + Thrips 拆類 + Aphid 過濾建置 v5.5；每類獨立切分 |
| `tools/verify_dataset_v5r.py` | v5r 建置後的驗收 |
| `tools/analyze_dataset.py` | 建置後的資料集分析：組成、框尺寸分位數、分佈位移、增強比例 |
| `tools/check_dataset_leakage.py` | 近重複影像跨 split 洩漏查驗（dHash，位元級 md5 檢查抓不到的重複） |
| `tools/quantify_leak_impact.py` | 把洩漏對 P3 分層指標（中位 IoU/AP50-95）的影響量化成「洩漏 vs 乾淨」子集對比 |
| `tools/dedupe_leaked_train_images.py` | 只從 train 移除洩漏影像的修正工具，預設 dry-run，`--apply` 才真的刪除 |
| `Train Code/v10/train_v10.ipynb` | **v10 訓練（v5.5 / 9 類）**，由 v9 的 A0 臂改寫；自足、無自訂模組注入 |
| `Train Code/v9/train_ablation.ipynb` | Kaggle 訓練（改一行 `ARM` 切換臂別） |
| `Train Code/v9/RESUME.ipynb` | Kaggle 續跑 |
| `Train Code/v9/v9_modules.py` | 自訂 loss、模組、六臂定義的單一真實來源 |
