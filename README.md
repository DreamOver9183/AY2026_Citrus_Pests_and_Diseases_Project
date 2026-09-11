# 一葉知病 OneLeaf — 病蟲害辨識模型

以 **YOLO26-n-P2** 偵測柑橘葉片上的九種病蟲害，部署目標是手機／邊緣裝置。
本庫是「一葉知病」專題的**影像辨識這條線**：訓練、評估與部署量測。

| 這個專題的其他部分 | |
| --- | --- |
| **對外研究報告** | [`OneLeaf-dx/report`](https://github.com/OneLeaf-dx/report) —— 七章 27 篇，含 RAG／SLM 與行動端 App |
| **組織首頁** | [`OneLeaf-dx`](https://github.com/OneLeaf-dx) |

> 本庫原本掛在個人帳號下，2026-09-10 轉入組織並改名為 `detection`。
> 舊網址會自動轉址，但請把本機 remote 換成新的：
> `git remote set-url origin https://github.com/OneLeaf-dx/detection.git`

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
| v11 | v5.6（9 類） | 已訓練完畢 | 0.822 / 0.613（v5.5 子集 0.800 / 0.603） |
| **v11.5** | v5.6（9 類） | **已訓練完畢。九類 ±2SE ≤ 0.10 達成** | 0.810 / 0.607（`last.pt`） |
| **v12s** | v5.6（9 類） | **已訓練完畢。判準未通過 → 容量不是瓶頸** | 0.813 / 0.618（`last.pt`） |

**v5.6 的招牌目標已經達成**——v11.5 用 `patience=0` ＋ 固定 70 輪重跑，
**九類的 pooled ±2SE 全部 ≤ 0.10**（最差 0.099），而且比 v11 少 1.36 小時（3.06 h）。
見 [docs/v11.5_結果_評估精度驗證.md](docs/v11.5_結果_評估精度驗證.md)。

**v12s 把「模型容量」這條路也關掉了**：3.84× 參數只換到 **+0.003** mAP50，
遠低於預先登記的門檻 0.03，也低於 run 間全距 0.021。
連同 v9 六臂與 v10→v11.5，**模型端已連續四輪、共十一種改動全部無效**。
見 [docs/v12s_結果_模型容量探索.md](docs/v12s_結果_模型容量探索.md)。

**部署端也量完了**：平台 1（Snapdragon 662）**達不到 30 FPS**，
任何官方匯出參數組合都不行；精度可用的最佳是 `fp32 @ 320` 的 15.4 FPS。
見 [docs/v12.1_結果_匯出參數掃描.md](docs/v12.1_結果_匯出參數掃描.md)。

**所以下一步是人工工作包 A**（`Thrips_Damage` 的雙人一致性測試）。
它是唯一剩下的顯著 valid/test 矛盾，三輪都沒有隨評估集擴大而收斂
（0.285 → 0.190 → 0.202）——成因是框定義不一致，補樣本量救不了。
**在模型端與部署端都被排除之後，資料端是目前唯一還沒試過的方向。**

> **往後的訓練協定**：一律 `patience=0` ＋ 固定輪數 ＋ 報 `last.pt`。
> v11.5 實測 `best.pt` 與 `last.pt` 差距 ≤ 0.004 且方向不一致，
> 改用 `last.pt` 沒有代價，卻讓 valid 可合法與 test 併計。

---

## 目錄

```
.
├── README.md            ← 你在這裡
├── AGENTS.md            AI agent 的操作規則：紅線、訓練協定、判準、收工前要跑什麼
├── .claude/skills/      benchmark skill（`.claude/` 底下唯一進版控的東西）
├── docs/                所有文件。索引在 docs/README.md
│   └── archive/         已失效但仍有用途的舊文件（附封存理由）
├── tools/               建置 / 驗收 / 診斷 / 標註流程的腳本
├── Benchmark/           手機端 TFLite 延遲量測（部署目標 30 FPS ±5）
│   ├── platform-tools/  ADB 與 benchmark APK（已進版控，clone 即可用）
│   ├── export/          .pt → .tflite 的 Docker 匯出管線
│   ├── Model/           待測的 .tflite（*.tflite 未進版控）
│   └── report/          量測報告
├── Train Code/          各版本的訓練 notebook 與其輸出
│   ├── yolo26-p2.yaml   官方架構快照（訓練時實際讀的是已安裝的 ultralytics）
│   ├── v9/              六臂消融與長跑 —— 交付權重的來源
│   ├── v10/             v5.5 的首次訓練
│   ├── v11/             v5.6 的首次訓練（patience=30，評估精度報不出來）
│   ├── v11.5/           **目前的基準**：patience=0 + 70 輪，±2SE 目標達成
│   └── v12s/            放大模型的實驗（yolo26**s**-p2）——**已完成，容量不是瓶頸**
│       每個版本一律是 `train_<版本>.ipynb` + `Train_output/`；後者放 Kaggle 下載的
│       zip（未進版控）與解壓出來的 `extracted/`，只有權重、超參數與評估數據進版控
├── Train Records/       v8 時代的訓練產出（已封存，數字與現行不可比）
├── _History/            2026 年 6–7 月的歸檔（5 個 zip、27.6 GB，未進版控；只有索引在版控裡）
└── Datasets/            未進版控，約 54 GB。依**處理階段**分三層：
    ├── 1_原始影像/        人工標註過的來源影像，還沒切分、沒增強
    │   ├── v5r/ v5.5/ v5.6/      各自 {Diseases, Healthy, Pests}
    │   ├── _人工標註待辦/          發給組員的工作包，回收後併進上面
    │   └── _外部資料集/            論文引用的公開資料集（全部沒有標註）
    ├── 2_處理與切分/      切分＋增強後，可以直接餵給 YOLO
    │   └── v5/ v5r/ v5.5/ v5.6/  各自 {train, valid, test} + data.yaml
    └── 3_最終輸出/        打包好、可上傳 Kaggle 或交付的壓縮檔
```

**為什麼按階段分而不是按版本**：這三層是資料的三個生命階段，
而且**只有第一層是不可重生的**——第二層可以由建置腳本從第一層重建，
第三層可以從第二層壓出來。按階段分，「哪些東西丟了會真的救不回來」一眼就看得出來。

**`Datasets/` 不進版控**（`.gitignore` 排除），因為它有數十 GB。
路徑由 `tools/dataset_paths.py` 統一定義，**所有腳本都從那裡取**——
以前九支腳本各自寫死路徑，搬一次資料夾就要改九個地方。

---

### `docs/` —— 從哪裡開始讀

| 想知道什麼 | 讀這份 |
| --- | --- |
| **專案現況與下一步** | [v5.6_計畫_無新拍攝的資料強化.md](docs/v5.6_計畫_無新拍攝的資料強化.md) |
| **最新一輪的訓練結果** | [v12s_結果_模型容量探索.md](docs/v12s_結果_模型容量探索.md) |
| **要口頭／面對面報告** | [v12.1_報告_口頭簡報.md](docs/v12.1_報告_口頭簡報.md)（重點版，約 8 分鐘，附備答） |
| 要查依據或存檔紀錄 | [v12.1_報告_階段性進度週報.md](docs/v12.1_報告_階段性進度週報.md)（詳細版，**§0 是全案盤點**） |
| **查歷史數字／確認能不能比較** | [通用_參考_歷史訓練與資料集紀錄.md](docs/通用_參考_歷史訓練與資料集紀錄.md) |
| **v10 之前的全部成果與教訓** | [v9_報告_v10之前的技術總結.md](docs/v9_報告_v10之前的技術總結.md)（由七份封存文件併成） |
| 要動資料集之前 | [v5.6_結果_資料集分析與驗收.md](docs/v5.6_結果_資料集分析與驗收.md) |
| 要發派標註工作 | [v5.6_說明_人工標註需求.md](docs/v5.6_說明_人工標註需求.md) |
| 寫報告要列公式 | [通用_參考_評估指標與公式.md](docs/通用_參考_評估指標與公式.md) |

完整清單與命名規則見 **[docs/README.md](docs/README.md)**。
檔名格式是 `{範疇}_{類型}_{主題}.md`，不帶日期（日期寫在內文表頭）。

---

### `Benchmark/` —— 部署端的延遲量測

到 v11.5 為止，這個專案只量得出**精度**。`Benchmark/` 補上另一半：
模型放到手機上到底跑多快。**部署目標 30 FPS ±5，即每張 28.6–40 ms。**

由 [`DreamOver9183/TFLite_Benchmark_skills`](https://github.com/DreamOver9183/TFLite_Benchmark_skills)
合併而來（上游 `694b95f`），**不是 submodule**。合併時把上游的 `tools/` 改名成
`platform-tools/`——本專案根目錄已經有一個 `tools/`，撞名會讓 skill 指到錯的地方。

| 想做什麼 | 看這裡 |
| --- | --- |
| 整體流程與本專案的追加規範 | [Benchmark/README.md](Benchmark/README.md) |
| 把 `.pt` 轉成 `.tflite` | [Benchmark/export/README.md](Benchmark/export/README.md) |
| 手動跑一次量測 | [Benchmark/Benchmark_process.md](Benchmark/Benchmark_process.md) |
| Agent 的行為邊界 | [Benchmark/AGENTS.md](Benchmark/AGENTS.md) |

Skill 本體在 `.claude/skills/tflite_mobile_benchmark/`（對 Claude Code 說
「跑 benchmark」即觸發）。它是 `.claude/` 底下**唯一**進版控的東西，
靠 `.gitignore` 的一條負向規則放行。

---

### `tools/` —— 依用途分組

**建置資料集**

| 檔案 | 用途 |
| --- | --- |
| `dataset_paths.py` | **`Datasets/` 版面配置的單一真實來源。** 要搬資料夾只改這一支 |
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

**部署**

| 檔案 | 用途 |
| --- | --- |
| `export_tflite.py` | `.pt` → `.tflite` 的四軸掃描（量化 × 解析度 × end2end × max_det），含逐框比對與完整 mAP。**只能在 `Benchmark/export/` 的容器裡跑**——ultralytics 的 LiteRT 匯出在 Windows 被平台斷言擋住 |
| `judge_deploy_stoploss.py` | 即時辨識功能的採用門檻＝部署線的止損判準（[v12.2 計畫](docs/v12.2_計畫_部署線止損標準.md)），**門檻寫死在程式裡**。精度讀匯出報告、延遲讀 `run_mobile_benchmark.py` 的 JSON（唯一會交錯重跑取最小值的入口），自動判過線、第二階段准入與期限 |

**人工標註流程**

| 檔案 | 用途 |
| --- | --- |
| `prepare_manual_packages.py` | 產生待標註影像（含 dHash 去重與等距取樣） |
| `make_annotation_guide.py` | 由既有標註渲染正例／反例對照圖 |
| `check_annotation_return.py` | **收件端**：格式預檢 ＋ 正式驗收 ＋ 併回來源樹（類別 id 重映） |
| `score_annotation_agreement.py` | 兩人標註一致性（貪婪配對 ＋ 中位 IoU） |

**已被 `Benchmark/` 取代**

| 檔案 | 為什麼保留 |
| --- | --- |
| `run_mobile_benchmark.py` · `export_other_formats.py` | 手機延遲量測、其他匯出格式的舊入口。現行流程走 `Benchmark/`（見上方 Benchmark 段） |

**實測後否決，保留作紀錄**

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
.venv/Scripts/python.exe tools/analyze_dataset.py Datasets/2_處理與切分/v5.6
```

`--dry-run` 會先印出切分數字、巢狀相容性與預測誤差——**數字對不上就不要寫檔**。
八道 Gate 全綠才算完成；任何一道紅燈都要回頭修腳本重建，**不接受手動改資料**。

---

## 三條專案層級的規則

完整的操作規範（紅線、目錄職責、收工關卡、兩個容易踩的坑）在 **[AGENTS.md](AGENTS.md)**。

1. **v8 的歷史數字與現行完全不可比。** 不只量尺不同——v5r 的 valid 有 44.5%、
   test 有 47.5% 的影像落在 v5 的 train 裡，**不存在對 v8 乾淨的評估集**。
   唯一合法的基準是重訓的 A0。理由見 [docs/archive/README.md](docs/archive/README.md)。

2. **紅燈只有兩種處置：要嘛資料真的有問題（就重建），要嘛量法有問題（就改量法）。**
   不因為看起來刺眼就調寬門檻放行。v5.6 的驗收過程兩種都發生過，
   記錄在 [v5.6_結果](docs/v5.6_結果_資料集分析與驗收.md) §4。

3. **對外報告一律同時列 `best.pt` 與平台期平均。** `best.pt` 是在數十個
   檢查點裡挑最大值，帶有選擇偏誤——v10 實測高出平台期 2.8σ。
