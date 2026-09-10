# AGENTS.md

本檔是所有 AI agent 操作此 repo 的**唯一入口**，不分廠商。動手前先讀完。
姊妹庫 [`OneLeaf-dx/report`](https://github.com/OneLeaf-dx/report) 有自己的一份，規範不同，**不要互相套用**。

## 這是什麼

「一葉知病 OneLeaf」的**影像辨識線**：YOLO26-n-P2 偵測柑橘葉片九類病蟲害，
目標是手機／邊緣裝置。這裡的「正確性」指的是**實驗可對照**與**資料可重現**，
不是測試綠燈——本庫沒有測試套件。

## 紅線（任何情況都不得跨越）

| # | 禁止 | 原因 |
| --- | --- | --- |
| R1 | **手動修改實驗產出**：`Train_output/` 底下的 `summary.json`、`results.csv`、`per_class*.csv`、`args.yaml` | 那是量測結果本身。數字不對只有兩條路：資料真的有問題就重建，量法有問題就改量法再重跑 |
| R2 | 動 `ultralytics` 的版本（釘死 `8.4.121`） | v9 之後的所有數字都在這個版本上取得，換版本等於全部失去可對照性 |
| R3 | `git push --force`、`rebase` 或 `amend` 已推送的 commit | 遠端是組織共用 |
| R4 | 把 `Datasets/`、`runs/`、`_History/*.zip` 加進版控 | 數十 GB。`.gitignore` 已排除，不要用 `git add -f` 繞過 |
| R5 | 在腳本裡寫死 `Datasets/` 的路徑 | 一律走 `tools/dataset_paths.py`。以前九支腳本各寫各的，搬一次資料夾要改九個地方 |

## 訓練協定（v11.5 起）

**一律 `patience=0` ＋ 固定輪數 ＋ 報 `last.pt`。**
`best.pt` 是在數十個檢查點裡挑最大值，帶選擇偏誤（v10 實測高出平台期 2.8σ）；
v11.5 實測 `best.pt` 與 `last.pt` 差距 ≤ 0.004 且方向不一致，改用 `last.pt` 沒有代價，
卻讓 valid 可以合法與 test 併計。**對外報告一律同時列 `best.pt` 與平台期平均。**

## 判準

改動要不要採納，**先登記門檻再看結果**，不要看到數字再決定怎麼算贏。
現行門檻是 2σ（`tools/compare_arms.py` 會自動套用）。
到目前為止模型端連續四輪、共十一種改動全部未過——**提新的模型端改動之前，
先讀 [docs/v12s_結果_模型容量探索.md](docs/v12s_結果_模型容量探索.md)**。

## 目錄職責

| 目錄 | 放什麼 | 不放什麼 |
| --- | --- | --- |
| `docs/` | 內部文件。檔名 `{範疇}_{類型}_{主題}.md`，不帶日期 | 對外研究報告（那是 `report` 庫的事） |
| `tools/` | 建置 / 驗收 / 診斷 / 匯出 / 標註流程的腳本 | 一次性的探索腳本 |
| `Benchmark/` | 延遲量測的**方法**與工具鏈 | 量測**數字**（回到 `docs/` 用 `v12_結果_*` 命名） |
| `Train Code/<版本>/` | `train_<版本>.ipynb` ＋ `Train_output/` | 跨版本共用的程式（抽到 `tools/`） |

`Train_output/` 的版面固定為：Kaggle 下載的 zip（未進版控）＋ `extracted/`（只有
`best.pt`／`last.pt`／`args.yaml`／`*.json`／`*.csv` 進版控，其餘由 zip 重現）。

## 收工前

沒有 CI，所以下面這些要自己跑。**不要以「我檢查過了」代替實際執行。**

```bash
.venv/Scripts/python.exe tools/verify_dataset_v5_6.py
```

動過資料集就要跑，**八道 Gate 全綠才算完成**。任何一道紅燈都回頭修腳本重建，
不接受手動改資料，也不因為看起來刺眼就調寬門檻。

```bash
git status --short
```

確認沒有把權重、資料集或解壓工作區帶進索引。新增訓練輸出後特別要看，
`.gitignore` 的白名單只放行 `best.pt`／`last.pt`／`args.yaml`／JSON／CSV。

## 兩個容易踩的坑

1. **`resume=True` 會靜默換掉權重。** `ultralytics 8.4.121` 的續跑會在不報錯的情況下
   換成 COCO 預訓練權重，**辨識法是看轉移率是不是 902/902**。已知的正確做法寫在
   `Train Code/v9/RESUME.ipynb`，要續跑先照那份的檢查步驟走。
2. **本庫指向 `DreamOver9183/TFLite_Benchmark_skills` 與
   `DreamOver9183/Citrus_Pest_and_Disease_Tools` 的連結是對的，不要「順手」改成組織網址**
   ——那兩個 repo 沒有搬進 `OneLeaf-dx`。

## 提交

正體中文，第一行講清楚做了什麼，空行後列變更點。涉及指標變化時把數字寫進 message。
**推送前取得使用者確認**——遠端是組織共用的公開 repo。
