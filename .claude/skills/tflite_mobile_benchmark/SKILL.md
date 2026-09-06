---
name: tflite_mobile_benchmark
description: >
  使用 ADB + android_aarch64_benchmark_model.apk 對 Android 設備執行 TFLite 模型效能基準測試。
  觸發情境：使用者要求「跑 benchmark」、「測試模型效能」、「測試推論速度」、
  「比較 tflite 模型」、「測 FPS」、「測延遲」或任何涉及 ADB、TFLite、
  mobile benchmark 的請求時，均應使用本 Skill。
---

# TFLite Mobile Benchmark — 標準執行流程

本 Skill 定義了使用 ADB 工具對 Android 設備執行 TFLite 模型效能測試的完整五階段流程。
執行前請先確認已讀取並遵循 `Benchmark/AGENTS.md` 中的行為邊界規範。

---

## 🔧 工具與路徑速查

```
工作根目錄  : D:\Claude Code\AY2026 Citrus Pests and Diseases Project\
ADB 路徑    : .\Benchmark\platform-tools\adb.exe
APK 路徑    : .\Benchmark\platform-tools\android_aarch64_benchmark_model.apk
模型目錄    : .\Benchmark\Model\                        ← 唯一合法的模型來源
手機暫存    : /data/local/tmp/
報告輸出    : .\Benchmark\report\benchmark_report_YYYY-MM-DD.md
報告範本    : .\Benchmark\report\report_sample.md       ← 格式參考（唯讀）
```

**標準測試參數（預設值）：**
```
--num_threads=4
--num_runs=25
--use_gpu=false
--use_nnapi=false
```

> ⚠️ **模型來源限制**：只允許讀取 `.\Benchmark\Model\` 目錄下的 `.tflite` 檔案。
> 若目錄為空，必須提示使用者先將模型放入該目錄，再重新啟動流程。

---

## 📌 本專案的追加規範（AY2026 柑橘病蟲害專案）

以下三點**覆寫**上游的預設行為。與上面的通用流程衝突時，以本節為準。

### A. 每個模型量兩組設定，不是一組

上游預設只量 CPU。本專案兩組都要，因為它們回答不同的問題：

| 設定 | 追加參數 | 回答什麼 |
| --- | --- | --- |
| **CPU-only 4T** | `--use_gpu=false --use_nnapi=false` | **保底 FPS**。決策門檻用這條 |
| **Delegate** | `--use_gpu=true`（再跑一次 `--use_nnapi=true`） | **部署 FPS**，且**必須**記錄節點替換率 |

兩組都用 `--num_threads=4 --num_runs=25`。報告的摘要表格要有「設定」欄區分兩者。

> **節點替換率一定要看。** YOLO26 是 `end2end=True`（NMS-free），偵測頭含 topk。
> 這類算子若不被 delegate 支援，整段會退回 CPU——delegate 的 FPS
> **有可能比純 CPU 更差**。這是要量的，不是能推論的。

### B. `__int8` 那一列不能跟其他變體直接比

匯出腳本產出的檔名是 `<stem>__<variant>.tflite`，variant 有三種：

| variant | `end2end` | 量到的延遲**包含 NMS** 嗎 |
| --- | --- | --- |
| `__fp32`  | 保留 | ✅ 包含 |
| `__w8a32` | 保留 | ✅ 包含 |
| `__int8`  | **被 exporter 關掉** | ❌ **不包含**，NMS 要在 App 端另外做 |

`ultralytics/engine/exporter.py` 在 `quantize in {8, "w8a16"}` 時會強制
`model.end2end = False`。所以 `__int8` 少做了一段工作，它的 FPS 天生偏高。
**報告裡務必標註這一點**，否則會得出錯誤的結論。

要跟 fp32 做 apples-to-apples 的 INT8 比較，用 **`__w8a32`**。

### C. 目標值

**30 FPS ±5，即每張 28.6–40 ms。** 摘要表格每一列都要標「達標 / 未達標」，
並以 **CPU-only 那一列**作為判定依據。

### D. 模型從哪來

`Benchmark\Model\` 預設為空，且 `*.tflite` 已被 gitignore。
本專案的 `.pt` 要先經 `Benchmark\export\` 的 Docker 管線轉換：

```
docker run --rm -v "%cd%":/work citrus-tflite-export python tools/export_tflite.py --weights "<.pt>" --variants fp32,w8a32 --verify
```

> ⚠️ **沒通過 `--verify` 的檔案不准進 benchmark。** 匯出報告在
> `Benchmark\Model\<stem>__export_report.json`，執行 Phase 1 時要一併確認
> 對應變體的 `verify.verdict` 是「通過」。量一個壞掉的圖只會產生看起來很正常的假數字。

---

## Phase 1 — 前置驗證（硬邊界）

> ⚠️ **此階段任何一項失敗均須立即停止並回報錯誤，不得繼續執行後續 Phase。**

### 1.1 確認設備連線

```powershell
# 在工作根目錄執行
.\Benchmark\platform-tools\adb.exe devices -l
```

**驗收標準：**
- ✅ 輸出中包含 `device`（例如 `ebe3968d device product:...`）→ 繼續
- ❌ 輸出為空或僅有 `List of devices attached` → 停止，提示 USB 連線問題
- ❌ 輸出包含 `unauthorized` → 停止，提示使用者在手機解鎖授權
- ❌ 輸出包含 `offline` → 停止，提示重新插拔 USB

### 1.2 確認 APK 存在

```powershell
Test-Path ".\Benchmark\platform-tools\android_aarch64_benchmark_model.apk"
```

**驗收標準：**
- ✅ 回傳 `True` → 繼續
- ❌ 回傳 `False` → 停止，提示 `Benchmark\platform-tools\` 目錄缺少 APK 檔案

### 1.3 確認模型目錄與模型存在

```powershell
# 列出 Model\ 目錄中可用的模型
Get-ChildItem ".\Benchmark\Model\*.tflite" | Select-Object Name, @{N="MB";E={[math]::Round($_.Length/1MB,2)}}
```

**驗收標準：**
- ✅ 目標模型存在於 `Benchmark\Model\` 清單中 → 繼續
- ❌ `Benchmark\Model\` 目錄為空（無 .tflite）→ 停止，提示：「請將 .tflite 模型檔案放入 `Benchmark\Model\` 目錄後重試」
- ❌ 目標模型不在清單中 → 停止，列出可用模型讓使用者選擇

---

## Phase 2 — 安裝與部署

### 2.1 安裝 APK

```powershell
.\Benchmark\platform-tools\adb.exe install -r -d -g ".\Benchmark\platform-tools\android_aarch64_benchmark_model.apk"
```

**參數說明：**
- `-r`：允許重新安裝（覆蓋）
- `-d`：允許降版本安裝
- `-g`：自動授予所有權限

**驗收標準：**
- ✅ 輸出包含 `Success` → 繼續
- ❌ 輸出包含 `Failure` → 停止並回報錯誤訊息

### 2.2 推送模型至手機

```powershell
# 模型必須來自 .\Benchmark\Model\ 目錄
.\Benchmark\platform-tools\adb.exe push ".\Benchmark\Model\<model_name>.tflite" /data/local/tmp/
```

**驗收標準：**
- ✅ 輸出包含 `pushed` 及傳輸速率 → 繼續
- ❌ 輸出包含 `error` 或 `failed` → 停止並回報

---

## Phase 3 — 測試執行

### 3.1 清除舊日誌（必要）

```powershell
.\Benchmark\platform-tools\adb.exe logcat -c
```

> ⚠️ 此步驟必須在啟動測試前執行，否則可能混入上一次的測試結果。

### 3.2 啟動 Benchmark Activity

**PowerShell 語法：**
```powershell
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/<model_name>.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```

**CMD 語法：**
```cmd
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args "\"--graph=/data/local/tmp/<model_name>.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\""
```

### 3.3 等待測試完成

根據模型大小決定等待時間：

| 模型類型 | 預估等待時間 |
|---|---|
| 小型模型（< 10MB）| 15 秒 |
| 中型模型（10–50MB）| 30 秒 |
| 大型模型（> 50MB）| 60 秒以上（如 yolo26l 需 ~180 秒）|

```powershell
Start-Sleep -Seconds <wait_seconds>
```

---

## Phase 4 — Log 擷取與解析

### 4.1 讀取 TFLite Log

```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s tflite
```

**關鍵輸出行（必須找到的欄位）：**

```
Inference timings in us: Init: <X>, First inference: <Y>, Warmup (avg): <Z>, Inference (avg): <W>
Memory footprint delta from the start of the tool (MB): init=<A> overall=<B>
Replacing <N> out of <T> node(s) with delegate
```

### 4.2 數據解析規範

從 Log 提取以下原始數值（單位：微秒 us）：

| Log 欄位 | 變數名稱 | 換算方式 |
|---|---|---|
| `Init` | `init_us` | `÷ 1000` → ms |
| `First inference` | `first_us` | `÷ 1000` → ms |
| `Warmup (avg)` | `warmup_us` | `÷ 1000` → ms |
| `Inference (avg)` | `inference_us` | `÷ 1000` → ms |
| `min=` | `min_us` | `÷ 1000` → ms |
| `max=` | `max_us` | `÷ 1000` → ms |
| `std=` | `std_us` | `÷ 1000` → ms |
| `init=` (MB) | `init_mem_mb` | 直接讀取 |
| `overall=` (MB) | `overall_mem_mb` | 直接讀取 |
| delegate nodes `N` | `delegate_n` | 直接讀取 |
| total nodes `T` | `total_n` | 直接讀取 |

**計算衍生指標：**
```
首輪推論 (ms)  = first_us / 1000
最快 (ms)      = min_us / 1000
最慢 (ms)      = max_us / 1000
平均推論 (ms)  = inference_us / 1000
預估 FPS       = 1000000 / inference_us
標準差 (ms)    = std_us / 1000
節點替代率 (%) = (delegate_n / total_n) × 100
Init 記憶體    = init_mem_mb
Overall 記憶體 = overall_mem_mb
運算開銷差值   = overall_mem_mb - init_mem_mb
```

> ⚠️ **注意**：當 Log 中 Inference avg 以科學計數法顯示（如 `2.44949e+06`），
> 必須先轉換為普通數值再換算。

---

## Phase 5 — 報告生成

### 5.1 摘要表格格式

```markdown
| 模型名稱 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 (%) | Init 記憶體 (MB) | Overall 記憶體 (MB) | 運算開銷差值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<model_name>` | <val> | <val> | <val> | <val> | <val> | <val> | <val>% | <val> | <val> | <val> |
```

### 5.2 報告寫入規則

- **輸出位置**：一律寫入 `.\Benchmark\report\` 目錄
- **命名格式**：`benchmark_report_YYYY-MM-DD.md`
- **同日多次**：加上版本號，如 `benchmark_report_2026-07-21_v2.md`
- **格式參考**：`.\Benchmark\report\report_sample.md`（禁止修改此範本）
- **批次測試**：單一報告包含所有模型結果

### 5.3 完整報告結構（參考 report_sample.md）

```markdown
# Mobile Benchmark 效能測試報告

## 1. 測試環境 (Environment)
* **測試設備代號 (Device ID)**: <from adb devices>
* **產品型號 (Product/Model)**: <from adb devices -l>
* **設備名稱 (Device)**: <from adb devices -l>
* **作業系統**: Android
* **測試工具**: TFLite Android AArch64 Benchmark Model (`android_aarch64_benchmark_model.apk`)
* **測試日期**: <YYYY-MM-DD>

## 2. 測試主題 (Test Topic)
## 3. 測試參數 (Test Parameters)
## 4. 輸出概要 (Output Summary)   ← 摘要表格
## 5. 完整輸出日誌 (Full Output Log) ← <details> 摺疊 Log
```

---

## 批次測試流程（多模型）

```
FOR EACH model IN model_list:
  1. Phase 1.3 確認模型在 Model\ 目錄中存在
  2. Phase 2.2 推送模型（從 Model\ 目錄讀取）
  3. Phase 3.1 清空日誌（每次均需執行）
  4. Phase 3.2 啟動測試
  5. Phase 3.3 等待
  6. Phase 4.1 讀取 Log
  7. Phase 4.2 解析數據（累積至表格）
  8. 繼續下一個模型
END FOR
9. Phase 5 生成彙整報告並寫入 report\ 目錄
```

---

## 常見問題排解

### Q1：Model\ 目錄為空
- 提示使用者：「請將 `.tflite` 模型檔案放入 `Benchmark\Model\` 目錄後重試」
- 禁止從其他位置（如 `Benchmark\platform-tools\`）讀取 `.tflite` 檔案代替

### Q2：logcat 輸出為空
- 等待 5 秒後重新執行 `.\Benchmark\platform-tools\adb.exe logcat -d -s tflite`
- 若仍為空，執行 `.\Benchmark\platform-tools\adb.exe logcat -d -s AndroidRuntime` 查看崩潰訊息

### Q3：找不到 `Inference timings` 行
- 可能測試尚未完成，延長等待時間再重試
- 確認 APK 是否安裝成功（`.\Benchmark\platform-tools\adb.exe shell pm list packages | grep tensorflow`）

### Q4：ADB 找不到設備
- 確認手機已啟用「開發者選項」>「USB 偵錯」
- 確認已在手機上允許 USB 偵錯授權對話框
- 嘗試重新插拔 USB 線後再執行 `.\Benchmark\platform-tools\adb.exe devices`

### Q5：科學計數法數值解析
- Log 中出現 `2.44949e+06` 等格式時，需轉換：`2.44949e+06 = 2449490`（微秒）
- 換算為毫秒：`2449490 / 1000 = 2449.49 ms`

---

## 參考文件

- ADB 指令速查：`references/adb_commands_reference.md`
- Log 解析規範：`references/log_parsing_guide.md`
- 報告模板：`references/report_template.md`
- 單一模型範例：`examples/single_model_example.md`
- 批次測試範例：`examples/multi_model_example.md`
