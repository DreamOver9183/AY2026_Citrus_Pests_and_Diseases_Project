# 單一模型測試執行範例

> 本範例展示完整的單一模型 TFLite Benchmark 執行流程，
> 以 `yolo26n_fp16.tflite` 為範例，涵蓋全部 5 個 Phase 的實際操作。

---

## 情境說明

**使用者請求**：「請幫我跑 yolo26n_fp16.tflite 的效能測試」

**觸發條件**：單一模型，使用標準測試參數

---

## Phase 1 — 前置驗證

### 1.1 確認設備連線

```powershell
# Agent 執行
.\Benchmark\platform-tools\adb.exe devices -l
```

**實際輸出：**
```
List of devices attached
ebe3968d               device product:OP5B16L1 model:CPH2641 device:OP5B16L1 transport_id:1
```

**Agent 判斷：** ✅ 包含 `device`，設備 `ebe3968d` 正常連線，繼續執行。

---

### 1.2 確認 APK 存在

```powershell
Test-Path ".\Benchmark\Model\android_aarch64_benchmark_model.apk"
```

**實際輸出：**
```
True
```

**Agent 判斷：** ✅ APK 存在，繼續執行。

---

### 1.3 確認模型檔案存在

```powershell
Get-ChildItem ".\Benchmark\Model\*.tflite" | Select-Object Name
```

**實際輸出（節錄）：**
```
Name
----
yolo26n_fp16.tflite
yolo26n_p2_fp16.tflite
...
```

**Agent 判斷：** ✅ `yolo26n_fp16.tflite` 存在，繼續執行。

---

## Phase 2 — 安裝與部署

### 2.1 安裝 APK

```powershell
.\Benchmark\platform-tools\adb.exe install -r -d -g ".\Benchmark\Model\android_aarch64_benchmark_model.apk"
```

**實際輸出：**
```
Performing Streamed Install
Success
```

**Agent 判斷：** ✅ 安裝成功（包含 `Success`），繼續執行。

---

### 2.2 推送模型

```powershell
.\Benchmark\platform-tools\adb.exe push ".\Benchmark\Model\yolo26n_fp16.tflite" /data/local/tmp/
```

**實際輸出：**
```
.\Benchmark\Model\yolo26n_fp16.tflite: 1 file pushed, 0 skipped. 42.7 MB/s (5058721 bytes in 0.113s)
```

**Agent 判斷：** ✅ 傳輸成功，繼續執行。

---

## Phase 3 — 測試執行

### 3.1 清除舊日誌

```powershell
.\Benchmark\platform-tools\adb.exe logcat -c
```

**Agent 動作：** 執行後立即繼續（無輸出為正常）。

---

### 3.2 啟動 Benchmark Activity

```powershell
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/yolo26n_fp16.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```

**實際輸出：**
```
Stopping: org.tensorflow.lite.benchmark
Starting: Intent { cmp=org.tensorflow.lite.benchmark/.BenchmarkModelActivity }
```

**Agent 判斷：** ✅ Activity 成功啟動。

---

### 3.3 等待測試完成

**模型大小評估：** `yolo26n_fp16.tflite` = 5.06 MB（小型模型）
**等待時間決策：** 但 yolo 模型每次推論約 250ms，25 輪 ≈ 6.5 秒；加上初始化與 Warmup ≈ 20 秒

```powershell
Start-Sleep -Seconds 20
```

---

## Phase 4 — Log 擷取與解析

### 4.1 讀取 Log

```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s tflite
```

**實際輸出：**
```log
07-14 13:23:08.133 25289 25289 I tflite  : Log parameter values verbosely: [0]
07-14 13:23:08.133 25289 25289 I tflite  : Min num runs: [25]
07-14 13:23:08.133 25289 25289 I tflite  : Num threads: [4]
07-14 13:23:08.133 25289 25289 I tflite  : Graph: [/data/local/tmp/yolo26n_fp16.tflite]
07-14 13:23:08.133 25289 25289 I tflite  : Signature to run: []
07-14 13:23:08.134 25289 25289 I tflite  : #threads used for CPU inference: [4]
07-14 13:23:08.134 25289 25289 I tflite  : Use gpu: [0]
07-14 13:23:08.134 25289 25289 I tflite  : Use NNAPI: [0]
07-14 13:23:08.135 25289 25289 I tflite  : Loaded model /data/local/tmp/yolo26n_fp16.tflite
07-14 13:23:08.136 25289 25289 I tflite  : Initialized TensorFlow Lite runtime.
07-14 13:23:08.145 25289 25289 I tflite  : Created TensorFlow Lite XNNPACK delegate for CPU.
07-14 13:23:08.148 25289 25289 I tflite  : Replacing 642 out of 685 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 19 partitions for subgraph 0.
07-14 13:23:08.229 25289 25289 I tflite  : The input model file size (MB): 5.05872
07-14 13:23:08.229 25289 25289 I tflite  : Initialized session in 95.112ms.
07-14 13:23:08.257 25289 25289 I tflite  : Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
07-14 13:23:08.980 25289 25289 I tflite  : count=3 first=237814 curr=236852 min=236852 max=242195 avg=238954 std=2325 p5=236852 median=237814 p95=242195
07-14 13:23:08.980 25289 25289 I tflite  : Running benchmark for at least 25 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
07-14 13:23:15.443 25289 25289 I tflite  : count=25 first=243180 curr=227146 min=220934 max=326273 avg=257131 std=27081 p5=226088 median=252767 p95=303407
07-14 13:23:15.443 25289 25289 I tflite  : Inference timings in us: Init: 95112, First inference: 237814, Warmup (avg): 238954, Inference (avg): 257131
07-14 13:23:15.443 25289 25289 I tflite  : Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
07-14 13:23:15.443 25289 25289 I tflite  : Memory footprint delta from the start of the tool (MB): init=65.4141 overall=90.25
```

---

### 4.2 數據解析過程

**Step 1：提取 Delegate 資訊**
```
行：Replacing 642 out of 685 node(s) with delegate
→ delegate_n = 642, total_n = 685
→ 節點替代率 = 642/685 × 100 = 93.72%
```

**Step 2：提取 Inference timings**
```
行：Inference timings in us: Init: 95112, First inference: 237814, Warmup (avg): 238954, Inference (avg): 257131
→ init_us = 95112
→ first_us = 237814
→ warmup_us = 238954
→ inference_us = 257131
```

**Step 3：提取正式推論統計行**
```
行：count=25 first=243180 curr=227146 min=220934 max=326273 avg=257131 std=27081
→ run_count = 25 ✅（等於 num_runs，確認這是正式推論那行）
→ min_us = 220934
→ max_us = 326273
→ std_us = 27081
```

**Step 4：提取記憶體**
```
行：Memory footprint delta from the start of the tool (MB): init=65.4141 overall=90.25
→ init_mem_mb = 65.4141
→ overall_mem_mb = 90.25
```

**Step 5：計算衍生指標**
```
首輪推論 (ms) = 237814 / 1000 = 237.81 ms
最快 (ms)     = 220934 / 1000 = 220.93 ms
最慢 (ms)     = 326273 / 1000 = 326.27 ms
平均推論 (ms) = 257131 / 1000 = 257.13 ms
預估 FPS      = 1000 / 257.13 = 3.89 FPS
標準差 (ms)   = 27081 / 1000 = 27.08 ms
節點替代率    = 93.72%
Init 記憶體   = 65.41 MB
Overall 記憶體= 90.25 MB
運算開銷差值  = 90.25 - 65.41 = 24.84 MB
```

---

## Phase 5 — 報告生成

**Agent 生成的報告輸出：**

```markdown
# Mobile Benchmark 效能測試報告

## 1. 測試環境 (Environment)
* **測試設備代號 (Device ID)**: ebe3968d
* **產品型號 (Product/Model)**: CPH2641
* **設備名稱 (Device)**: OP5B16L1
* **作業系統**: Android
* **測試工具**: TFLite Android AArch64 Benchmark Model (`android_aarch64_benchmark_model.apk`)
* **測試日期**: 2026-07-21

## 2. 測試主題 (Test Topic)
* **主題**: yolo26n_fp16 單模型效能評測
* **測試模型清單**:
  * `yolo26n_fp16.tflite`

## 3. 測試參數 (Test Parameters)
* **硬體加速**: CPU 運算 (未使用 GPU 與 NNAPI)
* **核心數量 (num_threads)**: 4
* **推論次數 (num_runs)**: 25 輪
* **指令配置**: `--num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false`

---

## 4. 輸出概要 (Output Summary)

| 模型名稱 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 (%) | Init 記憶體 (MB) | Overall 記憶體 (MB) | 運算開銷差值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo26n_fp16.tflite` | 237.81 | 220.93 | 326.27 | 257.13 | 3.89 | 27.08 | 93.72% | 65.41 | 90.25 | 24.84 |

## 5. 完整輸出日誌 (Full Output Log)

<details>
<summary><b>模型：yolo26n_fp16.tflite</b></summary>

[原始 log 內容]

</details>
```

---

## 驗證：與已知正確數據比對

| 指標 | 範例計算結果 | benchmark_report.md 記錄值 | 符合？ |
|---|---|---|---|
| 首輪推論 (ms) | 237.81 | 237.81 | ✅ |
| 最快 (ms) | 220.93 | 220.93 | ✅ |
| 最慢 (ms) | 326.27 | 326.27 | ✅ |
| 平均推論 (ms) | 257.13 | 257.13 | ✅ |
| 預估 FPS | 3.89 | 3.89 | ✅ |
| 標準差 (ms) | 27.08 | 27.08 | ✅ |
| 節點替代率 | 93.72% | 93.72% | ✅ |
| Init 記憶體 (MB) | 65.41 | 65.41 | ✅ |
| Overall 記憶體 (MB) | 90.25 | 90.25 | ✅ |
| 運算開銷差值 (MB) | 24.84 | 24.84 | ✅ |

**結論：** 所有計算結果與歷史報告完全吻合，解析公式驗證正確。 ✅
