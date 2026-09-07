# Mobile Benchmark 效能測試報告

## 1. 測試環境 (Environment)

* **測試設備代號 (Device ID)**: `ebe3968d`
* **產品型號 (Product/Model)**: OPPO CPH2641
* **裝置代號 (Device)**: OP5B16L1
* **SoC**: Qualcomm **SM6115**（Snapdragon 662，2020 年入門級，11 nm）
* **GPU**: Adreno 610
* **作業系統**: Android 14（SDK 34）
* **ABI**: `arm64-v8a`
* **記憶體**: 3,676,928 kB ≈ 3.5 GiB
* **測試工具**: TFLite Android AArch64 Benchmark Model (`android_aarch64_benchmark_model.apk`)
* **測試日期**: 2026-09-07

---

## 2. 測試主題 (Test Topic)

量測 **YOLO26n-P2**（v11.5 `last.pt`，9 類柑橘病蟲害）在測試平台 1 上的推論延遲，
建立部署基準線。

**部署目標：30 FPS ±5，即每張 28.6–40 ms。**

受測模型兩個，皆由 `tools/export_tflite.py` 於 Docker 內轉出並通過與 PyTorch 的比對驗收：

| 檔案 | 大小 | 量化 | `end2end` |
| --- | ---: | --- | --- |
| `last__fp32.tflite` | 9.85 MB | 無 | 保留（NMS 在圖裡） |
| `last__w8a32.tflite` | 3.10 MB | 動態 INT8（權重 INT8／啟動值 FP32） | 保留 |

---

## 3. 測試參數 (Test Parameters)

| 參數 | 值 |
| --- | --- |
| `--num_runs` | 25 |
| `--num_threads` | 4 |
| 設定 A：CPU-only | `--use_gpu=false --use_nnapi=false` |
| 設定 B：GPU delegate | `--use_gpu=true --use_nnapi=false` |
| 設定 C：NNAPI delegate | `--use_gpu=false --use_nnapi=true` |

每組測試前均執行 `adb logcat -c` 清空日誌。

---

## 4. 輸出概要 (Output Summary)

| 模型 | 設定 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 | Init 記憶體 (MB) | Overall 記憶體 (MB) | 達標 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :-: |
| `last__fp32` | CPU-only 4T | 319.8 | 267.7 | 294.0 | **272.1** | **3.68** | 7.1 | XNNPACK 510/559 = 91.2% | 95.41 | 122.73 | ✗ |
| `last__w8a32` | CPU-only 4T | 378.9 | 265.0 | 383.5 | **277.5** | **3.60** | 28.3 | XNNPACK 500/559 = 89.4% | 66.49 | 104.73 | ✗ |
| `last__fp32` | GPU delegate | 333.0 | 299.1 | 324.7 | 306.9 | 3.26 | 6.0 | **GPU 54/559 = 9.7%** ＋ XNNPACK 456/506 | 119.50 | 176.78 | ✗ |
| `last__w8a32` | GPU delegate | 303.2 | 268.2 | 336.6 | 280.5 | 3.56 | 16.3 | **GPU 54/559 = 9.7%** ＋ XNNPACK 448/506 | 135.22 | 165.21 | ✗ |
| `last__fp32` | NNAPI（實際未生效） | 362.2 | 269.2 | 393.5 | 281.8 | 3.55 | 27.8 | XNNPACK 510/559 = 91.2% | 95.46 | 122.79 | ✗ |
| `last__w8a32` | NNAPI（實際未生效） | 367.0 | 265.1 | 333.7 | 273.5 | 3.66 | 14.8 | XNNPACK 500/559 = 89.4% | 66.96 | 103.79 | ✗ |

**Init 時間**：CPU-only 115–128 ms、GPU delegate **2,051–2,228 ms**（約 17 倍）。

**最快的組合：`last__fp32` @ CPU-only 4T，272.1 ms = 3.68 FPS。
距離達標上限（40 ms）還差 6.8 倍。六組全部未達標。**

---

## 5. 三個必須讀懂的細節

### 5.1 NNAPI 那兩列不是 NNAPI 的成績

```
NNAPI accelerators available: [nnapi-reference]
Created TensorFlow Lite delegate for NNAPI.
Though NNAPI delegate is explicitly applied, the model graph will not be executed by the delegate.
```

`nnapi-reference` 是 NNAPI 的 **CPU 參考實作**，代表這台裝置**沒有對外開放任何
NPU/DSP 驅動**。整張圖最後仍是由 XNNPACK 在 CPU 上跑（節點替代率與 CPU-only 那兩列
完全相同：510/559 與 500/559）。

**所以那兩列應該讀成「CPU-only ＋ 多餘的 NNAPI 初始化開銷」，不是 NNAPI 的效能。**
它們與 CPU-only 的差異（281.8 vs 272.1、273.5 vs 277.5）落在量測抖動內。

### 5.2 GPU delegate 幫倒忙，而且原因很具體

GPU 只吃下 **54/559 個節點（9.7%）**，圖被切成 3 段，其餘 506 個節點退回 CPU。
來回搬運的成本高於 GPU 省下的時間 —— fp32 因此**比純 CPU 慢 12.8%**（306.9 vs 272.1）。

不支援的算子清單：

```
GATHER_ND      : Operation is not supported.
FLOOR_MOD      : OP is supported, but tensor type/shape isn't compatible.
CAST           : Input type INT32→INT64 / INT64→FLOAT32 / INT64→INT32
SELECT_V2      : Cond must be float or bool type ...
LESS / NOT_EQUAL : Not supported logical op case.
ADD            : Max version supported: 2. Requested version 4.
```

**這些正是 `end2end=True`（NMS-free）topk 頭的實作**：`GATHER_ND` 是索引取值、
`FLOOR_MOD` 與 `CAST INT64` 是索引運算、`SELECT_V2`/`LESS`/`NOT_EQUAL` 是條件篩選。

換句話說，**是 NMS-free 的頭擋住了 GPU 委派**。這是量出來的，不是推論的 ——
計畫階段就把它列為必須實測的風險項，結果它確實發生了。

另外 GPU delegate 的 **init 要 2.05–2.23 秒**（CPU-only 只要 0.12 秒），
對「開 App 就要能用」的體驗也是負擔。

### 5.3 `w8a32` 在 CPU 上不會比較快

277.5 vs 272.1 ms —— 動態量化**沒有加速效果**。

原因是 `w8a32` 的啟動值仍是 FP32，XNNPACK 實際做的仍然是浮點運算，
INT8 只省在權重的儲存與記憶體頻寬上。**它的價值是體積（9.85 → 3.10 MB，3.18×）
與記憶體佔用（122.7 → 104.7 MB），不是速度。**

---

## 6. 結論

| 問題 | 答案 |
| --- | --- |
| 現行模型能不能在平台 1 達到 30 FPS？ | **不能，差 6.8 倍** |
| 換 delegate 有沒有用？ | **沒有。** GPU 更慢，NNAPI 根本沒有可用的加速器 |
| 動態量化有沒有用？ | **對速度沒有**，對體積有（3.18×） |
| 瓶頸在哪？ | 兩層：① SM6115 是 2020 年入門級 SoC；② P2 頭在 640×640 的 stride-4 特徵圖是 160×160，成本極高；③ end2end 頭讓 GPU 委派失效 |

**這個結果直接否決了 v12s（`yolo26s-p2`，3.49× GFLOPs）在本平台的部署可行性**
—— 依此推估約 950 ms/張。v12s 仍值得跑，但只能當作「精度上限探索」，
不能當作候選部署模型。

---

## 7. 完整輸出日誌 (Full Output Log)

<details>
<summary>展開六組測試的 tflite logcat 摘要</summary>

以下逐行取自 `adb logcat -d -s tflite`，未經改寫。

```
── last__fp32 / CPU-only 4T ────────────────────────────────────────
Replacing 510 out of 559 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 11 partitions for subgraph 0.
The input model file size (MB): 10.327
Initialized session in 115.396ms.
count=25 first=293263 curr=268255 min=267674 max=293983 avg=272100 std=7102 p5=267779 median=268922 p95=293263
Inference timings in us: Init: 115396, First inference: 319801, Warmup (avg): 307158, Inference (avg): 272100
Memory footprint delta from the start of the tool (MB): init=95.4102 overall=122.727

── last__w8a32 / CPU-only 4T ───────────────────────────────────────
Replacing 500 out of 559 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 19 partitions for subgraph 0.
The input model file size (MB): 3.24988
Initialized session in 119.609ms.
count=25 first=383497 curr=266917 min=264993 max=383497 avg=277456 std=28280 p5=265634 median=267094 p95=358595
Inference timings in us: Init: 119609, First inference: 378862, Warmup (avg): 362766, Inference (avg): 277456
Memory footprint delta from the start of the tool (MB): init=66.4922 overall=104.73

── last__fp32 / GPU delegate ───────────────────────────────────────
Following operations are not supported by GPU delegate:
ADD: Max version supported: 2. Requested version 4.
CAST: Not supported Cast case. Input type: INT32 and output type: INT64
CAST: Not supported Cast case. Input type: INT64 and output type: FLOAT32
CAST: Not supported Cast case. Input type: INT64 and output type: INT32
CONCATENATION: OP is supported, but tensor type/shape isn't compatible.
DIV: OP is supported, but tensor type/shape isn't compatible.
FLOOR_MOD: OP is supported, but tensor type/shape isn't compatible.
GATHER_ND: Operation is not supported.
LESS: Not supported logical op case.
NOT_EQUAL: Not supported logical op case.
RESHAPE: OP is supported, but tensor type/shape isn't compatible.
SELECT: OP is supported, but tensor type/shape isn't compatible.
SELECT_V2: Cond must be float or bool type, if, else tensors must be either be same the shape as output or constant, scalar.
SIGN: OP is supported, but tensor type/shape isn't compatible.
Replacing 54 out of 559 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 3 partitions for subgraph 0.
Replacing 456 out of 506 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 13 partitions for subgraph 0.
Initialized session in 2227.67ms.
count=25 first=313325 curr=308050 min=299074 max=324703 avg=306869 std=6003 p5=299213 median=305582 p95=313707
Inference timings in us: Init: 2227672, First inference: 332961, Warmup (avg): 318238, Inference (avg): 306869
Memory footprint delta from the start of the tool (MB): init=119.5 overall=176.777

── last__w8a32 / GPU delegate ──────────────────────────────────────
Replacing 54 out of 559 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 3 partitions for subgraph 0.
Replacing 448 out of 506 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 21 partitions for subgraph 0.
Initialized session in 2050.93ms.
count=25 first=278274 curr=269119 min=268190 max=336545 avg=280520 std=16290 p5=268501 median=275657 p95=309019
Inference timings in us: Init: 2050927, First inference: 303170, Warmup (avg): 287809, Inference (avg): 280520
Memory footprint delta from the start of the tool (MB): init=135.215 overall=165.211

── last__fp32 / NNAPI（未生效）─────────────────────────────────────
NNAPI accelerators available: [nnapi-reference]
Created TensorFlow Lite delegate for NNAPI.
Though NNAPI delegate is explicitly applied, the model graph will not be executed by the delegate.
Replacing 510 out of 559 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 11 partitions for subgraph 0.
Initialized session in 123.396ms.
count=25 first=393529 curr=269246 min=269246 max=393529 avg=281831 std=27837 p5=269417 median=270283 p95=340777
Inference timings in us: Init: 123396, First inference: 362227, Warmup (avg): 351518, Inference (avg): 281831
Memory footprint delta from the start of the tool (MB): init=95.4609 overall=122.793

── last__w8a32 / NNAPI（未生效）────────────────────────────────────
Though NNAPI delegate is explicitly applied, the model graph will not be executed by the delegate.
Replacing 500 out of 559 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 19 partitions for subgraph 0.
Initialized session in 127.987ms.
count=25 first=333694 curr=268137 min=265129 max=333694 avg=273503 std=14793 p5=265310 median=267648 p95=302997
Inference timings in us: Init: 127987, First inference: 367038, Warmup (avg): 360349, Inference (avg): 273503
Memory footprint delta from the start of the tool (MB): init=66.957 overall=103.793
```

</details>

---

## 8. 量測方式的兩個注意事項

1. **六組是連續跑的，沒有交錯重測。** 機身溫度會隨測試累積上升，後面的組別可能
   略微偏慢。但本次的差距是 6.8 倍等級，熱效應（通常幾個百分點）不影響結論。
2. **從 Git Bash 執行 adb 必須設 `MSYS_NO_PATHCONV=1`**，否則 `/data/local/tmp/`
   會被改寫成 Windows 路徑。`adb push` 會回報 success 但檔案不在裝置上 ——
   本次就踩到，靠 `adb shell ls` 才發現。細節見 SKILL.md §D。
