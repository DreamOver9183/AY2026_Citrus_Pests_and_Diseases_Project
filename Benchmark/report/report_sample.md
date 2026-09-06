# Mobile Benchmark 效能測試報告（範本）

> **說明**：這是 Agent 自動生成報告的標準樣本。
> 執行測試後，Agent 將依此格式在 `report/` 目錄下生成 `benchmark_report_YYYY-MM-DD.md`。
> 本樣本的數據取自 2026-07-14 的實際測試，可作為格式參考。

---

## 1. 測試環境 (Environment)
* **測試設備代號 (Device ID)**: ebe3968d
* **產品型號 (Product/Model)**: CPH2641
* **設備名稱 (Device)**: OP5B16L1
* **作業系統**: Android
* **測試工具**: TFLite Android AArch64 Benchmark Model (`android_aarch64_benchmark_model.apk`)
* **測試日期**: 2026-07-14

## 2. 測試主題 (Test Topic)
* **主題**: 多模型 TFLite 推論效能比較評測
* **測試模型清單**:
  * `best_int8.tflite`
  * `ssd_mobilenetv3_large_fp16.tflite`
  * `ssd_mobilenetv3_large_fp32.tflite`
  * `ssd_mobilenetv3_small_fp16.tflite`
  * `ssd_mobilenetv3_small_fp32.tflite`
  * `yolo26l_fp16.tflite`
  * `yolo26n_fp16.tflite`
  * `yolo26n_p2_fp16.tflite`

## 3. 測試參數 (Test Parameters)
* **硬體加速**: CPU 運算 (未使用 GPU 與 NNAPI)
* **核心數量 (num_threads)**: 4
* **推論次數 (num_runs)**: 25 輪
* **指令配置**: `--num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false`

---

## 4. 輸出概要 (Output Summary)

*(下表彙整了 8 個模型完成 25 輪推論後的核心指標數據，其中包含延遲的極值、穩定度、節點替代率以及記憶體變化等)*

| 模型名稱 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 (%) | Init 記憶體 (MB) | Overall 記憶體 (MB) | 運算開銷差值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `best_int8.tflite` | 235.16 | 140.07 | 195.12 | 145.53 | 6.87 | 13.58 | 98.36% | 54.29 | 73.41 | 19.12 |
| `ssd_mobilenetv3_large_fp16.tflite` | 70.23 | 45.88 | 62.00 | 52.97 | 18.88 | 3.11 | 93.49% | 34.86 | 48.73 | 13.87 |
| `ssd_mobilenetv3_large_fp32.tflite` | 84.63 | 47.41 | 220.48 | 74.90 | 13.35 | 38.50 | 88.59% | 35.80 | 47.11 | 11.31 |
| `ssd_mobilenetv3_small_fp16.tflite` | 77.75 | 30.82 | 100.97 | 49.64 | 20.15 | 19.29 | 93.13% | 25.18 | 37.48 | 12.30 |
| `ssd_mobilenetv3_small_fp32.tflite` | 44.62 | 23.35 | 35.52 | 24.35 | 41.07 | 1.94 | 88.11% | 23.55 | 34.08 | 10.53 |
| `yolo26l_fp16.tflite` | 2562.79 | 2422.49 | 2510.68 | 2449.49 | 0.41 | 19.37 | 94.44% | 286.96 | 358.59 | 71.63 |
| `yolo26n_fp16.tflite` | 237.81 | 220.93 | 326.27 | 257.13 | 3.89 | 27.08 | 93.72% | 65.41 | 90.25 | 24.84 |
| `yolo26n_p2_fp16.tflite` | 321.65 | 279.65 | 310.76 | 284.02 | 3.52 | 7.78 | 94.74% | 81.23 | 115.05 | 33.82 |

> [!NOTE]
> **欄位指標說明：**
> * **首輪推論 (First Inference)**：模型載入後第一次進行推論的耗時，反映按下按鈕時是否會遇到「首幀卡頓」。
> * **預估 FPS (Estimated FPS)**：基於平均推論時間推算的每秒吞吐量，公式為 `1000 / 平均推論 (ms)`。
> * **標準差 (Std Dev)**：反映推論速度的波動程度；數值越低代表運行越穩定。
> * **節點替代率 (Delegate Ratio)**：被 XNNPACK 接管的運算節點比例。比例過低代表存在許多設備不支援的自定義算子。
> * **運算開銷差值**：`Overall 記憶體增量` 減去 `Init 記憶體增量`，顯示推論過程中因暫存特徵圖所額外佔用的記憶體峰值。

---

## 5. 完整輸出日誌 (Full Output Log)
*(在此收錄每個模型透過 `adb logcat -d -s tflite` 產生的原始 Log 資訊，確保數據具備可追溯性。)*

<details>
<summary><b>模型：best_int8.tflite</b></summary>

```log
07-14 13:14:52.097 22838 22838 I tflite  : Log parameter values verbosely: [0]
07-14 13:14:52.097 22838 22838 I tflite  : Min num runs: [25]
07-14 13:14:52.097 22838 22838 I tflite  : Num threads: [4]
07-14 13:14:52.097 22838 22838 I tflite  : Graph: [/data/local/tmp/best_int8.tflite]
07-14 13:14:52.097 22838 22838 I tflite  : #threads used for CPU inference: [4]
07-14 13:14:52.097 22838 22838 I tflite  : Use gpu: [0]
07-14 13:14:52.097 22838 22838 I tflite  : Use NNAPI: [0]
07-14 13:14:52.098 22838 22838 I tflite  : Loaded model /data/local/tmp/best_int8.tflite
07-14 13:14:52.099 22838 22838 I tflite  : Initialized TensorFlow Lite runtime.
07-14 13:14:52.113 22838 22838 I tflite  : Created TensorFlow Lite XNNPACK delegate for CPU.
07-14 13:14:52.115 22838 22838 I tflite  : Replacing 539 out of 548 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 19 partitions for subgraph 0.
07-14 13:14:52.190 22838 22838 I tflite  : The input model file size (MB): 3.06409
07-14 13:14:52.190 22838 22838 I tflite  : Initialized session in 92.318ms.
07-14 13:14:52.859 22838 22838 I tflite  : count=3 first=235158 curr=200386 min=199462 max=235158 avg=211669 std=16613 p5=199462 median=200386 p95=235158
07-14 13:14:56.520 22838 22838 I tflite  : count=25 first=187533 curr=140990 min=140071 max=195121 avg=145528 std=13583 p5=140545 median=141253 p95=187533
07-14 13:14:56.521 22838 22838 I tflite  : Inference timings in us: Init: 92318, First inference: 235158, Warmup (avg): 211669, Inference (avg): 145528
07-14 13:14:56.521 22838 22838 I tflite  : Memory footprint delta from the start of the tool (MB): init=54.2852 overall=73.4102
```

</details>

<details>
<summary><b>模型：ssd_mobilenetv3_large_fp16.tflite</b></summary>

```log
07-14 13:16:26.182 23002 23002 I tflite  : Num threads: [4]
07-14 13:16:26.182 23002 23002 I tflite  : Graph: [/data/local/tmp/ssd_mobilenetv3_large_fp16.tflite]
07-14 13:16:26.182 23002 23002 I tflite  : Use gpu: [0]
07-14 13:16:26.182 23002 23002 I tflite  : Use NNAPI: [0]
07-14 13:16:26.184 23002 23002 I tflite  : Loaded model /data/local/tmp/ssd_mobilenetv3_large_fp16.tflite
07-14 13:16:26.193 23002 23002 I tflite  : Created TensorFlow Lite XNNPACK delegate for CPU.
07-14 13:16:26.195 23002 23002 I tflite  : Replacing 431 out of 461 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 58 partitions for subgraph 0.
07-14 13:16:26.241 23002 23002 I tflite  : Initialized session in 58.312ms.
07-14 13:16:28.097 23002 23002 I tflite  : count=25 first=53752 curr=45884 min=45884 max=62002 avg=52970 std=3114 p5=46034 median=53524 p95=55365
07-14 13:16:28.097 23002 23002 I tflite  : Inference timings in us: Init: 58312, First inference: 70225, Warmup (avg): 57237.2, Inference (avg): 52970
07-14 13:16:28.097 23002 23002 I tflite  : Memory footprint delta from the start of the tool (MB): init=34.8633 overall=48.7305
```

</details>

<details>
<summary><b>其餘模型 Log（略）</b></summary>

完整 Log 請參見 `report/benchmark_report.md`。

</details>
