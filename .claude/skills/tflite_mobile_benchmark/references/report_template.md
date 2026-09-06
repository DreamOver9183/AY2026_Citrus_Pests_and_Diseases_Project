# 標準化報告模板 (Report Template)

> 本文件提供 TFLite Benchmark 測試報告的標準化輸出格式模板。
> Agent 生成報告時必須完全遵循此模板結構。

---

## 完整報告 Markdown 模板

```markdown
# Mobile Benchmark 效能測試報告

## 1. 測試環境 (Environment)
* **測試設備代號 (Device ID)**: {{DEVICE_SERIAL}}
* **產品型號 (Product/Model)**: {{PRODUCT_MODEL}}
* **設備名稱 (Device)**: {{DEVICE_NAME}}
* **作業系統**: Android {{ANDROID_VERSION}}
* **測試工具**: TFLite Android AArch64 Benchmark Model (`android_aarch64_benchmark_model.apk`)
* **測試日期**: {{TEST_DATE}}

## 2. 測試主題 (Test Topic)
* **主題**: {{TEST_TOPIC}}
* **測試模型清單**:
{{#each models}}
  * `{{this}}.tflite`
{{/each}}

## 3. 測試參數 (Test Parameters)
* **硬體加速**: {{ACCELERATION_DESC}}
* **核心數量 (num_threads)**: {{NUM_THREADS}}
* **推論次數 (num_runs)**: {{NUM_RUNS}} 輪
* **指令配置**: `--num_threads={{NUM_THREADS}} --num_runs={{NUM_RUNS}} --use_gpu={{USE_GPU}} --use_nnapi={{USE_NNAPI}}`

---

## 4. 輸出概要 (Output Summary)

*(下表彙整了 {{MODEL_COUNT}} 個模型完成 {{NUM_RUNS}} 輪推論後的核心指標數據)*

| 模型名稱 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 (%) | Init 記憶體 (MB) | Overall 記憶體 (MB) | 運算開銷差值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{{#each results}}
| `{{name}}` | {{first_ms}} | {{min_ms}} | {{max_ms}} | {{avg_ms}} | {{fps}} | {{std_ms}} | {{delegate_ratio}}% | {{init_mem}} | {{overall_mem}} | {{mem_overhead}} |
{{/each}}

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

{{#each results}}
<details>
<summary><b>模型：{{name}}.tflite</b></summary>

```log
{{raw_log}}
```

</details>
{{/each}}
```

---

## 模板變數說明

| 變數 | 取得方式 | 範例值 |
|---|---|---|
| `{{DEVICE_SERIAL}}` | `adb get-serialno` | `ebe3968d` |
| `{{PRODUCT_MODEL}}` | `adb shell getprop ro.product.model` | `CPH2641` |
| `{{DEVICE_NAME}}` | `adb devices -l` 中的 `device:` | `OP5B16L1` |
| `{{ANDROID_VERSION}}` | `adb shell getprop ro.build.version.release` | `14` |
| `{{TEST_DATE}}` | 當前日期 | `2026-07-21` |
| `{{TEST_TOPIC}}` | 使用者描述或模型名稱組合 | `多模型 TFLite 推論效能比較評測` |
| `{{NUM_THREADS}}` | 測試參數 | `4` |
| `{{NUM_RUNS}}` | 測試參數 | `25` |
| `{{USE_GPU}}` | 測試參數 | `false` |
| `{{USE_NNAPI}}` | 測試參數 | `false` |
| `{{ACCELERATION_DESC}}` | 根據 gpu/nnapi 參數生成 | `CPU 運算 (未使用 GPU 與 NNAPI)` |

---

## 數值精度規範

| 指標 | 小數位數 | 範例 |
|---|---|---|
| 首輪推論 (ms) | 2 位 | `187.53` |
| 最快 (ms) | 2 位 | `140.07` |
| 最慢 (ms) | 2 位 | `195.12` |
| 平均推論 (ms) | 2 位 | `145.53` |
| 預估 FPS | 2 位 | `6.87` |
| 標準差 (ms) | 2 位 | `13.58` |
| 節點替代率 (%) | 2 位 | `98.36` |
| 記憶體 (MB) | 2 位 | `54.29` |
| 運算開銷差值 (MB) | 2 位 | `19.12` |

---

## 加速模式描述對照表

| 參數組合 | `{{ACCELERATION_DESC}}` |
|---|---|
| `use_gpu=false, use_nnapi=false` | `CPU 運算 (未使用 GPU 與 NNAPI)` |
| `use_gpu=true, use_nnapi=false` | `GPU 加速 (TFLite GPU Delegate)` |
| `use_gpu=false, use_nnapi=true` | `NNAPI 加速 (Neural Networks API)` |
| `use_gpu=true, use_nnapi=true` | `GPU + NNAPI 加速` |

---

## 報告命名規範

| 情境 | 檔案名稱格式 |
|---|---|
| 一般新報告 | `benchmark_report_YYYY-MM-DD.md` |
| 同日多次報告 | `benchmark_report_YYYY-MM-DD_v2.md` |
| 特定主題報告 | `benchmark_report_<topic>_YYYY-MM-DD.md` |
| 追加至現有報告 | `benchmark_report.md`（需使用者確認）|
