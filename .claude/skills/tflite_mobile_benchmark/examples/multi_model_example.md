# 批次多模型測試執行範例

> 本範例展示如何使用批次模式同時測試多個 TFLite 模型，
> 以工作目錄中 5 個 SSD/YOLO 模型為例，說明迭代流程與最終彙整報告生成。

---

## 情境說明

**使用者請求**：「請幫我測試所有 SSD MobileNetV3 的模型，比較 fp16 和 fp32 的差異」

**觸發條件**：多模型批次測試

**批次清單（共 4 個模型）：**
1. `ssd_mobilenetv3_large_fp16.tflite`
2. `ssd_mobilenetv3_large_fp32.tflite`
3. `ssd_mobilenetv3_small_fp16.tflite`
4. `ssd_mobilenetv3_small_fp32.tflite`

---

## Phase 1 — 前置驗證（僅執行一次）

```powershell
# 1.1 確認設備連線
.\Benchmark\platform-tools\adb.exe devices -l
# 輸出：ebe3968d  device product:OP5B16L1 model:CPH2641 ...  ✅

# 1.2 確認 APK 存在
Test-Path ".\Benchmark\Model\android_aarch64_benchmark_model.apk"
# 輸出：True  ✅

# 1.3 確認所有目標模型存在
$models = @(
    "ssd_mobilenetv3_large_fp16.tflite",
    "ssd_mobilenetv3_large_fp32.tflite",
    "ssd_mobilenetv3_small_fp16.tflite",
    "ssd_mobilenetv3_small_fp32.tflite"
)
$models | ForEach-Object {
    $exists = Test-Path ".\Benchmark\Model\$_"
    Write-Host "$_ : $exists"
}
```

**預期輸出：**
```
ssd_mobilenetv3_large_fp16.tflite : True
ssd_mobilenetv3_large_fp32.tflite : True
ssd_mobilenetv3_small_fp16.tflite : True
ssd_mobilenetv3_small_fp32.tflite : True
```

**Agent 判斷：** ✅ 所有 4 個模型均存在，繼續執行。

---

## Phase 2 — 安裝 APK（僅執行一次）

```powershell
.\Benchmark\platform-tools\adb.exe install -r -d -g ".\Benchmark\Model\android_aarch64_benchmark_model.apk"
# 輸出：Success  ✅
```

---

## 批次迭代流程

以下為每個模型的完整迭代流程（以 `ssd_mobilenetv3_large_fp16.tflite` 為例）：

---

### 迭代 1/4：ssd_mobilenetv3_large_fp16.tflite

**2.2 推送模型**
```powershell
.\Benchmark\platform-tools\adb.exe push ".\Benchmark\Model\ssd_mobilenetv3_large_fp16.tflite" /data/local/tmp/
# 輸出：1 file pushed, 0 skipped. 38.2 MB/s (4776696 bytes in 0.119s)  ✅
```

**3.1 清除日誌**
```powershell
.\Benchmark\platform-tools\adb.exe logcat -c
```

**3.2 啟動測試**
```powershell
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/ssd_mobilenetv3_large_fp16.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```

**3.3 等待（模型 4.78MB → 小型，等待 15 秒）**
```powershell
Start-Sleep -Seconds 15
```

**4.1 讀取 Log**
```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s tflite
```

**4.2 解析結果：**
```
delegate_n=431, total_n=461 → 節點替代率 = 93.49%
Inference timings: Init=58312, First=70225, Inference(avg)=52970 (us)
min=45884, max=62002, std=3114 (us)
Memory: init=34.8633, overall=48.7305 (MB)

計算結果：
  首輪推論: 70.23 ms
  最快: 45.88 ms
  最慢: 62.00 ms
  平均推論: 52.97 ms
  FPS: 18.88
  標準差: 3.11 ms
  節點替代率: 93.49%  [注：實際 log 顯示 431/461 = 93.49%]
  Init 記憶體: 34.86 MB
  Overall 記憶體: 48.73 MB
  運算開銷差值: 13.87 MB
```

> ⚠️ **注意**：此處為範例數值，實際測試以 logcat 實際輸出為準。

---

### 迭代 2/4：ssd_mobilenetv3_large_fp32.tflite

（重複以上流程，模型 9.44MB，等待 15 秒）

---

### 迭代 3/4：ssd_mobilenetv3_small_fp16.tflite

（重複以上流程，模型 3.37MB，等待 15 秒）

---

### 迭代 4/4：ssd_mobilenetv3_small_fp32.tflite

（重複以上流程，模型 6.63MB，等待 15 秒）

---

## Phase 5 — 彙整報告生成

**Agent 收集所有模型結果後生成報告：**

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
* **主題**: SSD MobileNetV3 fp16 vs fp32 效能比較評測
* **測試模型清單**:
  * `ssd_mobilenetv3_large_fp16.tflite`
  * `ssd_mobilenetv3_large_fp32.tflite`
  * `ssd_mobilenetv3_small_fp16.tflite`
  * `ssd_mobilenetv3_small_fp32.tflite`

## 3. 測試參數 (Test Parameters)
* **硬體加速**: CPU 運算 (未使用 GPU 與 NNAPI)
* **核心數量 (num_threads)**: 4
* **推論次數 (num_runs)**: 25 輪
* **指令配置**: `--num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false`

---

## 4. 輸出概要 (Output Summary)

| 模型名稱 | 首輪推論 (ms) | 最快 (ms) | 最慢 (ms) | 平均推論 (ms) | 預估 FPS | 標準差 (ms) | 節點替代率 (%) | Init 記憶體 (MB) | Overall 記憶體 (MB) | 運算開銷差值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ssd_mobilenetv3_large_fp16.tflite` | [值] | [值] | [值] | [值] | [值] | [值] | [值]% | [值] | [值] | [值] |
| `ssd_mobilenetv3_large_fp32.tflite` | [值] | [值] | [值] | [值] | [值] | [值] | [值]% | [值] | [值] | [值] |
| `ssd_mobilenetv3_small_fp16.tflite` | [值] | [值] | [值] | [值] | [值] | [值] | [值]% | [值] | [值] | [值] |
| `ssd_mobilenetv3_small_fp32.tflite` | [值] | [值] | [值] | [值] | [值] | [值] | [值]% | [值] | [值] | [值] |
```

---

## 批次模式特殊處理規則

### 模型清單確認流程

```
Agent 收到批次測試請求時：

1. 解析模型清單（使用者明確列出，或從 Model\ 目錄掃描）
2. 顯示清單並請使用者確認（若清單 > 5 個模型）
3. 預估總耗時：
   - 小型模型（<10MB）≈ 20 秒/模型
   - 中型模型（10-50MB）≈ 35 秒/模型
   - 大型模型（>50MB）≈ 120 秒/模型
4. 回報給使用者：「共 N 個模型，預計耗時 X 分鐘，確認開始？」
5. 收到確認後開始迭代
```

### 失敗處理策略（批次模式）

```
若某個模型測試失敗：

情況 A：Phase 1.3 失敗（模型不存在）
  → 標記該模型為「檔案不存在」，繼續下一個

情況 B：adb push 失敗
  → 標記該模型為「上傳失敗」，繼續下一個

情況 C：logcat 輸出為空
  → 重試一次（等待 +10 秒後再讀取）
  → 若仍空白，標記為「測試失敗（Log 為空）」，繼續下一個

情況 D：Log 解析失敗（缺少關鍵行）
  → 標記為「解析失敗」，保存原始 log，繼續下一個

所有模型完成後，在報告中標示失敗模型，
並在報告末尾附上失敗摘要。
```

### 批次報告中的失敗標記格式

```markdown
| `failed_model.tflite` | ⚠️ 解析失敗 | — | — | — | — | — | — | — | — | — |
```

---

## 完整批次自動化腳本（PowerShell 參考）

```powershell
# 批次測試腳本（Agent 可參考此邏輯生成指令序列）

$adb = ".\Benchmark\platform-tools\adb.exe"
$apk = ".\Benchmark\Model\android_aarch64_benchmark_model.apk"
$models = @(
    "ssd_mobilenetv3_large_fp16.tflite",
    "ssd_mobilenetv3_large_fp32.tflite",
    "ssd_mobilenetv3_small_fp16.tflite",
    "ssd_mobilenetv3_small_fp32.tflite"
)

# Phase 1：前置驗證
& $adb devices -l
& $adb install -r -d -g $apk

# 批次迭代
foreach ($model in $models) {
    Write-Host "=== 開始測試 $model ==="
    
    # 推送模型
    & $adb push ".\Benchmark\Model\$model" /data/local/tmp/
    
    # 清除日誌
    & $adb logcat -c
    
    # 啟動測試
    & $adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity `
        --es args "\"--graph=/data/local/tmp/$model --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\""
    
    # 等待（依模型大小調整）
    $fileSize = (Get-Item ".\Benchmark\Model\$model").Length / 1MB
    $wait = if ($fileSize -lt 10) { 15 } elseif ($fileSize -lt 50) { 30 } else { 90 }
    Start-Sleep -Seconds $wait
    
    # 讀取 Log
    & $adb logcat -d -s tflite
    
    Write-Host "=== $model 完成 ==="
}
```
