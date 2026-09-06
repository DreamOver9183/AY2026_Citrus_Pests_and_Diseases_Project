為了方便您在不同的 Windows 終端機環境下無縫切換，以下為您分別整理 **PowerShell** 與 **CMD（命令提示字元）** 的完整實作流程。

在此流程中，我們會在執行測試前**先清除手機舊日誌**，並在執行後利用 ADB 原生的 `-s tflite` 標籤過濾功能，將最完整的 TFLite 推論 Log 一併輸出。

---

### 方案 A：Windows PowerShell 執行流程

在 PowerShell 中，請在檔案（APK 與模型）所在的資料夾內執行以下指令：

#### 1. 連線、安裝與上傳
```powershell
# 檢查手機是否成功連線 (需顯示 device)
adb devices -l

# 安裝 APK (PowerShell 環境本地檔案建議加上 .\)
adb install -r -d -g .\android_aarch64_benchmark_model.apk

# 上傳模型至手機暫存目錄
adb push .\your_model.tflite /data/local/tmp/
```

#### 2. 執行效能測試指令（4核心 / 25輪推論）
為避免舊日誌干擾，建議在啟動前先清空日誌：
```powershell
# 測試前：清空手機舊日誌
adb logcat -c

# 執行測試（特別針對 PowerShell 轉義引號，限制 4 核心、進行 25 輪推論，僅用 CPU）
adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/your_model.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```
*(💡 註：指令送出後手機畫面會短暫閃爍一下，代表測試正在背景運行，請靜待約 10 秒讓其完成 25 輪推論。)*

#### 3. 顯示完整推論 Log 輸出
使用 ADB 原生的標籤過濾器，完整輸出該次測試的效能報告：
```powershell
# -d 表示印出目前日誌後即結束，-s tflite 表示僅篩選標籤為 tflite 的所有資訊
adb logcat -d -s tflite
```

---

### 方案 B：Windows CMD (命令提示字元) 執行流程

在 CMD 視窗中，請在檔案所在的資料夾內執行以下指令：

#### 1. 連線、安裝與上傳
```cmd
:: 檢查手機是否成功連線
adb devices -l

:: 安裝 APK
adb install -r -d -g android_aarch64_benchmark_model.apk

:: 上傳模型至手機暫存目錄
adb push your_model.tflite /data/local/tmp/
```

#### 2. 執行效能測試指令（4核心 / 25輪推論）
```cmd
:: 測試前：清空手機舊日誌
adb logcat -c

:: 執行測試（針對 CMD 的引號轉義，限制 4 核心、進行 25 輪推論，僅用 CPU）
adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args "\"--graph=/data/local/tmp/your_model.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\""
```

#### 3. 顯示完整推論 Log 輸出
```cmd
:: 顯示該次完整推論報告
adb logcat -d -s tflite
```

---

### 💡 輸出 Log 預期結果說明
不論在 PowerShell 或 CMD，最後一個指令執行的輸出都將包含如下的完整分析報告：
1. **Loaded model** 與 **Initialized session**：確認載入成功與初始化耗時。
2. **Replacing nodes with delegate**：顯示成功使用 XNNPACK 優化了多少個運算節點。
3. **Inference timings in us**：顯示 25 輪推論後的 `Init`（初始化時間）、`First inference`（首輪推論延遲）、`Warmup`（預熱平均）以及 `Inference (avg)`（平均推論延遲）。
4. **Memory footprint**：提供初始化與整體的記憶體增量數據。