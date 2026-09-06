# ADB 指令速查表 (ADB Commands Reference)

> 本文件為 TFLite Mobile Benchmark Skill 的 ADB 指令完整參考。
> 所有指令均以 `.\Benchmark\platform-tools\adb.exe` 為前綴（在工作根目錄執行時）。

---

## 1. 設備連線與狀態

### 列出所有已連線設備
```powershell
.\Benchmark\platform-tools\adb.exe devices
```

### 列出詳細設備資訊
```powershell
.\Benchmark\platform-tools\adb.exe devices -l
```

**典型輸出（正常）：**
```
List of devices attached
ebe3968d               device product:OP5B16L1 model:CPH2641 device:OP5B16L1 transport_id:1
```

**輸出狀態說明：**
| 狀態 | 說明 | 處理方式 |
|---|---|---|
| `device` | 正常連線 | 可繼續 |
| `unauthorized` | 等待使用者授權 | 在手機上點選「允許」|
| `offline` | 設備離線 | 重新插拔 USB |
| 空白 | 未偵測到設備 | 確認 USB 連線與驅動程式 |

### 取得特定設備資訊
```powershell
# 獲取 ADB 序號
.\Benchmark\platform-tools\adb.exe get-serialno

# 獲取設備型號
.\Benchmark\platform-tools\adb.exe shell getprop ro.product.model

# 獲取 Android 版本
.\Benchmark\platform-tools\adb.exe shell getprop ro.build.version.release

# 獲取 CPU 架構
.\Benchmark\platform-tools\adb.exe shell getprop ro.product.cpu.abi
```

---

## 2. APK 管理

### 安裝 APK
```powershell
# 標準安裝（-r 覆蓋, -d 允許降版, -g 授予所有權限）
.\Benchmark\platform-tools\adb.exe install -r -d -g ".\Benchmark\platform-tools\android_aarch64_benchmark_model.apk"
```

### 確認 APK 已安裝
```powershell
.\Benchmark\platform-tools\adb.exe shell pm list packages | Select-String "tensorflow"
```

**期望輸出：**
```
package:org.tensorflow.lite.benchmark
```

### 解除安裝 APK
```powershell
# ⚠️ 需要使用者確認才可執行
.\Benchmark\platform-tools\adb.exe uninstall org.tensorflow.lite.benchmark
```

---

## 3. 檔案傳輸

### 推送模型至手機
```powershell
# 推送單一模型
.\Benchmark\platform-tools\adb.exe push ".\Benchmark\Model\<model_name>.tflite" /data/local/tmp/

# 推送多個模型（範例）
@("model1.tflite", "model2.tflite") | ForEach-Object {
    .\Benchmark\platform-tools\adb.exe push ".\Benchmark\Model\$_" /data/local/tmp/
}
```

**期望輸出（成功）：**
```
.\Benchmark\Model\yolo26n_fp16.tflite: 1 file pushed, 0 skipped. 45.2 MB/s (5058721 bytes in 0.107s)
```

### 列出手機暫存目錄檔案
```powershell
.\Benchmark\platform-tools\adb.exe shell ls -la /data/local/tmp/*.tflite
```

### 從手機拉取檔案（如有需要）
```powershell
# ⚠️ 需要使用者確認才可執行
.\Benchmark\platform-tools\adb.exe pull /data/local/tmp/<filename> ".\<local_path>"
```

---

## 4. 測試執行

### 4.1 清除 Logcat 緩衝區（必須在測試前執行）
```powershell
.\Benchmark\platform-tools\adb.exe logcat -c
```

### 4.2 啟動 Benchmark Activity

**PowerShell 完整語法：**
```powershell
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/<MODEL>.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```

**CMD 完整語法：**
```cmd
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args "\"--graph=/data/local/tmp/<MODEL>.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\""
```

**啟動參數說明：**
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--graph` | 模型路徑（手機端絕對路徑）| 必填 |
| `--num_threads` | CPU 執行緒數量 | 4 |
| `--num_runs` | 正式推論次數 | 25 |
| `--use_gpu` | 是否使用 GPU delegate | false |
| `--use_nnapi` | 是否使用 NNAPI delegate | false |
| `--num_warmup_runs` | 預熱次數（未指定則自動）| 不設定 |

### 4.3 非標準參數組合（需使用者確認）

```powershell
# GPU 加速模式（需確認）
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/<MODEL>.tflite --num_threads=4 --num_runs=25 --use_gpu=true\"'

# NNAPI 加速模式（需確認）
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/<MODEL>.tflite --num_threads=4 --num_runs=25 --use_nnapi=true\"'

# 單核測試模式（需確認）
.\Benchmark\platform-tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/<MODEL>.tflite --num_threads=1 --num_runs=25 --use_gpu=false --use_nnapi=false\"'
```

---

## 5. Logcat 操作

### 讀取 TFLite 測試結果（主要指令）
```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s tflite
```

### 即時監控（測試進行中）
```powershell
# 即時串流 tflite log（Ctrl+C 結束）
.\Benchmark\platform-tools\adb.exe logcat -s tflite
```

### 儲存 Log 至本地檔案
```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s tflite > ".\<model_name>_log.txt"
```

### 查看 App 崩潰訊息（排錯用）
```powershell
.\Benchmark\platform-tools\adb.exe logcat -d -s AndroidRuntime
```

### 查看所有 App 相關日誌（排錯用）
```powershell
.\Benchmark\platform-tools\adb.exe logcat -d | Select-String "tensorflow"
```

---

## 6. 常用輔助指令

### 取得設備螢幕截圖
```powershell
.\Benchmark\platform-tools\adb.exe shell screencap /data/local/tmp/screen.png
.\Benchmark\platform-tools\adb.exe pull /data/local/tmp/screen.png ".\screen.png"
```

### 確認 Activity 是否已啟動
```powershell
.\Benchmark\platform-tools\adb.exe shell dumpsys activity activities | Select-String "BenchmarkModel"
```

### 查看手機剩餘儲存空間
```powershell
.\Benchmark\platform-tools\adb.exe shell df /data/local/tmp
```

---

## 7. 多設備環境（指定設備 ID）

若同時連接多台設備，需加上 `-s <serial>` 參數：

```powershell
# 指定設備
.\Benchmark\platform-tools\adb.exe -s ebe3968d devices
.\Benchmark\platform-tools\adb.exe -s ebe3968d install -r -d -g ".\Benchmark\platform-tools\android_aarch64_benchmark_model.apk"
.\Benchmark\platform-tools\adb.exe -s ebe3968d logcat -d -s tflite
```
