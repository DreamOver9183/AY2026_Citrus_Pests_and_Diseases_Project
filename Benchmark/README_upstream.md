# TFLite Mobile Benchmark

> 使用 ADB + TFLite Benchmark APK 對 Android 設備進行推論效能測試的標準化工作流程，內建 AI Agent 自動化支援。

---

## 📁 專案結構

```
TFLite Mobile Benchmark/
│
├── .agents/                          # AI Agent 行為規範與 Skill 定義
│   ├── AGENTS.md                     # 三層行為邊界規範（必讀）
│   └── skills/
│       └── tflite_mobile_benchmark/  # Benchmark Skill 標準流程
│           ├── SKILL.md              # 主要 Skill 定義（五階段流程）
│           ├── references/           # 參考文件
│           │   ├── adb_commands_reference.md
│           │   ├── log_parsing_guide.md
│           │   └── report_template.md
│           └── examples/             # 執行範例
│               ├── single_model_example.md
│               └── multi_model_example.md
│
├── tools/                            # ADB 工具套件（不可修改）
│   ├── adb.exe                       # Android Debug Bridge
│   ├── android_aarch64_benchmark_model.apk  # TFLite Benchmark APK
│   └── ...                           # 其他 platform-tools 工具
│
├── Model/                            # ⚠️ 放置您的 .tflite 模型於此（預設為空）
│   └── .gitkeep
│
├── report/                           # 測試報告輸出目錄
│   ├── report_sample.md              # 報告格式範本（不可修改）
│   └── benchmark_report_*.md         # Agent 生成的測試報告（已 gitignore）
│
├── README.md                         # 本文件
└── .gitignore
```

---

## 🚀 快速開始

### 前置需求

- **Windows** 系統（PowerShell 5.1+）
- **Android 手機**，已啟用「開發者選項」>「USB 偵錯」
- USB 連線線（資料傳輸模式）

### 步驟一：放置模型

將您的 `.tflite` 模型檔案放入 `Model/` 目錄中：

```
Model/
├── your_model_v1.tflite
├── your_model_v2.tflite
└── ...
```

> ⚠️ **注意**：`Model/` 目錄下的 `.tflite` 檔案已被 `.gitignore` 排除，不會上傳至 GitHub。

### 步驟二：連線設備

1. 將手機以 USB 連接至電腦
2. 手機上允許「USB 偵錯」授權對話框

確認連線：
```powershell
.\tools\adb.exe devices -l
# 應看到類似：ebe3968d  device product:... 的輸出
```

### 步驟三：執行測試

**方案 A：使用 AI Agent（推薦）**

對 Agent 說：
> 「請幫我跑 `your_model.tflite` 的效能測試」

Agent 將自動完成：安裝 APK → 推送模型 → 執行測試 → 解析 Log → 生成報告

**方案 B：手動執行**

```powershell
# 1. 安裝 APK
.\tools\adb.exe install -r -d -g ".\tools\android_aarch64_benchmark_model.apk"

# 2. 推送模型
.\tools\adb.exe push ".\Model\your_model.tflite" /data/local/tmp/

# 3. 清除舊日誌
.\tools\adb.exe logcat -c

# 4. 啟動測試（PowerShell）
.\tools\adb.exe shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph=/data/local/tmp/your_model.tflite --num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false\"'

# 5. 等待約 15-60 秒後讀取結果
Start-Sleep -Seconds 20
.\tools\adb.exe logcat -d -s tflite
```

---

## 📊 報告格式

測試完成後，Agent 會在 `report/` 目錄下生成 `benchmark_report_YYYY-MM-DD.md`，
包含以下資訊：

| 欄位 | 說明 |
|---|---|
| **首輪推論 (ms)** | 首幀延遲，反映使用者體感卡頓 |
| **平均推論 (ms)** | 穩定推論速度 |
| **預估 FPS** | 基於平均推論計算的吞吐量 |
| **標準差 (ms)** | 推論穩定性（越低越好）|
| **節點替代率 (%)** | XNNPACK 加速覆蓋比例 |
| **Init/Overall 記憶體 (MB)** | 模型佔用記憶體 |

格式範本請參考 [`report/report_sample.md`](report/report_sample.md)。

---

## ⚙️ 標準測試參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--num_threads` | `4` | CPU 執行緒數量 |
| `--num_runs` | `25` | 正式推論次數 |
| `--use_gpu` | `false` | GPU delegate |
| `--use_nnapi` | `false` | NNAPI delegate |

如需修改標準參數，請先告知 Agent，確認後方可執行。

---

## 🤖 AI Agent 邊界說明

本專案已配置 AI Agent 自動化規範（`.agents/AGENTS.md`），Agent 遵循三層邊界：

| 層級 | 操作 |
|---|---|
| ✅ **自主執行** | 設備確認、APK 安裝、模型推送、測試執行、報告生成 |
| ⚠️ **需確認** | 修改測試參數、刪除手機檔案、覆蓋已有報告 |
| ❌ **絕對禁止** | 遞迴刪除、存取個人資料、讀取 Model/ 以外的模型 |

---

## 🔒 GitHub 上傳限制與版控原則

`.gitignore` 針對安全性與隱私設定了以下排除項目：

- **`Model/*.tflite`** — 模型權重（私有資產、檔案較大，預設不上傳）
- **`report/benchmark_report*.md`** — 本地測試報告（預設不上傳，僅保留 `report_sample.md` 範本）

> 💡 **註：`tools/` 目錄（包含 `adb.exe` 與 Benchmark APK）已包含在 Git 版控中**，他人 `git clone` 本專案後即可**開箱即用**，無需手動下載或配置 ADB 環境。

---

## 📖 進階文件

| 文件 | 說明 |
|---|---|
| [`.agents/AGENTS.md`](.agents/AGENTS.md) | Agent 行為邊界規範 |
| [`.agents/skills/tflite_mobile_benchmark/SKILL.md`](.agents/skills/tflite_mobile_benchmark/SKILL.md) | Benchmark Skill 完整流程 |
| [`.agents/skills/tflite_mobile_benchmark/references/adb_commands_reference.md`](.agents/skills/tflite_mobile_benchmark/references/adb_commands_reference.md) | ADB 指令速查 |
| [`.agents/skills/tflite_mobile_benchmark/references/log_parsing_guide.md`](.agents/skills/tflite_mobile_benchmark/references/log_parsing_guide.md) | TFLite Log 解析規範 |
| [`report/report_sample.md`](report/report_sample.md) | 報告格式範本 |

---

## ⚠️ 已知限制

- 本工具僅支援 **AArch64 (ARM64)** 架構的 Android 設備
- 測試結果受設備溫度、背景 App 等環境因素影響，建議關閉不必要的後台程式
- `yolo26l_fp16.tflite` 等大型模型需等待較長時間（~3 分鐘）
