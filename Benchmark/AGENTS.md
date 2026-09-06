# Mobile Benchmark Agent 行為規範 (AGENTS.md)

> 本文件定義 Agent 在執行 TFLite Mobile Benchmark 相關任務時的行為邊界與約束規則。
> Agent 必須在每次觸發 `tflite_mobile_benchmark` Skill 前讀取並遵循本規範。

---

## 工作目錄規範

```
D:\Claude Code\AY2026 Citrus Pests and Diseases Project\   ← 工作根目錄（＝專案根目錄）
├── .claude\skills\tflite_mobile_benchmark\  ← 本 Skill 的正本（Claude Code 由此載入）
├── tools\                                   ← ⚠ 本專案自己的 19 支 Python 腳本，**不是** ADB 工具
└── Benchmark\                               ← benchmark 的全部內容都在這底下
    ├── AGENTS.md                            ← 本文件
    ├── Benchmark_process.md                 ← 手動執行流程（PowerShell / CMD）
    ├── platform-tools\                      ← ADB 工具包（原上游的 tools\，改名避免撞名）
    │   ├── adb.exe
    │   └── android_aarch64_benchmark_model.apk
    ├── Model\                               ← 使用者放置 .tflite 模型的目錄（預設為空）
    ├── export\                              ← .pt → .tflite 的 Docker 匯出管線
    └── report\                              ← 測試報告輸出目錄
        ├── report_sample.md                 ← 報告格式範本（唯讀參考）
        └── benchmark_report_YYYY-MM-DD.md   ← Agent 生成的測試報告
```

> ⚠️ **`tools\` 撞名警告**：本專案根目錄的 `tools\` 是專案自己的 Python 腳本目錄。
> ADB 與 APK 一律在 `Benchmark\platform-tools\`。上游文件寫的 `.\tools\adb.exe`
> 在本專案是**錯的路徑**，已全數改寫。

**路徑速查：**
- **ADB 工具路徑**: `.\Benchmark\platform-tools\adb.exe`
- **APK 路徑**: `.\Benchmark\platform-tools\android_aarch64_benchmark_model.apk`
- **模型存放目錄**: `.\Benchmark\Model\`（**唯一允許讀取模型的位置**）
- **報告輸出目錄**: `.\Benchmark\report\`
- **手機暫存路徑**: `/data/local/tmp/`（唯一允許的手機寫入位置）

---

## 三層行為邊界

### ✅ Level 1 — 自主執行（Agent 可直接執行，無需確認）

以下操作 Agent 可以自主完成，無需事先詢問使用者：

1. **環境探查**
   - 執行 `.\Benchmark\platform-tools\adb.exe devices -l` 確認設備連線狀態
   - 讀取 `Benchmark\Model\` 目錄內的 `.tflite` 模型檔案清單
   - 執行 `Test-Path` 確認本地檔案是否存在

2. **測試部署**（前提：Phase 1 前置驗證全數通過）
   - 執行 `adb install -r -d -g` 安裝 APK（APK 來源：`tools\android_aarch64_benchmark_model.apk`）
   - 執行 `adb push <model>.tflite /data/local/tmp/` 推送模型（模型來源：`Benchmark\Model\` 目錄）

3. **測試執行**（使用標準參數）
   - 執行 `adb logcat -c` 清除舊日誌
   - 使用**標準參數**啟動測試 Activity（`--num_threads=4 --num_runs=25 --use_gpu=false --use_nnapi=false`）
   - 執行 `adb logcat -d -s tflite` 讀取測試結果

4. **數據處理與報告**
   - 解析 TFLite logcat 輸出
   - 計算衍生指標（FPS、標準差、節點替代率、記憶體差值）
   - 在 `Benchmark\report\` 目錄下生成 `benchmark_report_YYYY-MM-DD.md` 報告

---

### ⚠️ Level 2 — 需確認執行（Agent 必須先提示使用者，等待明確許可）

以下操作涉及非標準配置或破壞性動作，**必須先獲得使用者明確確認**：

1. **測試參數變更**
   - 修改 `num_threads`（非 4 的數值）
   - 修改 `num_runs`（非 25 的數值）
   - 啟用 `use_gpu=true`
   - 啟用 `use_nnapi=true`

2. **設備操作**
   - 重新安裝 APK（覆蓋已存在的版本時需再次確認）
   - 刪除手機上 `/data/local/tmp/` 目錄內的特定 `.tflite` 檔案

3. **報告操作**
   - 覆蓋 `Benchmark\report\` 目錄下已存在的報告文件（必須先確認是否備份）
   - ⚠️ 禁止修改 `report\report_sample.md`（此為格式範本，應保持唯讀）

4. **批次測試**
   - 若批次清單超過 5 個模型，需確認預估耗時後方可執行

---

### ❌ Level 3 — 硬性禁止（任何情況下均不得執行）

以下操作被**絕對禁止**，違反時 Agent 必須立即停止並回報錯誤：

1. **模型來源限制**
   - **禁止讀取 `Benchmark\Model\` 目錄以外的 `.tflite` 模型檔案**（包含 `Benchmark\platform-tools\` 中可能殘留的 .tflite 檔案）
   - 禁止從外部下載或引入未經使用者手動放置於 `Benchmark\Model\` 的模型

2. **破壞性指令**
   - 禁止執行 `adb shell rm -rf` 或任何遞迴刪除
   - 禁止對 `/data/local/tmp/` 以外的手機路徑執行任何寫入操作
   - 禁止執行 `adb shell reboot` 或其他重啟指令

3. **未驗證執行**
   - 禁止在 `adb devices -l` 未回傳 `device` 狀態的情況下執行測試
   - 禁止在模型檔案不存在的情況下執行 `adb push`
   - 禁止在 `Benchmark\Model\` 目錄為空（無 `.tflite` 檔案）時啟動測試

4. **系統與安全**
   - 禁止存取手機 `/sdcard/DCIM/`、`/sdcard/Pictures/` 等個人資料目錄
   - 禁止修改 `Benchmark\platform-tools\` 目錄內的任何執行檔（adb.exe, APK 等）
   - 禁止下載或替換 APK（只能使用 `Benchmark\platform-tools\` 目錄內既有的 APK）

---

## 錯誤處理規範

| 錯誤情境 | Agent 應採取的動作 |
|---|---|
| `adb devices` 回傳空結果或 `offline` | 停止執行，提示使用者檢查 USB 連線與開發者選項 |
| `adb devices` 回傳 `unauthorized` | 停止執行，提示使用者在手機上允許 USB 偵錯授權 |
| `Benchmark\Model\` 目錄為空或無 `.tflite` 檔案 | 停止執行，提示使用者將 `.tflite` 模型檔案放入 `Benchmark\Model\` 目錄 |
| 目標模型不在 `Benchmark\Model\` 目錄中 | 停止執行，列出 `Benchmark\Model\` 中目前可用的模型清單 |
| APK 安裝失敗 | 回報錯誤訊息，不繼續推送模型 |
| logcat 輸出為空 | 等待 5 秒後重試一次；若仍空白則回報可能測試未完成 |
| logcat 缺少關鍵欄位 | 標記該模型結果為「解析失敗」並繼續下一個模型（批次模式）|

---

## 報告輸出規範

- **輸出位置**：一律寫入 `Benchmark\report\` 目錄（禁止寫入根目錄或其他位置）
- **命名格式**：`benchmark_report_YYYY-MM-DD.md`（新報告）
- **同日多次測試**：`benchmark_report_YYYY-MM-DD_v2.md`（版本遞增）
- **報告必須包含**：測試環境、測試參數、摘要表格、完整 Log（`<details>` 摺疊）
- **格式參考**：`report\report_sample.md`（禁止修改此範本）
