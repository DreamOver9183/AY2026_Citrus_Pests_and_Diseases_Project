# Benchmark — 手機端 TFLite 推論效能量測

在實體 Android 裝置上量測 `.tflite` 模型的推論延遲，用來回答一個本專案到目前為止
**沒有能力回答**的問題：這個模型放到手機上跑得動嗎？

**部署目標：30 FPS ±5，即每張 28.6–40 ms。**

---

## 0. 這份東西從哪來

由 [`DreamOver9183/TFLite_Benchmark_skills`](https://github.com/DreamOver9183/TFLite_Benchmark_skills)
合併而來，上游 commit `694b95f`（2026-07-21）。

**不是 submodule**，是複製進來的一般檔案——依使用者要求「以後作為同 repo」。
上游後續若有更新，需手動比對合併。

### 合併時改了什麼

| 上游 | 本專案 | 原因 |
| --- | --- | --- |
| `tools/` | **`Benchmark/platform-tools/`** | 本專案根目錄已有 `tools/`（19 支 Python 腳本），**會撞名** |
| `.agents/skills/tflite_mobile_benchmark/` | **`.claude/skills/tflite_mobile_benchmark/`** | Claude Code 只從 `.claude/skills/` 載入 |
| `.agents/AGENTS.md` | **`Benchmark/AGENTS.md`** | `.agents/` 在本專案的 `.gitignore` 裡 |
| 工作根目錄 = benchmark repo 根 | **專案根目錄** | 同 repo 後只有一個根 |

> 上述路徑在 skill 與文件內共改寫 **115 處**。
> 若你看到任何地方還寫著 `.\tools\adb.exe`，那是漏網的，**它在本專案指向錯的目錄**。

`.gitignore` 也為此開了一個白名單（`.claude/*` ＋ `!.claude/skills/`）——
註解寫在那裡，改動前先讀。

---

## 1. 目錄

```
Benchmark/
├── README.md                    ← 本文件
├── AGENTS.md                    ← Agent 行為邊界（三層）
├── Benchmark_process.md         ← 手動執行流程（PowerShell / CMD 兩版）
├── platform-tools/              ← ADB 工具包，已進版控，clone 下來即可用
│   ├── adb.exe
│   └── android_aarch64_benchmark_model.apk
├── export/                      ← .pt → .tflite 的 Docker 匯出管線（本專案新增）
├── Model/                       ← 待測的 .tflite（*.tflite 已 gitignore）
└── report/                      ← 量測報告
    └── report_sample.md         ← 格式範本（唯讀）
```

Skill 本體在 `.claude/skills/tflite_mobile_benchmark/`，對 Claude Code 說
「跑 benchmark」「測 FPS」即會觸發。

---

## 2. 前置檢查

```bash
./Benchmark/platform-tools/adb.exe devices -l
```

| 輸出 | 意義 | 怎麼辦 |
| --- | --- | --- |
| `<id> device` | 可以開始 | — |
| `<id> unauthorized` | 手機沒有接受 USB 偵錯授權 | **在手機螢幕上按「允許」並勾選「一律允許」** |
| `<id> offline` | 連線異常 | 重新插拔 USB |
| 只有標題列 | 沒偵測到裝置 | 確認 USB 為「檔案傳輸」模式、開發者選項已開 USB 偵錯 |

授權對話框只會出現在**已解鎖**的手機螢幕上。授權後再跑一次上面的指令確認變成 `device`。

---

## 3. 本專案的量測協定（與上游預設不同）

上游 skill 的預設是**只量 CPU**。本專案**每個模型量兩組**，因為這兩個數字回答的是不同的問題：

| 設定 | 參數 | 回答什麼 |
| --- | --- | --- |
| **CPU-only 4T** | `--num_threads=4 --use_gpu=false --use_nnapi=false` | **保底 FPS**。決策門檻用這條，因為它不依賴任何裝置專屬的加速器 |
| **Delegate** | `--use_gpu=true` 或 `--use_nnapi=true` | **部署 FPS**。另外記錄「節點替換率」 |

兩組都用 `--num_runs=25`。

> **節點替換率要看，不能假設。** YOLO26 是 `end2end=True`（NMS-free），偵測頭裡含 topk。
> 這類算子若不被 delegate 支援，整段會退回 CPU 執行——delegate 的 FPS **有可能比純 CPU 更差**。
> 這是要量出來的事實，不是可以推論的。

Log 解析與報告格式沿用上游規範，見
[`.claude/skills/tflite_mobile_benchmark/references/log_parsing_guide.md`](../.claude/skills/tflite_mobile_benchmark/references/log_parsing_guide.md)。

---

## 4. 模型從哪來

`Model/` 一開始是空的。本專案的 `.pt` 權重要先轉成 `.tflite`：

見 [`export/README.md`](export/README.md)。

**簡短版**：Windows 上轉不出來（ultralytics 的 litert 匯出有
`assert MACOS or (LINUX and not ARM64)` 的平台檢查），所以走一個
`linux/amd64` 的 Docker 映像。**不要走 ONNX → saved_model 那條路**，
理由寫在 `export/requirements.txt` 的註解裡。

---

## 5. 已知限制

- 只支援 **AArch64 (ARM64)** 的 Android 裝置
- 量測值受機身溫度與背景程式影響。連續量多個模型時，後面的通常比較慢；
  要比較的模型建議**交錯重跑**而不是一次跑完一個
- 大模型（>50 MB）要等 60 秒以上才會出結果

---

## 6. 上游原始說明

`README_upstream.md` 保留了上游 README 的原文，**其中的路徑對本專案是錯的**，
僅供追溯出處時參考。日常使用一律以本文件為準。
