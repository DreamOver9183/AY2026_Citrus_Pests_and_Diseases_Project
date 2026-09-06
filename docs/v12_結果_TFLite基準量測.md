# v12 結果 — TFLite 匯出與手機端基準量測

| | |
| --- | --- |
| 日期 | 2026-09-07 |
| 狀態 | **匯出管線已完成並驗證；手機端量測待裝置授權** |
| 對應程式 | `tools/export_tflite.py`、`Benchmark/export/`、`.claude/skills/tflite_mobile_benchmark/` |
| 部署目標 | **30 FPS ±5，即每張 28.6–40 ms** |

---

## 0. 摘要

這一輪要補上專案缺的另一半量尺。到 v11.5 為止，這個專案量得出**精度**
（九類 pooled ±2SE ≤ 0.10），但完全量不出**延遲**——所以「要不要換更大的模型」
一直無法討論，因為沒有預算可以花。

| 項目 | 狀態 |
| --- | --- |
| Benchmark 工具鏈合併進本 repo | ✅ 完成 |
| `.pt` → `.tflite` 匯出管線 | ✅ 完成 |
| 匯出產物與 PyTorch 的等價性 | ✅ **完整 401 張 test 上，兩個變體的 mAP50 都與 PyTorch 在雜訊內** |
| 手機端量測 | ⛔ **卡在裝置授權**，見 §4 |

**兩個關鍵結論**

1. **`end2end=True`（NMS-free）的 YOLO26 可以經 LiteRT 直轉**，且**不掉精度**。
   這在動工前是最大的未知數——若 topk 轉不過去，整條部署路線都要重新設計。
2. **`w8a32`（動態 INT8）是主力量化選項，不是退路。** 它把模型從 9.85 MB 壓到 3.10 MB
   （3.18×），精度沒有可辨識的損失，而且**保留 end2end**——靜態 INT8 做不到這一點。

---

## 1. 匯出管線

### 1.1 為什麼需要一個容器

ultralytics 的 LiteRT 匯出函式第一行就是平台斷言：

```python
assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
```

開發機是 Windows，**本機不可能轉**。因此建了一個最小化的 `linux/amd64` 映像
（857 MB，`python:3.12-slim` ＋ torch 2.12.1 CPU ＋ litert 三件組）。

### 1.2 為什麼不走 ONNX → saved_model

這條路原本是首選，研究階段確認它是**相依死路**：

> `ultralytics[export-tensorflow]` 要求 `numpy<2.0.0`，
> 而 ultralytics 傳遞相依的 `opencv-python 5.x` 要求 `numpy>=2`。

本專案 venv 是 numpy 2.5.2 + opencv 5.0.0，根目錄 `tools/` 有 20 支腳本吃這組相依。
走那條路等於在跑著的推論堆疊底下抽換套件。**LiteRT 是 PyTorch 直轉、不經 ONNX，
沒有 numpy 限制。**

這個結論不是自己推的——`DreamOver9183/Citrus_Pest_and_Disease_Tools` 的
`requirements-docker.txt` 已經記載過，版本釘選也直接沿用它驗證過的組合。

### 1.3 三種變體，以及一個會影響量測解讀的差異

`ultralytics/engine/exporter.py` 有這一段：

```python
if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
    # Static activation quantization collapses the end2end class-index output
    model.end2end = False
```

| 變體 | `quantize` | 校正資料 | `end2end` | 量到的延遲**含 NMS** |
| --- | --- | --- | --- | --- |
| `fp32` | `None` | 不需要 | **保留** | ✅ 含（topk 在圖裡） |
| `w8a32` | `"w8a32"` | **不需要** | **保留** | ✅ 含 |
| `int8` | `8` | 需要 | **被關掉** | ❌ 不含，NMS 要在 App 端另外做 |
| `w8a16` | `"w8a16"` | 需要 | **被關掉** | ❌ 同上 |

> **這一點修正了計畫階段的判斷。** 原本把 `w8a32` 當成「靜態 INT8 失敗時的退路」，
> 讀了 exporter 原始碼之後結論相反：**`w8a32` 才是主力**——它同時做到
> 「權重 INT8」與「保留 end2end」，是唯一能跟 fp32 做 apples-to-apples 比較的量化選項。
>
> 靜態 `int8` 的 FPS 天生偏高（它少做了 NMS 那段工作）。日後若要量，
> **報告必須標註這一點**，否則會得出錯誤的結論。

---

## 2. 匯出結果（v11.5 `last.pt`）

來源：`Train Code/v11.5/Train Output/extracted/runs/detect/v5.6_v11_5/weights/last.pt`

| 變體 | 檔案大小 | 對 `.pt`（5.3 MB） | 對 fp32 | 匯出耗時 | `end2end` |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp32` | **9.85 MB** | 1.86× | — | 21 s | 保留 |
| `w8a32` | **3.10 MB** | 0.58× | **0.31×** | 18 s | 保留 |

fp32 比 `.pt` 大是正常的——`.pt` 存的是 fp16 權重加上壓縮，`.tflite` 是攤平的 fp32 圖。

匯出時回報的輸出形狀是 **`(1, 300, 6)`**——這就是 end2end 的 post-NMS 輸出
（最多 300 個框，每框 `[x1, y1, x2, y2, score, cls]`），確認 topk 分支確實轉進去了。

### 2.1 決定性的證據：完整 401 張 test 的 mAP

`batch=1`（litert 追蹤時把 batch 維度寫死成 1，三個模型都用 1 才是同一條量測路徑）：

| 模型 | mAP50 | Δ | mAP50-95 | Δ | 耗時 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch `last.pt` | 0.80897 | — | 0.60527 | — | 73 s |
| `fp32` `.tflite` | **0.81554** | **+0.0066** | **0.61014** | **+0.0049** | 48 s |
| `w8a32` `.tflite` | **0.81936** | **+0.0104** | **0.60892** | **+0.0037** | 50 s |

**兩個變體都不比 PyTorch 差，名目上還略高。** 這個「略高」不要解讀成改善——
0.0066 遠低於本專案量過的 run 間全距 0.021，屬於同一批雜訊。

逐類 AP50：

| 類別 | PyTorch | fp32 | Δ | w8a32 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oily_Spot | 0.9446 | 0.9518 | +0.0072 | 0.9518 | +0.0072 |
| Canker | 0.8610 | 0.8768 | +0.0158 | 0.8776 | +0.0165 |
| Sooty_Mold | 0.9950 | 0.9950 | +0.0000 | 0.9950 | +0.0000 |
| Black_Spot | 0.9503 | 0.9815 | +0.0312 | 0.9853 | +0.0350 |
| Scale_Insect | 0.6682 | 0.6817 | +0.0134 | 0.6876 | +0.0194 |
| Citrus_Leaf_Miner | 0.6795 | 0.6883 | +0.0089 | 0.6851 | +0.0056 |
| Thrips | 0.7828 | 0.7760 | −0.0068 | 0.7656 | −0.0172 |
| Aphid | 0.9479 | 0.9310 | −0.0169 | 0.9317 | −0.0161 |
| Thrips_Damage | 0.4515 | 0.4577 | +0.0062 | **0.4947** | **+0.0432** |

全距 −0.017 ～ +0.043。**per-class 的雜訊地板是 ±0.04**（v11 實測），
所以除了 `Thrips_Damage` 的 w8a32 剛好踩在門檻上，其餘全部不可辨識。
那一格也不該當成「量化讓模型變好」——`Thrips_Damage` 本來就是九類裡最不穩的一類。

> 註：PyTorch 這一列的 0.80897 與 v11.5 報告的 0.80959 差 0.0006，
> 因為那份是 `batch=20` 跑的。同一份權重、同一批影像，差異來自 batch 大小。

### 2.2 逐框比對（`--verify`）

20 張等距取樣的 v5.6 test 影像（橫跨全部九類 ＋ Background），`conf=0.25`：

| 變體 | PyTorch 框 | tflite 框 | 配對 | 配對 IoU 平均 | 最低 | 類別一致 | 未配對率 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| `fp32` | 40 | 38 | 37 | **0.9763** | 0.9114 | 37/37 | 5.1% | ✅ |
| `w8a32` | 40 | 38 | 36 | **0.9727** | 0.9141 | 36/36 | 7.7% | ✅ |

判準三條全部要過：配對框 IoU 平均 ≥ 0.90、類別 100% 一致、未配對率 ≤ 10%。
0.90 比偵測常用的 0.50 嚴得多，因為這裡比的是**同一個模型的兩種序列化**，
不是兩個不同的模型。

### 2.3 逐框比對抓到的東西，以及為什麼最後不算問題

**這一節記錄的是「差點寫成錯誤結論」的過程**，因為判斷順序本身有參考價值。

`Scale_Insect_00026.jpg`（18 個真值框，密集小目標）上，fp32 相對 PyTorch：

- **三個框完全消失**，其中一個 score 0.5645、與任何 tflite 框的 IoU 都是 **0.000**
- 另外三個框幾何幾乎一致（IoU ≈ 0.98）但**信心值掉了 0.19 ～ 0.26**

檢查過的假設，以及結果：

| 假設 | 結果 |
| --- | --- |
| 是重複框被合併（像 `Aphid_00008` 那樣） | **否。** 那個 0.5645 的框與其他 PyTorch 框的最大重疊只有 0.000，是獨立物件 |
| 是 letterbox 不一致——PyTorch 走矩形、tflite 走正方形 | **否。** `pre_transform` 的 `auto` 需要 `self.args.rect`，而 predict 的 `rect` 預設是 `False`，**兩邊都是正方形 640** |
| 這對部署有影響嗎 | **沒有。** 完整 401 張的 mAP50 反而高 0.0066（見 §2.1） |

**結論**：逐框的差異是密集小目標上的分數抖動，聚合到 mAP 之後不但沒有代價，
名目上還略正。**20 張影像的逐框比對只能擋住「明顯壞掉」的產物，
回答不了「這對部署有沒有影響」**——那要靠完整的 `val()`。

> **順帶修掉三個自己挖的坑**，都是跑了才發現的：
>
> 1. `--verify` 把 20 張影像一次餵給 `predict()`，但 litert 追蹤時 batch 維度就寫死成 1，
>    直接 `Dimension mismatch. Got 20 but expected 1`。改成一次一張。
> 2. 只比「最高分框」。`Aphid_00008` 上兩個分數接近的框在量化後換了名次
>    （0.8166/0.7906 → 0.8345/0.8192），指標掉成 **IoU = 0.0**——但兩隻蚜蟲兩邊都找到了，
>    IoU 分別是 0.978 與 0.967。改成**全體框貪婪配對**，最低 IoU 才變成有意義的 0.9571。
> 3. 取 test 的**前** 20 張。檔名帶類別前綴、排序後同類會連在一起，
>    結果 20 張**全部是 Aphid**，等於只驗了九分之一的類別。改成等距取樣。
>
> 第 3 點才是讓 `Scale_Insect_00026` 浮出來的原因。前兩個版本的驗收都會給出漂亮的
> 「0/0 未配對」，而那只是因為抽到的都是單目標的簡單影像。

---

## 3. 測試平台 1 規格

| 項目 | 值 |
| --- | --- |
| ADB 序號 | `ebe3968d` |
| 型號 / SoC | **尚未取得**（`getprop` 需要授權） |
| 連線狀態 | **`unauthorized`** |

---

## 4. ⛔ 目前的阻塞：裝置未授權

```
$ Benchmark/platform-tools/adb.exe devices -l
List of devices attached
ebe3968d               unauthorized transport_id:1
```

手機被偵測到了，但**沒有接受 USB 偵錯的 RSA 授權**。已嘗試
`adb kill-server` ＋ `start-server` 重新出示金鑰，狀態不變——
授權對話框只會出現在**已解鎖的手機螢幕上**，必須由人實體操作。

**解除方式**：

1. 解鎖手機
2. 出現「允許 USB 偵錯嗎？」對話框時，勾選**「一律允許使用這台電腦進行偵錯」**→ 允許
3. 若沒有跳出對話框：開發者選項 →「**撤銷 USB 偵錯授權**」，然後重新插拔 USB
4. 確認：`Benchmark/platform-tools/adb.exe devices -l` 應顯示 `ebe3968d  device`

授權完成後，`Benchmark/Model/` 裡的兩個 `.tflite` 已通過驗收，可直接進量測流程。

---

## 5. 待量測的內容（授權後）

每個變體量**兩組設定**，因為它們回答不同的問題：

| 設定 | 參數 | 回答什麼 |
| --- | --- | --- |
| **CPU-only 4T** | `--num_threads=4 --use_gpu=false --use_nnapi=false` | **保底 FPS**。達標判定用這條 |
| **Delegate** | `--use_gpu=true`（再跑 `--use_nnapi=true`） | **部署 FPS**，且必須記錄**節點替換率** |

兩組都 `--num_runs=25`。

> **節點替換率是要量的，不是能推論的。** YOLO26 的 end2end 頭含 topk，
> 這類算子若不被 delegate 支援，整段會退回 CPU——**delegate 的 FPS 有可能比純 CPU 更差**。

**不要拿桌機的數字外推。** §2.1 那三個「耗時」欄位（PyTorch 73 s、fp32 48 s、
w8a32 50 s / 401 張）是 x86 容器裡的 CPU 時間，包含資料載入與 mAP 計算，
**與手機的推論延遲沒有可換算的關係**。

### 5.1 這些數字要拿來決定什麼

`v12s`（`yolo26s-p2`，3.49× GFLOPs）的 notebook 已備妥但尚未執行。
它的**精度**判準是預先登記的 `test mAP50 ≥ 0.83959`（v11.5 + 0.03）。

FPS **不是** v12s 的通過條件（已確認若過不了就放寬 FPS 目標），
但這裡量出的 n 模型基準線，決定了 s 模型的延遲代價要怎麼解讀：

- 若 n 模型在 CPU-only 就已經接近 40 ms，s 模型（3.49×）幾乎不可能達標
- 若 n 模型有大幅餘裕，s 模型才有討論空間

---

## 6. 已知限制

1. **沒有量靜態 INT8**。它會關掉 end2end，量出來的數字與另外兩個不可比，
   而 `w8a32` 已經提供了「權重 INT8」而不付這個代價。若日後 `w8a32` 的
   FPS 不夠，再回來量 `int8` 並在 App 端補 NMS。
2. **等價性只在 v5.6 test 上驗過**，沒有驗 valid。這一關的用途是確認
   「序列化沒有動到模型」，不是重新評估模型——後者 v11.5 已經做過。
3. **`.tflite` 不進版控**（`.gitignore` 已排除）。每個數 MB 到數十 MB，
   且可由對應的 `last.pt` 用一行指令重現。
4. **上游 benchmark 工具是複製進來的，不是 submodule**（依「以後作為同 repo」的要求）。
   上游若更新需手動比對合併。來源 commit 記在 `Benchmark/README.md`。

---

## 7. 重現方式

```bash
docker build -t citrus-tflite-export -f Benchmark/export/Dockerfile .
```

```bash
docker run --rm -v "${PWD}:/work" citrus-tflite-export python tools/export_tflite.py --weights "Train Code/v11.5/Train Output/extracted/runs/detect/v5.6_v11_5/weights/last.pt" --variants fp32,w8a32 --verify
```

只想重驗已存在的檔案（不重轉）：把 `--verify` 換成 `--verify-only`。

原始數據：

| 檔案 | 內容 |
| --- | --- |
| `Benchmark/Model/last__export_report.json` | 匯出與逐框比對的結果 |
| `Benchmark/Model/last__val_comparison.json` | 三個模型在完整 test 上的 mAP 與逐類 AP50 |

方法與流程的細節見 [`Benchmark/README.md`](../Benchmark/README.md) 與
[`Benchmark/export/README.md`](../Benchmark/export/README.md)。
