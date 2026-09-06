# `.pt` → `.tflite` 匯出管線

把訓練出來的 YOLO26 權重轉成手機端能跑的 `.tflite`，並**驗證轉出來的還是同一個模型**。

---

## 為什麼要一個容器

ultralytics 的 LiteRT 匯出函式第一行就是平台斷言：

```python
assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
```

本專案的開發機是 Windows，**不可能在本機轉**。這個容器是最小化的 `linux/amd64`
環境，只做這一件事——沒有 web、沒有資料庫、沒有前端。

### 為什麼不走 ONNX → saved_model → TFLite

這條路看起來比較「標準」，但在本專案是**已經有人踩過的死路**：

> `ultralytics[export-tensorflow]` 要求 `numpy<2.0.0`，
> 而 ultralytics 傳遞相依的 `opencv-python 5.x` 要求 `numpy>=2`。

本專案的 venv 是 numpy 2.5.2 + opencv 5.0.0，根目錄 `tools/` 有 19 支腳本吃這組相依。
走那條路等於在跑著的推論堆疊底下抽換套件。**LiteRT 路徑沒有 numpy 限制。**

完整理由與版本釘選的來由寫在 [`requirements.txt`](requirements.txt) 的註解裡。

---

## 用法

### 1. 建映像（只要做一次，約 10 分鐘）

先確認 Docker Desktop 已啟動。

```bash
docker build -t citrus-tflite-export -f Benchmark/export/Dockerfile .
```

> 根目錄的 `.dockerignore` 把 build context 縮到只剩一個 `requirements.txt`。
> 沒有它的話，光是把數 GB 的 `Datasets/` 傳給 daemon 就要好幾分鐘。

### 2. 轉換

PowerShell：

```bash
docker run --rm -v "${PWD}:/work" citrus-tflite-export python tools/export_tflite.py --weights "Train Code/v11.5/Train Output/extracted/runs/detect/v5.6_v11_5/weights/last.pt" --variants fp32,w8a32 --verify
```

CMD：

```bash
docker run --rm -v "%cd%":/work citrus-tflite-export python tools/export_tflite.py --weights "Train Code/v11.5/Train Output/extracted/runs/detect/v5.6_v11_5/weights/last.pt" --variants fp32,w8a32 --verify
```

產物進 `Benchmark/Model/`，命名為 `<stem>__<variant>.tflite`，
另附一份 `<stem>__export_report.json`。

---

## 三種變體，以及一個**會影響 benchmark 解讀**的差異

`ultralytics/engine/exporter.py` 有這一段：

```python
if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
    # Static activation quantization collapses the end2end class-index output
    model.end2end = False
```

也就是說：

| 變體 | `--variants` | 校正資料 | `end2end` | 量到的延遲**包含 NMS** 嗎 |
| --- | --- | --- | --- | --- |
| FP32 | `fp32` | 不需要 | **保留** | ✅ 包含（topk 在圖裡） |
| 動態 INT8 | `w8a32` | **不需要** | **保留** | ✅ 包含 |
| 靜態 INT8 | `int8` | 需要（v5.6 train） | **被關掉** | ❌ **不包含**，NMS 要在 App 端另外做 |
| 靜態 W8A16 | `w8a16` | 需要 | **被關掉** | ❌ 同上 |

**所以 `__int8` 的 FPS 不能直接跟另外兩個比**——它少做了一段工作。

計畫階段我把 `w8a32` 當成「INT8 失敗時的退路」。讀了 exporter 的原始碼之後
這個判斷要修正：**`w8a32` 才是主力**，因為它同時做到「權重 INT8」與「保留 end2end」，
是唯一能跟 fp32 做 apples-to-apples 比較的量化選項。

實測（v11.5 `last.pt`，完整 401 張 v5.6 test）也支持這個判斷：

| 變體 | 大小 | 對 fp32 | mAP50 | 對 PyTorch |
| --- | ---: | ---: | ---: | ---: |
| PyTorch `.pt` | 5.3 MB | — | 0.80897 | — |
| `fp32` | 9.85 MB | — | 0.81554 | +0.0066 |
| **`w8a32`** | **3.10 MB** | **0.31×** | **0.81936** | **+0.0104** |

**壓到 0.31 倍而精度沒有可辨識的損失。** 那個 +0.01 不要解讀成改善——
它低於本專案量過的 run 間全距 0.021。

預設 `--variants fp32`（最保守）。建議實際用 `fp32,w8a32`。

---

## `--verify` 在驗什麼（以及它驗不到什麼）

拿同一批 v5.6 test 影像（預設 20 張，**等距取樣**所以橫跨全部九類）
分別跑 PyTorch 與 `.tflite`，把兩邊的框做**貪婪配對**後比對：

- 配對框的 IoU（平均與最小值）
- 配對框的類別是否一致
- 配不上的框有多少

**判準**（三條都要過）：

| 條件 | 門檻 | 為什麼 |
| --- | --- | --- |
| 配對框 IoU 平均 | **≥ 0.90** | 比偵測常用的 0.50 嚴得多，因為比的是**同一個模型的兩種序列化**，不是兩個模型 |
| 類別一致 | **100%** | 量化不該改變分類結果 |
| 未配對率 | **≤ 10%** | 留餘裕：end2end 偶爾對同一物件吐兩個框，量化把重複合併掉反而是好事 |

沒通過的變體，腳本會以 exit code 1 結束並明說**不要拿去 benchmark**。
量一個壞掉的圖只會產生看起來很正常的假數字。

### ⚠ 它驗不到「精度有沒有掉」

`--verify` 是**擋壞產物的煙霧測試**，不是精度評估。實測過一個具體案例：
`Scale_Insect_00026.jpg`（18 個密集小目標）上 fp32 相對 PyTorch 掉了三個框、
另三個框的信心值掉了 0.19–0.26，看起來很嚴重——但在**完整 401 張 test** 上，
fp32 的 mAP50 反而比 PyTorch 高 0.0066。

要回答「這對部署有沒有影響」，必須跑完整的 `val()`。
數字與方法見 [`docs/v12_結果_TFLite基準量測.md`](../../docs/v12_結果_TFLite基準量測.md) §2.1。

### 兩個常見的誤判（都踩過）

- **只比最高分框**：兩個分數接近的框在量化後換名次，指標會掉成 IoU = 0.0，
  但兩個物件其實都找到了。所以改用全體框配對。
- **取前 N 張**：檔名帶類別前綴、排序後同類會連在一起，取前 20 張會**全部是 Aphid**。
  所以改用等距取樣。

---

## 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--weights` | （必填） | 來源 `.pt`，相對於專案根目錄或絕對路徑 |
| `--variants` | `fp32` | 逗號分隔：`fp32` / `w8a32` / `int8` / `w8a16` |
| `--dataset` | `v5.6` | 校正與驗證用的資料集版本（走 `tools/dataset_paths.py`） |
| `--fraction` | `0.05` | 靜態量化的校正取樣比例（v5.6 train 7,264 張 → 約 363 張） |
| `--imgsz` | 不指定 | 不指定則沿用訓練解析度。**建議不要指定** |
| `--verify` | 關閉 | 轉完後與 PyTorch 比對 |
| `--verify-n` | `20` | 比對用幾張 test 影像 |

---

## 出問題時

| 症狀 | 原因 | 怎麼辦 |
| --- | --- | --- |
| 腳本說「這個環境轉不出 LiteRT」 | 在本機而不是容器裡跑 | 用上面的 `docker run` |
| 「缺少匯出相依」 | 映像沒建成功 | 重跑 `docker build`，看 pip 有沒有逾時 |
| `docker build` 在 pip 逾時 | 連線慢，輪子很大 | Dockerfile 已設 `--timeout 180 --retries 10`；仍失敗就重跑（有 layer 快取） |
| fp32 就轉不出來 | end2end 的 topk 轉不過去 | **回報並重新討論部署格式**。`ncnn` / `mnn` 支援 Windows，但 benchmark APK 只吃 `.tflite`，等於要換量測工具 |
| `--verify` 不通過 | 轉換動到了幾何 | 不要拿去 benchmark。先確認是哪個變體，靜態量化的變體失敗就改用 `w8a32` |

### 另一條可用的路

Kaggle 本身就是 Linux x86，在訓練 notebook 裡直接加一個匯出 cell 也能產出 `.tflite`。
但那對「已經訓練好的權重」要重跑一次 notebook，而容器對本機任何 `.pt` 都能直接用。
**容器當主路，Kaggle 當退路。**
