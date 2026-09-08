# _History — 2026 年 6–7 月實作歸檔

這裡是專案早期（v8 之前）的實作檔案，**5 個壓縮檔、27.6 GB，維持未解壓狀態**。

**完整盤點與結論寫在 [docs/通用_參考_2026年6-7月歷史歸檔.md](../docs/通用_參考_2026年6-7月歷史歸檔.md)**，
本檔只是放在資料夾裡的索引，方便直接看到裡面有什麼。

> **這批資料與現行的 v5r / v5.5 / v5.6 訓練線不可直接比較**：
> 資料集是 12 類（其中 2 類零標註），切分與評估協定都不同。

---

## 檔案清單

| 檔案 | 大小 | 內容 |
| --- | ---: | --- |
| `模型成果報告.zip` | 27.3 GB | **主檔案**。12 類資料集（YOLO ＋ COCO 兩種格式）、6 個 YOLO26 run、2 個 SSD run、`模型訓練數據報告.md`、`docs/各項數據計算.md` |
| `TFLite-20260908T063545Z-1-001.zip` | 159 MB | 8 個 `.tflite` 匯出產物（與下一項重複） |
| `Mobile Benchmark.zip` | 149 MB | 8 個 `.tflite` ＋ adb platform-tools ＋ benchmark APK ＋ `Benchmark_process.md`、`benchmark_report.md` |
| `YOLO26-20260908T063537Z-1-001.zip` | 878 MB | YOLO26 訓練輸出（與主檔案重複） |
| `MobilenetV3-20260908T063542Z-1-001.zip` | 15 MB | 兩個 SSD 訓練輸出（與主檔案重複） |

## 裡面最值得看的六個檔案

```
模型成果報告.zip
 ├─ 模型訓練數據報告.md              ← 6 個模型的彙整報告（注意：「最優」其實是最後一輪）
 ├─ docs/各項數據計算.md             ← Detection Jaccard 的原始定義出處
 └─ Model/
     ├─ SSD-MobilenetV3-large_train/training_report.md   ← AMP NaN、LR 排程器兩個坑的解法
     ├─ SSD-MobilenetV3-small_train/training_report.md
     └─ YOLO26-nano-p2-w8a32/runs/detect/YOLO26n_P2_Citrus_MuSGD_FT-2/weights/best_int8.tflite
                                                          ← 就是 7 月 benchmark 裡那顆模型
Mobile Benchmark.zip
 ├─ Benchmark_process.md            ← 現行 ADB 量測流程的原始出處
 └─ benchmark_report.md             ← 7 月實機數據（表格有兩列與自身 log 不符，已更正）
```

## 三個要記住的重點

1. **`P_TP_LD`（薊馬葉害）從 6 月起就在類別清單裡，但一個標註框都沒有。**
   它就是現行最弱的 `Thrips_Damage`。
2. **`P_SI`（介殼蟲）佔 valid 框數的 68.8%。** 現行的類別偏斜問題從這時就存在。
3. **27.5 GB 裡只有約 6.9 GB 是唯一內容**，其餘是同一批影像的 COCO 格式副本與 zip 副本。

## 怎麼看裡面的東西（不要整包解壓）

```bash
python -c "import zipfile;print('\n'.join(zipfile.ZipFile(r'模型成果報告.zip').namelist()[:50]))"
```

單獨抽一個檔案：

```bash
python -c "import zipfile;zipfile.ZipFile(r'模型成果報告.zip').extract('模型成果報告/模型訓練數據報告.md')"
```
