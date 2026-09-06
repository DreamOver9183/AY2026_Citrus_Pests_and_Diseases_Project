# TFLite Logcat 輸出解析規範 (Log Parsing Guide)

> 本文件詳細說明如何解讀 `adb logcat -d -s tflite` 的完整輸出格式，
> 包含每個欄位的含義、單位換算以及衍生指標的計算公式。

---

## 1. 完整 Log 結構解析

一次完整的 TFLite Benchmark Log 由以下幾個區塊組成：

```log
07-14 13:14:52.097 22838 22838 I tflite  : [區塊A] 測試配置參數
07-14 13:14:52.098 22838 22838 I tflite  : [區塊B] 模型載入狀態
07-14 13:14:52.113 22838 22838 I tflite  : [區塊C] XNNPACK Delegate 初始化
07-14 13:14:52.190 22838 22838 I tflite  : [區塊D] 模型資訊
07-14 13:14:52.859 22838 22838 I tflite  : [區塊E] Warmup 統計
07-14 13:14:56.520 22838 22838 I tflite  : [區塊F] 正式推論統計
07-14 13:14:56.521 22838 22838 I tflite  : [區塊G] 時間摘要（關鍵行）
07-14 13:14:56.521 22838 22838 I tflite  : [區塊H] 記憶體摘要（關鍵行）
```

---

## 2. 各區塊詳細解析

### 區塊 A：測試配置參數

```log
I tflite  : Log parameter values verbosely: [0]
I tflite  : Min num runs: [25]
I tflite  : Num threads: [4]
I tflite  : Graph: [/data/local/tmp/best_int8.tflite]
I tflite  : Signature to run: []
I tflite  : #threads used for CPU inference: [4]
I tflite  : Use gpu: [0]
I tflite  : Use NNAPI: [0]
```

**用途**：確認測試使用的參數是否與預期一致。
**驗收**：`Num threads: [4]`, `Use gpu: [0]`, `Use NNAPI: [0]`

---

### 區塊 B：模型載入狀態

```log
I tflite  : Loaded model /data/local/tmp/best_int8.tflite
I tflite  : Initialized TensorFlow Lite runtime.
```

**用途**：確認模型成功載入。若此行不存在，代表模型路徑錯誤或檔案損壞。

---

### 區塊 C：XNNPACK Delegate 節點替代

```log
I tflite  : Created TensorFlow Lite XNNPACK delegate for CPU.
I tflite  : Replacing 539 out of 548 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 19 partitions for subgraph 0.
```

**關鍵正則表達式（用於解析）：**
```regex
Replacing (\d+) out of (\d+) node\(s\) with delegate
```

**提取變數：**
- `delegate_n` = 539（已替代節點數）
- `total_n` = 548（總節點數）

**計算節點替代率：**
```
節點替代率 = delegate_n / total_n × 100 = 539 / 548 × 100 = 98.36%
```

**欄位含義：**
| 數值 | 含義 |
|---|---|
| `delegate_n`（539）| 被 XNNPACK 加速的節點數 |
| `total_n`（548）| 模型總計算節點數 |
| `partitions`（19）| 被分割為幾個連續執行群 |

**指標判讀：**
- 替代率 > 95%：優秀，幾乎全部節點被加速
- 替代率 85–95%：良好，有少數自定義算子
- 替代率 < 85%：偏低，模型含較多設備不支援的算子

---

### 區塊 D：模型資訊

```log
I tflite  : The input model file size (MB): 3.06409
I tflite  : Initialized session in 92.318ms.
```

**提取變數：**
- `model_size_mb` = 3.06409
- `session_init_ms` = 92.318（已是毫秒單位）

---

### 區塊 E：Warmup 統計行

```log
I tflite  : count=3 first=235158 curr=200386 min=199462 max=235158 avg=211669 std=16613 p5=199462 median=200386 p95=235158
```

**正則表達式：**
```regex
count=(\d+) first=(\d+) curr=(\d+) min=(\d+) max=(\d+) avg=([\d.e+]+) std=(\d+)
```

> ⚠️ 注意：這行是 **Warmup 階段**統計，不用於最終報告計算。

---

### 區塊 F：正式推論統計行（重要）

```log
I tflite  : count=25 first=187533 curr=140990 min=140071 max=195121 avg=145528 std=13583 p5=140545 median=141253 p95=187533
```

**正則表達式（比對正式推論那行）：**
```regex
count=(\d+) first=(\d+) curr=(\d+) min=(\d+) max=(\d+) avg=([\d.e+]+) std=(\d+) p5=(\d+) median=(\d+) p95=(\d+)
```

> ⚠️ 識別方式：正式推論行的 `count` 值應等於 `num_runs`（預設 25）。
> Warmup 行的 `count` 通常遠小於 25。

**提取變數（單位均為微秒 us）：**

| 欄位 | 變數 | 說明 |
|---|---|---|
| `count` | `run_count` | 實際執行的推論次數 |
| `first` | `first_us` | 第一次推論耗時（us）|
| `min` | `min_us` | 最快推論耗時（us）|
| `max` | `max_us` | 最慢推論耗時（us）|
| `avg` | `avg_us` | 平均推論耗時（us）|
| `std` | `std_us` | 標準差（us）|
| `p5` | `p5_us` | 第 5 百分位（us）|
| `median` | `median_us` | 中位數（us）|
| `p95` | `p95_us` | 第 95 百分位（us）|

---

### 區塊 G：時間摘要（最關鍵行）

```log
I tflite  : Inference timings in us: Init: 92318, First inference: 187533, Warmup (avg): 211669, Inference (avg): 145528
```

**正則表達式：**
```regex
Inference timings in us: Init: ([\d.]+), First inference: ([\d.]+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): ([\d.e+]+)
```

**提取變數（單位：微秒 us）：**

| 欄位 | 變數 | 換算 ms | 說明 |
|---|---|---|---|
| `Init` | `init_us` | `÷ 1000` | 模型初始化 + 圖編譯耗時 |
| `First inference` | `first_us` | `÷ 1000` | 第一次推論耗時（含記憶體分配）|
| `Warmup (avg)` | `warmup_us` | `÷ 1000` | 預熱階段平均耗時 |
| `Inference (avg)` | `inference_us` | `÷ 1000` | 正式推論平均耗時（最重要指標）|

---

### 區塊 H：記憶體摘要（最關鍵行）

```log
I tflite  : Memory footprint delta from the start of the tool (MB): init=54.2852 overall=73.4102
```

**正則表達式：**
```regex
Memory footprint delta from the start of the tool \(MB\): init=([\d.]+) overall=([\d.]+)
```

**提取變數（單位：MB）：**

| 欄位 | 變數 | 說明 |
|---|---|---|
| `init=` | `init_mem_mb` | 模型初始化後的記憶體增量 |
| `overall=` | `overall_mem_mb` | 推論過程中的記憶體增量（峰值）|

---

## 3. 衍生指標計算公式

以下是所有衍生指標的完整計算公式：

```python
# 輸入原始數值（從 logcat 提取）
init_us        = 92318      # Init（微秒）
first_us       = 187533     # First inference（微秒）
warmup_us      = 211669     # Warmup avg（微秒）
inference_us   = 145528     # Inference avg（微秒）
min_us         = 140071     # 最快推論（微秒）
max_us         = 195121     # 最慢推論（微秒）
std_us         = 13583      # 標準差（微秒）
init_mem_mb    = 54.2852    # Init 記憶體增量（MB）
overall_mem_mb = 73.4102    # Overall 記憶體增量（MB）
delegate_n     = 539        # 被替代節點數
total_n        = 548        # 總節點數

# --- 換算公式 ---

# 報告欄位（毫秒）
first_ms      = first_us / 1000           # 首輪推論 → 187.53 ms
min_ms        = min_us / 1000             # 最快 → 140.07 ms
max_ms        = max_us / 1000             # 最慢 → 195.12 ms
inference_ms  = inference_us / 1000       # 平均推論 → 145.53 ms
std_ms        = std_us / 1000             # 標準差 → 13.58 ms

# 衍生指標
fps           = 1000 / inference_ms       # 預估 FPS → 1000/145.53 = 6.87
              # 或等效：fps = 1000000 / inference_us

delegate_ratio = (delegate_n / total_n) * 100  # 節點替代率 → 98.36%
mem_overhead  = overall_mem_mb - init_mem_mb    # 運算開銷差值 → 19.12 MB
```

---

## 4. 特殊情況處理

### 4.1 科學計數法數值

某些大型模型的 `avg` 值會以科學計數法顯示：

```log
avg=2.44949e+06
```

**轉換方式：**
```python
# 2.44949e+06 = 2449490 微秒 = 2449.49 毫秒
import re

def parse_us(value_str):
    """解析可能含科學計數法的微秒值"""
    return float(value_str)  # Python float() 可直接處理科學計數法
```

### 4.2 Log 不完整（測試未完成）

若 logcat 輸出缺少 `Inference timings in us:` 行，可能原因：
1. 測試尚未完成 → 延長等待時間（+30s）後重試
2. APK 崩潰 → 執行 `adb logcat -d -s AndroidRuntime` 查看崩潰日誌
3. 記憶體不足 → 模型過大，考慮使用較小模型

### 4.3 有多段 Inference timings（歷史殘留）

若 `adb logcat -d -s tflite` 輸出含有多次測試結果：
- **以最後一段** `Inference timings in us:` 為準（最新的那次）
- 判斷方式：比對時間戳或 `Graph:` 行中的模型名稱

---

## 5. 解析優先順序

當 Log 中存在多個候選行時，遵循以下優先順序：

1. `Inference timings in us:` 行（最權威，直接使用）
2. 正式推論統計行（`count=25` 那行，作為備用驗證）
3. Warmup 統計行（僅用於確認 `Warmup (avg)`，不直接用於 FPS 計算）
