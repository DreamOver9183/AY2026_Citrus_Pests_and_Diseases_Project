# .pt → .tflite 的最小無頭匯出映像。
#
# 只做一件事：跑 tools/export_tflite.py。沒有 web、沒有資料庫、沒有前端。
# 版本組合沿用 DreamOver9183/Citrus_Pest_and_Disease_Tools 已驗證過的釘選，
# 理由全部寫在 requirements.txt 的註解裡——改動前先讀那份。

# --platform 是硬需求，不是最佳化：litert-torch / ai-edge-litert 沒有 aarch64 輪子。
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /work

# libgl1 / libglib2.0-0 給 opencv；ultralytics 匯入時就會 import cv2。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# YOLO_AUTOINSTALL=0：ultralytics 預設在缺套件時會於執行中直接 pip install。
#   匯出跑到一半才裝套件，會讓「這次產物是用哪個版本轉的」變成不可考。
# YOLO_CONFIG_DIR：避免 ultralytics 往唯讀的 HOME 寫 settings.json。
#   值要給「已存在且可寫」的目錄——ultralytics 會在它底下再建一層 Ultralytics/。
#   給 /tmp/ultralytics 會噴 "is not writable" 的警告然後自己退回 /tmp/Ultralytics，
#   功能沒事但每次執行都多一行雜訊，直接給 /tmp 就乾淨了。
ENV YOLO_AUTOINSTALL=0 \
    YOLO_CONFIG_DIR=/tmp \
    PYTHONUNBUFFERED=1

# torch 單獨一道，且指定 CPU index。
# 若與 requirements.txt 合併成一道，pip 解析時會從預設索引拉 CUDA 版
# （約 2 GB 的 nvidia-* 相依，而這個容器沒有 GPU runtime）。
#
# --timeout / --retries：首次建置在 files.pythonhosted.org 讀取逾時而失敗。
# 這裡的輪子很大（torch CPU 版約 200 MB，litert-torch 與 ai-edge-* 合計數百 MB），
# pip 預設的 15 秒逾時在慢連線上不夠用。
RUN pip install --no-cache-dir --timeout 180 --retries 10 \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.12.1' torchvision

COPY Benchmark/export/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r /tmp/requirements.txt

# ncnn 必須 --no-deps：它的相依含 opencv-python，會覆蓋掉映像裡由 ultralytics
# 帶進來的 opencv 5.x。ultralytics 自己呼叫 check_requirements("ncnn", cmds="--no-deps")
# 也是同樣理由，這裡只是把它預先裝好（YOLO_AUTOINSTALL=0 時它不會自己裝）。
RUN pip install --no-cache-dir --timeout 180 --retries 10 --no-deps ncnn

# 程式碼由 -v 掛進來，映像本身不含專案檔案——
# 改一行腳本不必重建映像（重建要 10 分鐘以上）。
CMD ["python", "tools/export_tflite.py", "--help"]
