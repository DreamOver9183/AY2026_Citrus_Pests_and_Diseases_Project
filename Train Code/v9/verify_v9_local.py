# -*- coding: utf-8 -*-
"""v9 架構本機隔離驗證。

內容 = model-train_v9.ipynb 的 Step.2 + Step.3 + Step.4 + Step.4.5 合併版，
差別只在 yaml 輸出路徑改到本機暫存目錄。目的是在燒 Kaggle GPU 時數之前，
先確認模型可建構、可前向、可算出有限損失，並量到參數量 / GFLOPs / 活化記憶體。

用法（專案根目錄下）：
    .venv/Scripts/python.exe "Train Code/verify_v9_local.py"
"""

import os
import sys
import math
import pathlib
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import yaml as _yaml

import ultralytics
import ultralytics.utils.loss
import ultralytics.utils.metrics
import ultralytics.nn.tasks
import ultralytics.nn.modules
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.conv import Conv

OUT_DIR = pathlib.Path(os.environ.get("V9_OUT", tempfile.gettempdir())) / "v9_verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 往上找出含有 other/yolo26-p2.yaml 的專案根目錄，避免本檔被搬動後失效
REPO = next((p for p in pathlib.Path(__file__).resolve().parents
             if (p / "other" / "yolo26-p2.yaml").exists()),
            pathlib.Path(__file__).resolve().parent.parent)
V8_YAML = REPO / "other" / "yolo26-p2.yaml"          # v8 用的官方 yolo26-p2.yaml
V9_YAML = OUT_DIR / "yolo26-citrus-p2-v9.yaml"

BAR = "=" * 74


def section(title):
    print("\n" + BAR + "\n  " + title + "\n" + BAR)


# ══════════════════════════════════════════════════════════════════════════
section("Step.2  Wise-Inner-MPDIoU")
# ══════════════════════════════════════════════════════════════════════════

_BBOX_IOU_ORIG = ultralytics.utils.metrics.bbox_iou   # 留著做量級對照

_WIOU_STATE = {"iou_mean": 1.0}


def bbox_wise_inner_mpdiou(
    box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7,
    ratio=0.7, lambda_mpd=1.0, lambda_inner=1.0,
    alpha=1.9, delta=3.0, momentum=0.02,
):
    if xywh:
        (cx1, cy1, w1, h1), (cx2, cy2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        hw1, hh1, hw2, hh2 = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = cx1 - hw1, cx1 + hw1, cy1 - hh1, cy1 + hh1
        b2_x1, b2_x2, b2_y1, b2_y2 = cx2 - hw2, cx2 + hw2, cy2 - hh2, cy2 + hh2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1).clamp(0), (b1_y2 - b1_y1).clamp(0)
        w2, h2 = (b2_x2 - b2_x1).clamp(0), (b2_y2 - b2_y1).clamp(0)
        cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    r = ratio / 2
    i1_x1, i1_x2, i1_y1, i1_y2 = cx1 - w1 * r, cx1 + w1 * r, cy1 - h1 * r, cy1 + h1 * r
    i2_x1, i2_x2, i2_y1, i2_y2 = cx2 - w2 * r, cx2 + w2 * r, cy2 - h2 * r, cy2 + h2 * r
    inner_inter = (i1_x2.minimum(i2_x2) - i1_x1.maximum(i2_x1)).clamp(0) * \
                  (i1_y2.minimum(i2_y2) - i1_y1.maximum(i2_y1)).clamp(0)
    inner_union = (w1 * h1 + w2 * h2) * ratio ** 2 - inner_inter + eps
    inner_iou = inner_inter / inner_union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c_diag_sq = cw.pow(2) + ch.pow(2) + eps
    d1_sq = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2_sq = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)
    mpdiou_penalty = (d1_sq + d2_sq) / c_diag_sq

    l_iou = 1.0 - iou
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)
    r_wiou = torch.exp(center_dist_sq / c_diag_sq.detach())

    if torch.is_grad_enabled() and l_iou.numel() > 0:
        _WIOU_STATE["iou_mean"] = (1 - momentum) * _WIOU_STATE["iou_mean"] \
                                  + momentum * l_iou.detach().mean().item()
    beta = l_iou.detach() / max(_WIOU_STATE["iou_mean"], eps)
    focus = beta / (delta * alpha ** (beta - delta))
    loss_wiou = focus * r_wiou * l_iou

    total_loss = loss_wiou + lambda_mpd * mpdiou_penalty + lambda_inner * (1.0 - inner_iou)
    return 1.0 - total_loss


ultralytics.utils.loss.bbox_iou = bbox_wise_inner_mpdiou
ultralytics.utils.metrics.bbox_iou = bbox_wise_inner_mpdiou

_p = torch.tensor([[10.0, 10.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]], requires_grad=True)
_t = torch.tensor([[12.0, 12.0, 32.0, 32.0], [50.0, 50.0, 60.0, 60.0]])
_iou = ultralytics.utils.loss.bbox_iou(_p, _t, xywh=False, CIoU=True)
assert _iou.shape == (2, 1), "回傳形狀必須為 [N,1]，實際為 " + str(tuple(_iou.shape))
assert torch.isfinite(_iou).all(), "回傳值含 NaN/Inf"
(1.0 - _iou).sum().backward()
assert torch.isfinite(_p.grad).all(), "梯度含 NaN/Inf"
print("[1/7] 簽名/形狀/梯度 OK  loss(重疊框)={:.4f}  loss(不相交框)={:.4f}".format(
    float((1 - _iou[0]).detach()), float((1 - _iou[1]).detach())))

# 量級對照：同一批框，內建 CIoU 損失 vs 自訂損失
torch.manual_seed(0)
_n = 4096
_cxy = torch.rand(_n, 2)
_wh = torch.rand(_n, 2) * 0.08 + 0.01                     # 貼近本資料集的微小目標
_gt = torch.cat([_cxy, _wh], 1)
_pred = _gt + torch.randn(_n, 4) * 0.01                   # 模擬預測抖動
_pred[:, 2:] = _pred[:, 2:].clamp(min=1e-3)
with torch.no_grad():
    _l_ciou = (1.0 - _BBOX_IOU_ORIG(_pred, _gt, xywh=True, CIoU=True)).mean().item()
    _l_new = (1.0 - bbox_wise_inner_mpdiou(_pred, _gt, xywh=True)).mean().item()
print("      量級對照  CIoU={:.4f}   Wise-Inner-MPDIoU={:.4f}   倍率={:.2f}x".format(
    _l_ciou, _l_new, _l_new / max(_l_ciou, 1e-9)))

# ══════════════════════════════════════════════════════════════════════════
section("Step.3  StarTripletBlock 定義與註冊")
# ══════════════════════════════════════════════════════════════════════════


class TripletAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.cw = Conv(2, 1, k, 1, act=False)
        self.hc = Conv(2, 1, k, 1, act=False)
        self.hw = Conv(2, 1, k, 1, act=False)

    @staticmethod
    def _pool(t):
        return torch.cat([t.max(1, keepdim=True)[0], t.mean(1, keepdim=True)], dim=1)

    def forward(self, x):
        p1 = x.permute(0, 2, 1, 3).contiguous()
        o1 = (p1 * self.cw(self._pool(p1)).sigmoid()).permute(0, 2, 1, 3).contiguous()
        p2 = x.permute(0, 3, 2, 1).contiguous()
        o2 = (p2 * self.hc(self._pool(p2)).sigmoid()).permute(0, 3, 2, 1).contiguous()
        o3 = x * self.hw(self._pool(x)).sigmoid()
        return (o1 + o2 + o3) / 3.0


class StarBlock(nn.Module):
    def __init__(self, c, mlp_ratio=3, k=7):
        super().__init__()
        c_ = int(c * mlp_ratio)
        self.dw1 = Conv(c, c, k, 1, g=c, act=False)
        self.f1 = Conv(c, c_, 1, 1, act=False)
        self.f2 = Conv(c, c_, 1, 1, act=False)
        self.act = nn.ReLU6()
        self.ta = TripletAttention(k)
        self.g = Conv(c_, c, 1, 1, act=False)
        self.dw2 = Conv(c, c, k, 1, g=c, act=False)

    def forward(self, x):
        y = self.dw1(x)
        y = self.act(self.f1(y)) * self.f2(y)
        y = self.ta(y)
        return x + self.dw2(self.g(y))


class StarTripletBlock(nn.Module):
    def __init__(self, c1, c2, n=1, mlp_ratio=3, k=7):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(StarBlock(c2, mlp_ratio, k) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.proj(x))


for _cls in (TripletAttention, StarBlock, StarTripletBlock):
    _cls.__module__ = "ultralytics.nn.tasks"
    setattr(ultralytics.nn.tasks, _cls.__name__, _cls)
    setattr(ultralytics.nn.modules, _cls.__name__, _cls)

_C2F_ORIG = ultralytics.nn.tasks.C2f
ultralytics.nn.tasks.C2f = StarTripletBlock

assert StarTripletBlock(128, 32, n=1)(torch.randn(2, 128, 32, 32)).shape == (2, 32, 32, 32)
assert StarTripletBlock(64, 64, n=2)(torch.randn(2, 64, 32, 32)).shape == (2, 64, 32, 32)
assert ultralytics.nn.tasks.ADown(64, 64)(torch.randn(2, 64, 32, 32)).shape == (2, 64, 16, 16)
print("[2/7] StarTripletBlock (c1==c2 / c1!=c2) 與內建 ADown 形狀 OK")

# ══════════════════════════════════════════════════════════════════════════
section("Step.4  產生並回讀 yolo26-citrus-p2-v9.yaml")
# ══════════════════════════════════════════════════════════════════════════

yaml_v9_content = """
nc: 8
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2                    -> 16
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4                    -> 32
  - [-1, 2, C2f, [256]]                # 2-P2/4  StarTripletBlock  -> 64
  - [-1, 1, ADown, [256]]              # 3-P3/8                    -> 64
  - [-1, 2, C2f, [512]]                # 4-P3/8  StarTripletBlock  -> 128
  - [-1, 1, ADown, [512]]              # 5-P4/16                   -> 128
  - [-1, 2, C3k2, [512, True]]         # 6-P4/16                   -> 128
  - [-1, 1, ADown, [1024]]             # 7-P5/32                   -> 256
  - [-1, 2, C3k2, [1024, True]]        # 8-P5/32                   -> 256
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9                         -> 256
  - [-1, 2, C2PSA, [1024]]             # 10                        -> 256

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]          # 12  256+128               -> 384
  - [-1, 2, C3k2, [512, True]]         # 13                        -> 128

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]          # 15  128+128               -> 256
  - [-1, 2, C3k2, [256, True]]         # 16 (P3/8)                 -> 64

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]          # 18  64+64                 -> 128
  - [-1, 2, C2f, [128]]                # 19 (P2/4-xsmall) STB      -> 32

  - [-1, 1, ADown, [128]]              # 20                        -> 32
  - [[-1, 16], 1, Concat, [1]]         # 21  32+64                 -> 96
  - [-1, 2, C3k2, [256, True]]         # 22 (P3/8)                 -> 64

  - [-1, 1, ADown, [256]]              # 23                        -> 64
  - [[-1, 13], 1, Concat, [1]]         # 24  64+128                -> 192
  - [-1, 2, C3k2, [512, True]]         # 25 (P4/16)                -> 128

  - [-1, 1, ADown, [512]]              # 26                        -> 128
  - [[-1, 10], 1, Concat, [1]]         # 27  128+256               -> 384
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32)            -> 256

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

V9_YAML.write_text(yaml_v9_content.strip(), encoding="utf-8")
_d = _yaml.safe_load(V9_YAML.read_text(encoding="utf-8"))
assert _d["nc"] == 8 and _d["end2end"] is True and _d["reg_max"] == 1, _d
_layers = _d["backbone"] + _d["head"]
assert len(_layers) == 30, "層數 = " + str(len(_layers)) + "，應為 30"
assert _layers[-1][2] == "Detect" and _layers[-1][0] == [19, 22, 25, 28], _layers[-1]
print("[3/7] yaml 回讀 OK：{} 層，Detect 取用 {}，STB@{}，ADown@{}".format(
    len(_layers), _layers[-1][0],
    [i for i, l in enumerate(_layers) if l[2] == "C2f"],
    [i for i, l in enumerate(_layers) if l[2] == "ADown"]))

# ══════════════════════════════════════════════════════════════════════════
section("Step.4.5  建構 / 前向 / 損失")
# ══════════════════════════════════════════════════════════════════════════

model = YOLO(str(V9_YAML))
core = model.model
print("[4/7] 模型建構成功")

det = core.model[-1]
assert getattr(det, "end2end", False), "Detect 不是 end2end"
assert det.reg_max == 1, "reg_max = " + str(det.reg_max)
assert det.nc == 8 and det.nl == 4, "nc={} nl={}".format(det.nc, det.nl)
strides = [int(s) for s in det.stride]
assert strides == [4, 8, 16, 32], "strides = " + str(strides)
print("[5/7] end2end=True, reg_max={}, nc={}, 偵測頭={}, strides={}".format(
    det.reg_max, det.nc, det.nl, strides))

types = {i: m.type.split(".")[-1] for i, m in enumerate(core.model)}
for idx in (2, 4, 19):
    assert types[idx] == "StarTripletBlock", "第 {} 層是 {}，別名注入沒生效".format(idx, types[idx])
for idx in (3, 5, 7, 20, 23, 26):
    assert types[idx] == "ADown", "第 {} 層是 {}".format(idx, types[idx])
print("      模組就位：StarTripletBlock @ [2,4,19]、ADown @ [3,5,7,20,23,26]")

core.eval()
with torch.no_grad():
    core(torch.zeros(1, 3, 640, 640))
print("[6/7] Forward pass (640x640) 成功")

core.args = get_cfg(overrides={"box": 5.0, "cls": 1.0, "dfl": 1.5})
core.train()
fake_batch = {
    "img": torch.rand(2, 3, 640, 640),
    "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    "cls": torch.tensor([[4.0], [6.0], [1.0]]),
    "bboxes": torch.tensor([[0.50, 0.50, 0.04, 0.04],
                            [0.22, 0.31, 0.02, 0.03],
                            [0.71, 0.68, 0.15, 0.12]]),
}
loss, loss_items = core.loss(fake_batch)
if isinstance(loss_items, dict):
    items = {k: float(v) for k, v in loss_items.items()}
else:
    items = {k: float(v) for k, v in zip(("box", "cls", "l1/dfl"), loss_items.flatten())}
assert torch.isfinite(loss).all(), "總損失 NaN/Inf"
assert all(math.isfinite(v) for v in items.values()), "分項損失 NaN/Inf: " + str(items)
print("[7/7] E2EDetectLoss 完整路徑 OK（含自訂 Wise-Inner-MPDIoU）")
print("      " + "  ".join("{}={:.4f}".format(k, v) for k, v in items.items())
      + "  | total={:.4f}".format(float(loss.sum().detach())))

# ══════════════════════════════════════════════════════════════════════════
section("成本對照  v8 (官方 yolo26-p2) vs v9")
# ══════════════════════════════════════════════════════════════════════════

from ultralytics.utils.torch_utils import get_flops, get_num_params


def activation_elems(m, imgsz=640):
    """粗估一次 forward 產生的活化張量總元素數（所有葉子模組輸出加總）。"""
    total = [0]
    hooks = []

    def hook(_mod, _inp, out):
        if torch.is_tensor(out):
            total[0] += out.numel()

    for sub in m.modules():
        if len(list(sub.children())) == 0:
            hooks.append(sub.register_forward_hook(hook))
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, imgsz, imgsz))
    for h in hooks:
        h.remove()
    return total[0]


# v8 基準：官方 yolo26-p2.yaml，nc 改成 8 以便公平比較
v8_cfg = _yaml.safe_load(V8_YAML.read_text(encoding="utf-8"))
v8_cfg["nc"] = 8
V8_TMP = OUT_DIR / "yolo26-p2-nc8.yaml"
V8_TMP.write_text(_yaml.safe_dump(v8_cfg, sort_keys=False), encoding="utf-8")

ultralytics.nn.tasks.C2f = _C2F_ORIG          # v8 yaml 不含 C2f，還原比較保險
v8 = YOLO(str(V8_TMP)).model
ultralytics.nn.tasks.C2f = StarTripletBlock

rows = []
for name, m in (("v8 (官方 yolo26-p2)", v8), ("v9 (STB + ADown)", core)):
    rows.append((name, get_num_params(m), get_flops(m, imgsz=640), activation_elems(m)))

print("{:<24}{:>12}{:>10}{:>18}{:>13}".format("模型", "參數量", "GFLOPs", "活化元素(B=1)", "B=24 fp16"))
for name, p, f, a in rows:
    print("{:<24}{:>12,}{:>10.2f}{:>18,}{:>12.2f}G".format(name, p, f, a, a * 24 * 2 / 1024 ** 3))

p8, f8, a8 = rows[0][1:]
p9, f9, a9 = rows[1][1:]
print("\n▷ v9 / v8 倍率：參數 {:.2f}x   GFLOPs {:.2f}x   活化 {:.2f}x".format(
    p9 / p8, f9 / max(f8, 1e-9), a9 / a8))

print("\n" + BAR + "\n  全部通過。yaml 產物：" + str(V9_YAML) + "\n" + BAR)
