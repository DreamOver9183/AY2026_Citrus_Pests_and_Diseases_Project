# -*- coding: utf-8 -*-
"""v9 消融實驗的單一真實來源：自訂損失、自訂模組、六臂定義。

在此之前 Step.2（Wise-Inner-MPDIoU）與 Step.3（StarTripletBlock）同時存在於
`model-train_v9.ipynb`、`RESUME.ipynb`、`verify_v9_local.py` 三個檔案裡，
任一份漂移都會讓續跑或驗證的模型與訓練時不一致，而且不會報錯（問題 C3）。
本檔把它們收斂成一份，三處呼叫端一律改成 import 這裡。

Kaggle 端取得方式（需開啟 internet）：

    !curl -sSL -o v9_modules.py \\
      https://raw.githubusercontent.com/DreamOver9183/AY2026_Citrus_Pests_and_Diseases_Project/main/Train%20Code/v9/v9_modules.py

典型用法：

    import v9_modules as v9
    hp = v9.install_arm("A1")          # 依臂裝好 loss / 模組，回傳超參數
    cfg = v9.make_arm_yaml(**v9.ARMS["A1"]["arch"])
    ...
    v9.restore()                       # 還原全域 patch（同一行程要測下一臂時必須呼叫）

相依：ultralytics 8.4.121。整套注入機制建立在該版本的實作細節上
（`parse_model` 的 frozenset、`BboxLoss` 的呼叫簽名），換版本前務必重跑
`verify_v9_local.py`。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import yaml as _yaml

import ultralytics
import ultralytics.nn.modules
import ultralytics.nn.tasks
import ultralytics.utils.loss
import ultralytics.utils.metrics
from ultralytics.nn.modules.conv import Conv

__version__ = "1.0.0"
REQUIRED_ULTRALYTICS = "8.4.121"

# ══════════════════════════════════════════════════════════════════════════
# Wise-Inner-MPDIoU
#
#   Inner-IoU : 依 ratio 縮放輔助框後計算 IoU。ratio < 1 放大「已對準」樣本的
#               梯度（精修）；ratio > 1 讓遠距離的框也能重疊（救漏檢）。
#   MPDIoU    : 左上/右下頂點的對角約束。
#   WIoU v3   : 依離群度動態聚焦，抑制低品質樣本。
#
# ── 相容性要求（照 ultralytics 8.4.121 原始碼）─────────────────────────────
# 1. `utils/loss.py:133` 的呼叫方式是
#        iou = bbox_iou(pred[fg], target[fg], xywh=False, CIoU=True)
#    → 簽名必須吃得下 GIoU/DIoU/CIoU，否則第一個 batch 就 TypeError。
# 2. 同一行後面是 `loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum`，
#    而 weight 形狀為 [N,1]。因此本函式必須回傳 [N,1]；若用 box1[..., 0] 索引
#    會退化成 [N]，相乘廣播成 [N,N] → 損失錯誤且記憶體暴增。一律用 chunk(4, -1)。
# 3. 回傳值語意 = 1 - total_loss，如此 (1.0 - iou) 恰好等於 total_loss。
# 4. `utils/tal.py:9` 在 import 時就把原始 bbox_iou 綁進自己的命名空間，
#    所以標籤分配器與驗證階段的 mAP 不受本檔影響 —— v8/v9 的 mAP 可直接對照。
# ══════════════════════════════════════════════════════════════════════════

CFG: dict = dict(
    ratio=0.7,          # Inner-IoU 輔助框縮放比
    lambda_wiou=1.0,    # WIoU v3 權重（設 0 可單獨隔離其餘項，驗收用）
    lambda_mpd=1.0,     # MPDIoU 頂點懲罰權重
    lambda_inner=1.0,   # Inner-IoU 損失權重
    alpha=1.9,          # WIoU v3 聚焦係數
    delta=3.0,          # WIoU v3 聚焦係數（beta == delta 時 focus 恰為 1）
    momentum=0.02,      # L_IoU 滑動平均的動量
    beta_max=20.0,      # 離群度上限，見下方 FIX 2b
)
_CFG_DEFAULTS = dict(CFG)   # restore() 用來還原

# iou_mean 以「留在 GPU 的純量 tensor」維護，避免每次呼叫都做 .item() 同步。
_STATE: dict = {"iou_mean": None, "init": 1.0}


def _running_mean(l_iou: torch.Tensor, momentum: float, update: bool) -> torch.Tensor:
    """取得（必要時就地更新）L_IoU 的滑動平均。"""
    st = _STATE["iou_mean"]
    if st is None or st.device != l_iou.device:
        st = torch.full((), float(_STATE["init"]), device=l_iou.device, dtype=torch.float32)
        _STATE["iou_mean"] = st
    if update:
        st.mul_(1.0 - momentum).add_(l_iou.detach().mean().float(), alpha=momentum)
    return st


def wiou_state() -> float:
    """讀出目前的離群度統計量（會觸發一次 GPU→CPU 同步，僅供存檔時呼叫）。"""
    st = _STATE["iou_mean"]
    return float(_STATE["init"]) if st is None else float(st)


def set_wiou_state(value: float) -> None:
    """還原離群度統計量。續跑時必須呼叫，否則第二場會從 1.0 重新暖機。"""
    _STATE["init"] = float(value)
    if _STATE["iou_mean"] is not None:
        _STATE["iou_mean"].fill_(float(value))


def bbox_wise_inner_mpdiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """回傳 [N,1]，語意為 1 - (λ_wiou·WIoU + λ_mpd·MPD + λ_inner·(1-InnerIoU))。"""
    ratio = CFG["ratio"]
    alpha, delta = CFG["alpha"], CFG["delta"]

    # FIX 2a（原問題 B4）：全程在 fp32 計算，最後才轉回輸入 dtype。
    # AMP 之下輸入是 fp16，而 c_diag_sq = cw² + ch² 在座標達到特徵圖尺度時
    # 就會逼近 fp16 上限 65504：imgsz=640 的 P2 是 160（2×160²=51,200，勉強在界內），
    # imgsz=800 則是 200（2×200²=80,000，直接溢位成 inf）。
    # 一旦 c_diag_sq 變 inf，MPD 與 r_wiou 同時歸零，損失靜默失效。
    # 成本只有 N×4 的轉型，換掉一整類 AMP 脆弱性。
    in_dtype = box1.dtype
    box1, box2 = box1.float(), box2.float()

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

    # ── 1. 標準 IoU ────────────────────────────────────────────────────
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # ── 2. Inner-IoU：兩框各自以中心點依 ratio 縮放後再算 IoU ──────────
    r = ratio / 2
    i1_x1, i1_x2, i1_y1, i1_y2 = cx1 - w1 * r, cx1 + w1 * r, cy1 - h1 * r, cy1 + h1 * r
    i2_x1, i2_x2, i2_y1, i2_y2 = cx2 - w2 * r, cx2 + w2 * r, cy2 - h2 * r, cy2 + h2 * r
    inner_inter = (i1_x2.minimum(i2_x2) - i1_x1.maximum(i2_x1)).clamp(0) * \
                  (i1_y2.minimum(i2_y2) - i1_y1.maximum(i2_y1)).clamp(0)
    inner_union = (w1 * h1 + w2 * h2) * ratio ** 2 - inner_inter + eps
    inner_iou = inner_inter / inner_union

    # ── 3. 外接框對角線 ────────────────────────────────────────────────
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c_diag_sq = cw.pow(2) + ch.pow(2) + eps

    # ── 4. MPDIoU 頂點對角約束 ─────────────────────────────────────────
    # FIX 1（原問題 B3）：分母必須 detach。否則梯度可以靠「把外接框撐大」
    # 來降低懲罰項，方向與收斂相反 —— 這正是 WIoU 那一項早就 detach 的理由，
    # 只是 MPD 這一項先前漏了。
    d1_sq = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2_sq = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)
    mpdiou_penalty = (d1_sq + d2_sq) / c_diag_sq.detach()

    # ── 5. WIoU v3 ────────────────────────────────────────────────────
    l_iou = 1.0 - iou
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)
    r_wiou = torch.exp(center_dist_sq / c_diag_sq.detach())

    # 僅在訓練階段更新統計量，避免驗證污染（validator 走 no_grad）
    update = bool(torch.is_grad_enabled() and l_iou.numel() > 0)
    mean = _running_mean(l_iou, CFG["momentum"], update)

    # FIX 2b（原問題 B4）：離群度夾上限。
    # 訓練後期 iou_mean 變小時 beta 會衝很高，`alpha ** (beta - delta)` 隨之爆增；
    # beta_max=20 對應 focus ≈ 1e-4，已等同完全抑制，再大沒有意義只會逼近數值邊界。
    beta = (l_iou.detach() / mean.clamp_min(eps)).clamp(max=CFG["beta_max"])
    focus = beta / (delta * alpha ** (beta - delta))
    loss_wiou = focus * r_wiou * l_iou

    total_loss = CFG["lambda_wiou"] * loss_wiou \
        + CFG["lambda_mpd"] * mpdiou_penalty \
        + CFG["lambda_inner"] * (1.0 - inner_iou)
    return (1.0 - total_loss).to(in_dtype)


# ══════════════════════════════════════════════════════════════════════════
# StarTripletBlock（StarNet 星形升維 + Triplet Attention）
# ══════════════════════════════════════════════════════════════════════════


class TripletAttention(nn.Module):
    """無降維三元注意力：C-W、H-C、H-W 三個分支的空間/通道交互權重平均。"""

    def __init__(self, k: int = 7):
        super().__init__()
        # act=False：Conv 預設帶 SiLU，接 sigmoid 會變成雙重非線性，注意力會被壓平
        self.cw = Conv(2, 1, k, 1, act=False)
        self.hc = Conv(2, 1, k, 1, act=False)
        self.hw = Conv(2, 1, k, 1, act=False)

    @staticmethod
    def _pool(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t.max(1, keepdim=True)[0], t.mean(1, keepdim=True)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = x.permute(0, 2, 1, 3).contiguous()                       # B,H,C,W → 沿 H 池化
        o1 = (p1 * self.cw(self._pool(p1)).sigmoid()).permute(0, 2, 1, 3).contiguous()

        p2 = x.permute(0, 3, 2, 1).contiguous()                       # B,W,H,C → 沿 W 池化
        o2 = (p2 * self.hc(self._pool(p2)).sigmoid()).permute(0, 3, 2, 1).contiguous()

        o3 = x * self.hw(self._pool(x)).sigmoid()                     # 沿 C 池化
        return (o1 + o2 + o3) / 3.0


class StarBlock(nn.Module):
    """StarNet 星形運算單元（通道數不變）。

    DW → 1x1 雙路升維後逐元素相乘（隱式高維映射）→ TripletAttention → 1x1 降維 → DW → 殘差。
    """

    def __init__(self, c: int, mlp_ratio: int = 3, k: int = 7):
        super().__init__()
        c_ = int(c * mlp_ratio)
        self.dw1 = Conv(c, c, k, 1, g=c, act=False)
        self.f1 = Conv(c, c_, 1, 1, act=False)
        self.f2 = Conv(c, c_, 1, 1, act=False)
        self.act = nn.ReLU6()
        self.ta = TripletAttention(k)
        self.g = Conv(c_, c, 1, 1, act=False)
        self.dw2 = Conv(c, c, k, 1, g=c, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw1(x)
        y = self.act(self.f1(y)) * self.f2(y)   # star operation
        y = self.ta(y)
        return x + self.dw2(self.g(y))


class StarTripletBlock(nn.Module):
    """c1→c2 投影 + n 個 StarBlock；簽名刻意與 C2f 對齊 (c1, c2, n, ...)。

    先用 1x1 投影對齊通道，之後所有 StarBlock 都在 c2 上做等通道殘差 ——
    否則 head 第 19 層（c1=96 / c2=32）的殘差形狀不符，而且
    `Conv(c1, c2, g=c1)` 在 c2 不被 c1 整除時連建構都會失敗。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, mlp_ratio: int = 3, k: int = 7):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(StarBlock(c2, mlp_ratio, k) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


# ══════════════════════════════════════════════════════════════════════════
# 注入與還原
#
# 為什麼模組要綁到 `C2f` 這個既有名字：
#   parse_model 內部的 base_modules / repeat_modules 是在「函式執行時」用
#   tasks 模組的 globals 動態建立的 frozenset（tasks.py:1973 / 2011），
#   但成員名單寫死。新名稱永遠不在名單內，只會落到 else 分支 ——
#   拿不到 c1、args 不做 width 縮放 → `TypeError: missing 'c2'`。
#   覆寫既有名稱則能完整取得與 C3k2 相同的待遇。
#   本專案的 yaml 完全沒用到 C2f，且 tasks.py 中 C2f 僅出現於 import 與
#   那兩個 frozenset，無其他副作用。
# ══════════════════════════════════════════════════════════════════════════

_ORIG = {"bbox_iou_loss": None, "bbox_iou_metrics": None, "C2f": None}


def install_loss(ratio: float | None = None, **overrides) -> None:
    """把 BboxLoss 用的 bbox_iou 換成 Wise-Inner-MPDIoU。"""
    if ratio is not None:
        CFG["ratio"] = float(ratio)
    CFG.update({k: v for k, v in overrides.items() if k in CFG})

    if _ORIG["bbox_iou_loss"] is None:
        _ORIG["bbox_iou_loss"] = ultralytics.utils.loss.bbox_iou
        _ORIG["bbox_iou_metrics"] = ultralytics.utils.metrics.bbox_iou

    # loss.py 是 `from .metrics import bbox_iou`，覆寫 loss 模組命名空間才有效。
    ultralytics.utils.loss.bbox_iou = bbox_wise_inner_mpdiou
    # metrics.py 本身不呼叫 bbox_iou，這行純粹為了一致性，不影響任何路徑。
    ultralytics.utils.metrics.bbox_iou = bbox_wise_inner_mpdiou
    _STATE["iou_mean"] = None          # 換臂時重置統計量


def install_modules() -> None:
    """註冊三個自訂類別並把 StarTripletBlock 綁到 C2f。"""
    for cls in (TripletAttention, StarBlock, StarTripletBlock):
        # 讓 checkpoint pickle 成 ultralytics.nn.tasks.<name> 而非 __main__.<name>
        cls.__module__ = "ultralytics.nn.tasks"
        setattr(ultralytics.nn.tasks, cls.__name__, cls)
        setattr(ultralytics.nn.modules, cls.__name__, cls)

    if _ORIG["C2f"] is None:
        _ORIG["C2f"] = ultralytics.nn.tasks.C2f
    ultralytics.nn.tasks.C2f = StarTripletBlock       # ★ 別名注入


def restore() -> None:
    """還原所有全域 patch。同一個行程要測下一臂時必須先呼叫（問題 C5）。"""
    if _ORIG["bbox_iou_loss"] is not None:
        ultralytics.utils.loss.bbox_iou = _ORIG["bbox_iou_loss"]
        ultralytics.utils.metrics.bbox_iou = _ORIG["bbox_iou_metrics"]
        _ORIG["bbox_iou_loss"] = _ORIG["bbox_iou_metrics"] = None
    if _ORIG["C2f"] is not None:
        ultralytics.nn.tasks.C2f = _ORIG["C2f"]
        _ORIG["C2f"] = None
    _STATE["iou_mean"] = None
    _STATE["init"] = 1.0
    CFG.update(_CFG_DEFAULTS)


# ══════════════════════════════════════════════════════════════════════════
# 模型組態
# ══════════════════════════════════════════════════════════════════════════

# v9 相對官方 yolo26-p2 的兩處架構替換，層號與通道數完全對齊，
# 確保 mAP 差異可歸因到改動本身而非頭部配置。
ADOWN_LAYERS = ((3, 256), (5, 512), (7, 1024), (20, 128), (23, 256), (26, 512))
STB_LAYERS = ((2, 256), (4, 512), (19, 128))


def base_p2_cfg() -> dict:
    """讀取已安裝 ultralytics 內建的 yolo26-p2.yaml（與 other/yolo26-p2.yaml 位元相同）。"""
    from ultralytics.utils import ROOT
    path = ROOT / "cfg" / "models" / "26" / "yolo26-p2.yaml"
    return _yaml.safe_load(path.read_text(encoding="utf-8"))


def make_arm_yaml(adown: bool = False, stb: bool = False, nc: int = 8,
                  scale: str = "n") -> dict:
    """產生指定臂的模型組態 dict，可直接餵 DetectionModel(cfg=...) 或 dump 成檔。"""
    d = base_p2_cfg()
    d["nc"] = nc
    d["scale"] = scale
    # end2end / reg_max 必須保留：YOLO26 靠這兩個 key 決定走 NMS-free 的
    # E2EDetectLoss（box + cls + l1）還是舊的 DFL(reg_max=16) 路徑。
    # 漏掉會靜默變成另一個體系的模型，與 v8 完全無法對照。
    d.setdefault("end2end", True)
    d.setdefault("reg_max", 1)

    layers = d["backbone"] + d["head"]
    if adown:
        for i, ch in ADOWN_LAYERS:
            layers[i] = [layers[i][0], 1, "ADown", [ch]]
    if stb:
        for i, ch in STB_LAYERS:
            layers[i] = [layers[i][0], layers[i][1], "C2f", [ch]]
    d["backbone"], d["head"] = layers[:11], layers[11:]
    return d


# ══════════════════════════════════════════════════════════════════════════
# 六臂定義 —— 各臂與 A0 只差一項，這是唯一定義差異的地方
# ══════════════════════════════════════════════════════════════════════════

# 所有臂共用，與 v8 對齊
HYPERS: dict = dict(
    optimizer="MuSGD", lr0=0.008, lrf=0.01, momentum=0.937, cos_lr=True,
    warmup_epochs=3.0, mosaic=0.7, cls=0.8, dfl=1.5,
    batch=20, seed=0, cache="ram", workers=4, plots=True, device=0,
)

ARMS: dict = {
    "A0": dict(
        desc="基準：官方 yolo26-p2 + 內建 CIoU",
        arch=dict(adown=False, stb=False), loss_ratio=None,
        train=dict(box=8.0, imgsz=640), s_per_epoch=165,
    ),
    "A4": dict(
        desc="解析度：imgsz 640 → 800",
        arch=dict(adown=False, stb=False), loss_ratio=None,
        train=dict(box=8.0, imgsz=800), s_per_epoch=270,
    ),
    "A2": dict(
        desc="ADown：stride-2 Conv → ADown ×6",
        arch=dict(adown=True, stb=False), loss_ratio=None,
        train=dict(box=8.0, imgsz=640), s_per_epoch=160,
    ),
    "A1": dict(
        desc="loss：CIoU → Wise-Inner-MPDIoU (ratio=0.70)",
        arch=dict(adown=False, stb=False), loss_ratio=0.70,
        train=dict(box=3.7, imgsz=640), s_per_epoch=175,
    ),
    "A1b": dict(
        desc="loss 修向：同 A1 但 ratio=1.25（救低 IoU 漏檢）",
        arch=dict(adown=False, stb=False), loss_ratio=1.25,
        train=dict(box=4.15, imgsz=640), s_per_epoch=175,
    ),
    "A3": dict(
        desc="StarTripletBlock：C3k2 → STB ×3",
        arch=dict(adown=False, stb=True), loss_ratio=None,
        train=dict(box=8.0, imgsz=640), s_per_epoch=360,
    ),
}

# A1 / A1b 的 box 是刻意校準的：Wise-Inner-MPDIoU 的損失量級隨 ratio 而不同
# （本機實測微小框：ratio=0.70 為 CIoU 的 2.17x、ratio=1.25 為 1.93x），
# 因此 box 取 8.0 / 倍率 → 3.7 與 4.15，讓兩臂的「等效」權重都對齊 A0 的 8.0。
# 不校準的話就同時改了損失形狀與量級，又多一個混淆變數。
# verify_v9_local.py 會重新量測並印出建議值，換 ratio 時記得對照。


def close_mosaic_for(epochs: int) -> int:
    """v8 是 epochs=160 / close_mosaic=30。按比例縮放以維持相同的排程形狀，
    讓 60 輪的短跑能預測 160 輪長跑的末段行為。"""
    return max(1, round(30 * epochs / 160))


def install_arm(arm: str, epochs: int = 60) -> dict:
    """依臂裝好 loss 與模組，回傳可直接展開給 model.train() 的超參數。"""
    if arm not in ARMS:
        raise KeyError(f"未知的臂 {arm!r}，可用：{list(ARMS)}")
    spec = ARMS[arm]
    restore()

    if spec["loss_ratio"] is not None:
        install_loss(ratio=spec["loss_ratio"])
    if spec["arch"]["stb"]:
        install_modules()

    hp = dict(HYPERS)
    hp.update(spec["train"])
    hp.update(
        epochs=epochs,
        close_mosaic=close_mosaic_for(epochs),
        # 消融臂一律跑滿，否則各臂的平台期窗口不一致就無法比較
        patience=epochs,
        name=f"v5r_{arm}_{epochs}e",
    )
    return hp


def arm_summary(arm: str, epochs: int = 60) -> str:
    spec = ARMS[arm]
    hours = spec["s_per_epoch"] * epochs / 3600
    loss = "內建 CIoU" if spec["loss_ratio"] is None else f"W-I-MPDIoU ratio={spec['loss_ratio']}"
    return (f"{arm}  {spec['desc']}\n"
            f"    架構 adown={spec['arch']['adown']} stb={spec['arch']['stb']}   loss {loss}\n"
            f"    box={spec['train']['box']} imgsz={spec['train']['imgsz']} "
            f"epochs={epochs} close_mosaic={close_mosaic_for(epochs)}\n"
            f"    預估 {spec['s_per_epoch']} s/epoch → {hours:.2f} h")
