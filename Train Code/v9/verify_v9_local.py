# -*- coding: utf-8 -*-
"""階段 0 的驗收閘門：在燒 Kaggle GPU 時數之前，於本機把六個消融臂全部驗過一遍。

每臂檢查：建構 → 層別與 strides → end2end/reg_max → forward → 有限損失 →
params/GFLOPs 對照預期值。另外驗證 v9_modules 的三個數值修正與狀態往返。

CPU 可跑，不需要 GPU。用法（專案根目錄下）：

    .venv/Scripts/python.exe "Train Code/v9/verify_v9_local.py"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ultralytics
import ultralytics.utils.loss
import ultralytics.utils.metrics
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params

import v9_modules as v9

BAR = "═" * 78
FAIL: list[str] = []

# 本機實測基準（ultralytics 8.4.121, imgsz=640, nc=8, scale=n）
EXPECTED = {
    "A0":  (2_517_824, 7.65),
    "A4":  (2_517_824, 7.65),          # 與 A0 同架構，只有訓練 imgsz 不同
    "A2":  (2_031_936, 6.54),
    "A1":  (2_517_824, 7.65),
    "A1b": (2_517_824, 7.65),
    "A3":  (2_711_604, 12.11),
}


def section(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def check(cond: bool, msg: str) -> bool:
    if not cond:
        FAIL.append(msg)
        print(f"    ✗ {msg}")
    return cond


# ══════════════════════════════════════════════════════════════════════════
section("環境")
# ══════════════════════════════════════════════════════════════════════════
print(f"  ultralytics {ultralytics.__version__}   torch {torch.__version__}")
print(f"  v9_modules  {v9.__version__}  (要求 ultralytics {v9.REQUIRED_ULTRALYTICS})")
check(ultralytics.__version__ == v9.REQUIRED_ULTRALYTICS,
      f"ultralytics 版本不符：{ultralytics.__version__} != {v9.REQUIRED_ULTRALYTICS}")

# ══════════════════════════════════════════════════════════════════════════
section("數值修正驗證（B3 / B4）")
# ══════════════════════════════════════════════════════════════════════════

v9.restore()
v9.install_loss(ratio=0.7)

# ── 1. 簽名與形狀：完全比照 utils/loss.py:133 的呼叫方式 ──────────────
_p = torch.tensor([[10.0, 10.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]], requires_grad=True)
_t = torch.tensor([[12.0, 12.0, 32.0, 32.0], [50.0, 50.0, 60.0, 60.0]])
_iou = ultralytics.utils.loss.bbox_iou(_p, _t, xywh=False, CIoU=True)
check(_iou.shape == (2, 1), f"回傳形狀必須是 [N,1]，實際 {tuple(_iou.shape)}")
check(bool(torch.isfinite(_iou).all()), "回傳值含 NaN/Inf")
(1.0 - _iou).sum().backward()
check(bool(torch.isfinite(_p.grad).all()), "梯度含 NaN/Inf")
print(f"  [1/6] 簽名/形狀/梯度 OK   loss(重疊)={float(1 - _iou[0]):.4f}  "
      f"loss(不相交)={float(1 - _iou[1]):.4f}")

# ── 2. FIX 1：MPD 分母已 detach ────────────────────────────────────────
# 用 λ 把 WIoU 與 Inner 關掉，單獨隔離 MPD 項再與解析解比對。
# detach 之後 d(MPD)/d(b1_x1) 應恰為 2*(b1_x1-b2_x1)/c²；
# 若分母未 detach，還會多一項來自 c 隨 b1_x1 變動的貢獻。
_saved = {k: v9.CFG[k] for k in ("lambda_wiou", "lambda_mpd", "lambda_inner")}
v9.CFG.update(lambda_wiou=0.0, lambda_mpd=1.0, lambda_inner=0.0)
try:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
    b = torch.tensor([[4.0, 3.0, 16.0, 12.0]])
    g = torch.autograd.grad(v9.bbox_wise_inner_mpdiou(a, b, xywh=False).sum(), a)[0]
    cw, ch = 16.0 - 0.0, 12.0 - 0.0          # 外接框：b2 的右下、b1 的左上
    c2 = cw ** 2 + ch ** 2 + 1e-7
    # 回傳值 = 1 - MPD，故 d(ret)/d(b1_x1) = -2*(b1_x1-b2_x1)/c²
    analytic = -2.0 * (0.0 - 4.0) / c2
    check(math.isclose(float(g[0, 0]), analytic, rel_tol=1e-4),
          f"MPD 分母未 detach：d/d(b1_x1)={float(g[0,0]):.6e}，解析解={analytic:.6e}")
    print(f"  [2/6] MPD 分母已 detach   d/d(b1_x1)={float(g[0,0]):.6e}  "
          f"解析解={analytic:.6e}")
finally:
    v9.CFG.update(_saved)

# ── 3. FIX 2：AMP 之下不溢位 ───────────────────────────────────────────
# 兩種溢位來源：
#   (a) c_diag_sq = cw² + ch²，座標到特徵圖尺度就逼近 fp16 上限 65504
#       （imgsz=800 的 P2 是 200 → 2×200² = 80,000，直接爆掉）
#   (b) iou_mean 收斂到 0.01 時 beta = 1/0.01 = 100，alpha**(beta-delta) 隨之爆增
# 現在全程 fp32 計算並夾 beta 上限，兩者都不會發生。
v9.set_wiou_state(0.01)
far_p = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
far_t = torch.tensor([[400.0, 400.0, 404.0, 404.0]])   # c_diag_sq ≈ 3.3e5，fp16 必爆
with torch.no_grad():
    out32 = v9.bbox_wise_inner_mpdiou(far_p, far_t, xywh=False)
check(bool(torch.isfinite(out32).all()), "極端離群度下 fp32 出現 NaN/Inf")
ok16 = "略過（此平台不支援 CPU fp16）"
try:
    with torch.no_grad():
        out16 = v9.bbox_wise_inner_mpdiou(far_p.half(), far_t.half(), xywh=False)
    if check(bool(torch.isfinite(out16).all()), "極端離群度下 fp16 出現 NaN/Inf"):
        check(out16.dtype == torch.float16, f"回傳 dtype 應為 fp16，實際 {out16.dtype}")
        ok16 = f"fp16 輸入亦有限（回傳 dtype {out16.dtype}）"
except RuntimeError:
    pass
print(f"  [3/6] AMP 不溢位   loss={float(1 - out32[0]):.4f}   {ok16}")

# ── 4. FIX 3：離群度統計量可往返（供 RESUME 續跑還原）──────────────────
v9.set_wiou_state(0.37)
check(math.isclose(v9.wiou_state(), 0.37, rel_tol=1e-6),
      f"set/get 往返失敗：{v9.wiou_state()}")
_p2 = torch.rand(64, 4) * 0.1 + 0.4
_t2 = _p2 + torch.randn(64, 4) * 0.01
v9.bbox_wise_inner_mpdiou(_p2.requires_grad_(True), _t2, xywh=True).sum().backward()
moved = v9.wiou_state()
check(0.30 < moved < 0.45, f"滑動平均更新後偏離過大：{moved:.4f}")
print(f"  [4/6] 狀態往返 OK   set 0.37 → 一次更新後 {moved:.4f}（momentum={v9.CFG['momentum']}）")

# ── 5. 量級對照：確認 box=3.7 的等效權重校準仍成立 ─────────────────────
torch.manual_seed(0)
_n = 4096
_wh = torch.rand(_n, 2) * 0.08 + 0.01                  # 貼近本資料集的微小目標
_gt = torch.cat([torch.rand(_n, 2), _wh], 1)
_pr = _gt + torch.randn(_n, 4) * 0.01
_pr[:, 2:] = _pr[:, 2:].clamp(min=1e-3)
orig_iou = v9._ORIG["bbox_iou_metrics"]
with torch.no_grad():
    l_ciou = (1.0 - orig_iou(_pr, _gt, xywh=True, CIoU=True)).mean().item()
    rows = []
    for r in (0.70, 1.25):
        v9.CFG["ratio"] = r
        v9.set_wiou_state(1.0)
        rows.append((r, (1.0 - v9.bbox_wise_inner_mpdiou(_pr, _gt, xywh=True)).mean().item()))
v9.CFG["ratio"] = 0.7
_arm_box = {0.70: v9.ARMS["A1"]["train"]["box"], 1.25: v9.ARMS["A1b"]["train"]["box"]}
print(f"  [5/6] 量級對照（校準 box 讓等效權重對齊 A0 的 8.0）   CIoU={l_ciou:.4f}")
for r, l in rows:
    mult = l / l_ciou
    box = _arm_box[r]
    eff = box * mult
    want = 8.0 / mult
    ok = abs(eff - 8.0) < 0.4
    check(ok, f"ratio={r} 的 box={box} 等效權重為 {eff:.1f}，應接近 8.0（建議改用 {want:.2f}）")
    print(f"          ratio={r:<5} loss={l:.4f}  倍率={mult:.2f}x  "
          f"box={box} → 等效權重 {eff:.1f}   建議 box={want:.2f}  {'✓' if ok else '✗'}")

# ── 6. restore() 確實還原 ──────────────────────────────────────────────
v9.restore()
check(ultralytics.utils.loss.bbox_iou.__name__ != "bbox_wise_inner_mpdiou",
      "restore() 後 bbox_iou 未還原")
check(ultralytics.nn.tasks.C2f.__name__ == "C2f", "restore() 後 C2f 未還原")
print("  [6/6] restore() 還原 bbox_iou 與 C2f OK")

# ══════════════════════════════════════════════════════════════════════════
section("六臂逐一驗證")
# ══════════════════════════════════════════════════════════════════════════

fake_batch = {
    "img": torch.rand(2, 3, 640, 640),
    "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    "cls": torch.tensor([[4.0], [6.0], [1.0]]),      # Scale_Insect / Thrips / Canker
    "bboxes": torch.tensor([[0.50, 0.50, 0.04, 0.04],   # 刻意用微小框，貼近真實分佈
                            [0.22, 0.31, 0.02, 0.03],
                            [0.71, 0.68, 0.15, 0.12]]),
}

results = []
for arm in ("A0", "A4", "A2", "A1", "A1b", "A3"):
    spec = v9.ARMS[arm]
    print(f"\n── {arm}  {spec['desc']}")
    hp = v9.install_arm(arm, epochs=60)

    # 建構
    cfg = v9.make_arm_yaml(**spec["arch"])
    model = DetectionModel(cfg=cfg, ch=3, nc=8, verbose=False)

    # 體系與層別
    det = model.model[-1]
    check(getattr(det, "end2end", False), f"{arm}: Detect 不是 end2end")
    check(det.reg_max == 1, f"{arm}: reg_max={det.reg_max}，應為 1")
    check(det.nc == 8 and det.nl == 4, f"{arm}: nc={det.nc} nl={det.nl}")
    strides = [int(s) for s in det.stride]
    check(strides == [4, 8, 16, 32], f"{arm}: strides={strides}")

    types = {i: m.type.split(".")[-1] for i, m in enumerate(model.model)}
    want_adown = [i for i, _ in v9.ADOWN_LAYERS] if spec["arch"]["adown"] else []
    want_stb = [i for i, _ in v9.STB_LAYERS] if spec["arch"]["stb"] else []
    for i in want_adown:
        check(types[i] == "ADown", f"{arm}: 第 {i} 層是 {types[i]}，應為 ADown")
    for i in want_stb:
        check(types[i] == "StarTripletBlock",
              f"{arm}: 第 {i} 層是 {types[i]}，別名注入沒生效")
    if not spec["arch"]["stb"]:
        check("StarTripletBlock" not in types.values(),
              f"{arm}: 不該出現 StarTripletBlock（前一臂的注入沒還原）")

    # loss 是否照該臂的設定安裝
    patched = ultralytics.utils.loss.bbox_iou.__name__ == "bbox_wise_inner_mpdiou"
    check(patched == (spec["loss_ratio"] is not None),
          f"{arm}: 自訂 loss 安裝狀態不符（patched={patched}）")
    if patched:
        check(math.isclose(v9.CFG["ratio"], spec["loss_ratio"]),
              f"{arm}: ratio={v9.CFG['ratio']}，應為 {spec['loss_ratio']}")

    # forward
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))

    # 完整損失路徑
    model.args = get_cfg(overrides={"box": hp["box"], "cls": hp["cls"], "dfl": hp["dfl"]})
    model.train()
    loss, items = model.loss(fake_batch)
    vals = ({k: float(v) for k, v in items.items()} if isinstance(items, dict)
            else {k: float(v) for k, v in zip(("box", "cls", "l1"), items.flatten())})
    check(bool(torch.isfinite(loss).all()), f"{arm}: 總損失 NaN/Inf")
    check(all(math.isfinite(v) for v in vals.values()), f"{arm}: 分項損失 NaN/Inf {vals}")

    # 成本
    model.eval()
    p, f = get_num_params(model), get_flops(model, imgsz=640)
    exp_p, exp_f = EXPECTED[arm]
    check(p == exp_p, f"{arm}: params={p:,}，預期 {exp_p:,}")
    check(abs(f - exp_f) < 0.05, f"{arm}: GFLOPs={f:.2f}，預期 {exp_f:.2f}")

    results.append((arm, p, f, hp, vals, float(loss.sum().detach())))
    print(f"    建構/層別/forward/損失 OK   params={p:,}  GFLOPs={f:.2f}")
    print(f"    " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
          + f"  | total={float(loss.sum().detach()):.4f}")
    v9.restore()

# ══════════════════════════════════════════════════════════════════════════
section("消融設計總表")
# ══════════════════════════════════════════════════════════════════════════
print(f"{'臂':<6}{'params':>11}{'GFLOPs':>9}{'imgsz':>7}{'box':>6}{'close_mos':>11}"
      f"{'patience':>10}{'s/epoch':>9}{'60 輪時數':>11}")
total_h = 0.0
for arm, p, f, hp, _v, _t in results:
    h = v9.ARMS[arm]["s_per_epoch"] * hp["epochs"] / 3600
    total_h += h
    print(f"{arm:<6}{p:>11,}{f:>9.2f}{hp['imgsz']:>7}{hp['box']:>6.1f}"
          f"{hp['close_mosaic']:>11}{hp['patience']:>10}"
          f"{v9.ARMS[arm]['s_per_epoch']:>9}{h:>10.2f} h")
print(f"\n  六臂合計 {total_h:.2f} GPU-h（Kaggle 單週配額 30 h）")
print(f"  建議順序 A0 → A4 → A2 → A1 → A1b → A3；"
      f"前四臂 {sum(v9.ARMS[a]['s_per_epoch'] for a in ('A0','A4','A2','A1'))*60/3600:.2f} h")

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
if FAIL:
    print(f"  驗收未通過，{len(FAIL)} 項問題：")
    for m in FAIL:
        print(f"    ✗ {m}")
    print(BAR)
    sys.exit(1)
print("  驗收通過：六臂全部可建構、可前向、可算出有限損失；三項數值修正均生效。")
print(BAR)
