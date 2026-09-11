# -*- coding: utf-8 -*-
"""部署線的止損判準（v12.2）＝即時辨識功能的採用門檻。

教授的說法：達不到這個標準，就不採用即時辨識的功能（App 只做拍照辨識）。

讀取匯出報告（精度）與手機量測（延遲），套用
docs/v12.2_計畫_部署線止損標準.md 預先登記的過線標準與停損點。
與 compare_arms.py 同一個用意：**門檻寫死在程式裡，看結果之前就定好**。

  過線   延遲 ≤ 40 ms（end2end=False 另加 2 ms 後處理保留）
         且 test mAP50 ≥ 0.764
         且沒有任何一類 AP50 比 fp32@320 低超過 0.10
  停損   S1 有組合過線 → 採用即時辨識（用其中 mAP50 最高者），停止搜尋
         S2 第一階段量完仍沒有組合還有機會（或第二階段那一次重訓仍未過）
            → 不採用即時辨識，結案
         S3 過了 2026-09-22 → 無論進度如何都不採用即時辨識，結案

常數於 2026-09-11 登記，早於任何新量測。看到結果後若要改，
必須在計畫書 §6 記一筆「事後修改」並寫明理由，不能只改這裡的數字。

用法（專案根目錄下）：
    .venv/Scripts/python.exe tools/judge_deploy_stoploss.py
    .venv/Scripts/python.exe tools/judge_deploy_stoploss.py --only i224,i256,i288
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import dataset_paths as P  # noqa: E402

MODEL_DIR = P.REPO / "Benchmark" / "Model"
REPORT_DIR = P.REPO / "Benchmark" / "report"
BAR = "═" * 100

# ── 過線標準（三條都要過）─────────────────────────────────────────────
LAT_LIMIT_MS = 40.0       # 30 FPS −5 的上緣。比 28.6 ms 更快不算失格
E2E0_RESERVE_MS = 2.0     # end2end=False 的圖不含後處理，NMS 要在 App 端做。
                          # 圖內 topk 在 320 實測佔 1.6 ms（66.4 vs 64.8），進位成 2
MAP50_FLOOR = 0.764       # 現行可交付的 fp32@320（0.78511）− run 間全距 0.021
CLASS_MAX_DROP = 0.10     # v5.6 評估集設計的解析度（±2SE ≤ 0.10）
# 逐類的參照：320 的組合裡唯一存了逐類 AP50 的一筆（fp32__i320__e2e1__md100）
REF_PER_CLASS = {
    "Oily_Spot": 0.98556, "Canker": 0.86203, "Sooty_Mold": 0.995,
    "Black_Spot": 0.97322, "Scale_Insect": 0.54852, "Citrus_Leaf_Miner": 0.62233,
    "Thrips": 0.7206, "Aphid": 0.87264, "Thrips_Damage": 0.43556,
}

# ── 量測協定：只有這組設定的數字算數 ─────────────────────────────────
REQ_SETTING, REQ_THREADS, REQ_RUNS = "cpu", 4, 25
REQ_REPEATS = 3           # 判「未過」需要 ≥3 輪（取最小值）
MAX_SPREAD = 0.39         # 實測過的最大熱散佈（md50@320 的 [67.58, 94.17]）

# ── 計畫書 §3.3 的路線判定 ───────────────────────────────────────────
# 這些變體的關閉理由在 v12.1 的文字分析裡，不在量測 JSON 裡
# （w8a16 在 JSON 中只有一筆「無結果」），所以要明寫，否則會被當成「還沒量」
CLOSED_VARIANTS = {
    "w8a16": "v12.1 已關：XNNPACK 沒有 INT16 核心，慢 27 倍",
    "w8a32": "v12.1 已關：只省體積，不省時間",
}
# 第一階段登記的 5 個候選（v11.5 last.pt 匯出）。沒匯出完之前不判 S2
PHASE1 = ("fp32__i224__e2e1", "fp32__i224__e2e0",
          "int8__i288__e2e0", "int8__i256__e2e0", "int8__i224__e2e0")
PHASE1_PREFIX = "last"

# ── 停損點 ───────────────────────────────────────────────────────────
NEAR_MISS = 0.032         # 第二階段准入：延遲已過、mAP50 差距在此之內。
                          # = 本模型 640→320 的整段損失（PyTorch 0.80897 → 0.77663）
DEADLINE = date(2026, 9, 22)


def load_latency() -> dict[str, dict]:
    """把所有符合協定的手機量測按模型彙整。

    跨檔合併後取最小值：干擾只會讓數字變大，多一輪只會更接近乾淨的值。
    帶 `--extra-flags` 的量測（例如 force_fp16）不是同一個組合，一律不算。
    """
    lat: dict[str, dict] = {}
    for js in sorted(REPORT_DIR.glob("benchmark_*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        rows = d.get("results") if isinstance(d, dict) else None
        if not isinstance(rows, list):
            continue
        if not (d.get("threads") == REQ_THREADS and d.get("num_runs") == REQ_RUNS
                and not d.get("extra_flags")):
            continue
        for r in rows:
            if r.get("setting") != REQ_SETTING:
                continue
            e = lat.setdefault(r["model"], {"ms": [], "lower_bound": None})
            if r.get("ok"):
                e["ms"] += r.get("all_ms") or [r["avg_ms"]]
            elif r.get("too_slow"):
                lb = r["lower_bound_ms"]
                e["lower_bound"] = lb if e["lower_bound"] is None else min(lb, e["lower_bound"])
    return lat


def load_accuracy() -> list[dict]:
    """讀 Benchmark/Model/*__export_report.json 裡每個有 mAP 的組合。"""
    rows = []
    for rep in sorted(MODEL_DIR.glob("*__export_report.json")):
        prefix = rep.name[: -len("__export_report.json")]
        d = json.loads(rep.read_text(encoding="utf-8"))
        for tag, r in d.get("results", {}).items():
            val = r.get("val") or {}
            if not r.get("exported") or "mAP50" not in val:
                continue
            rows.append({"model": r.get("file") or f"{prefix}__{tag}.tflite",
                         "variant": r.get("variant"), "end2end": bool(r.get("end2end")),
                         "mAP50": val["mAP50"], "per_class": val.get("per_class_ap50")})
    return rows


def judge_latency(row: dict, lat: dict) -> tuple[str, float | None, int]:
    """回傳 (判定, 有效延遲, 輪數)。有效延遲 = 最小值 + 後處理保留。"""
    reserve = 0.0 if row["end2end"] else E2E0_RESERVE_MS
    e = lat.get(row["model"])
    if not e or (not e["ms"] and e["lower_bound"] is None):
        return "未量", None, 0
    if not e["ms"]:
        # 連 25 輪都跑不完，只有下界——下界都已經遠超 40 ms
        return "未過", e["lower_bound"] + reserve, 0
    eff, n = min(e["ms"]) + reserve, len(e["ms"])
    if eff <= LAT_LIMIT_MS:
        return "通過", eff, n       # 干擾只會讓數字變大，單輪過線即成立
    # 判未過要夠確定：≥3 輪，或就算扣掉實測最大散佈仍超過
    if n >= REQ_REPEATS or eff / (1 + MAX_SPREAD) > LAT_LIMIT_MS:
        return "未過", eff, n
    return "需補量", eff, n


def judge_classes(pc: dict | None) -> tuple[str, list[str]]:
    if not pc:
        return "無資料", []
    bad = [f"{c} −{REF_PER_CLASS[c] - v:.3f}" for c, v in pc.items()
           if c in REF_PER_CLASS and REF_PER_CLASS[c] - v > CLASS_MAX_DROP]
    return ("未過", bad) if bad else ("通過", [])


def main() -> int:
    ap = argparse.ArgumentParser(description="部署線止損判準（v12.2）")
    ap.add_argument("--only", default="",
                    help="逗號分隔的子字串，只看檔名含其一的模型（例：i224,i256,i288）")
    a = ap.parse_args()
    only = [s.strip() for s in a.only.split(",") if s.strip()]

    lat = load_latency()
    everything = load_accuracy()
    exported = {r["model"] for r in everything}
    missing = [t for t in PHASE1 if f"{PHASE1_PREFIX}__{t}.tflite" not in exported]
    rows = [r for r in everything if not only or any(o in r["model"] for o in only)]
    if not rows:
        raise SystemExit(f"✗ {MODEL_DIR} 底下沒有符合條件、且含精度的匯出紀錄。"
                         "先跑 tools/export_tflite.py --val。")

    for r in rows:
        r["lat_v"], r["lat_ms"], r["n"] = judge_latency(r, lat)
        r["acc_v"] = "通過" if r["mAP50"] >= MAP50_FLOOR else "未過"
        r["cls_v"], r["cls_bad"] = judge_classes(r["per_class"])
        vs = (r["lat_v"], r["acc_v"], r["cls_v"])
        if r["variant"] in CLOSED_VARIANTS:
            r["verdict"] = "✗ 已關"
        else:
            r["verdict"] = ("✓ 過線" if all(v == "通過" for v in vs)
                            else "✗ 未過" if "未過" in vs else "… 待定")
        r["near"] = (r["variant"] == "fp32" and r["lat_v"] == "通過"
                     and MAP50_FLOOR - NEAR_MISS <= r["mAP50"] < MAP50_FLOOR)

    rows.sort(key=lambda r: (r["lat_ms"] is None, r["lat_ms"] or 0.0))
    print(BAR)
    print(f"  過線：延遲 ≤ {LAT_LIMIT_MS:.0f} ms（e2e0 另加 {E2E0_RESERVE_MS:.0f}）"
          f"　mAP50 ≥ {MAP50_FLOOR}　逐類跌幅 ≤ {CLASS_MAX_DROP:.2f}（對 fp32@320）")
    print(BAR)
    print(f"  {'模型':<42}{'有效ms':>8}{'輪':>4}{'延遲':>6}{'mAP50':>9}{'精度':>6}{'逐類':>6}  總判")
    print(BAR)
    for r in rows:
        ms = f"{r['lat_ms']:.1f}" if r["lat_ms"] is not None else "—"
        print(f"  {r['model']:<42}{ms:>8}{r['n']:>4}{r['lat_v']:>6}{r['mAP50']:>9.5f}"
              f"{r['acc_v']:>6}{r['cls_v']:>6}  {r['verdict']}")
        if r["verdict"] == "✗ 已關":
            print(f"  {'':<42}└ {CLOSED_VARIANTS[r['variant']]}")
        elif r["cls_bad"]:
            print(f"  {'':<42}└ 逐類否決：{'、'.join(r['cls_bad'])}")
    print(BAR)

    passed = [r for r in rows if r["verdict"] == "✓ 過線"]
    near = [r for r in rows if r["near"] and r["verdict"] != "✗ 已關"]
    # 還沒有可下結論的延遲、且精度面仍有機會的組合，才值得接手機去量
    worth = [r for r in rows if r["verdict"] == "… 待定"
             and ((r["acc_v"] == "通過" and r["cls_v"] != "未過")
                  or (r["variant"] == "fp32" and r["mAP50"] >= MAP50_FLOOR - NEAR_MISS))]
    today = date.today()
    if passed:
        best = max(passed, key=lambda r: r["mAP50"])
        print(f"  S1 觸發：{len(passed)} 組過線 → 採用即時辨識，用 mAP50 最高的一組："
              f"{best['model']}（{best['lat_ms']:.1f} ms / mAP50 {best['mAP50']:.3f}）")
    elif today > DEADLINE:
        print(f"  S3 觸發：已過期限 {DEADLINE} → 不採用即時辨識，App 只做拍照辨識（計畫書 §5）")
    else:
        print(f"  尚未觸發停損點。距期限 {DEADLINE} 還有 {(DEADLINE - today).days} 天。")
        if missing:
            print(f"  第一階段尚未匯出（{len(missing)}/{len(PHASE1)}）：" + "、".join(missing))
        if worth:
            print("  值得上手機量：" + "、".join(r["model"] for r in worth))
        if near:
            print("  第二階段准入（原生低解析度重訓一次）：" + "、".join(r["model"] for r in near))
        if not (missing or worth or near):
            print("  S2：第一階段已量完，沒有組合還有過線的機會。"
                  "若第二階段也已用掉或無人准入 → 不採用即時辨識，結案")
    return 0


if __name__ == "__main__":
    sys.exit(main())
