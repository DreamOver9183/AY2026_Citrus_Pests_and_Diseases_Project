# -*- coding: utf-8 -*-
"""消融各臂的判準比較表。

讀取 `Train Code/v9/Train_output/ablation_*.zip`（Step.7 產出的分析包），
以 A0 為基準套用 docs/v9_記錄_實驗重整與六臂消融.md 階段 1 的決策規則：

  主判準  mAP@0.5:0.95 的平台期平均（60 輪取最後 16 輪）
  門檻    需超出 A0 達 2σ 才納入；1σ 以內記為無差異
  σ       用 A0 自己平台期的標準差，不沿用 v8 時代的舊值
  次判準  Thrips AP@0.5、Scale_Insect FP 數、s/epoch

用法（專案根目錄下）：
    .venv/Scripts/python.exe tools/compare_arms.py
    .venv/Scripts/python.exe tools/compare_arms.py --dir <放 zip 的目錄>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO / "Train Code" / "v9" / "Train_output"
ORDER = ["A0", "A4", "A2", "A1", "A1b", "A3"]
PRIMARY = "mAP50_95"
BAR = "═" * 96


def load_zip(path: Path) -> dict:
    """從一個 ablation_{ARM}.zip 讀出 summary.json 與 per_class.csv。"""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base == "summary.json":
                out["summary"] = json.loads(z.read(name).decode("utf-8"))
            elif base == "per_class.csv":
                rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))
                out["per_class"] = {r["class"]: r for r in rows}
    if "summary" not in out:
        raise ValueError(f"{path.name} 裡沒有 summary.json")
    return out


def verdict(delta: float, sigma: float) -> tuple[str, str]:
    """依決策規則判定。回傳 (標記, 說明)。

    正負兩側都要標清楚：−1.9σ 若寫成「不足」會被誤讀成「正向但差一點」，
    實際上它是「接近顯著變差」。
    """
    if sigma <= 0:
        return "?", "σ 無效"
    n = delta / sigma
    if n >= 2:
        return "納入", f"+{n:.1f}σ ≥ 2σ"
    if n <= -2:
        return "顯著變差", f"{n:.1f}σ ≤ −2σ"
    if n >= 1:
        return "正向不足", f"+{n:.1f}σ（未達 2σ）"
    if n <= -1:
        return "疑似變差", f"{n:.1f}σ（接近 −2σ）"
    return "無差異", f"{n:+.1f}σ（<1σ）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    src = Path(args.dir)

    arms: dict[str, dict] = {}
    for p in sorted(src.glob("ablation_*.zip")):
        arm = p.stem.replace("ablation_", "")
        arms[arm] = load_zip(p)

    if "A0" not in arms:
        print(f"找不到 ablation_A0.zip（{src}）—— A0 是基準，必須先跑。")
        return 1

    base = arms["A0"]["summary"]["plateau"]
    b_mean = base[PRIMARY]["mean"]
    sigma = base[PRIMARY]["std"]
    bar2 = b_mean + 2 * sigma

    present = [a for a in ORDER if a in arms] + [a for a in arms if a not in ORDER]

    print(BAR)
    print(f"  消融判準表　　基準 A0：{PRIMARY} 平台期 = {b_mean:.5f} ± {sigma:.5f}"
          f"　　2σ 門檻 = {bar2:.5f}（增益 ≥ +{2 * sigma:.4f}）")
    print(BAR)
    print(f"{'臂':<5}{'說明':<34}{'mAP50-95':>10}{'Δ vs A0':>10}{'σ 倍數':>9}"
          f"{'判定':>8}{'mAP50':>9}{'s/ep':>8}{'成本':>7}")
    print("─" * 96)

    a0_spe = base.get("s_per_epoch") or 0
    for arm in present:
        s = arms[arm]["summary"]
        pl = s["plateau"]
        mean = pl[PRIMARY]["mean"]
        d = mean - b_mean
        spe = pl.get("s_per_epoch") or 0
        cost = f"{spe / a0_spe:.2f}x" if a0_spe else "—"
        if arm == "A0":
            mark, why = "基準", ""
        else:
            mark, why = verdict(d, sigma)
        desc = s["desc"][:32]
        print(f"{arm:<5}{desc:<34}{mean:>10.5f}{d:>+10.5f}{d / sigma:>+8.1f}σ"
              f"{mark:>8}{pl['mAP50']['mean']:>9.5f}{spe:>8.1f}{cost:>7}")

    # ── 次判準 ────────────────────────────────────────────────────────
    print(f"\n{'臂':<5}{'Thrips AP50':>13}{'Δ':>9}{'Scale FP':>10}{'Δ':>7}"
          f"{'Leaf_Miner AP50':>17}{'Δ':>9}{'最佳 conf':>10}")
    print("─" * 96)
    a0p = arms["A0"].get("per_class", {})
    a0_fp = arms["A0"]["summary"]["fp_from_background"]
    for arm in present:
        s = arms[arm]["summary"]
        pc = arms[arm].get("per_class", {})
        def ap(cls):
            return float(pc[cls]["ap50"]) if cls in pc else float("nan")
        def ap0(cls):
            return float(a0p[cls]["ap50"]) if cls in a0p else float("nan")
        fp = s["fp_from_background"].get("Scale_Insect", 0)
        print(f"{arm:<5}{ap('Thrips'):>13.4f}{ap('Thrips') - ap0('Thrips'):>+9.4f}"
              f"{fp:>10}{fp - a0_fp.get('Scale_Insect', 0):>+7}"
              f"{ap('Citrus_Leaf_Miner'):>17.4f}"
              f"{ap('Citrus_Leaf_Miner') - ap0('Citrus_Leaf_Miner'):>+9.4f}"
              f"{s.get('best_f1_conf', float('nan')):>10.4f}")

    # ── 逐類 AP50 全表 ────────────────────────────────────────────────
    classes = list(a0p.keys())
    if classes:
        print(f"\n{'類別':<20}" + "".join(f"{a:>10}" for a in present))
        print("─" * (20 + 10 * len(present)))
        for c in classes:
            line = f"{c:<20}"
            for arm in present:
                pc = arms[arm].get("per_class", {})
                line += f"{float(pc[c]['ap50']):>10.4f}" if c in pc else f"{'—':>10}"
            print(line)

    # ── 結論 ──────────────────────────────────────────────────────────
    print(f"\n{BAR}")
    passed = [a for a in present if a != "A0"
              and arms[a]["summary"]["plateau"][PRIMARY]["mean"] >= bar2]
    todo = [a for a in ORDER if a not in arms]
    if passed:
        print(f"  通過 2σ 門檻：{', '.join(passed)}")
    else:
        done = [a for a in present if a != "A0"]
        print(f"  尚無任何臂通過 2σ 門檻"
              + (f"（已測：{', '.join(done)}）" if done else ""))
    if todo:
        print(f"  尚未執行：{', '.join(todo)}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
