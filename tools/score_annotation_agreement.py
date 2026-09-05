# -*- coding: utf-8 -*-
"""兩人標註一致性評分（人工介入包 A 用）。

`Thrips_Damage` 的 R@0.75 只有 0.290–0.375——模型找得到病徵、但框畫不準。
這通常代表**既有的框本身就不一致**：同樣的受害程度，不同時候框的大小不同。

工作包 A 讓兩個人照同一份書面約定各標同樣 20 張，這支腳本量他們畫得像不像。
判準（沿用 docs/v11_計畫_資料擴充與切分重新設計.md §7）：

    中位 IoU >= 0.85  ->  邊界可以被一致定義，值得投入全類 208 張重標
    中位 IoU <  0.85  ->  這個類別的邊界本質上定不出來，應改為刪除該類別
                          （P3 §2.2 已分析過刪除的代價接近零）

配對方式是貪婪最大 IoU：兩邊都有的框才算數，落單的框（一個人畫了、
另一個人沒畫）以 IoU = 0 計入——**漏標也是不一致**，不能只算兩邊都畫到的部分。

用法：
    .venv/Scripts/python.exe tools/score_annotation_agreement.py 甲的資料夾 乙的資料夾
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

THRESHOLD = 0.85


def read_boxes(p: Path) -> list[tuple[float, float, float, float]]:
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8-sig").splitlines():
        q = ln.split()
        if len(q) < 5:
            continue
        cx, cy, w, h = (float(v) for v in q[1:5])
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-12 else 0.0


def match(A: list, B: list) -> list[float]:
    """貪婪配對，回傳每個框的 IoU。落單的框算 0。"""
    if not A and not B:
        return []
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)),
                   reverse=True)
    ua, ub, scores = set(), set(), []
    for v, i, j in pairs:
        if v <= 0 or i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        scores.append(v)
    scores += [0.0] * (len(A) - len(ua) + len(B) - len(ub))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", help="第一位標註者的『完成後放這裡』資料夾")
    ap.add_argument("dir_b", help="第二位標註者的『完成後放這裡』資料夾")
    args = ap.parse_args()
    da, db = Path(args.dir_a), Path(args.dir_b)
    for d in (da, db):
        if not d.is_dir():
            raise SystemExit(f"找不到資料夾：{d}")

    stems = sorted({p.stem for p in da.glob("*.txt")} | {p.stem for p in db.glob("*.txt")}
                   - {"classes"})
    if not stems:
        raise SystemExit("兩個資料夾裡都找不到 .txt 標註檔")

    print("═" * 70)
    print("  兩人標註一致性")
    print("═" * 70)
    print(f"\n{'照片':<10}{'甲的框數':>9}{'乙的框數':>9}{'中位 IoU':>10}  備註")

    all_scores, per_img = [], []
    for s in stems:
        A, Bx = read_boxes(da / f"{s}.txt"), read_boxes(db / f"{s}.txt")
        sc = match(A, Bx)
        all_scores += sc
        m = float(np.median(sc)) if sc else float("nan")
        per_img.append(m)
        note = ""
        if len(A) != len(Bx):
            note = f"框數不同（差 {abs(len(A) - len(Bx))} 個）"
        elif sc and min(sc) == 0:
            note = "有框完全對不上"
        print(f"{s:<10}{len(A):>9}{len(Bx):>9}{m:>10.3f}  {note}")

    med = float(np.median(all_scores)) if all_scores else 0.0
    n_img = sum(1 for m in per_img if not np.isnan(m))
    good = sum(1 for m in per_img if not np.isnan(m) and m >= THRESHOLD)

    print("\n" + "═" * 70)
    print(f"  照片數           {n_img}")
    print(f"  配對框數         {len(all_scores)}")
    print(f"  **全部框的中位 IoU  {med:.3f}**   （判準 {THRESHOLD}）")
    print(f"  單張達標比例     {good}/{n_img}")

    print("\n" + "─" * 70)
    if med >= THRESHOLD:
        print("結論：**兩人畫得夠像。**")
        print(f"      中位 IoU {med:.3f} >= {THRESHOLD}，代表這份標註約定講得夠清楚、")
        print("      不同的人照著做會得到差不多的框。")
        print("\n下一步：值得投入把 Thrips_Damage 全類 208 張依這份約定重標（屬 v5.7）。")
    else:
        print("結論：**兩人畫得不夠像。**")
        print(f"      中位 IoU {med:.3f} < {THRESHOLD}，代表照同一份約定做，")
        print("      不同的人還是會畫出差很多的框。")
        print("\n下一步：先看上面『備註』欄——")
        print("      * 多數是「框數不同」  -> 約定沒講清楚『什麼時候要分成兩個框』，")
        print("        補強規則之後可以再測一次。")
        print("      * 多數是框數相同但 IoU 低 -> 邊界本身就模糊，補規則也救不了。")
        print("        依 v11 §7 的中止條件，應改為**刪除 Thrips_Damage 這個類別**，")
        print("        而不是繼續投入重標。")


if __name__ == "__main__":
    main()
