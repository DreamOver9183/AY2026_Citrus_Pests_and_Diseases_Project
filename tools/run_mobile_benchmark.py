# -*- coding: utf-8 -*-
r"""在實體 Android 裝置上量測 `.tflite` 的推論延遲，並產出報告。

把先前散在 scratchpad 的一次性腳本收成可重複執行的工具。
流程與判準沿用 `.claude/skills/tflite_mobile_benchmark/SKILL.md`。

用法（專案根目錄）：

    python tools/run_mobile_benchmark.py --models "last__*.tflite" --settings cpu,gpu
    python tools/run_mobile_benchmark.py --models last__fp32__i640__e2e1 --settings cpu \
        --extra-flags "--xnnpack_force_fp16=true" --repeat 2

──────────────────────────────────────────────────────────────────────
兩個非踩不可的坑（都實際踩過）
──────────────────────────────────────────────────────────────────────
**1. `MSYS_NO_PATHCONV=1`**

從 Git Bash 呼叫 adb 時，MSYS 會把 `/data/local/tmp/` 這種**裝置端**路徑
改寫成 Windows 路徑（`C:/Program Files/Git/data/local/tmp/`）。失敗是**靜默**的：

    $ adb push model.tflite /data/local/tmp/
    model.tflite: 1 file pushed, 0 skipped. 25.4 MB/s      ← 回報成功
    $ adb shell ls /data/local/tmp/*.tflite
    （檔案不在裡面）

所以本腳本一律用 `subprocess` 直接呼叫 adb.exe（繞過 shell），
且**推送後一定用 `adb shell ls` 實際確認**，不相信 push 的回報。

**2. `--es args` 仍然要自己加引號**

一開始以為用 subprocess 的 list 形式就能避開引號問題。**不對。**
`adb shell` 不保留參數邊界：它把 argv 用空白接成一個字串丟給**裝置端的 shell**
重新解析。所以那一整串 benchmark 參數到了手機上會被拆開，`am` 只收到
`--graph=...`，其餘變成散落的 token —— benchmark 根本沒啟動，logcat 空的，
而且**沒有任何錯誤訊息**。

正確寫法是把整串包一層引號，讓裝置端的 shell 還原成單一 token：

    adb("shell", "am", "start", "-S", "-n", ACTIVITY, "--es", "args", f'"{args}"')

subprocess 的 list 形式仍然值得用，但它擋掉的是**本機**的 shell，
不是裝置端那一層。

──────────────────────────────────────────────────────────────────────
熱漂移
──────────────────────────────────────────────────────────────────────
連續量測會讓機身升溫，後面的組別偏慢。差距是數倍時無所謂，
但**要比較接近的組合就必須 `--repeat 3` 以上**：本腳本會把同一組設定
分散在不同輪次執行（而不是連著跑兩次），再**取最小值**。

取最小值而非中位數，是因為干擾只會讓數字變大不會變小。
實測 `md50@320` 兩輪是 `[67.58, 94.17]`（散佈 39%）——n=2 時中位數等於平均數，
一次熱尖峰就能憑空造出一個 20% 的「差異」。散佈超過 10% 的列會標 ⚠，
那種數字不可用於接近組合的比較。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dataset_paths as P  # noqa: E402

ADB = P.REPO / "Benchmark" / "platform-tools" / "adb.exe"
APK = P.REPO / "Benchmark" / "platform-tools" / "android_aarch64_benchmark_model.apk"
MODEL_DIR = P.REPO / "Benchmark" / "Model"
REPORT_DIR = P.REPO / "Benchmark" / "report"
DEVICE_TMP = "/data/local/tmp"
ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"

# 目標 30 FPS ±5
TARGET_LO_MS, TARGET_HI_MS = 28.6, 40.0

SETTINGS = {
    "cpu": "--use_gpu=false --use_nnapi=false",
    "gpu": "--use_gpu=true --use_nnapi=false",
    "nnapi": "--use_gpu=false --use_nnapi=true",
}


def adb(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """直接呼叫 adb.exe（不經 shell），因此不受 MSYS 路徑改寫影響。"""
    return subprocess.run([str(ADB), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


# ══════════════════════════════════════════════════════════════════════
#  Phase 1 — 前置驗證（硬邊界，任何一項失敗都要停）
# ══════════════════════════════════════════════════════════════════════
def device_gate() -> dict:
    """確認裝置已連線且**已授權**，順便取回規格。"""
    out = adb("devices", "-l").stdout
    rows = [l for l in out.splitlines()[1:] if l.strip()]
    if not rows:
        raise SystemExit("✗ 沒有偵測到裝置。確認 USB 已接、開發者選項的 USB 偵錯已開。")
    first = rows[0].split()
    serial, state = first[0], first[1]
    if state == "unauthorized":
        raise SystemExit(
            f"✗ 裝置 {serial} 尚未授權。\n"
            "  請在**已解鎖**的手機上接受「允許 USB 偵錯」並勾選「一律允許」。\n"
            "  沒跳出對話框就把 USB 模式改成「檔案傳輸」，或到開發者選項\n"
            "  「撤銷 USB 偵錯授權」後重新插拔。")
    if state != "device":
        raise SystemExit(f"✗ 裝置 {serial} 狀態是 {state}，需要 device。重新插拔 USB 再試。")

    props = {}
    for k in ("ro.product.model", "ro.product.manufacturer", "ro.board.platform",
              "ro.soc.model", "ro.build.version.release", "ro.product.cpu.abi"):
        props[k] = adb("shell", "getprop", k).stdout.strip()
    print(f"▷ 裝置 {serial}  {props['ro.product.manufacturer']} {props['ro.product.model']}"
          f"  SoC {props['ro.soc.model']}  Android {props['ro.build.version.release']}")
    return {"serial": serial, **props}


def ensure_apk() -> None:
    if not APK.is_file():
        raise SystemExit(f"✗ 找不到 APK：{APK}")
    installed = adb("shell", "pm", "list", "packages").stdout
    if "org.tensorflow.lite.benchmark" in installed:
        print("▷ benchmark APK 已安裝")
        return
    print("▷ 安裝 benchmark APK …")
    r = adb("install", "-r", "-d", "-g", str(APK), timeout=300)
    if "Success" not in r.stdout:
        raise SystemExit(f"✗ APK 安裝失敗：{r.stdout.strip()} {r.stderr.strip()}")


def push_model(local: Path) -> str:
    """推送並**實際確認檔案在裝置上**——push 的 success 訊息不可信。"""
    remote = f"{DEVICE_TMP}/{local.name}"
    adb("push", str(local), remote, timeout=300)
    listed = adb("shell", "ls", "-l", remote).stdout.strip()
    if local.name not in listed:
        raise SystemExit(
            f"✗ {local.name} 推送後不在裝置上。\n"
            f"  adb shell ls 回傳：{listed!r}\n"
            "  若你是從 Git Bash 手動跑 adb，記得 MSYS_NO_PATHCONV=1。")
    size = int(listed.split()[4]) if len(listed.split()) > 4 else -1
    if size != local.stat().st_size:
        raise SystemExit(f"✗ {local.name} 大小不符：裝置 {size} vs 本機 {local.stat().st_size}")
    return remote


# ══════════════════════════════════════════════════════════════════════
#  Phase 3/4 — 執行與解析
# ══════════════════════════════════════════════════════════════════════
def run_once(remote: str, setting: str, num_runs: int, threads: int,
             extra: str, wait_s: int) -> tuple[dict, str]:
    args = (f"--graph={remote} --num_runs={num_runs} --num_threads={threads} "
            f"{SETTINGS[setting]} {extra}").strip()
    adb("logcat", "-c")
    # **必須自己加引號。** `adb shell` 不會保留參數邊界：它把 argv 用空白接成
    # 一個字串丟給裝置端的 shell 重新解析。所以即使用 subprocess 的 list 形式，
    # args 那一整串到了手機上仍會被拆開，`am` 只會收到 `--graph=...`，
    # 其餘變成散落的 token —— 結果是 benchmark 根本沒跑，logcat 空的。
    # 包一層雙引號讓裝置端的 shell 把它還原成單一 token。
    adb("shell", "am", "start", "-S", "-n", ACTIVITY, "--es", "args", f'"{args}"')
    time.sleep(wait_s)
    log = adb("logcat", "-d", "-s", "tflite").stdout
    return parse_log(log), log


def parse_log(log: str) -> dict:
    """依 references/log_parsing_guide.md 的規範解析。單位 us → ms。"""
    r: dict = {"delegates": [], "unsupported_ops": []}
    for line in log.splitlines():
        s = line.split("tflite  : ")[-1].strip()
        if m := re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]+)\)", s):
            r["delegates"].append({"name": m.group(3), "replaced": int(m.group(1)),
                                   "total": int(m.group(2)),
                                   "rate": round(int(m.group(1)) / int(m.group(2)) * 100, 1)})
        if m := re.search(r"Inference timings in us: Init: ([\d.e+]+), First inference: ([\d.e+]+), "
                          r"Warmup \(avg\): ([\d.e+]+), Inference \(avg\): ([\d.e+]+)", s):
            r["init_us"], r["first_us"], r["warmup_us"], r["avg_us"] = (float(x) for x in m.groups())
        if m := re.search(r"count=(\d+) first=[\d.e+]+ curr=[\d.e+]+ min=([\d.e+]+) "
                          r"max=([\d.e+]+) avg=([\d.e+]+) std=([\d.e+]+)", s):
            if int(m.group(1)) >= 25:               # 只取正式那一輪，不要 warmup
                r["min_us"], r["max_us"], r["std_us"] = (float(m.group(i)) for i in (2, 3, 5))
        # 極慢的模型連 warmup 都跑不完 25 輪（benchmark 工具本身有 150 秒上限）。
        # 這種情況下只有 `count=1 curr=...`，沒有 "Inference timings" 那行。
        # 抓下來當**下界**回報，比「無結果」有用得多——w8a16 就是這樣被發現
        # 每張要 7.2 秒（XNNPACK 沒有 INT16 啟動值的核心，整張圖退回參考實作）。
        if m := re.search(r"count=1 curr=([\d.e+]+)", s):
            r["warmup_only_us"] = float(m.group(1))
        if m := re.search(r"Memory footprint delta.*init=([\d.]+) overall=([\d.]+)", s):
            r["mem_init_mb"], r["mem_overall_mb"] = float(m.group(1)), float(m.group(2))
        # delegate 沒吃下整張圖時，這一行說明了為什麼
        if "not supported by GPU delegate" in s:
            r["_collect_unsupported"] = True
            continue
        if r.get("_collect_unsupported"):
            if re.match(r"^[A-Z_0-9]+:", s):
                r["unsupported_ops"].append(s)
            else:
                r.pop("_collect_unsupported", None)
        # NNAPI 明說它沒有真的執行這張圖時，那一列就不是 NNAPI 的成績
        if "will not be executed by the delegate" in s:
            r["delegate_inactive"] = True
        if "NNAPI accelerators available" in s:
            r["nnapi_accelerators"] = s.split(":", 1)[-1].strip()
    r.pop("_collect_unsupported", None)
    return r


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Android 端 TFLite 延遲量測")
    ap.add_argument("--models", default="*.tflite",
                    help="Benchmark/Model/ 底下的 glob 或逗號分隔的檔名（可省略 .tflite）")
    ap.add_argument("--settings", default="cpu,gpu",
                    help="逗號分隔：" + "、".join(SETTINGS))
    ap.add_argument("--num-runs", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extra-flags", default="", help="額外傳給 benchmark 的旗標")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每組跑幾輪取中位數。**要比較接近的組合時務必 ≥2**——"
                         "本腳本會把重複分散到不同輪次，抵銷機身升溫")
    ap.add_argument("--wait", type=int, default=0,
                    help="每次測試後等待秒數，0 = 依模型大小自動決定")
    ap.add_argument("--tag", default="", help="報告檔名的後綴")
    a = ap.parse_args()

    settings = [s.strip() for s in a.settings.split(",") if s.strip()]
    if bad := [s for s in settings if s not in SETTINGS]:
        raise SystemExit(f"✗ 未知的設定：{bad}。可用：{list(SETTINGS)}")

    if "*" in a.models or "?" in a.models:
        models = sorted(MODEL_DIR.glob(a.models))
    else:
        models = []
        for n in a.models.split(","):
            n = n.strip()
            p = MODEL_DIR / (n if n.endswith(".tflite") else f"{n}.tflite")
            if not p.is_file():
                raise SystemExit(f"✗ 找不到 {p}")
            models.append(p)
    if not models:
        raise SystemExit(f"✗ {MODEL_DIR} 底下沒有符合 {a.models} 的 .tflite。\n"
                         "  先用 tools/export_tflite.py 轉出模型。")

    print("═" * 70)
    device = device_gate()
    ensure_apk()
    print(f"▷ 待測 {len(models)} 個模型 × {len(settings)} 種設定 × {a.repeat} 輪")
    print("═" * 70)

    remotes = {}
    for m in models:
        remotes[m.name] = push_model(m)
        print(f"▷ 已推送並確認 {m.name}  ({m.stat().st_size / 1048576:.2f} MB)")

    # 把重複分散到不同輪次，而不是連著跑同一組兩次
    samples: dict[tuple[str, str], list[dict]] = {}
    logs: dict[tuple[str, str], str] = {}
    for rep in range(a.repeat):
        for m in models:
            for st in settings:
                wait = a.wait or max(25, min(90, int(m.stat().st_size / 1048576 * 4) + 20))
                print(f"  [{rep + 1}/{a.repeat}] {m.name} @ {st} …", end="", flush=True)
                res, log = run_once(remotes[m.name], st, a.num_runs, a.threads,
                                    a.extra_flags, wait)
                if "avg_us" in res:
                    print(f" {res['avg_us'] / 1000:.1f} ms")
                else:
                    print(" ✗ 沒有推論結果")
                samples.setdefault((m.name, st), []).append(res)
                logs[(m.name, st)] = log

    rows = []
    for (name, st), reps in samples.items():
        ok = [r for r in reps if "avg_us" in r]
        if not ok:
            # 沒跑完 25 輪，但可能有 warmup 的單次計時可以當下界
            slow = [r["warmup_only_us"] for r in reps if "warmup_only_us" in r]
            row = {"model": name, "setting": st, "ok": False}
            if slow:
                lb = min(slow) / 1000
                row.update(too_slow=True, lower_bound_ms=round(lb, 1),
                           delegates=reps[0].get("delegates", []),
                           note=("連 warmup 都跑不完 25 輪（工具上限 150 秒）。"
                                 "下界取自單次 warmup 計時"))
            rows.append(row)
            continue
        # **取最小值，不是中位數。** 量測干擾（升溫、背景程式）只會讓數字變大，
        # 不會讓它變小，所以最小值才是最接近「乾淨」的一次。
        # 實測 md50@320 兩輪是 [67.58, 94.17]——散佈 39%，而 n=2 時中位數就是平均數，
        # 一次熱尖峰就把結果拉到 80.9 ms，足以憑空造出一個不存在的差異。
        all_ms = sorted(r["avg_us"] / 1000 for r in ok)
        avg_ms = all_ms[0]
        spread = (all_ms[-1] - all_ms[0]) / all_ms[0] if len(all_ms) > 1 else 0.0
        best = min(ok, key=lambda r: r["avg_us"])
        rows.append({
            "model": name, "setting": st, "ok": True, "repeats": len(ok),
            "avg_ms": round(avg_ms, 2), "fps": round(1000 / avg_ms, 2),
            "stat": "min of repeats",
            "spread": round(spread, 4),
            "all_ms": [round(r["avg_us"] / 1000, 2) for r in ok],
            "init_ms": round(best["init_us"] / 1000, 2),
            "first_ms": round(best["first_us"] / 1000, 2),
            "min_ms": round(best.get("min_us", 0) / 1000, 2),
            "max_ms": round(best.get("max_us", 0) / 1000, 2),
            "std_ms": round(best.get("std_us", 0) / 1000, 2),
            "delegates": best["delegates"],
            "delegate_inactive": best.get("delegate_inactive", False),
            "nnapi_accelerators": best.get("nnapi_accelerators"),
            "unsupported_ops": best["unsupported_ops"],
            "mem_overall_mb": best.get("mem_overall_mb"),
            "meets_target": TARGET_LO_MS <= avg_ms <= TARGET_HI_MS,
        })

    rows.sort(key=lambda r: r.get("avg_ms", r.get("lower_bound_ms", 1e9)))
    print("\n" + "═" * 96)
    print(f"  {'模型':<40}{'設定':<8}{'平均ms':>9}{'FPS':>8}{'節點替換':>26}{'達標':>6}")
    print("═" * 96)
    for r in rows:
        if not r["ok"]:
            if r.get("too_slow"):
                dg = (" / ".join(f"{d['name'].replace('TfLite', '').replace('Delegate', '')}"
                                 f" {d['rate']:.0f}%" for d in r.get("delegates", []))
                      or "無（整張圖走參考實作）")
                print(f"  {r['model']:<40}{r['setting']:<8}"
                      f"{'>' + str(r['lower_bound_ms']):>9}{'—':>8}{dg[:24]:>26}{'✗':>6}"
                      f"  太慢，未完成 25 輪")
            else:
                print(f"  {r['model']:<40}{r['setting']:<8}  ✗ 無結果")
            continue
        dg = " / ".join(f"{d['name'].replace('TfLite', '').replace('Delegate', '')} {d['rate']:.0f}%"
                        for d in r["delegates"]) or "無"
        mark = "✓" if r["meets_target"] else "✗"
        note = " ⚠未生效" if r["delegate_inactive"] else ""
        # 散佈太大代表這次量測受了干擾，該列不能拿去做接近的比較
        if r.get("spread", 0) > 0.10:
            note += f" ⚠散佈{r['spread']:.0%}"
        print(f"  {r['model']:<40}{r['setting']:<8}{r['avg_ms']:>9.1f}{r['fps']:>8.2f}"
              f"{dg[:24]:>26}{mark:>6}{note}")
    print("═" * 96)
    print(f"  目標 {TARGET_LO_MS}–{TARGET_HI_MS} ms/張（30 FPS ±5）")
    if ok_rows := [r for r in rows if r["ok"]]:
        b = ok_rows[0]
        print(f"  最快：{b['model']} @ {b['setting']}  {b['avg_ms']:.1f} ms = {b['fps']:.2f} FPS"
              f"  （距達標 {b['avg_ms'] / TARGET_HI_MS:.1f}x）")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat() + (f"_{a.tag}" if a.tag else "")
    js = REPORT_DIR / f"benchmark_{stamp}.json"
    js.write_text(json.dumps({"device": device, "settings": settings,
                              "num_runs": a.num_runs, "threads": a.threads,
                              "extra_flags": a.extra_flags, "repeat": a.repeat,
                              "results": rows}, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    raw = REPORT_DIR / f"benchmark_{stamp}_raw.log"
    raw.write_text("\n\n".join(f"{'=' * 70}\n{k[0]} @ {k[1]}\n{'=' * 70}\n{v}"
                               for k, v in logs.items()), encoding="utf-8")
    print(f"  數據 {js.relative_to(P.REPO)}")
    print(f"  原始 log {raw.relative_to(P.REPO)}")


if __name__ == "__main__":
    main()
