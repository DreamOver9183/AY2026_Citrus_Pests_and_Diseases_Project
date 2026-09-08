// 柑橘病蟲害辨識 — 口頭報告簡報產生器
// 數據來源：docs/v12.1_報告_口頭簡報.md、docs/v12.1_報告_階段性進度週報.md §9
const pptxgen = require("pptxgenjs");

// ---- 調色盤：柑橘園（深葉綠主導 + 果橙點綴）------------------------------
const C = {
  dark:   "0F2E22",  // 深葉綠（暗底）
  mid:    "2C5F45",  // 中綠
  tint:   "EAF2EC",  // 淡綠卡片底
  tint2:  "F6F9F7",  // 更淡
  accent: "E8963C",  // 果橙（唯一銳利重點色）
  clay:   "B3453A",  // 陶紅（負向 / 未通過）
  white:  "FFFFFF",
  ink:    "1A2620",  // 正文
  muted:  "5E6F66",  // 次要文字
  line:   "C9DCD0",
};
const F = "Microsoft JhengHei";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.333 x 7.5 吋
pres.author = "Claude Code";
pres.title = "柑橘病蟲害辨識 階段性進度報告";

const W = 13.333, H = 7.5, M = 0.7;    // 邊界 0.7"

// ---- 共用元件 --------------------------------------------------------------
function darkBg(s) {
  s.background = { color: C.dark };
}
function lightBg(s) {
  s.background = { color: C.white };
}

// 標題（淺色版）
function title(s, text, sub) {
  s.addText(text, {
    x: M, y: 0.45, w: W - 2 * M, h: 0.72, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 34, bold: true, color: C.dark, align: "left",
  });
  if (sub) {
    s.addText(sub, {
      x: M, y: 1.19, w: W - 2 * M, h: 0.42, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 15, color: C.muted, align: "left",
    });
  }
}

// 圓形號碼徽章（視覺母題）
function badge(s, n, x, y, d, fill, txtColor) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color: fill || C.mid },
  });
  s.addText(String(n), {
    x, y, w: d, h: d, isTextBox: true, margin: 0,
    fontFace: F, fontSize: Math.round(d * 26), bold: true,
    color: txtColor || C.white, align: "center", valign: "middle",
  });
}

// 圓角卡片（視覺母題）
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: fill || C.tint },
    line: { color: fill || C.tint, width: 0 },
  });
}

function pageNote(s, txt) {
  s.addText(txt, {
    x: M, y: H - 0.62, w: W - 2 * M, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 10, color: C.muted, align: "left",
  });
}

const tblBase = {
  fontFace: F, fontSize: 12, color: C.ink, border: { type: "solid", color: C.line, pt: 0.5 },
  valign: "middle", autoPage: false,
};
function hdr(t) {
  return { text: t, options: { bold: true, color: C.white, fill: { color: C.mid }, fontSize: 12 } };
}

// =====================================================================
// 1. 封面
// =====================================================================
{
  const s = pres.addSlide(); darkBg(s);
  s.addText("柑橘病蟲害辨識　115 資工四A", {
    x: M, y: 1.35, w: W - 2 * M, h: 0.38, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.accent, charSpacing: 2,
  });
  s.addText("階段性進度報告", {
    x: M, y: 1.78, w: W - 2 * M, h: 1.0, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 46, bold: true, color: C.white,
  });
  s.addText("v11.5 → v12.1　·　2026-09-07　週二下午進度會議", {
    x: M, y: 2.82, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 16, color: "A9C4B4",
  });

  card(s, M, 3.62, W - 2 * M, 1.72, "17402F");
  s.addText("這一週用預先登記的判準，把「換更大的模型」和「調匯出參數」兩條路同時關掉了。", {
    x: M + 0.45, y: 3.86, w: W - 2 * M - 0.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 19, bold: true, color: C.white,
  });
  s.addText("剩下唯一還沒試過的方向是資料端 —— 而它卡在兩包還沒發出去的人工標註。", {
    x: M + 0.45, y: 4.42, w: W - 2 * M - 0.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 19, bold: true, color: C.accent,
  });

  s.addText("正文約 8 分鐘　·　備答另計", {
    x: M, y: 5.9, w: W - 2 * M, h: 0.35, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12, color: "7E9788",
  });
  s.addNotes("開場先講這句話，再進三個數字。全場的骨架就是：兩條路關掉、剩資料端、兩個決策要當場定。");
}

// =====================================================================
// 2. 三個數字
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "開場：三個數字", "這一週的結論可以壓縮成這三個");

  const items = [
    { n: "+0.003", lab: "模型參數放大 3.84 倍換到的 mAP50", so: "判準門檻是 +0.03\n容量不是瓶頸", col: C.clay },
    { n: "15.4", unit: " FPS", lab: "手機上精度可用的最快組合", so: "目標 30 FPS\n這台裝置做不到", col: C.clay },
    { n: "0", lab: "人工工作包 A / B 的回收檔案數", so: "唯一沒試過的方向\n還沒動工", col: C.accent },
  ];
  const cw = 3.82, gap = 0.42, x0 = M;
  items.forEach((it, i) => {
    const x = x0 + i * (cw + gap);
    card(s, x, 1.85, cw, 3.62, C.tint);
    s.addText([
      { text: it.n, options: { fontSize: 54, bold: true, color: it.col } },
      { text: it.unit || "", options: { fontSize: 22, bold: true, color: it.col } },
    ], {
      x: x + 0.3, y: 2.12, w: cw - 0.6, h: 1.0, isTextBox: true, margin: 0,
      fontFace: F, align: "left",
    });
    s.addText(it.lab, {
      x: x + 0.3, y: 3.18, w: cw - 0.6, h: 0.78, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, color: C.ink,
    });
    s.addShape(pres.ShapeType.rect, {
      x: x + 0.3, y: 4.02, w: cw - 0.6, h: 0.012, fill: { color: C.line },
    });
    s.addText(it.so, {
      x: x + 0.3, y: 4.2, w: cw - 0.6, h: 1.0, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, bold: true, color: C.mid,
    });
  });
  s.addNotes("三個數字講完就往下走，不要在這裡展開。細節在後面。");
}

// =====================================================================
// 3. 進度：三條線
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "進度：三條線分開講", "混在一起講會失焦");

  const rows = [
    { k: "資料集", v: "v5.6　9 類 / train 7,264 / valid 401 / test 401",
      p: "量尺已經修好：九類的評估誤差 ±2SE 全部 ≤ 0.10，而且沒有新拍任何一張照片" },
    { k: "模型", v: "v11.5 = 交付基準（test mAP50 0.810）",
      p: "已經走到底：模型端連四輪、十一種改動全部無效" },
    { k: "部署", v: "本週新開，測試平台 1 已量完",
      p: "量清楚了：最好 15.4 FPS，離 30 FPS 差 2 倍" },
  ];
  rows.forEach((r, i) => {
    const y = 1.82 + i * 1.28;
    card(s, M, y, W - 2 * M, 1.12, i === 2 ? C.tint : C.tint2);
    badge(s, i + 1, M + 0.3, y + 0.29, 0.55, i === 2 ? C.accent : C.mid);
    s.addText(r.k, {
      x: M + 1.05, y: y + 0.17, w: 1.5, h: 0.4, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 19, bold: true, color: C.dark,
    });
    s.addText(r.v, {
      x: M + 2.5, y: y + 0.19, w: 9.0, h: 0.36, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, color: C.muted,
    });
    s.addText(r.p, {
      x: M + 1.05, y: y + 0.61, w: 10.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, bold: true, color: C.mid,
    });
  });

  s.addText("本週最大的變化是第三條線從無到有。在這之前只量得出精度、量不出延遲，所以「要不要換更大的模型」根本無法討論 —— 因為不知道有多少預算可以花。",
    { x: M, y: 5.78, w: W - 2 * M, h: 0.72, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, color: C.ink });
  s.addNotes("強調第三條線是本週的結構性變化：以前沒有延遲這把尺。");
}

// =====================================================================
// 4. 關掉的方向一：模型容量
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "關掉的方向一：模型容量不是瓶頸", "v12s：只改一個變因，模型尺度 n → s，其餘逐字相同");

  card(s, M, 1.9, 6.1, 2.72, C.tint);
  s.addText("判準裁決", {
    x: M + 0.35, y: 2.08, w: 3.0, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, bold: true, color: C.muted,
  });
  const verdict = [
    ["達標線（開跑前寫死）", "≥ 0.83959", C.mid],
    ["實得 test mAP50", "0.81266", C.ink],
    ["Δ vs v11.5", "+0.00307", C.clay],
  ];
  verdict.forEach((v, i) => {
    const y = 2.52 + i * 0.62;
    s.addText(v[0], {
      x: M + 0.35, y, w: 3.5, h: 0.42, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, color: C.ink, valign: "middle",
    });
    s.addText(v[1], {
      x: M + 3.85, y, w: 1.95, h: 0.42, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 19, bold: true, color: v[2], align: "right", valign: "middle",
    });
  });

  card(s, M + 6.5, 1.9, W - 2 * M - 6.5, 2.72, "F7E7E2");
  s.addText("未通過", {
    x: M + 6.85, y: 2.28, w: 4.9, h: 0.72, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 40, bold: true, color: C.clay,
  });
  s.addText("3.84 倍參數只換到門檻的十分之一，而且低於同設定重跑一次的隨機變異（0.021）。", {
    x: M + 6.85, y: 3.08, w: 4.9, h: 1.2, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, color: C.ink,
  });

  s.addText([
    { text: "九類裡只有一類的變化超過雜訊地板，四好四壞一平", options: { bold: true, color: C.dark } },
    { text: " —— 若容量真的是瓶頸不會長這樣，尤其第二弱的潛葉蛾反而變差 0.024。", options: { color: C.ink } },
  ], { x: M, y: 4.85, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0, fontFace: F, fontSize: 15 });

  card(s, M, 5.5, W - 2 * M, 1.05, C.tint2);
  s.addText([
    { text: "順帶結論：m 尺度不用試了。", options: { bold: true, color: C.accent } },
    { text: "　s 的 3.84 倍都只有雜訊等級，m（21 M 參數）沒有理由不同，成本卻會再上去。", options: { color: C.ink } },
  ], { x: M + 0.35, y: 5.74, w: W - 2 * M - 0.7, h: 0.6, isTextBox: true, margin: 0, fontFace: F, fontSize: 14 });
  s.addNotes("重點：門檻是開跑前訂的，不是事後挑的。0.003 跟擲骰子分不出來。");
}

// =====================================================================
// 5. 逐類別圖
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "九個類別：沒有系統性改善", "四類變好、四類變差、一類不動 —— 這就是「沒有改善」的樣子");

  const labels = ["Sooty_Mold", "Black_Spot", "Aphid", "Oily_Spot", "Canker",
                  "Thrips", "Scale_Insect", "Citrus_Leaf_Miner", "Thrips_Damage"];
  const v115  = [0.9950, 0.9534, 0.9478, 0.9446, 0.8677, 0.7795, 0.6704, 0.6764, 0.4515];
  const v12s  = [0.9950, 0.9486, 0.9436, 0.9204, 0.9016, 0.7526, 0.7112, 0.6523, 0.4885];

  s.addChart(pres.ChartType.bar, [
    { name: "v11.5", labels: labels, values: v115 },
    { name: "v12s",  labels: labels, values: v12s },
  ], {
    x: M, y: 1.78, w: W - 2 * M, h: 4.15,
    barDir: "bar", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: [C.line, C.mid],
    showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: C.ink,
    catAxisLabelColor: C.ink, catAxisLabelFontSize: 11, catAxisLabelFontFace: F,
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 10, valAxisLabelFontFace: F,
    valAxisMinVal: 0, valAxisMaxVal: 1.0,
    valGridLine: { color: "E4EDE7", size: 1 },
    catGridLine: { style: "none" },
    showTitle: false,
  });
  s.addText("只有 Scale_Insect（+0.041）勉強越過 ±0.04 的 per-class 雜訊地板；第二弱的 Citrus_Leaf_Miner 反而掉了 0.024。",
    { x: M, y: 6.05, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, color: C.ink });
  s.addNotes("圖的重點是兩排幾乎重疊。若容量是瓶頸，應該看到多數類別同向改善。");
}

// =====================================================================
// 6. 關掉的方向二：30 FPS
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "關掉的方向二：30 FPS 達不到", "官方匯出文件的參數表整份測過：22 組量完整 mAP、18 組量手機延遲、外加 4 種格式");

  s.addTable([
    [hdr("組合"), hdr("延遲 (ms)"), hdr("FPS"), hdr("mAP50"), hdr("判定")],
    [{ text: "int8 @ 320（最快）" }, { text: "43.5" }, { text: "23.0" }, { text: "0.719" },
     { text: "精度不可用", options: { color: C.clay, bold: true } }],
    [{ text: "fp32 @ 320（最佳可用）", options: { bold: true } },
     { text: "64.8", options: { bold: true } }, { text: "15.4", options: { bold: true } },
     { text: "0.785", options: { bold: true } },
     { text: "現在就能交付", options: { color: C.mid, bold: true } }],
    [{ text: "fp32 @ 640（原始）" }, { text: "266.3" }, { text: "3.8" }, { text: "0.816" },
     { text: "太慢", options: { color: C.muted } }],
  ], {
    x: M, y: 1.92, w: W - 2 * M, colW: [3.9, 1.9, 1.6, 1.8, 2.73], rowH: 0.44,
    ...tblBase, fontSize: 13, align: "left",
  });

  card(s, M, 4.05, 5.95, 1.28, C.tint);
  s.addText([
    { text: "有效的槓桿只有一個：降解析度", options: { bold: true, color: C.dark } },
    { text: "\n640 → 320 加速 4.11 倍，只掉 0.032 mAP。量化、GPU、NNAPI 全部無效或倒退。", options: { color: C.ink } },
  ], { x: M + 0.32, y: 4.24, w: 5.3, h: 0.95, isTextBox: true, margin: 0, fontFace: F, fontSize: 13 });

  card(s, M + 6.35, 4.05, W - 2 * M - 6.35, 1.28, "FBEEDF");
  s.addText([
    { text: "最反直覺的一條", options: { bold: true, color: C.accent } },
    { text: "\n相同延遲下 fp32@320（0.785）勝過 int8@416（0.728）。算力預算該花在降解析度，不是花在量化。", options: { color: C.ink } },
  ], { x: M + 6.67, y: 4.24, w: 5.3, h: 0.95, isTextBox: true, margin: 0, fontFace: F, fontSize: 13 });

  s.addText("測試平台 1：OPPO CPH2641 / Snapdragon 662 (SM6115) / Adreno 610 / Android 14　·　CPU only 4 threads　·　目標 30 FPS ±5 = 每張 28.6–40 ms",
    { x: M, y: 5.62, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 12, color: C.muted });
  s.addNotes("若被問「換框架會不會比較快」：框架差幾十個百分點，我們缺的是 2 倍。");
}

// =====================================================================
// 7. 順帶兩件事
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "順帶的兩件事", "可講可略");

  card(s, M, 1.85, 5.85, 3.35, C.tint2);
  badge(s, "A", M + 0.35, 2.12, 0.5, C.mid);
  s.addText("官方文件與實際不符四處", {
    x: M + 1.0, y: 2.16, w: 4.6, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 18, bold: true, color: C.dark,
  });
  s.addText("都有程式碼證據。最值得記的是 w8a16：文件列為支援、精度也真的無損，但實測慢 27 倍 —— XNNPACK 一個節點都沒接手。", {
    x: M + 0.35, y: 2.85, w: 5.15, h: 1.1, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, color: C.ink,
  });
  s.addText("「文件說支援」不等於「實務上可用」。", {
    x: M + 0.35, y: 4.18, w: 5.15, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.accent,
  });

  card(s, M + 6.25, 1.85, W - 2 * M - 6.25, 3.35, C.tint2);
  badge(s, "B", M + 6.6, 2.12, 0.5, C.mid);
  s.addText("與 7 月手動紀錄交叉驗證", {
    x: M + 7.25, y: 2.16, w: 4.6, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 18, bold: true, color: C.dark,
  });
  s.addText("同一台手機、相隔兩個月、不同工具鏈，延遲落在 1.2% 以內 —— 本輪的量測方法可信。", {
    x: M + 6.6, y: 2.85, w: 5.15, h: 0.8, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, color: C.ink,
  });
  s.addText("同時發現該紀錄有兩列數據與它自己附的原始 log 不符，其中一列正好是報告摘要第一句（會讓人以為 60 FPS 已達成，實際 6.87）。已逐欄比對、更正並推送。", {
    x: M + 6.6, y: 3.62, w: 5.15, h: 1.4, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, color: C.muted,
  });

  pageNote(s, "更正 commit：730e4ef、1b947d8　·　依 AGENTS.md 紅線 R1，修改前已取得指名授權");
  s.addNotes("這一頁時間不夠可以整頁跳過。");
}

// =====================================================================
// 8. 決策一
// =====================================================================
{
  const s = pres.addSlide(); darkBg(s);
  s.addText("要在這場會議上決定的事", {
    x: M, y: 0.5, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.accent, charSpacing: 2,
  });
  badge(s, "1", M, 1.2, 0.72, C.accent, C.dark);
  s.addText("30 FPS 是不是真的需要？", {
    x: M + 0.95, y: 1.24, w: 10.5, h: 0.68, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 34, bold: true, color: C.white,
  });

  card(s, M, 2.35, 5.85, 2.5, "17402F");
  s.addText("若互動是「對準葉片 → 拍照 → 出結果」", {
    x: M + 0.35, y: 2.58, w: 5.15, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.accent,
  });
  s.addText("65 ms 使用者根本不會察覺。\n\nfp32 @ 320（15.4 FPS / mAP50 0.785）現在就可以交付。", {
    x: M + 0.35, y: 3.08, w: 5.15, h: 1.55, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.white,
  });

  card(s, M + 6.25, 2.35, W - 2 * M - 6.25, 2.5, "17402F");
  s.addText("若必須是即時串流", {
    x: M + 6.6, y: 2.58, w: 5.15, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.accent,
  });
  s.addText("這台裝置做不到。要嘛換硬體，要嘛重訓 —— QAT 或直接在 320 解析度上訓練，是僅剩的兩條路。", {
    x: M + 6.6, y: 3.08, w: 5.15, h: 1.55, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.white,
  });

  s.addText("這是產品決策，不是技術問題 —— 技術面已經量清楚了。", {
    x: M, y: 5.35, w: W - 2 * M, h: 0.55, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 20, bold: true, color: C.accent,
  });
  s.addNotes("把球丟回去。技術面沒有更多可以做的了，需要產品端定義互動形式。");
}

// =====================================================================
// 9. 決策二
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  s.addText("要在這場會議上決定的事", {
    x: M, y: 0.45, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, color: C.accent, charSpacing: 2,
  });
  badge(s, "2", M, 1.0, 0.66, C.accent, C.dark);
  s.addText("人工工作包 A / B 要不要發？", {
    x: M + 0.88, y: 1.02, w: 10.5, h: 0.62, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 32, bold: true, color: C.dark,
  });
  s.addText("容量既然已被排除，資料端是目前唯一還沒試過的方向，而它已經掛了兩週。", {
    x: M, y: 1.82, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.ink,
  });

  s.addTable([
    [hdr("最弱的類別"), hdr("AP50"), hdr("成因"), hdr("對應")],
    [{ text: "Thrips_Damage", options: { bold: true } },
     { text: "0.489", options: { color: C.clay, bold: true } },
     { text: "框定義不一致（三輪都沒隨評估集擴大而收斂）" },
     { text: "工作包 A：兩人各標 20 張", options: { bold: true, color: C.mid } }],
    [{ text: "Citrus_Leaf_Miner", options: { bold: true } },
     { text: "0.652", options: { color: C.clay, bold: true } },
     { text: "樣本量少（原始只有 111 張）" },
     { text: "工作包 B：外部 150 張補標", options: { bold: true, color: C.mid } }],
  ], {
    x: M, y: 2.38, w: W - 2 * M, colW: [2.8, 1.1, 4.5, 3.533], rowH: 0.5,
    ...tblBase, align: "left",
  });

  card(s, M, 4.35, W - 2 * M, 1.92, C.tint);
  s.addText([
    { text: "工作包 A 有明確的分歧點：", options: { bold: true, color: C.dark } },
    { text: "兩人的中位 IoU ≥ 0.85 才值得全類重標，否則依既定規則直接刪掉這個類別。", options: { color: C.ink } },
  ], { x: M + 0.35, y: 4.55, w: W - 2 * M - 0.7, h: 0.42, isTextBox: true, margin: 0, fontFace: F, fontSize: 15 });
  s.addText([
    { text: "本週新查到：", options: { bold: true, color: C.dark } },
    { text: "Thrips_Damage 在 2026-06 就有類別代號，但 train / valid / test 全部零標註 —— 它是九類裡唯一從零開始、最晚建立的類別。", options: { color: C.ink } },
  ], { x: M + 0.35, y: 5.03, w: W - 2 * M - 0.7, h: 0.62, isTextBox: true, margin: 0, fontFace: F, fontSize: 14 });
  s.addText("這條路怎麼走都會有結論，不會白做。", {
    x: M + 0.35, y: 5.68, w: W - 2 * M - 0.7, h: 0.45, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 17, bold: true, color: C.accent,
  });
  s.addNotes("重點：這不是「投入了可能沒結果」的賭注，兩種結果都有明確的下一步。");
}

// =====================================================================
// 10. 下週
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "下週", "取決於上面兩個決策");

  card(s, M, 1.9, 5.85, 1.9, C.tint);
  s.addText("若兩包都發出去", {
    x: M + 0.35, y: 2.12, w: 5.15, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 17, bold: true, color: C.mid,
  });
  s.addText("等回收 ＋ 標註一致性分析。", {
    x: M + 0.35, y: 2.6, w: 5.15, h: 0.85, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.ink,
  });

  card(s, M + 6.25, 1.9, W - 2 * M - 6.25, 1.9, C.tint);
  s.addText("若改走部署路線", {
    x: M + 6.6, y: 2.12, w: 5.15, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 17, bold: true, color: C.mid,
  });
  s.addText("QAT 或 320 重訓，兩者都要重訓一輪。", {
    x: M + 6.6, y: 2.6, w: 5.15, h: 0.85, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, color: C.ink,
  });

  card(s, M, 4.1, W - 2 * M, 1.85, "F7E7E2");
  s.addText("明確不再做的", {
    x: M + 0.35, y: 4.32, w: 5.0, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 17, bold: true, color: C.clay,
  });
  s.addText([
    { text: "m 尺度模型", options: { bullet: true, breakLine: true } },
    { text: "任何模型端改動", options: { bullet: true, breakLine: true } },
    { text: "GPU delegate 與 NNAPI", options: { bullet: true } },
  ], {
    x: M + 0.42, y: 4.82, w: 5.5, h: 0.95, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 14, color: C.ink, paraSpaceAfter: 2,
  });
  s.addText("三者都已有結論 —— 不是還沒做，是做完了而且否定。", {
    x: M + 6.6, y: 4.9, w: 5.15, h: 0.8, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.clay,
  });
  s.addNotes("這頁講完就進備答。歷史數據那兩頁是被問到才翻。");
}

// =====================================================================
// 11. 備答分隔頁
// =====================================================================
{
  const s = pres.addSlide(); darkBg(s);
  s.addText("備答", {
    x: M, y: 2.6, w: W - 2 * M, h: 1.0, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 48, bold: true, color: C.white,
  });
  s.addText("以下三頁是被問到才翻的資料　·　歷史數據比較與常見提問", {
    x: M, y: 3.68, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 17, color: C.accent,
  });
  s.addNotes("正文到這裡結束。");
}

// =====================================================================
// 11b. 全期 mAP50 總覽（圖為主）
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "全期 mAP50 總覽（2026-06 至今）", "valid split —— 唯一每一期都有的量測。顏色 = 資料集世代，跨顏色不可比");

  const labels = ["06 large", "06 nano", "06 n+P2", "06 微調", "06 SSD-L", "06 SSD-S",
                  "v8", "v9", "v10", "v11", "v11.5", "v12s"];
  const N = null;
  s.addChart(pres.ChartType.bar, [
    { name: "12 類資料集（06/07）", labels: labels,
      values: [0.871, 0.854, 0.851, 0.846, 0.472, 0.453, N, N, N, N, N, N] },
    { name: "v5・v5r（8 類）", labels: labels,
      values: [N, N, N, N, N, N, 0.821, 0.867, N, N, N, N] },
    { name: "v5.5・v5.6（9 類，現行）", labels: labels,
      values: [N, N, N, N, N, N, N, N, 0.799, 0.835, 0.829, 0.837] },
  ], {
    x: M, y: 1.82, w: W - 2 * M, h: 4.05,
    barDir: "col", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: ["A8C4B4", "5E8F73", "2C5F45"],
    showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: C.ink,
    showValue: true, dataLabelPosition: "inEnd", dataLabelColor: "FFFFFF",
    dataLabelFontSize: 10, dataLabelFontFace: F, dataLabelFormatCode: "0.000",
    catAxisLabelColor: C.ink, catAxisLabelFontSize: 11, catAxisLabelFontFace: F,
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 10, valAxisLabelFontFace: F,
    valAxisMinVal: 0, valAxisMaxVal: 1.0,
    valGridLine: { color: "E4EDE7", size: 1 }, catGridLine: { style: "none" },
    showTitle: false,
  });
  s.addText("兩個 SSD 的凹陷是架構限制（320 輸入抓不到極小目標）；其餘十個都落在 0.80–0.87，沒有趨勢可言。",
    { x: M, y: 6.02, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, color: C.ink });
  s.addNotes("重點只有一句：顏色不同就不能相減。SSD 那兩根是架構問題，不是資料問題。");
}

// =====================================================================
// 11c. 三套資料集的差別（解釋為何不可比）
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "為什麼跨期不能相減", "三套資料集量的根本不是同一件事");

  const BR = String.fromCharCode(10);
  const cols = [
    { h: "12 類（2026-06/07）", c: "A8C4B4", rows: [
        ["類別", "12 類，其中 2 類零標註" + BR + "另含 2 個健康葉類別"],
        ["評估集", "valid 438 張" + BR + "P_SI 一類佔框數 68.8%"],
        ["權重", "最後一輪" + BR + "（原報告標成「最優」）"],
      ] },
    { h: "8 類（v5・v5r）", c: "5E8F73", rows: [
        ["類別", "8 類" + BR + "沒有 Thrips_Damage"],
        ["評估集", "valid 438 張" + BR + "v5r 有 13.4% 跨 split 近重複"],
        ["權重", "best.pt"],
      ] },
    { h: "9 類（v5.5・v5.6，現行）", c: "2C5F45", rows: [
        ["類別", "9 類" + BR + "定義已固定"],
        ["評估集", "valid 401 ＋ test 401" + BR + "零洩漏、±2SE ≤ 0.10"],
        ["權重", "last.pt（無選擇偏誤）"],
      ] },
  ];
  const cw = 3.82, gap = 0.42;
  cols.forEach((col, i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.82, cw, 4.05, i === 2 ? C.tint : C.tint2);
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 1.82, w: cw, h: 0.62, rectRadius: 0.1, fill: { color: col.c },
    });
    s.addText(col.h, {
      x: x + 0.2, y: 1.82, w: cw - 0.4, h: 0.62, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, bold: true, color: i === 0 ? C.dark : C.white, valign: "middle",
    });
    col.rows.forEach((r, j) => {
      const y = 2.62 + j * 1.08;
      s.addText(r[0], {
        x: x + 0.25, y: y, w: cw - 0.5, h: 0.3, isTextBox: true, margin: 0,
        fontFace: F, fontSize: 11, bold: true, color: C.muted,
      });
      s.addText(r[1], {
        x: x + 0.25, y: y + 0.3, w: cw - 0.5, h: 0.7, isTextBox: true, margin: 0,
        fontFace: F, fontSize: 13, color: C.ink,
      });
    });
  });
  s.addText("量尺換過三次，刻度不能互換 —— 唯一能直接相減的是 v11.5 對 v12s。", {
    x: M, y: 6.05, w: W - 2 * M, h: 0.45, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.accent,
  });
  s.addNotes("被問「不是變差了嗎」就翻這頁，三欄對照講完就夠。");
}

// =====================================================================
// 12. 歷史數據 — test（與記錄版週報 §9 表相同）
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "歷史訓練成果總覽　test split", "與記錄版週報 §9 相同；各版本的主要結果");

  card(s, M, 1.72, W - 2 * M, 0.62, "F7E7E2");
  s.addText([
    { text: "本表的各列不可直接比較。", options: { bold: true, color: C.clay } },
    { text: "　三個版本區間用的是三套不同的資料集，類別數與評估集組成都不同，數值高低不代表模型能力的高低。", options: { color: C.ink } },
  ], { x: M + 0.3, y: 1.85, w: W - 2 * M - 0.6, h: 0.4, isTextBox: true, margin: 0, fontFace: F, fontSize: 12.5 });

  const na = { text: "未存檔", options: { color: C.muted, italic: true } };
  s.addTable([
    [hdr("版本"), hdr("資料集"), hdr("類別"), hdr("報告權重"), hdr("mAP50"), hdr("mAP50-95"), hdr("Precision"), hdr("Recall"), hdr("Jaccard")],
    [{ text: "v9（A0 交付版）" }, { text: "v5r" }, { text: "8" }, { text: "best.pt（ep69）" },
     { text: "0.89079" }, { text: "0.67829" }, { text: "0.89950" }, { text: "0.86139" }, { text: "0.66979" }],
    [{ text: "v10" }, { text: "v5.5" }, { text: "9" }, { text: "best.pt（ep54）" },
     { text: "0.80928" }, { text: "0.60605" }, { text: "0.85456" }, { text: "0.77909" }, { text: "0.71906" }],
    [{ text: "v11" }, { text: "v5.6" }, { text: "9" }, { text: "best.pt（ep69）" },
     { text: "0.82213" }, { text: "0.61294" }, na, na, na],
    [{ text: "v11.5（現行交付基準）", options: { bold: true } }, { text: "v5.6", options: { bold: true } },
     { text: "9", options: { bold: true } }, { text: "last.pt", options: { bold: true } },
     { text: "0.80959", options: { bold: true } }, { text: "0.60691", options: { bold: true } },
     { text: "0.85839", options: { bold: true } }, { text: "0.79502", options: { bold: true } },
     { text: "0.68627", options: { bold: true } }],
    [{ text: "v12s", options: { bold: true } }, { text: "v5.6", options: { bold: true } },
     { text: "9", options: { bold: true } }, { text: "last.pt", options: { bold: true } },
     { text: "0.81266", options: { bold: true } }, { text: "0.61836", options: { bold: true } },
     { text: "0.86643", options: { bold: true } }, { text: "0.80441", options: { bold: true } },
     { text: "0.72737", options: { bold: true } }],
  ], {
    x: M, y: 2.55, w: W - 2 * M, colW: [2.15, 0.85, 0.65, 1.55, 1.15, 1.25, 1.3, 1.13, 1.903], rowH: 0.44,
    ...tblBase, fontSize: 11.5, align: "left",
  });

  s.addText([
    { text: "只有最後兩列可以直接相減", options: { bold: true, color: C.mid } },
    { text: "（同資料集 v5.6、同排程、同權重協定，只差模型尺度）。v11 雖然同為 v5.6，但用 patience=30 與 best.pt，協定不同，且其 test 混淆矩陣未存檔。", options: { color: C.ink } },
  ], { x: M, y: 5.42, w: W - 2 * M, h: 0.72, isTextBox: true, margin: 0, fontFace: F, fontSize: 13 });

  pageNote(s, "Precision / Recall 取自 ultralytics val()（conf 0.001）；Detection Jaccard 取自混淆矩陣（conf 0.25）—— 兩者不在同一工作點，不可互推");
  s.addNotes("被問到歷史比較時翻這頁。先講「不可直接比較」再給數字。");
}

// =====================================================================
// 13. 歷史數據 — valid
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "歷史訓練成果總覽　valid split", "同上，附逐框計數");

  const na = { text: "未存檔", options: { color: C.muted, italic: true } };
  s.addTable([
    [hdr("版本"), hdr("mAP50"), hdr("mAP50-95"), hdr("Precision"), hdr("Recall"), hdr("Jaccard"), hdr("TP / FP / FN")],
    [{ text: "v9（A0 交付版）" }, { text: "0.86651" }, { text: "0.65356" }, { text: "0.88595" },
     { text: "0.80756" }, { text: "0.65353" }, { text: "1841 / 704 / 272" }],
    [{ text: "v10" }, { text: "0.79931" }, { text: "0.59268" }, { text: "0.85617" },
     { text: "0.77412" }, { text: "0.64277" }, { text: "556 / 136 / 173" }],
    [{ text: "v11" }, { text: "0.83543" }, { text: "0.62924" }, na, na,
     { text: "0.66056" }, { text: "685 / 169 / 183" }],
    [{ text: "v11.5", options: { bold: true } }, { text: "0.82899", options: { bold: true } },
     { text: "0.63580", options: { bold: true } }, { text: "0.86341", options: { bold: true } },
     { text: "0.80114", options: { bold: true } }, { text: "0.65425", options: { bold: true } },
     { text: "685 / 179 / 183", options: { bold: true } }],
    [{ text: "v12s", options: { bold: true } }, { text: "0.83652", options: { bold: true } },
     { text: "0.64345", options: { bold: true } }, { text: "0.86186", options: { bold: true } },
     { text: "0.82734", options: { bold: true } }, na, na],
  ], {
    x: M, y: 1.85, w: W - 2 * M, colW: [2.2, 1.35, 1.5, 1.5, 1.35, 1.4, 2.633], rowH: 0.46,
    ...tblBase, fontSize: 12, align: "left",
  });

  card(s, M, 4.85, W - 2 * M, 1.45, C.tint);
  s.addText([
    { text: "mAP 高不等於 Jaccard 高。", options: { bold: true, color: C.accent } },
    { text: "　v9 的 mAP50 全表最高，但 valid 的 Jaccard 只有 0.654 —— 因為它的 FP 高達 704，其中 Scale_Insect 一類就貢獻了 563 個背景誤報。兩個指標衡量的是不同的東西。", options: { color: C.ink } },
  ], { x: M + 0.35, y: 5.08, w: W - 2 * M - 0.7, h: 1.0, isTextBox: true, margin: 0, fontFace: F, fontSize: 14 });
  s.addNotes("v11 的 valid Jaccard 與 v12s 的 test Jaccard 是這次依存檔混淆矩陣新算出來的。");
}

// =====================================================================
// 14. 為什麼 v9 最高卻不代表最好
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "若有人指著 v9 的 0.891 說「不是變差了嗎」", "三個理由，照這個順序講");

  const reasons = [
    ["v5r 有 13.4% 跨 split 近重複影像",
     "v9 的評估集裡有模型訓練時看過的近乎相同照片，數字被灌水。v5.5 起已全類別清零。"],
    ["v9 只有 8 類，沒有 Thrips_Damage",
     "那是九類中最弱的一項（0.489），加進平均自然拉低。v10 取 8 類等效值是 0.864，與 v9 的 0.891 只差 0.027，而 v9 那側還有洩漏 —— 實質接近持平，不是退步。"],
    ["評估集本身換過",
     "v5.6 是為了把每類誤差壓到 ±2SE ≤ 0.10 而重新切分的。量尺變了，刻度就不能互換。"],
  ];
  reasons.forEach((r, i) => {
    const y = 1.85 + i * 1.5;
    card(s, M, y, W - 2 * M, 1.32, i % 2 === 0 ? C.tint2 : C.tint);
    badge(s, i + 1, M + 0.32, y + 0.36, 0.58, C.mid);
    s.addText(r[0], {
      x: M + 1.12, y: y + 0.18, w: 10.3, h: 0.42, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 17, bold: true, color: C.dark,
    });
    s.addText(r[1], {
      x: M + 1.12, y: y + 0.62, w: 10.3, h: 0.62, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, color: C.ink,
    });
  });

  s.addText("一句話：這張表適合回答「某一版當時量到什麼」，不適合回答「哪一版比較強」。", {
    x: M, y: 6.4, w: W - 2 * M, h: 0.45, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 15, bold: true, color: C.accent,
  });
  s.addNotes("跨資料集的強弱比較在本專題沒有合法基準，唯一例外是同資料集內的對照。");
}

// =====================================================================
// 15. 備答：常見提問
// =====================================================================
{
  const s = pres.addSlide(); lightBg(s);
  title(s, "備答：可能會被問到的", "一句話回答");

  s.addTable([
    [hdr("問題"), hdr("一句話回答")],
    [{ text: "不是說更大的模型比較準嗎？" },
     { text: "資料量不夠時不會。raw train 只有 2,353 張，模型早就不是限制。" }],
    [{ text: "為什麼 +0.003 算沒有進步？" },
     { text: "同一個設定重跑一次就會差 0.021。這個差距跟擲骰子分不出來，而門檻是開跑前訂好的。" }],
    [{ text: "換別的推論框架會不會比較快？" },
     { text: "框架之間通常差幾十個百分點，我們缺的是 2 倍。那一步的價值是排除框架因素，不是找解法。" }],
    [{ text: "量化不是應該會變快嗎？" },
     { text: "動態量化省的是體積不是算力，INT16 甚至慢 27 倍。只有靜態 INT8 真的快 1.81 倍，但掉 0.14 mAP。" }],
    [{ text: "這台手機是不是太爛了？" },
     { text: "Snapdragon 662 是 2020 年入門 SoC，確實舊。但這是實際會遇到的裝置，換機要重測。" }],
    [{ text: "工作包發下去要多久？" },
     { text: "標註本身兩人各 20 張，一次會議的時間。真正的成本是決定要不要全類重標之後的那一輪。" }],
    [{ text: "v12s 那顆權重還有用嗎？" },
     { text: "只當精度上限的參考。它在手機上約 950 ms 一張，不是部署候選。" }],
    [{ text: "6–7 月不是有 0.87 嗎？" },
     { text: "那是 12 類資料集（含 2 個健康葉類別、2 類零標註），而且是最後一輪被標成「最優」。不同量尺，不能相減。" }],
  ], {
    x: M, y: 1.85, w: W - 2 * M, colW: [4.2, 7.73], rowH: 0.61,
    ...tblBase, fontSize: 12.5, align: "left",
  });
  s.addNotes("這頁不用講，被問到再看。");
}

// =====================================================================
// 16. 結尾
// =====================================================================
{
  const s = pres.addSlide(); darkBg(s);
  s.addText("需要當場決定的兩件事", {
    x: M, y: 1.5, w: W - 2 * M, h: 0.7, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 34, bold: true, color: C.white,
  });

  const d = [
    ["1", "30 FPS 是不是真的需要？", "拍照式互動 → fp32 @ 320 現在就能交付　·　即時串流 → 換硬體或重訓"],
    ["2", "人工工作包 A / B 要不要發？", "容量已排除，資料端是唯一還沒試過的方向 —— 已經掛了兩週"],
  ];
  d.forEach((it, i) => {
    const y = 2.75 + i * 1.62;
    card(s, M, y, W - 2 * M, 1.4, "17402F");
    badge(s, it[0], M + 0.35, y + 0.4, 0.6, C.accent, C.dark);
    s.addText(it[1], {
      x: M + 1.15, y: y + 0.24, w: 10.2, h: 0.45, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 21, bold: true, color: C.white,
    });
    s.addText(it[2], {
      x: M + 1.15, y: y + 0.75, w: 10.2, h: 0.45, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 14, color: "A9C4B4",
    });
  });

  s.addText("技術面已經量清楚了。", {
    x: M, y: 6.25, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 20, bold: true, color: C.accent,
  });
  s.addNotes("收尾把兩個決策再念一次，然後開放討論。");
}

const OUT = process.argv[2] || "deck.pptx";
pres.writeFile({ fileName: OUT }).then(() => console.log("written:", OUT));
