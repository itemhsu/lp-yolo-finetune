#!/usr/bin/env python3
"""Four-way HTML report: Two-step vs Merged-v2 vs Merged-v3 vs Merged-v4."""

import argparse, csv, html
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

_sect_id = 0

def pct(n, d, sign=False):
    if not d: return "—"
    v = n/d*100
    return (f"{v:+.1f}%" if sign else f"{v:.1f}%")

def subdir_of(p):
    parts = Path(p).parts
    if "LPD" in parts:
        idx = parts.index("LPD")
        return parts[idx+1] if idx+1 < len(parts) else "_"
    return "_"

def count_match(rows, col):
    return sum(1 for r in rows if r.get(col,"0") == "1")

def section_table_html(rows, preview, sid):
    hdr = ("<tr><th>#</th><th>檔名</th><th>期望車牌</th>"
           "<th>TS OCR</th><th>TS✓</th>"
           "<th>v2 OCR</th><th>v2✓</th>"
           "<th>v3 OCR</th><th>v3✓</th>"
           "<th>v4 OCR</th><th>v4✓</th></tr>")
    body = ""
    for i, r in enumerate(rows):
        fname = Path(r["image_path"]).name
        hidden = ' class="hidden-row"' if i >= preview else ""
        def td_m(v): return f"<td class=\"{'pos' if v=='1' else 'neg'}\">{v}</td>"
        body += (f"<tr{hidden}>"
                 f"<td style='color:#64748b;font-size:11px'>{i+1}</td>"
                 f"<td title='{html.escape(r['image_path'])}'><code>{html.escape(fname)}</code></td>"
                 f"<td>{html.escape(r['expected_plate'])}</td>"
                 f"<td><code>{html.escape(r.get('twostep_ocr_clean',''))}</code></td>"
                 + td_m(r.get("twostep_match","0"))
                 + f"<td><code>{html.escape(r.get('v2_ocr_clean',''))}</code></td>"
                 + td_m(r.get("v2_match","0"))
                 + f"<td><code>{html.escape(r.get('v3_ocr_clean',''))}</code></td>"
                 + td_m(r.get("v3_match","0"))
                 + f"<td><code>{html.escape(r.get('v4_ocr_clean',''))}</code></td>"
                 + td_m(r.get("v4_match","0"))
                 + "</tr>")
    hidden_count = max(0, len(rows) - preview)
    btn = ""
    if hidden_count > 0:
        btn = (f'<button class="expand-btn" data-sid="{sid}" data-total="{hidden_count}" '
               f'onclick="toggleRows(this,\'{sid}\')">'
               f'展開全部 ▼ （還有 {hidden_count:,} 筆）</button>')
    return f'<div id="tbl-{sid}">{btn}<table><thead>{hdr}</thead><tbody>{body}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default=str(SCRIPT_DIR/"lpd_four_way_iou.csv"))
    ap.add_argument("--out",    default=str(SCRIPT_DIR/"lpd_four_way_report.html"))
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--iou",    action="store_true", default=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    n = len(rows)
    print(f"[load] {n:,} rows", flush=True)

    ts_col = "twostep_match_valid"
    v2_col = "v2_match_iou"
    v3_col = "v3_match_iou"
    v4_col = "v4_match_iou"

    ts = count_match(rows, ts_col)
    v2 = count_match(rows, v2_col)
    v3 = count_match(rows, v3_col)
    v4 = count_match(rows, v4_col)

    ts_det = sum(1 for r in rows if r.get("twostep_detected","0")=="1")
    v2_det = sum(1 for r in rows if r.get("v2_detected","0")=="1")
    v3_det = sum(1 for r in rows if r.get("v3_detected","0")=="1")
    v4_det = sum(1 for r in rows if r.get("v4_detected","0")=="1")

    # IoU upgrade stats
    ts_orig    = sum(1 for r in rows if r.get("twostep_match","0")=="1")
    ts_invalid = ts_orig - ts
    v2_up = sum(1 for r in rows if r.get("v2_match","0")=="0" and r.get("v2_match_iou","0")=="1")
    v3_up = sum(1 for r in rows if r.get("v3_match","0")=="0" and r.get("v3_match_iou","0")=="1")
    v4_up = sum(1 for r in rows if r.get("v4_match","0")=="0" and r.get("v4_match_iou","0")=="1")

    # ── v4 vs v3 key cells ───────────────────────────────────────────────────
    def cell4(ts_v, v2_v, v3_v, v4_v):
        return [r for r in rows
                if r.get(ts_col,"0")==ts_v
                and r.get(v2_col,"0")==v2_v
                and r.get(v3_col,"0")==v3_v
                and r.get(v4_col,"0")==v4_v]

    # v4 gains vs v3 (v3=0, v4=1)
    v4_gain = [r for r in rows if r.get(v3_col,"0")=="0" and r.get(v4_col,"0")=="1"]
    # v4 losses vs v3 (v3=1, v4=0)
    v4_loss = [r for r in rows if r.get(v3_col,"0")=="1" and r.get(v4_col,"0")=="0"]
    # all four correct
    all_ok   = [r for r in rows if all(r.get(c,"0")=="1" for c in [ts_col,v2_col,v3_col,v4_col])]
    # v4 only gains (ts=0, v2=0, v3=0, v4=1)
    v4_only  = cell4("0","0","0","1")
    # v4 regression vs all (ts=1, v2=1, v3=1, v4=0)
    v4_regr  = cell4("1","1","1","0")
    # all fail
    all_fail = cell4("0","0","0","0")

    # ── per-subdir ────────────────────────────────────────────────────────────
    sub = defaultdict(list)
    for r in rows: sub[subdir_of(r["image_path"])].append(r)

    sub_html = ""
    for s in sorted(sub, key=lambda x: -len(sub[x])):
        sr = sub[s]; sn = len(sr)
        sts = count_match(sr, ts_col)
        sv2 = count_match(sr, v2_col)
        sv3 = count_match(sr, v3_col)
        sv4 = count_match(sr, v4_col)
        d43 = sv4 - sv3
        d42 = sv4 - sv2
        d4t = sv4 - sts
        def dspan(v):
            c = "pos" if v>0 else ("neg" if v<0 else "")
            return f"<span class='{c}'>{v:+,}</span>"
        sub_html += (
            f"<tr><td><code>{html.escape(s)}</code></td><td>{sn:,}</td>"
            f"<td>{sts:,} ({pct(sts,sn)})</td>"
            f"<td>{sv2:,} ({pct(sv2,sn)})</td>"
            f"<td>{sv3:,} ({pct(sv3,sn)})</td>"
            f"<td>{sv4:,} ({pct(sv4,sn)})</td>"
            f"<td>{dspan(d43)} ({pct(abs(d43),sn)})</td>"
            f"<td>{dspan(d42)} ({pct(abs(d42),sn)})</td>"
            f"<td>{dspan(d4t)} ({pct(abs(d4t),sn)})</td></tr>"
        )

    # ── section builder ───────────────────────────────────────────────────────
    _ids = iter(range(100))
    def sect(title, rows_list, css="", preview=15):
        if not rows_list: return ""
        sid = next(_ids)
        effective_preview = len(rows_list) if len(rows_list) <= 200 else preview
        tbl = section_table_html(rows_list, effective_preview, sid)
        pct_s = pct(len(rows_list), n)
        note = "全部展開" if effective_preview >= len(rows_list) else f"前 {effective_preview} 筆，可展開全部"
        return f"""<div class="card">
          <h2 class="{css}">{html.escape(title)} — {len(rows_list):,} 張 ({pct_s})</h2>
          <p class="sub">{note}</p>{tbl}</div>"""

    sects = "".join([
        sect("✓✓✓✓ 四者皆正確", all_ok, "pos"),
        sect("v4 新增（v3=✗, v4=✓）", v4_gain, "pos"),
        sect("v4 退化（v3=✓, v4=✗）", v4_loss, "warn"),
        sect("✗✗✗✓ 僅 v4 正確", v4_only, "pos"),
        sect("✓✓✓✗ v4 退化（三者皆對）", v4_regr, "warn"),
        sect("✗✗✗✗ 四者皆錯", all_fail, ""),
    ])

    html_out = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Four-way: Two-step vs v2 vs v3 vs v4 (IoU修正)</title>
  <style>
    body{{font-family:"Noto Sans TC",Arial,sans-serif;margin:0;padding:24px;background:#0f172a;color:#e2e8f0}}
    h1{{margin:0 0 4px;font-size:24px}}
    h2{{margin:0 0 10px;font-size:13px;text-transform:uppercase;color:#cbd5e1}}
    h2.pos{{color:#86efac}} h2.neg{{color:#fca5a5}} h2.warn{{color:#fcd34d}}
    .sub{{color:#94a3b8;margin:4px 0 14px;font-size:13px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:18px 22px;margin-bottom:16px}}
    .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:14px}}
    .stat{{background:#0f172a;padding:12px 16px;border-radius:6px;border:1px solid #334155}}
    .sl{{font-size:11px;color:#94a3b8;text-transform:uppercase}}
    .sv{{font-size:22px;font-weight:700;margin-top:2px}}
    .sv.pos{{color:#86efac}} .sv.neg{{color:#fca5a5}} .sv.hi{{color:#f59e0b}} .sv.v4{{color:#a78bfa}}
    table{{width:100%;border-collapse:collapse;font-size:12px;overflow-x:auto;display:block}}
    th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #334155;white-space:nowrap}}
    th{{color:#94a3b8;font-weight:500;background:#0f172a}}
    .pos{{color:#86efac;font-weight:600}} .neg{{color:#fca5a5;font-weight:600}} .warn{{color:#fcd34d;font-weight:600}}
    code{{background:#0f172a;padding:1px 6px;border-radius:3px;font-size:11px;color:#fbbf24}}
    .bar{{height:8px;background:#334155;border-radius:3px;margin-top:6px;overflow:hidden}}
    .bar-fill{{height:100%;border-radius:3px}}
    .hidden-row{{display:none}}
    .expand-btn{{margin-bottom:8px;padding:6px 14px;background:#1e3a5f;color:#93c5fd;
      border:1px solid #2563eb;border-radius:5px;cursor:pointer;font-size:12px}}
    .expand-btn:hover{{background:#1e40af}}
    .expand-btn.open{{color:#fca5a5;border-color:#ef4444;background:#3b1a1a}}
  </style>
  <script>
  function toggleRows(btn, sid) {{
    var rows = document.querySelectorAll('#tbl-'+sid+' tr.hidden-row');
    var open = btn.classList.toggle('open');
    rows.forEach(function(r){{ r.style.display = open ? '' : 'none'; }});
    btn.textContent = open ? '收起 ▲'
      : '展開全部 ▼ （還有 '+parseInt(btn.dataset.total||0).toLocaleString()+' 筆）';
  }}
  </script>
</head>
<body>
<h1>Four-way Report <span style="font-size:14px;color:#a78bfa;background:#1e1a3a;padding:2px 10px;border-radius:4px;margin-left:8px">IoU≥0.5 + min-OCR-4 修正</span></h1>
<div class="sub">Two-step vs Merged-v2 (ep30) vs Merged-v3 (ep60) vs Merged-v4 (ep292) · {n:,} 張圖片</div>

<div class="card" style="border-color:#7c3aed">
  <h2 style="color:#c4b5fd">修正說明</h2>
  <p class="sub">① Two-step OCR &lt; 4碼視為誤判（-{ts_invalid:,}）&nbsp; ② IoU≥0.5 且 two-step 正確 → 匡偵測正確</p>
  <div class="stats">
    <div class="stat"><div class="sl">TS 過短扣除</div><div class="sv warn">-{ts_invalid:,}</div></div>
    <div class="stat"><div class="sl">v2 IoU 升格</div><div class="sv" style="color:#c4b5fd">+{v2_up:,}</div></div>
    <div class="stat"><div class="sl">v3 IoU 升格</div><div class="sv" style="color:#c4b5fd">+{v3_up:,}</div></div>
    <div class="stat"><div class="sl">v4 IoU 升格</div><div class="sv" style="color:#a78bfa">+{v4_up:,}</div></div>
  </div>
</div>

<div class="card">
  <h2>Match 總覽（IoU修正後）</h2>
  <div class="stats">
    <div class="stat">
      <div class="sl">總圖片</div><div class="sv">{n:,}</div>
    </div>
    <div class="stat">
      <div class="sl">Two-step</div>
      <div class="sv">{ts:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(ts,n)}</div>
      <div class="bar"><div class="bar-fill" style="width:{ts/n*100:.1f}%;background:#64748b"></div></div>
    </div>
    <div class="stat">
      <div class="sl">Merged-v2 (ep30)</div>
      <div class="sv {'pos' if v2>ts else 'neg'}">{v2:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(v2,n)} <span class="{'pos' if v2>ts else 'neg'}">({v2-ts:+,} vs TS)</span></div>
      <div class="bar"><div class="bar-fill" style="width:{v2/n*100:.1f}%;background:#22c55e"></div></div>
    </div>
    <div class="stat">
      <div class="sl">Merged-v3 (ep60)</div>
      <div class="sv hi">{v3:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(v3,n)} <span class="{'pos' if v3>ts else 'neg'}">({v3-ts:+,} vs TS)</span> <span class="{'pos' if v3>v2 else 'neg'}">({v3-v2:+,} vs v2)</span></div>
      <div class="bar"><div class="bar-fill" style="width:{v3/n*100:.1f}%;background:#f59e0b"></div></div>
    </div>
    <div class="stat" style="border-color:#7c3aed">
      <div class="sl">Merged-v4 (ep292) ★</div>
      <div class="sv v4">{v4:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(v4,n)}
        <span class="{'pos' if v4>ts else 'neg'}">({v4-ts:+,} vs TS)</span>
        <span class="{'pos' if v4>v3 else 'neg'}">({v4-v3:+,} vs v3)</span></div>
      <div class="bar"><div class="bar-fill" style="width:{v4/n*100:.1f}%;background:#a78bfa"></div></div>
    </div>
    <div class="stat"><div class="sl">偵測率 TS</div><div class="sv">{pct(ts_det,n)}</div></div>
    <div class="stat"><div class="sl">偵測率 v2</div><div class="sv">{pct(v2_det,n)}</div></div>
    <div class="stat"><div class="sl">偵測率 v3</div><div class="sv">{pct(v3_det,n)}</div></div>
    <div class="stat"><div class="sl">偵測率 v4</div><div class="sv">{pct(v4_det,n)}</div></div>
  </div>
</div>

<div class="card">
  <h2>v4 vs v3 差異摘要</h2>
  <div class="stats">
    <div class="stat"><div class="sl">v4 新增（v3✗→v4✓）</div><div class="sv pos">+{len(v4_gain):,}</div><div class="sub">{pct(len(v4_gain),n)}</div></div>
    <div class="stat"><div class="sl">v4 退化（v3✓→v4✗）</div><div class="sv neg">-{len(v4_loss):,}</div><div class="sub">{pct(len(v4_loss),n)}</div></div>
    <div class="stat"><div class="sl">淨增益</div>
      <div class="sv {'pos' if len(v4_gain)>len(v4_loss) else 'neg'}">{len(v4_gain)-len(v4_loss):+,}</div></div>
    <div class="stat"><div class="sl">僅 v4 正確</div><div class="sv v4">{len(v4_only):,}</div></div>
    <div class="stat"><div class="sl">v4 一人退化</div><div class="sv warn">{len(v4_regr):,}</div></div>
    <div class="stat"><div class="sl">四者皆正確</div><div class="sv pos">{len(all_ok):,}</div><div class="sub">{pct(len(all_ok),n)}</div></div>
    <div class="stat"><div class="sl">四者皆錯</div><div class="sv">{len(all_fail):,}</div><div class="sub">{pct(len(all_fail),n)}</div></div>
  </div>
</div>

<div class="card">
  <h2>各子目錄比較</h2>
  <table>
    <thead><tr>
      <th>子目錄</th><th>n</th>
      <th>Two-step</th><th>v2</th><th>v3</th><th>v4★</th>
      <th>v4 vs v3</th><th>v4 vs v2</th><th>v4 vs TS</th>
    </tr></thead>
    <tbody>{sub_html}</tbody>
  </table>
</div>

{sects}

</body></html>"""

    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"[done] {args.out}", flush=True)
    print(f"\nSummary (IoU) ({n:,} images):")
    print(f"  Two-step : {ts:,} ({pct(ts,n)})")
    print(f"  Merged-v2: {v2:,} ({pct(v2,n)})  {v2-ts:+,} vs TS")
    print(f"  Merged-v3: {v3:,} ({pct(v3,n)})  {v3-ts:+,} vs TS  {v3-v2:+,} vs v2")
    print(f"  Merged-v4: {v4:,} ({pct(v4,n)})  {v4-ts:+,} vs TS  {v4-v3:+,} vs v3  {v4-v2:+,} vs v2")
    print(f"\n  v4 新增(v3✗→v4✓): {len(v4_gain):,}  退化(v3✓→v4✗): {len(v4_loss):,}  淨: {len(v4_gain)-len(v4_loss):+,}")

if __name__ == "__main__":
    main()
