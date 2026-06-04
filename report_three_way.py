#!/usr/bin/env python3
"""Three-way HTML report: Two-step vs Merged-v2 vs Merged-v3.

Reads lpd_three_way.csv produced by bench_three_way.py.
"""

import argparse, csv, html
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

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

def count_match(rows, col): return sum(1 for r in rows if r[col]=="1")

_sect_id = 0

def section_table_html(rows, preview, sect_id):
    """Table with first `preview` rows visible; rest hidden behind a toggle button."""
    global _sect_id
    hdr = ("<tr><th>#</th><th>檔名</th><th>期望車牌</th>"
           "<th>Two-step OCR</th><th>TS✓</th>"
           "<th>v2 OCR</th><th>v2✓</th>"
           "<th>v3 OCR</th><th>v3✓</th></tr>")
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
                 + td_m(r.get("twostep_match","0")) +
                 f"<td><code>{html.escape(r.get('v2_ocr_clean',''))}</code></td>"
                 + td_m(r.get("v2_match","0")) +
                 f"<td><code>{html.escape(r.get('v3_ocr_clean',''))}</code></td>"
                 + td_m(r.get("v3_match","0")) +
                 f"</tr>")
    hidden_count = max(0, len(rows) - preview)
    btn = ""
    if hidden_count > 0:
        btn = (f'<button class="expand-btn" data-sid="{sect_id}" data-total="{hidden_count}" '
               f'onclick="toggleRows(this,\'{sect_id}\')">'
               f'展開全部 ▼ （還有 {hidden_count:,} 筆）</button>')
    return f'<div id="tbl-{sect_id}">{btn}<table><thead>{hdr}</thead><tbody>{body}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(SCRIPT_DIR/"lpd_three_way.csv"))
    ap.add_argument("--out", default=str(SCRIPT_DIR/"lpd_three_way_report.html"))
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--iou", action="store_true",
                    help="Use v2_match_iou/v3_match_iou columns (from recompute_iou_matches.py)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    n = len(rows)
    print(f"[load] {n:,} rows  iou={args.iou}", flush=True)

    ts_col = "twostep_match_valid" if args.iou else "twostep_match"
    v2_col = "v2_match_iou"       if args.iou else "v2_match"
    v3_col = "v3_match_iou"       if args.iou else "v3_match"

    ts  = count_match(rows, ts_col)
    v2  = count_match(rows, v2_col)
    v3  = count_match(rows, v3_col)
    ts_det = sum(1 for r in rows if r.get("twostep_detected","0")=="1")
    v2_det = sum(1 for r in rows if r.get("v2_detected","0")=="1")
    v3_det = sum(1 for r in rows if r.get("v3_detected","0")=="1")

    # ── 8-cell breakdown (3-way) ─────────────────────────────────────────
    def cell(ts_v, v2_v, v3_v):
        return [r for r in rows
                if r.get(ts_col,"0")==ts_v
                and r.get(v2_col,"0")==v2_v
                and r.get(v3_col,"0")==v3_v]

    c111 = cell("1","1","1")   # all correct
    c110 = cell("1","1","0")   # v3 regression
    c101 = cell("1","0","1")   # v2 regression, v3 recovered
    c100 = cell("1","0","0")   # both merged fail
    c011 = cell("0","1","1")   # merged both gained
    c010 = cell("0","1","0")   # only v2 gained (v3 lost)
    c001 = cell("0","0","1")   # only v3 gained
    c000 = cell("0","0","0")   # all fail

    # ── per-subdir ───────────────────────────────────────────────────────
    sub = defaultdict(list)
    for r in rows: sub[subdir_of(r["image_path"])].append(r)

    sub_html = ""
    for s in sorted(sub, key=lambda x: -len(sub[x])):
        sr = sub[s]; sn = len(sr)
        sts = count_match(sr, ts_col)
        sv2 = count_match(sr, v2_col)
        sv3 = count_match(sr, v3_col)
        d2  = sv2-sts; d3 = sv3-sts; d32 = sv3-sv2
        def dspan(v):
            c = "pos" if v>0 else ("neg" if v<0 else "")
            return f"<span class='{c}'>{v:+,}</span>"
        sub_html += (
            f"<tr><td><code>{html.escape(s)}</code></td><td>{sn:,}</td>"
            f"<td>{sts:,} ({pct(sts,sn)})</td>"
            f"<td>{sv2:,} ({pct(sv2,sn)})</td>"
            f"<td>{sv3:,} ({pct(sv3,sn)})</td>"
            f"<td>{dspan(d2)} ({pct(abs(d2),sn)})</td>"
            f"<td>{dspan(d3)} ({pct(abs(d3),sn)})</td>"
            f"<td>{dspan(d32)} ({pct(abs(d32),sn)})</td></tr>"
        )

    # ── cell sections ────────────────────────────────────────────────────
    _ids = iter(range(100))
    def sect(title, rows_list, css="", preview=15):
        if not rows_list: return ""
        sid = next(_ids)
        # auto-expand if small enough
        effective_preview = len(rows_list) if len(rows_list) <= 200 else preview
        tbl = section_table_html(rows_list, effective_preview, sid)
        pct_s = pct(len(rows_list), n)
        shown = min(effective_preview, len(rows_list))
        note = "全部展開" if shown == len(rows_list) else f"前 {shown} 筆，可展開全部"
        return f"""<div class="card">
          <h2 class="{css}">{html.escape(title)} — {len(rows_list):,} 張 ({pct_s})</h2>
          <p class="sub">{note}</p>{tbl}</div>"""

    sects = "".join([
        sect("✓✓✓ 三者皆正確", c111, "pos"),
        sect("✓✓✗ v3 退化（ts+v2 正確，v3 錯）", c110, "warn"),
        sect("✓✗✓ v2 退化後 v3 找回", c101, ""),
        sect("✓✗✗ ts 正確，merged 兩版皆錯", c100, "neg"),
        sect("✗✓✓ merged 兩版皆比 ts 好", c011, "pos"),
        sect("✗✓✗ 只有 v2 正確（v3 未保住）", c010, "warn"),
        sect("✗✗✓ 只有 v3 新增正確", c001, "pos"),
        sect("✗✗✗ 三者皆錯", c000, ""),
    ])

    iou_note = " + IoU 匡重疊修正" if args.iou else ""
    iou_badge = (
        f'<span style="background:#7c3aed;color:#ede9fe;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;margin-left:8px">IoU≥0.5 匡修正</span>'
        if args.iou else ""
    )

    # IoU upgrade stats for banner
    if args.iou:
        ts_orig  = sum(1 for r in rows if r.get("twostep_match","0")=="1")
        ts_valid = sum(1 for r in rows if r.get("twostep_match_valid","0")=="1")
        v2_upgrades = sum(1 for r in rows if r.get("v2_match","0")=="0" and r.get("v2_match_iou","0")=="1")
        v3_upgrades = sum(1 for r in rows if r.get("v3_match","0")=="0" and r.get("v3_match_iou","0")=="1")
        iou_stats_html = f"""<div class="card" style="border-color:#7c3aed">
  <h2 style="color:#c4b5fd">修正說明</h2>
  <p class="sub">① Two-step 最短車牌長度過濾（OCR &lt; 4碼視為誤判）&nbsp; ② v2/v3 匡重疊率 ≥ 0.5 → 匡偵測正確</p>
  <div class="stats">
    <div class="stat"><div class="sl">Two-step 過短誤判扣除</div><div class="sv warn">-{ts_orig-ts_valid:,}</div></div>
    <div class="stat"><div class="sl">v2 IoU 升格</div><div class="sv" style="color:#c4b5fd">+{v2_upgrades:,}</div></div>
    <div class="stat"><div class="sl">v3 IoU 升格</div><div class="sv" style="color:#c4b5fd">+{v3_upgrades:,}</div></div>
  </div>
</div>"""
    else:
        iou_stats_html = ""

    html_out = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Three-way: Two-step vs Merged-v2 vs Merged-v3{iou_note}</title>
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
    .sv.pos{{color:#86efac}} .sv.neg{{color:#fca5a5}} .sv.hi{{color:#93c5fd}}
    table{{width:100%;border-collapse:collapse;font-size:12px;overflow-x:auto;display:block}}
    th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #334155;white-space:nowrap}}
    th{{color:#94a3b8;font-weight:500;background:#0f172a}}
    .pos{{color:#86efac;font-weight:600}} .neg{{color:#fca5a5;font-weight:600}}
    .warn{{color:#fcd34d;font-weight:600}}
    code{{background:#0f172a;padding:1px 6px;border-radius:3px;font-size:11px;color:#fbbf24}}
    .bar{{height:6px;background:#334155;border-radius:3px;margin-top:6px;overflow:hidden}}
    .bar-fill{{height:100%;border-radius:3px}}
    .matrix{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}}
    .cell{{background:#0f172a;border-radius:6px;padding:10px 12px;border:1px solid #334155}}
    .cell-t{{font-size:10px;color:#94a3b8;margin-bottom:4px}}
    .cell-n{{font-size:20px;font-weight:800}}
    .cell-p{{font-size:11px;color:#64748b}}
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
    if (open) {{
      btn.textContent = '收起 ▲';
    }} else {{
      var total = parseInt(btn.dataset.total||0);
      btn.textContent = '展開全部 ▼ （還有 ' + total.toLocaleString() + ' 筆）';
    }}
  }}
  </script>
</head>
<body>
<h1>Three-way OCR Match Report{iou_badge}</h1>
<div class="sub">Two-step vs Merged-v2 (ep30) vs Merged-v3 (ep60) · {n:,} 張圖片{iou_note}</div>
{iou_stats_html}

<div class="card">
  <h2>OCR Match 總覽</h2>
  <div class="stats">
    <div class="stat">
      <div class="sl">總圖片</div><div class="sv">{n:,}</div>
    </div>
    <div class="stat">
      <div class="sl">Two-step match</div>
      <div class="sv">{ts:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(ts,n)}</div>
      <div class="bar"><div class="bar-fill" style="width:{ts/n*100:.1f}%;background:#64748b"></div></div>
    </div>
    <div class="stat">
      <div class="sl">Merged-v2 match (ep30)</div>
      <div class="sv {'pos' if v2>ts else 'neg'}">{v2:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(v2,n)} <span class="{'pos' if v2>ts else 'neg'}">({v2-ts:+,} vs TS)</span></div>
      <div class="bar"><div class="bar-fill" style="width:{v2/n*100:.1f}%;background:#22c55e"></div></div>
    </div>
    <div class="stat" style="border-color:#2563eb">
      <div class="sl">Merged-v3 match (ep60)</div>
      <div class="sv hi">{v3:,}</div>
      <div class="sub" style="margin:2px 0 0">{pct(v3,n)} <span class="{'pos' if v3>ts else 'neg'}">({v3-ts:+,} vs TS)</span> <span class="{'pos' if v3>v2 else 'neg'}">({v3-v2:+,} vs v2)</span></div>
      <div class="bar"><div class="bar-fill" style="width:{v3/n*100:.1f}%;background:#3b82f6"></div></div>
    </div>
    <div class="stat">
      <div class="sl">偵測率 Two-step</div><div class="sv">{pct(ts_det,n)}</div>
    </div>
    <div class="stat">
      <div class="sl">偵測率 Merged-v2</div><div class="sv">{pct(v2_det,n)}</div>
    </div>
    <div class="stat">
      <div class="sl">偵測率 Merged-v3</div><div class="sv">{pct(v3_det,n)}</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>8-cell 分類矩陣（Three-way）</h2>
  <div class="matrix">
    <div class="cell"><div class="cell-t">✓✓✓ 三者皆對</div><div class="cell-n pos">{len(c111):,}</div><div class="cell-p">{pct(len(c111),n)}</div></div>
    <div class="cell"><div class="cell-t">✓✓✗ v3 退化</div><div class="cell-n warn">{len(c110):,}</div><div class="cell-p">{pct(len(c110),n)}</div></div>
    <div class="cell"><div class="cell-t">✓✗✓ v2 退化 v3 救回</div><div class="cell-n">{len(c101):,}</div><div class="cell-p">{pct(len(c101),n)}</div></div>
    <div class="cell"><div class="cell-t">✓✗✗ merged 兩版皆錯</div><div class="cell-n neg">{len(c100):,}</div><div class="cell-p">{pct(len(c100),n)}</div></div>
    <div class="cell"><div class="cell-t">✗✓✓ merged 兩版皆贏</div><div class="cell-n pos">{len(c011):,}</div><div class="cell-p">{pct(len(c011),n)}</div></div>
    <div class="cell"><div class="cell-t">✗✓✗ 只 v2 正確</div><div class="cell-n warn">{len(c010):,}</div><div class="cell-p">{pct(len(c010),n)}</div></div>
    <div class="cell"><div class="cell-t">✗✗✓ 只 v3 新增</div><div class="cell-n pos">{len(c001):,}</div><div class="cell-p">{pct(len(c001),n)}</div></div>
    <div class="cell"><div class="cell-t">✗✗✗ 三者皆錯</div><div class="cell-n">{len(c000):,}</div><div class="cell-p">{pct(len(c000),n)}</div></div>
  </div>
  <p class="sub" style="margin-top:12px">
    v3 比 v2 多（c101+c001）：{len(c101)+len(c001):,} 張 &nbsp;|&nbsp;
    v3 比 v2 少（c110+c010）：{len(c110)+len(c010):,} 張 &nbsp;|&nbsp;
    淨增益：<span class="{'pos' if len(c101)+len(c001)-len(c110)-len(c010)>=0 else 'neg'}">{len(c101)+len(c001)-len(c110)-len(c010):+,}</span>
  </p>
</div>

<div class="card">
  <h2>各子目錄 OCR Match 比較</h2>
  <table>
    <thead><tr>
      <th>子目錄</th><th>n</th>
      <th>Two-step</th><th>Merged-v2</th><th>Merged-v3</th>
      <th>v2 vs TS</th><th>v3 vs TS</th><th>v3 vs v2</th>
    </tr></thead>
    <tbody>{sub_html}</tbody>
  </table>
</div>

{sects}

</body></html>"""

    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"[done] {args.out}", flush=True)
    tag = "(IoU)" if args.iou else "(OCR)"
    print(f"\nSummary {tag} ({n:,} images):")
    print(f"  Two-step : {ts:,} ({pct(ts,n)})")
    print(f"  Merged-v2: {v2:,} ({pct(v2,n)})  {v2-ts:+,} vs TS")
    print(f"  Merged-v3: {v3:,} ({pct(v3,n)})  {v3-ts:+,} vs TS  {v3-v2:+,} vs v2")
    v3g = len(c101)+len(c001); v3l = len(c110)+len(c010)
    print(f"\n  v3 比v2多 c101+c001={v3g:,}（其中修復c101={len(c101):,} 純增c001={len(c001):,}）")
    print(f"  v3 比v2少 c110+c010={v3l:,}  淨: {v3g-v3l:+,}")

if __name__ == "__main__":
    main()
