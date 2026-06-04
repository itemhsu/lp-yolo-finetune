#!/usr/bin/env python3
"""Live dashboard for merged-v4 training (300 epochs).

Writes merged_v4_data.js every loop (polled by merged_v4_dashboard.html).
Shows v4 curve vs v3 reference lines.
"""

import os, re, time, csv, json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
LOG_FILE    = SCRIPT_DIR / "merged_v4_train.log"
RESULTS_CSV = SCRIPT_DIR / "runs/merged-v4/train/results.csv"
OUT_HTML    = SCRIPT_DIR / "merged_v4_dashboard.html"
OUT_JS      = SCRIPT_DIR / "merged_v4_data.js"
TOTAL_EP    = 300

# History CSVs: v2 (ep1-30) and v3 continuation (ep31-60)
V2_CSV = SCRIPT_DIR / "artifacts/yolo26n-merged-v2-20260530-143438/runs/train/results.csv"
V3_CSV = SCRIPT_DIR / "runs/pose/runs/merged-v3/continue/results.csv"
V4_OFFSET = 60  # v4 epochs are displayed starting from ep 61

RE_DONE = re.compile(r'Results saved to|Training complete')
RE_ERR  = re.compile(r'Traceback|Error:|FAILED|Killed')


def load_csv(path):
    if not path.exists():
        return []
    try:
        return list(csv.DictReader(open(path)))
    except Exception:
        return []


def parse_log():
    state = {"cur": 0, "total": TOTAL_EP, "done": False, "error": "", "gpu_mem": "—"}
    if not LOG_FILE.exists():
        return state
    try:
        text = LOG_FILE.read_text(errors='replace')
    except Exception:
        return state
    pat = re.compile(rf'(\d+)/{TOTAL_EP}\s+([\d.]+)G')
    for line in text.splitlines():
        if RE_ERR.search(line):  state["error"] = line[:120]
        if RE_DONE.search(line): state["done"] = True
        m = pat.search(line)
        if m:
            state["cur"]     = int(m.group(1))
            state["gpu_mem"] = m.group(2) + "G"
    return state


def rows_to_arrays(rows, offset=0):
    ep, tb, tp, vb, vp, mb, mp = [], [], [], [], [], [], []
    for r in rows:
        try:
            ep.append(int(r['epoch']) + offset)
            tb.append(round(float(r['train/box_loss']), 5))
            tp.append(round(float(r['train/pose_loss']), 5))
            vb.append(round(float(r['val/box_loss']), 5))
            vp.append(round(float(r['val/pose_loss']), 5))
            mb.append(round(float(r['metrics/mAP50(B)']), 5))
            mp.append(round(float(r['metrics/mAP50(P)']), 5))
        except Exception:
            continue
    return ep, tb, tp, vb, vp, mb, mp


def write_js(state, v4_rows):
    # v4 new data only (offset +60)
    ep4, tb4, tp4, vb4, vp4, mb4, mp4 = rows_to_arrays(v4_rows, offset=V4_OFFSET)
    latest = ""
    if v4_rows:
        r = v4_rows[-1]
        try:
            latest = (f"ep {int(r['epoch'])+V4_OFFSET} | "
                      f"mAP50(B)={float(r['metrics/mAP50(B)']):.4f} "
                      f"mAP50(P)={float(r['metrics/mAP50(P)']):.4f} | "
                      f"val/box={float(r['val/box_loss']):.4f} "
                      f"val/pose={float(r['val/pose_loss']):.4f}")
        except Exception:
            pass
    data = {
        "updated":    datetime.now().strftime('%H:%M:%S'),
        "cur":        state["cur"],
        "total":      state["total"],
        "done":       state["done"],
        "error":      state["error"],
        "gpu_mem":    state["gpu_mem"],
        "n_rows":     len(v4_rows),
        "latest":     latest,
        "newEpochs":  ep4,
        "newTrainBox": tb4, "newTrainPose": tp4,
        "newValBox":  vb4, "newValPose":   vp4,
        "newMapB":    mb4, "newMapP":      mp4,
    }
    tmp = OUT_JS.with_suffix('.js.tmp')
    tmp.write_text("window.__v4data=" + json.dumps(data) + ";", encoding='utf-8')
    os.replace(tmp, OUT_JS)


def render_html(v2_rows, v3_rows):
    ep2, tb2, tp2, vb2, vp2, mb2, mp2 = rows_to_arrays(v2_rows, offset=0)
    ep3, tb3, tp3, vb3, vp3, mb3, mp3 = rows_to_arrays(v3_rows, offset=30)
    prev_ep  = ep2 + ep3
    prev_tb  = tb2 + tb3;  prev_tp = tp2 + tp3
    prev_vb  = vb2 + vb3;  prev_vp = vp2 + vp3
    prev_mb  = mb2 + mb3;  prev_mp = mp2 + mp3
    v3_best_vb = round(min(prev_vb[29:]), 5) if len(prev_vb) >= 30 else 1.05
    v3_best_mb = round(max(prev_mb[29:]), 5) if len(prev_mb) >= 30 else 0.853
    v3_best_mp = round(max(prev_mp[29:]), 5) if len(prev_mp) >= 30 else 0.685

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Merged-v4 訓練監控（300 ep）</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
  <style>
    body{{font-family:"Noto Sans TC",Arial,sans-serif;margin:0;padding:20px 24px;background:#0f172a;color:#e2e8f0}}
    h1{{font-size:21px;margin:0 0 3px}}
    .sub{{color:#94a3b8;font-size:12px;margin:0 0 18px}}
    .pill{{display:inline-block;padding:4px 14px;border-radius:16px;font-weight:800;font-size:13px;margin-left:8px}}
    .pill-run{{background:#14532d;color:#86efac;animation:pulse 2s infinite}}
    .pill-done{{background:#1e3a8a;color:#93c5fd}}
    .pill-err{{background:#7f1d1d;color:#fca5a5}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:9px;padding:14px 18px}}
    h2{{font-size:11px;text-transform:uppercase;color:#94a3b8;margin:0 0 10px}}
    canvas{{max-height:220px}}
    .stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}}
    .stat{{background:#1e293b;border:1px solid #334155;border-radius:7px;padding:10px 14px}}
    .sl{{font-size:10px;color:#94a3b8;text-transform:uppercase}}
    .sv{{font-size:18px;font-weight:700;margin-top:2px}}
    .prog{{width:100%;height:28px;background:#334155;border-radius:14px;overflow:hidden;position:relative;margin:10px 0 4px}}
    .progf{{height:100%;background:linear-gradient(90deg,#7c3aed,#06b6d4);transition:width .6s ease}}
    .progt{{position:absolute;left:0;right:0;top:0;line-height:28px;text-align:center;font-weight:700;font-size:13px;color:#fff}}
    .latest{{font-size:12px;color:#94a3b8;margin:6px 0 0;font-family:monospace}}
    .err{{background:#7f1d1d;color:#fecaca;padding:8px 12px;border-radius:4px;font-size:12px;font-family:monospace;margin-bottom:12px;display:none}}
    .ref-tag{{font-size:10px;color:#fcd34d;margin-left:6px}}
  </style>
</head>
<body>
<h1>Merged-v4 訓練（300 epochs）<span id="pill" class="pill pill-run">RUNNING</span></h1>
<div class="sub">更新：<span id="ts">—</span> · 輪詢 10s · 起點：v3 best.pt (ep59) · lr0=0.0001
  <span class="ref-tag">— 虛線 = v3 參考值</span></div>

<div id="errBox" class="err"></div>

<div class="stats">
  <div class="stat"><div class="sl">已完成 epoch</div><div class="sv" id="s-ep">—</div></div>
  <div class="stat"><div class="sl">進度</div><div class="sv" id="s-prog">—</div></div>
  <div class="stat"><div class="sl">GPU Mem</div><div class="sv" id="s-gpu">—</div></div>
  <div class="stat"><div class="sl">val/box best</div><div class="sv" id="s-vbest">—</div></div>
  <div class="stat"><div class="sl">mAP50(B) best</div><div class="sv" id="s-mbest">—</div></div>
</div>

<div class="card" style="margin-bottom:14px">
  <h2>進度</h2>
  <div class="prog"><div class="progf" id="progf" style="width:0%"></div>
    <div class="progt" id="progt">0/300 (0.0%)</div></div>
  <div class="latest" id="latest">等待第一個 epoch...</div>
</div>

<div class="grid">
  <div class="card"><h2>Box Loss</h2><canvas id="cBoxLoss"></canvas></div>
  <div class="card"><h2>Pose Loss</h2><canvas id="cPoseLoss"></canvas></div>
  <div class="card"><h2>mAP50 — Box &amp; Pose</h2><canvas id="cMap"></canvas></div>
  <div class="card"><h2>Val Loss</h2><canvas id="cValLoss"></canvas></div>
</div>

<script>
// ── v2+v3 歷史資料（ep 1-60）嵌入 ──
const P_EP  = {prev_ep};
const P_TB  = {prev_tb};  const P_TP = {prev_tp};
const P_VB  = {prev_vb};  const P_VP = {prev_vp};
const P_MB  = {prev_mb};  const P_MP = {prev_mp};

const V3_BEST_VB = {v3_best_vb};
const V3_BEST_MB = {v3_best_mb};
const V3_BEST_MP = {v3_best_mp};

const GRID = '#334155';
const EP60_ANN = {{
  type:'line', xMin:60, xMax:60,
  borderColor:'rgba(99,102,241,0.7)', borderWidth:1.5, borderDash:[4,3],
  label:{{content:'ep60 v3→v4', display:true, position:'start',
          color:'#a5b4fc', font:{{size:9}}, backgroundColor:'rgba(0,0,0,0.5)'}}
}};

function refLineY(val, label) {{
  return {{
    type:'line', yMin:val, yMax:val,
    borderColor:'rgba(252,211,77,0.55)', borderWidth:1.2, borderDash:[5,3],
    label:{{content:label+' '+val, display:true, position:'end',
            color:'#fcd34d', font:{{size:9}}, backgroundColor:'rgba(0,0,0,0.45)'}}
  }};
}}

const BASE = (anns) => ({{
  responsive:true, animation:false,
  plugins:{{
    legend:{{labels:{{color:'#94a3b8',font:{{size:10}}}}}},
    annotation:{{annotations:anns}}
  }},
  scales:{{
    x:{{ticks:{{color:'#64748b',font:{{size:9}},maxTicksLimit:20}},grid:{{color:GRID}},
       title:{{display:true,text:'Epoch (累計)',color:'#64748b',font:{{size:10}}}}}},
    y:{{ticks:{{color:'#64748b',font:{{size:9}}}},grid:{{color:GRID}}}}
  }}
}});

const charts = {{}};

function mkDS(label, data, color, radius) {{
  return {{label, data, borderColor:color, tension:0.3, pointRadius:radius||1.5,
           backgroundColor:color.replace('rgb(','rgba(').replace(')',',0.06)')}};
}}

function buildCharts(ep, tb, tp, vb, vp, mb, mp) {{
  function mk(id, dsets, anns) {{
    charts[id] = new Chart(document.getElementById(id), {{
      type:'line', data:{{labels:ep, datasets:dsets}}, options:BASE(anns)
    }});
  }}
  mk('cBoxLoss',
    [mkDS('train/box',tb,'rgb(59,130,246)'), mkDS('val/box',vb,'rgb(245,158,11)')],
    {{ep60:EP60_ANN, refVB:refLineY(V3_BEST_VB,'v3 best val/box')}});
  mk('cPoseLoss',
    [mkDS('train/pose',tp,'rgb(139,92,246)'), mkDS('val/pose',vp,'rgb(239,68,68)')],
    {{ep60:EP60_ANN}});
  mk('cMap',
    [mkDS('mAP50(B)',mb,'rgb(34,197,94)'), mkDS('mAP50(P)',mp,'rgb(6,182,212)')],
    {{ep60:EP60_ANN, refMB:refLineY(V3_BEST_MB,'v3 best mAP(B)'), refMP:refLineY(V3_BEST_MP,'v3 best mAP(P)')}});
  mk('cValLoss',
    [mkDS('val/box',vb,'rgb(245,158,11)'), mkDS('val/pose',vp,'rgb(239,68,68)')],
    {{ep60:EP60_ANN, refVB:refLineY(V3_BEST_VB,'v3 best val/box')}});
}}

function updCharts(ep, tb, tp, vb, vp, mb, mp) {{
  const u = (id, arrs) => {{ const c=charts[id]; c.data.labels=ep; arrs.forEach((d,i)=>c.data.datasets[i].data=d); c.update('none'); }};
  u('cBoxLoss',[tb,vb]); u('cPoseLoss',[tp,vp]); u('cMap',[mb,mp]); u('cValLoss',[vb,vp]);
}}

// 初始化：只用歷史資料先畫出 ep1-60
buildCharts([...P_EP],[...P_TB],[...P_TP],[...P_VB],[...P_VP],[...P_MB],[...P_MP]);

let lastRows = -1;

function applyData(d) {{
  document.getElementById('ts').textContent = d.updated;
  const pill=document.getElementById('pill');
  if(d.done)       {{pill.className='pill pill-done';pill.textContent='DONE';}}
  else if(d.error) {{pill.className='pill pill-err'; pill.textContent='ERROR';}}
  else             {{pill.className='pill pill-run'; pill.textContent='RUNNING';}}
  const eb=document.getElementById('errBox');
  if(d.error){{eb.style.display='block';eb.textContent=d.error;}}else{{eb.style.display='none';}}

  const pct=d.total?(d.cur/d.total*100).toFixed(1):0;
  document.getElementById('s-ep').textContent   = d.cur;
  document.getElementById('s-prog').textContent = d.cur+'/'+d.total;
  document.getElementById('s-gpu').textContent  = d.gpu_mem;
  document.getElementById('progf').style.width  = pct+'%';
  document.getElementById('progt').textContent  = d.cur+'/'+d.total+' ('+pct+'%)';
  if(d.latest) document.getElementById('latest').textContent=d.latest;

  if(d.newValBox && d.newValBox.length) {{
    const allVb = P_VB.concat(d.newValBox);
    const allMb = P_MB.concat(d.newMapB);
    document.getElementById('s-vbest').textContent=Math.min(...allVb).toFixed(4);
    document.getElementById('s-mbest').textContent=Math.max(...allMb).toFixed(4);
  }}

  if(d.n_rows === lastRows) return;
  lastRows = d.n_rows;

  // concat 歷史 + v4 新資料
  const ep = P_EP.concat(d.newEpochs);
  updCharts(ep,
    P_TB.concat(d.newTrainBox), P_TP.concat(d.newTrainPose),
    P_VB.concat(d.newValBox),   P_VP.concat(d.newValPose),
    P_MB.concat(d.newMapB),     P_MP.concat(d.newMapP));
}}

function poll() {{
  const s=document.createElement('script');
  s.src='merged_v4_data.js?t='+Date.now();
  s.onload=()=>{{if(window.__v4data)applyData(window.__v4data);s.remove();}};
  s.onerror=()=>{{document.getElementById('ts').textContent+=' ?';s.remove();}};
  document.head.appendChild(s);
}}
poll();
setInterval(poll,10000);
</script>
</body>
</html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', type=int, default=0)
    args = ap.parse_args()

    v2_rows = load_csv(V2_CSV)
    v3_rows = load_csv(V3_CSV)
    OUT_HTML.write_text(render_html(v2_rows, v3_rows), encoding='utf-8')
    print("[v4-dash] HTML written", flush=True)

    while True:
        state = parse_log()
        rows  = load_csv(RESULTS_CSV)
        write_js(state, rows)
        print(f"[v4-dash] ep={state['cur']}/{TOTAL_EP}  rows={len(rows)}  done={state['done']}", flush=True)
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
