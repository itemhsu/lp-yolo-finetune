#!/usr/bin/env python3
"""Live training dashboard for merged-v3 local training.

Writes merged_v3_data.json every loop iteration.
HTML shell (merged_v3_dashboard.html) is written once and polls the JSON via JS —
no page reload, no flicker. Charts update in-place only when new epoch data arrives.
"""

import os, re, time, csv, json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
LOG_FILE    = SCRIPT_DIR / "merged_v3_train.log"
RESULTS_CSV = SCRIPT_DIR / "runs/pose/runs/merged-v3/continue/results.csv"
OUT_HTML    = SCRIPT_DIR / "merged_v3_dashboard.html"
OUT_JS      = SCRIPT_DIR / "merged_v3_data.js"
PREV_CSV    = SCRIPT_DIR / "artifacts/yolo26n-merged-v2-20260530-143438/runs/train/results.csv"

RE_DONE = re.compile(r'Results saved to|Training complete')
RE_ERR  = re.compile(r'Traceback|Error:|FAILED|Killed')


def load_results_csv(path):
    if not path.exists():
        return []
    try:
        return list(csv.DictReader(open(path)))
    except Exception:
        return []


def parse_log():
    state = {"cur": 0, "total": 30, "done": False, "error": "", "gpu_mem": "—"}
    if not LOG_FILE.exists():
        return state
    try:
        text = LOG_FILE.read_text(errors='replace')
    except Exception:
        return state
    for line in text.splitlines():
        if RE_ERR.search(line):
            state["error"] = line[:120]
        if RE_DONE.search(line):
            state["done"] = True
        m = re.search(r'(\d+)/30\s+([\d.]+)G', line)
        if m:
            state["cur"] = int(m.group(1))
            state["gpu_mem"] = m.group(2) + "G"
    return state


def rows_to_arrays(rows, offset=0):
    epochs, tr_box, tr_pose, vl_box, vl_pose, map_b, map_p = [], [], [], [], [], [], []
    for r in rows:
        try:
            epochs.append(int(r['epoch']) + offset)
            tr_box.append(round(float(r['train/box_loss']), 5))
            tr_pose.append(round(float(r['train/pose_loss']), 5))
            vl_box.append(round(float(r['val/box_loss']), 5))
            vl_pose.append(round(float(r['val/pose_loss']), 5))
            map_b.append(round(float(r['metrics/mAP50(B)']), 5))
            map_p.append(round(float(r['metrics/mAP50(P)']), 5))
        except Exception:
            continue
    return epochs, tr_box, tr_pose, vl_box, vl_pose, map_b, map_p


def write_js(state, new_rows):
    ep2, tb2, tp2, vb2, vp2, mb2, mp2 = rows_to_arrays(new_rows, offset=30)
    latest = ""
    if new_rows:
        r = new_rows[-1]
        try:
            latest = (f"ep {int(r['epoch'])+30} | "
                      f"mAP50(B)={float(r['metrics/mAP50(B)']):.4f} "
                      f"mAP50(P)={float(r['metrics/mAP50(P)']):.4f} | "
                      f"val/box={float(r['val/box_loss']):.4f} "
                      f"val/pose={float(r['val/pose_loss']):.4f}")
        except Exception:
            pass
    data = {
        "updated":      datetime.now().strftime('%H:%M:%S'),
        "cur":          state["cur"],
        "total":        state["total"],
        "done":         state["done"],
        "error":        state["error"],
        "gpu_mem":      state["gpu_mem"],
        "new_rows":     len(new_rows),
        "latest":       latest,
        "newEpochs":    ep2,
        "newTrainBox":  tb2,
        "newTrainPose": tp2,
        "newValBox":    vb2,
        "newValPose":   vp2,
        "newMapB":      mb2,
        "newMapP":      mp2,
    }
    tmp = OUT_JS.with_suffix('.js.tmp')
    tmp.write_text("window.__dashData=" + json.dumps(data) + ";", encoding='utf-8')
    os.replace(tmp, OUT_JS)


def render_html(prev_rows):
    ep1, tb1, tp1, vb1, vp1, mb1, mp1 = rows_to_arrays(prev_rows, offset=0)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Merged-v3 訓練監控</title>
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
    canvas{{max-height:210px}}
    .stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}}
    .stat{{background:#1e293b;border:1px solid #334155;border-radius:7px;padding:10px 14px}}
    .sl{{font-size:10px;color:#94a3b8;text-transform:uppercase}}
    .sv{{font-size:18px;font-weight:700;margin-top:2px}}
    .prog{{width:100%;height:28px;background:#334155;border-radius:14px;overflow:hidden;position:relative;margin:10px 0 4px}}
    .progf{{height:100%;background:linear-gradient(90deg,#7c3aed,#06b6d4);transition:width .6s ease}}
    .progt{{position:absolute;left:0;right:0;top:0;line-height:28px;text-align:center;font-weight:700;font-size:13px;color:#fff}}
    .latest{{font-size:12px;color:#94a3b8;margin:6px 0 0;font-family:monospace}}
    .err{{background:#7f1d1d;color:#fecaca;padding:8px 12px;border-radius:4px;font-size:12px;font-family:monospace;margin-bottom:12px;display:none}}
  </style>
</head>
<body>
<h1>Merged-v3 繼續訓練（ep 31→60）<span id="pill" class="pill pill-run">RUNNING</span></h1>
<div class="sub">更新：<span id="ts">—</span> · 輪詢 10s（資料不變不重繪）· Base: merged-v2 best.pt · epochs=30+30</div>

<div id="errBox" class="err"></div>

<div class="stats">
  <div class="stat"><div class="sl">已完成 epoch</div><div class="sv" id="s-done">—</div></div>
  <div class="stat"><div class="sl">本輪進度</div><div class="sv" id="s-prog">—</div></div>
  <div class="stat"><div class="sl">GPU Mem</div><div class="sv" id="s-gpu">—</div></div>
  <div class="stat"><div class="sl">新增 results</div><div class="sv" id="s-rows">—</div></div>
  <div class="stat"><div class="sl">總 epoch</div><div class="sv" id="s-total">—</div></div>
</div>

<div class="card" style="margin-bottom:14px">
  <h2>本輪進度</h2>
  <div class="prog"><div class="progf" id="progf" style="width:0%"></div><div class="progt" id="progt">0/30 (0%)</div></div>
  <div class="latest" id="latest">等待第一個 epoch...</div>
</div>

<div class="grid">
  <div class="card"><h2>Box Loss（偵測框）</h2><canvas id="boxLoss"></canvas></div>
  <div class="card"><h2>Pose Loss（角點）</h2><canvas id="poseLoss"></canvas></div>
  <div class="card"><h2>mAP50 — Box &amp; Pose</h2><canvas id="map50"></canvas></div>
  <div class="card"><h2>Val Loss 收斂</h2><canvas id="valLoss"></canvas></div>
</div>

<script>
// ── Prev data (ep 1-30) embedded at generation time ──
const P_EP  = {ep1};
const P_TB  = {tb1};
const P_TP  = {tp1};
const P_VB  = {vb1};
const P_VP  = {vp1};
const P_MB  = {mb1};
const P_MP  = {mp1};

const GRID = '#334155';
const ANN = {{
  annotations: {{
    ep30: {{
      type:'line', xMin:30, xMax:30,
      borderColor:'#fcd34d', borderWidth:1.5, borderDash:[4,3],
      label:{{content:'ep30', display:true, position:'start',
              color:'#fcd34d', font:{{size:10}}, backgroundColor:'rgba(0,0,0,0.5)'}}
    }}
  }}
}};
const BASE = {{
  responsive:true, animation:false,
  plugins:{{legend:{{labels:{{color:'#94a3b8',font:{{size:10}}}}}}, annotation:ANN}},
  scales:{{
    x:{{ticks:{{color:'#64748b',font:{{size:9}}}},grid:{{color:GRID}},
       title:{{display:true,text:'Epoch (累計)',color:'#64748b',font:{{size:10}}}}}},
    y:{{ticks:{{color:'#64748b',font:{{size:9}}}},grid:{{color:GRID}}}}
  }}
}};

function mkDatasets(cfgs) {{
  return cfgs.map(([label, data, color]) => ({{
    label, data, borderColor:color, tension:0.3, pointRadius:2,
    backgroundColor: color.replace(')', ',0.06)').replace('rgb','rgba')
  }}));
}}

const charts = {{}};
function initCharts(ep, tb, tp, vb, vp, mb, mp) {{
  function mk(id, cfgs) {{
    charts[id] = new Chart(document.getElementById(id), {{
      type:'line', data:{{labels:ep, datasets:mkDatasets(cfgs)}}, options:BASE
    }});
  }}
  mk('boxLoss',  [['train/box',tb,'#3b82f6'],['val/box',vb,'#f59e0b']]);
  mk('poseLoss', [['train/pose',tp,'#8b5cf6'],['val/pose',vp,'#ef4444']]);
  mk('map50',    [['mAP50(B)',mb,'#22c55e'],['mAP50(P)',mp,'#06b6d4']]);
  mk('valLoss',  [['val/box',vb,'#f59e0b'],['val/pose',vp,'#ef4444']]);
}}

function updateCharts(ep, tb, tp, vb, vp, mb, mp) {{
  const sets = {{
    boxLoss:  [tb, vb],
    poseLoss: [tp, vp],
    map50:    [mb, mp],
    valLoss:  [vb, vp],
  }};
  for (const [id, datas] of Object.entries(sets)) {{
    const c = charts[id];
    c.data.labels = ep;
    datas.forEach((d, i) => c.data.datasets[i].data = d);
    c.update('none');
  }}
}}

// Init with prev data only
initCharts(P_EP, P_TB, P_TP, P_VB, P_VP, P_MB, P_MP);

// ── Polling via dynamic <script> (works with file://) ──
let lastNewRows = -1;

function applyData(d) {{
  document.getElementById('ts').textContent = d.updated;

  const pill = document.getElementById('pill');
  if (d.done)       {{ pill.className='pill pill-done'; pill.textContent='DONE'; }}
  else if (d.error) {{ pill.className='pill pill-err';  pill.textContent='ERROR'; }}
  else              {{ pill.className='pill pill-run';  pill.textContent='RUNNING'; }}

  const eb = document.getElementById('errBox');
  if (d.error) {{ eb.style.display='block'; eb.textContent=d.error; }}
  else         {{ eb.style.display='none'; }}

  const cur=d.cur, total=d.total;
  document.getElementById('s-done').textContent  = '30 + ' + cur;
  document.getElementById('s-prog').textContent  = cur + '/' + total;
  document.getElementById('s-gpu').textContent   = d.gpu_mem;
  document.getElementById('s-rows').textContent  = d.new_rows;
  document.getElementById('s-total').textContent = (30+cur) + '/60';
  const pct = total ? (cur/total*100).toFixed(1) : 0;
  document.getElementById('progf').style.width = pct + '%';
  document.getElementById('progt').textContent  = cur+'/'+total+' ('+pct+'%)';
  if (d.latest) document.getElementById('latest').textContent = d.latest;

  if (d.new_rows === lastNewRows) return;
  lastNewRows = d.new_rows;

  const ep = P_EP.concat(d.newEpochs);
  updateCharts(ep,
    P_TB.concat(d.newTrainBox),  P_TP.concat(d.newTrainPose),
    P_VB.concat(d.newValBox),    P_VP.concat(d.newValPose),
    P_MB.concat(d.newMapB),      P_MP.concat(d.newMapP));
}}

function poll() {{
  const s = document.createElement('script');
  s.src = 'merged_v3_data.js?t=' + Date.now();
  s.onload = () => {{ if (window.__dashData) applyData(window.__dashData); s.remove(); }};
  s.onerror = () => {{ document.getElementById('ts').textContent += ' ?'; s.remove(); }};
  document.head.appendChild(s);
}}

poll();
setInterval(poll, 10000);
</script>
</body>
</html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', type=int, default=0)
    args = ap.parse_args()

    prev_rows = load_results_csv(PREV_CSV)

    # Write HTML shell once
    OUT_HTML.write_text(render_html(prev_rows), encoding='utf-8')
    print("[dash] HTML written (static shell, polls merged_v3_data.json)", flush=True)

    while True:
        state    = parse_log()
        new_rows = load_results_csv(RESULTS_CSV)
        write_js(state, new_rows)
        print(f"[dash] ep={30+state['cur']}/60  new_rows={len(new_rows)}  done={state['done']}", flush=True)
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
