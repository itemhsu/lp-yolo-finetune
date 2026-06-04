#!/usr/bin/env python3
"""Test Two-step vs Merged-v4 on a directory of images.

Filename convention: <PLATE>_<idx>.jpg  (expected plate = stem before first '_')
Outputs: test_compare_report.html (self-contained with inline base64 images)
"""

import argparse, base64, re, sys, html as html_mod
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_lpr_two_step_compare import (
    decode_plate_det, preprocess_plate_det,
    preprocess_rectifier, rectifier_corners_to_image,
    sort_corners, warp_plate,
    preprocess_best, decode_best_detection, BEST_IMGSZ,
)

# ── PARSeq ────────────────────────────────────────────────────────────────────

_parseq = _device = _transform = None

def init_parseq():
    global _parseq, _device, _transform
    if _parseq: return
    import types
    from importlib.machinery import ModuleSpec
    if "nltk" not in sys.modules:
        s = types.ModuleType("nltk"); s.edit_distance = lambda a,b:0
        s.__spec__ = ModuleSpec("nltk",loader=None); s.__path__ = []
        sys.modules["nltk"] = s
    import torch
    from torchvision import transforms
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] PARSeq on {_device}", flush=True)
    _parseq = torch.hub.load("baudm/parseq","parseq",pretrained=True).eval().to(_device)
    _transform = transforms.Compose([
        transforms.Resize((32,128)), transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

def parseq_ocr(plate_bgr):
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    inp = _transform(pil).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = _parseq(inp).softmax(-1)
    preds = _parseq.tokenizer.decode(probs)
    if isinstance(preds,(tuple,list)) and len(preds)==2:
        preds,_ = preds
    return preds[0] if isinstance(preds,(list,tuple)) else preds

def clean(s):
    return re.sub(r"[^0-9A-Z]","",s.upper()) if s else ""

def is_plate_like(ocr):
    """Taiwan plate: 5-8 alphanumeric chars; empty / too short / too long = x."""
    return bool(ocr) and 5 <= len(ocr) <= 8

# ── helpers ───────────────────────────────────────────────────────────────────

def img_to_b64(bgr, max_w=400):
    if bgr is None: return ""
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h*max_w/w)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else ""

def make_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=opts,
                             providers=["CPUExecutionProvider"])
    return s

def expected_from_name(stem):
    # e.g. "ACX7999_0" → "ACX7999"
    part = stem.split("_")[0]
    return clean(part)

# ── two-step inference ────────────────────────────────────────────────────────

def run_two_step(img, det_sess, rect_sess, det_conf=0.5):
    det_in  = det_sess.get_inputs()[0].name
    det_out = [o.name for o in det_sess.get_outputs()]
    rect_in  = rect_sess.get_inputs()[0].name
    rect_out = [o.name for o in rect_sess.get_outputs()]

    tensor = preprocess_plate_det(img)
    outs = det_sess.run(det_out, {det_in: tensor})
    box = decode_plate_det(outs, img.shape, conf_thresh=det_conf)
    if not box:
        return None, None, None

    x1,y1,x2,y2 = box["bbox"]
    plate_crop = img[max(0,y1):y2, max(0,x1):x2]
    if plate_crop.size == 0:
        return None, None, None

    rect_tensor = preprocess_rectifier(plate_crop)
    rect_outs = rect_sess.run(rect_out, {rect_in: rect_tensor})
    corners_rel = rectifier_corners_to_image(rect_outs[0], (x1,y1,x2,y2))
    corners = sort_corners(corners_rel)
    warped = warp_plate(img, corners)
    return corners, warped, box["bbox"]

# ── v4 inference ──────────────────────────────────────────────────────────────

def run_v4(img, v4_sess, conf=0.25):
    in_name  = v4_sess.get_inputs()[0].name
    out_names = [o.name for o in v4_sess.get_outputs()]
    try:
        tensor, gain, pad = preprocess_best(img, BEST_IMGSZ)
        outs = v4_sess.run(out_names, {in_name: tensor})
        info = decode_best_detection(outs[0], gain, pad, img.shape, conf)
    except Exception:
        return None, None
    if info is None:
        return None, None
    corners = info["keypoints"]
    warped = warp_plate(img, corners)
    return corners, warped

# ── draw corners on thumbnail ─────────────────────────────────────────────────

def draw_corners(img, corners, color):
    out = img.copy()
    if corners is None: return out
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(out, [pts], True, color, 2)
    for p in pts:
        cv2.circle(out, tuple(p), 4, color, -1)
    return out

# ── HTML ──────────────────────────────────────────────────────────────────────

STYLE = """
body{font-family:"Noto Sans TC",Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}
h1{margin:0 0 8px;font-size:22px}
.sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
.summary{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.scard{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 18px;min-width:130px;cursor:pointer;transition:border-color .15s}
.scard:hover{border-color:#64748b}
.scard.active{border-color:#a78bfa;background:#1e1a3a}
.sl{font-size:11px;color:#94a3b8;text-transform:uppercase}
.sv{font-size:22px;font-weight:700;margin-top:2px}
.pos{color:#86efac}.neg{color:#fca5a5}.warn{color:#fcd34d}.v4{color:#a78bfa}
.filter-bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap;
  background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 16px}
.filter-bar label{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;user-select:none}
.filter-bar input[type=checkbox]{width:14px;height:14px;accent-color:#a78bfa;cursor:pointer}
.filter-bar .sep{color:#334155;margin:0 4px}
#count-info{font-size:12px;color:#94a3b8;margin-left:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #1e293b;vertical-align:middle}
th{background:#1e293b;color:#94a3b8;font-weight:500;position:sticky;top:0;
   cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:#e2e8f0}
th.sort-asc::after{content:" ▲";font-size:10px}
th.sort-desc::after{content:" ▼";font-size:10px}
tr:hover td{background:#1e2d40}
img.thumb{border-radius:4px;max-height:60px;cursor:pointer}
img.plate{border-radius:3px;max-height:32px;border:1px solid #334155}
code{background:#0f172a;padding:1px 6px;border-radius:3px;font-size:11px;color:#fbbf24}
.match1{color:#86efac;font-weight:700} .match0{color:#fca5a5}
.no-det{color:#64748b;font-style:italic}
tr.hidden-filter{display:none}
"""

JS = """
(function(){
  // ── match filter (group 1) ──
  var mChecks = {
    '11': document.getElementById('f-both'),
    '10': document.getElementById('f-only-ts'),
    '01': document.getElementById('f-only-v4'),
    '00': document.getElementById('f-neither')
  };
  // ── OCR plate-like filter (group 2) ──
  var oChecks = {
    '11': document.getElementById('f-ocr-vv'),
    '10': document.getElementById('f-ocr-vx'),
    '01': document.getElementById('f-ocr-xv'),
    '00': document.getElementById('f-ocr-xx')
  };

  function applyFilter(){
    var mShow = {}, oShow = {};
    Object.keys(mChecks).forEach(function(k){ mShow[k] = mChecks[k].checked; });
    Object.keys(oChecks).forEach(function(k){ oShow[k] = oChecks[k].checked; });

    var rows = document.querySelectorAll('tbody tr');
    var vis = 0;
    rows.forEach(function(r){
      var ok = mShow[r.dataset.cat] === true && oShow[r.dataset.ocat] === true;
      r.classList.toggle('hidden-filter', !ok);
      if(ok) vis++;
    });
    document.getElementById('count-info').textContent = '顯示 '+vis+' / '+rows.length+' 筆';
  }

  document.querySelectorAll('.filter-bar input[type=checkbox]')
    .forEach(function(cb){ cb.addEventListener('change', applyFilter); });
  applyFilter();

  // ── sorting ──
  var sortCol = -1, sortAsc = true;
  var thead = document.querySelector('thead tr');
  thead.querySelectorAll('th').forEach(function(th, ci){
    th.addEventListener('click', function(){
      if(sortCol===ci){ sortAsc=!sortAsc; }
      else { sortCol=ci; sortAsc=true; }
      thead.querySelectorAll('th').forEach(function(h){ h.className=''; });
      th.className = sortAsc ? 'sort-asc' : 'sort-desc';
      var tbody = document.querySelector('tbody');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort(function(a,b){
        var av = a.children[ci] ? a.children[ci].dataset.val||a.children[ci].textContent.trim() : '';
        var bv = b.children[ci] ? b.children[ci].dataset.val||b.children[ci].textContent.trim() : '';
        var n = parseFloat(av), m = parseFloat(bv);
        var cmp = (!isNaN(n)&&!isNaN(m)) ? n-m : av.localeCompare(bv,'zh-Hant');
        return sortAsc ? cmp : -cmp;
      });
      rows.forEach(function(r){ tbody.appendChild(r); });
    });
  });
})();
"""

def build_html(results, img_dir):
    n = len(results)
    ts_match  = sum(1 for r in results if r["ts_match"])
    v4_match  = sum(1 for r in results if r["v4_match"])
    both      = sum(1 for r in results if r["ts_match"] and r["v4_match"])
    only_ts   = sum(1 for r in results if r["ts_match"] and not r["v4_match"])
    only_v4   = sum(1 for r in results if not r["ts_match"] and r["v4_match"])
    neither   = sum(1 for r in results if not r["ts_match"] and not r["v4_match"])

    # OCR quality counts
    ocr_vv = sum(1 for r in results if r["ts_ocr_plate"] and r["v4_ocr_plate"])
    ocr_vx = sum(1 for r in results if r["ts_ocr_plate"] and not r["v4_ocr_plate"])
    ocr_xv = sum(1 for r in results if not r["ts_ocr_plate"] and r["v4_ocr_plate"])
    ocr_xx = sum(1 for r in results if not r["ts_ocr_plate"] and not r["v4_ocr_plate"])

    def pct(a, b): return f"{a/b*100:.1f}%" if b else "—"

    summary = f"""
<div class="summary">
  <div class="scard"><div class="sl">總圖片</div><div class="sv">{n}</div></div>
  <div class="scard"><div class="sl">Two-step match</div><div class="sv {'pos' if ts_match>0 else ''}">{ts_match}</div><div style="font-size:12px;color:#94a3b8">{pct(ts_match,n)}</div></div>
  <div class="scard"><div class="sl">v4 match</div><div class="sv v4">{v4_match}</div><div style="font-size:12px;color:#94a3b8">{pct(v4_match,n)}</div></div>
  <div class="scard"><div class="sl">兩者皆對 ✓✓</div><div class="sv pos">{both}</div></div>
  <div class="scard"><div class="sl">僅 TS 對 ✓✗</div><div class="sv warn">{only_ts}</div></div>
  <div class="scard"><div class="sl">僅 v4 對 ✗✓</div><div class="sv v4">{only_v4}</div></div>
  <div class="scard"><div class="sl">兩者皆錯 ✗✗</div><div class="sv neg">{neither}</div></div>
</div>"""

    rows_html = ""
    for idx, r in enumerate(results):
        fname  = Path(r["path"]).name
        ts_m   = r["ts_match"]
        v4_m   = r["v4_match"]
        ts_p   = r["ts_ocr_plate"]
        v4_p   = r["v4_ocr_plate"]
        cat    = ("1" if ts_m else "0") + ("1" if v4_m else "0")
        ocat   = ("1" if ts_p else "0") + ("1" if v4_p else "0")
        ts_cls = "match1" if ts_m else "match0"
        v4_cls = "match1 v4" if v4_m else "match0"

        thumb_tag = (f'<img class="thumb" src="data:image/jpeg;base64,{r["thumb_b64"]}">'
                     if r["thumb_b64"] else "—")

        def plate_tag(b64): return (f'<img class="plate" src="data:image/jpeg;base64,{b64}">'
                                    if b64 else '<span class="no-det">未偵測</span>')

        def ocr_disp(ocr, plate_like):
            badge = '<span style="color:#86efac;font-size:10px;margin-left:4px">v</span>' if plate_like \
                    else '<span style="color:#64748b;font-size:10px;margin-left:4px">x</span>'
            if ocr:
                return f'<code>{html_mod.escape(ocr)}</code>{badge}'
            return f'<span class="no-det">—</span>{badge}'

        rows_html += (
            f'<tr data-cat="{cat}" data-ocat="{ocat}">'
            f'<td data-val="{idx}">{thumb_tag}</td>'
            f'<td data-val="{html_mod.escape(fname)}"><code style="font-size:10px">{html_mod.escape(fname)}</code></td>'
            f'<td data-val="{html_mod.escape(r["expected"])}"><b>{html_mod.escape(r["expected"])}</b></td>'
            f'<td>{plate_tag(r["ts_plate_b64"])}</td>'
            f'<td data-val="{html_mod.escape(r["ts_ocr"])}">{ocr_disp(r["ts_ocr"], ts_p)}</td>'
            f'<td data-val="{1 if ts_m else 0}" class="{ts_cls}">{"✓" if ts_m else "✗"}</td>'
            f'<td>{plate_tag(r["v4_plate_b64"])}</td>'
            f'<td data-val="{html_mod.escape(r["v4_ocr"])}">{ocr_disp(r["v4_ocr"], v4_p)}</td>'
            f'<td data-val="{1 if v4_m else 0}" class="{v4_cls}">{"✓" if v4_m else "✗"}</td>'
            f'</tr>\n'
        )

    filter_bar = f"""
<div class="filter-bar">
  <b style="font-size:12px;color:#cbd5e1;margin-right:4px">Match：</b>
  <label><input type="checkbox" id="f-both" checked>
    <span class="pos">✓✓</span> 兩者皆對 ({both})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-only-ts" checked>
    <span class="warn">✓✗</span> 僅 TS 對 ({only_ts})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-only-v4" checked>
    <span class="v4">✗✓</span> 僅 v4 對 ({only_v4})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-neither" checked>
    <span class="neg">✗✗</span> 兩者皆錯 ({neither})</label>
  <span id="count-info" style="margin-left:auto;color:#94a3b8;font-size:12px"></span>
</div>
<div class="filter-bar" style="margin-top:-6px">
  <b style="font-size:12px;color:#cbd5e1;margin-right:4px">OCR像車牌：</b>
  <label><input type="checkbox" id="f-ocr-vv" checked>
    <span style="color:#86efac">vv</span> 兩者皆像 ({ocr_vv})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-ocr-vx" checked>
    <span style="color:#fcd34d">vx</span> 僅 TS 像 ({ocr_vx})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-ocr-xv" checked>
    <span style="color:#a78bfa">xv</span> 僅 v4 像 ({ocr_xv})</label>
  <span class="sep">|</span>
  <label><input type="checkbox" id="f-ocr-xx" checked>
    <span style="color:#64748b">xx</span> 兩者皆不像 ({ocr_xx})</label>
</div>"""

    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8">
<title>Test: Two-step vs Merged-v4 — {html_mod.escape(str(img_dir))}</title>
<style>{STYLE}</style>
</head><body>
<h1>Two-step vs Merged-v4 測試</h1>
<div class="sub">{html_mod.escape(str(img_dir))} · {n} 張圖片</div>
{summary}
{filter_bar}
<table>
<thead><tr>
  <th>#</th><th>檔名</th><th>期望車牌</th>
  <th>TS 板</th><th>TS OCR</th><th>TS✓</th>
  <th>v4 板</th><th>v4 OCR</th><th>v4✓</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<script>{JS}</script>
</body></html>"""

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", nargs="?",
                    default="/home/itemhsu/amtk/lppart/lp/lp_viewer_tool/lp_viewer_output/testImg/1111")
    ap.add_argument("--plate-det",  default="/home/itemhsu/amtk/lpr/0x1PlateDet/PlateDet.onnx")
    ap.add_argument("--rectifier",  default="/home/itemhsu/amtk/lpr/PlateRectifier.onnx")
    ap.add_argument("--v4",         default=str(SCRIPT_DIR/"artifacts/yolo26n-merged-v4-20260604/best.onnx"))
    ap.add_argument("--out",        default=str(SCRIPT_DIR/"test_compare_report.html"))
    ap.add_argument("--det-conf",   type=float, default=0.5)
    ap.add_argument("--v4-conf",    type=float, default=0.25)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    images  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"[init] {len(images)} images in {img_dir}", flush=True)

    init_parseq()
    det_sess  = make_session(args.plate_det)
    rect_sess = make_session(args.rectifier)
    v4_sess   = make_session(args.v4)
    print(f"[init] models loaded", flush=True)

    results = []
    ts_ok = v4_ok = 0
    for i, imgp in enumerate(images):
        img = cv2.imread(str(imgp))
        if img is None:
            continue

        expected = expected_from_name(imgp.stem)

        # two-step
        ts_corners, ts_warped, ts_bbox = run_two_step(img, det_sess, rect_sess, args.det_conf)
        ts_ocr = clean(parseq_ocr(ts_warped)) if ts_warped is not None else ""
        ts_match = bool(ts_ocr and expected and ts_ocr in expected)

        # v4
        v4_corners, v4_warped = run_v4(img, v4_sess, args.v4_conf)
        v4_ocr = clean(parseq_ocr(v4_warped)) if v4_warped is not None else ""
        v4_match = bool(v4_ocr and expected and v4_ocr in expected)

        if ts_match: ts_ok += 1
        if v4_match: v4_ok += 1

        # thumbnails
        vis = img.copy()
        if ts_corners is not None:
            vis = draw_corners(vis, ts_corners, (0, 200, 0))
        if v4_corners is not None:
            vis = draw_corners(vis, v4_corners, (180, 0, 255))

        results.append({
            "path":          str(imgp),
            "expected":      expected,
            "thumb_b64":     img_to_b64(vis, max_w=320),
            "ts_ocr":        ts_ocr,
            "ts_match":      ts_match,
            "ts_ocr_plate":  is_plate_like(ts_ocr),
            "ts_plate_b64":  img_to_b64(ts_warped, max_w=400) if ts_warped is not None else "",
            "v4_ocr":        v4_ocr,
            "v4_match":      v4_match,
            "v4_ocr_plate":  is_plate_like(v4_ocr),
            "v4_plate_b64":  img_to_b64(v4_warped, max_w=400) if v4_warped is not None else "",
        })

        if (i+1) % 20 == 0 or (i+1) == len(images):
            print(f"[progress] {i+1}/{len(images)}  ts={ts_ok}  v4={v4_ok}", flush=True)

    html_out = build_html(results, img_dir)
    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"[done] {args.out}", flush=True)
    n = len(results)
    print(f"\n  Two-step : {ts_ok}/{n} ({ts_ok/n*100:.1f}%)")
    print(f"  Merged-v4: {v4_ok}/{n} ({v4_ok/n*100:.1f}%)")

if __name__ == "__main__":
    main()
