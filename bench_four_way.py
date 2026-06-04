#!/usr/bin/env python3
"""Four-way benchmark: Two-step vs Merged-v2 vs Merged-v3 vs Merged-v4.

Re-uses two-step results from lpd_results.csv.
Runs v2, v3, v4 ONNX in one pass per image with shared PARSeq session.

Output: lpd_four_way.csv
"""

import argparse, csv, json, re, sys, time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_lpr_two_step_compare import (
    decode_best_detection, preprocess_best, sort_corners, warp_plate, BEST_IMGSZ,
)

# ── PARSeq ────────────────────────────────────────────────────────────────────

_parseq = _device = _transform = None

def _nltk_stub():
    import types
    from importlib.machinery import ModuleSpec
    if "nltk" in sys.modules:
        return
    s = types.ModuleType("nltk"); s.edit_distance = lambda a,b:0
    s.__spec__ = ModuleSpec("nltk",loader=None); s.__path__ = []
    sys.modules["nltk"] = s

def init_parseq():
    global _parseq, _device, _transform
    if _parseq: return
    _nltk_stub()
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
    if isinstance(preds, (tuple,list)) and len(preds)==2:
        preds, _ = preds
    return preds[0] if isinstance(preds,(list,tuple)) else preds

def clean(s):
    return re.sub(r"[^0-9A-Z]","",s.upper()) if s else ""

def is_match(ocr_clean, expected):
    return bool(ocr_clean and expected and ocr_clean in expected)

def corners_json(c):
    if c is None: return ""
    return json.dumps([[float(x),float(y)] for x,y in c], separators=(",",":"))

# ── per-model inference ───────────────────────────────────────────────────────

def run_model(sess, out_names, in_name, img, conf):
    try:
        tensor, gain, pad = preprocess_best(img, BEST_IMGSZ)
        outs = sess.run(out_names, {in_name: tensor})
        return decode_best_detection(outs[0], gain, pad, img.shape, conf)
    except Exception:
        return None

def process_model(info, img):
    if info is None:
        return "0","","","","0"
    corners = info["keypoints"]
    try:
        plate = warp_plate(img, corners)
        raw   = parseq_ocr(plate)
        clean_s = clean(raw)
        return "1", corners_json(corners), raw, clean_s, ""
    except Exception:
        return "1", corners_json(corners), "", "", ""

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2",    default=str(SCRIPT_DIR/"artifacts/yolo26n-merged-v2-20260530-143438/best.onnx"))
    ap.add_argument("--v3",    default=str(SCRIPT_DIR/"artifacts/yolo26n-merged-v3-20260602/best.onnx"))
    ap.add_argument("--v4",    default=str(SCRIPT_DIR/"artifacts/yolo26n-merged-v4-20260604/best.onnx"))
    ap.add_argument("--base",  default=str(SCRIPT_DIR/"lpd_results.csv"))
    ap.add_argument("--out",   default=str(SCRIPT_DIR/"lpd_four_way.csv"))
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--limit", type=int,   default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.base)))
    if args.limit: rows = rows[:args.limit]
    print(f"[init] {len(rows):,} images", flush=True)

    init_parseq()

    def load_sess(path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        s = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        return s, s.get_inputs()[0].name, [o.name for o in s.get_outputs()]

    sess_v2, in_v2, out_v2 = load_sess(args.v2)
    sess_v3, in_v3, out_v3 = load_sess(args.v3)
    sess_v4, in_v4, out_v4 = load_sess(args.v4)
    print(f"[init] v2: {Path(args.v2).parent.name}", flush=True)
    print(f"[init] v3: {Path(args.v3).parent.name}", flush=True)
    print(f"[init] v4: {Path(args.v4).parent.name}", flush=True)

    HEADER = [
        "image_path","expected_plate",
        "twostep_detected","twostep_corners","twostep_ocr_clean","twostep_match",
        "v2_detected","v2_corners","v2_ocr_raw","v2_ocr_clean","v2_match",
        "v3_detected","v3_corners","v3_ocr_raw","v3_ocr_clean","v3_match",
        "v4_detected","v4_corners","v4_ocr_raw","v4_ocr_clean","v4_match",
    ]

    fout = open(args.out, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(HEADER)

    t0 = time.perf_counter(); n = len(rows)
    for i, r in enumerate(rows):
        img = cv2.imread(r["image_path"])
        if img is None:
            writer.writerow([r["image_path"], r["expected_plate"],
                             r["twostep_detected"], r.get("twostep_corners",""),
                             r["twostep_ocr_clean"], r["twostep_match"],
                             "0","","","","0",
                             "0","","","","0",
                             "0","","","","0"])
            continue

        info_v2 = run_model(sess_v2, out_v2, in_v2, img, args.conf)
        info_v3 = run_model(sess_v3, out_v3, in_v3, img, args.conf)
        info_v4 = run_model(sess_v4, out_v4, in_v4, img, args.conf)

        v2_det, v2_c, v2_raw, v2_cl, _ = process_model(info_v2, img)
        v3_det, v3_c, v3_raw, v3_cl, _ = process_model(info_v3, img)
        v4_det, v4_c, v4_raw, v4_cl, _ = process_model(info_v4, img)

        v2_match = "1" if is_match(v2_cl, r["expected_plate"]) else "0"
        v3_match = "1" if is_match(v3_cl, r["expected_plate"]) else "0"
        v4_match = "1" if is_match(v4_cl, r["expected_plate"]) else "0"

        writer.writerow([
            r["image_path"], r["expected_plate"],
            r["twostep_detected"], r.get("twostep_corners",""),
            r["twostep_ocr_clean"], r["twostep_match"],
            v2_det, v2_c, v2_raw, v2_cl, v2_match,
            v3_det, v3_c, v3_raw, v3_cl, v3_match,
            v4_det, v4_c, v4_raw, v4_cl, v4_match,
        ])

        if (i+1) % 500 == 0:
            dt = time.perf_counter() - t0
            rate = (i+1)/dt
            eta  = (n-i-1)/rate
            fout.flush()
            print(f"[progress] {i+1:,}/{n:,}  rate={rate:.1f}/s  eta={int(eta//60)}m{int(eta%60)}s", flush=True)

    fout.close()
    print(f"[done] {args.out}", flush=True)

if __name__ == "__main__":
    main()
