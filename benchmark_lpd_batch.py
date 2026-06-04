#!/usr/bin/env python3
"""Batch-run 4 LPR pipelines (two-step + 3 YOLOv26) on every image under LPD,
PARSeq-OCR each detected plate, and record results in a resumable CSV.

For each batch of --batch-size images we process pipelines one at a time
(all images for pipeline A, then all images for pipeline B, ...) to minimize
session/cache thrash, then flush one batch of rows to disk.

Rows already present in the CSV are skipped on restart.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_lpr_two_step_compare import (
    decode_best_detection,
    decode_plate_det,
    make_session,
    preprocess_best,
    preprocess_plate_det,
    preprocess_rectifier,
    rectifier_corners_to_image,
    sort_corners,
    warp_plate,
    BEST_IMGSZ,
)


PIPELINE_KEYS = ["twostep", "yolo26m", "yolo26s", "yolo26n"]

YOLO26_LABEL_MAP = {
    "yolo26-train-1775509671/best.onnx": "yolo26m",
    "yolo26-train-1773128236/best.onnx": "yolo26s",
    "yolo26-train-1779339446/best.onnx": "yolo26n",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

CSV_HEADER = ["image_path", "expected_plate"] + [
    f"{key}_{field}"
    for key in PIPELINE_KEYS
    for field in ("detected", "corners", "ocr_raw", "ocr_clean", "match")
]


# ============================================================
# PARSeq (pretrained, baudm/parseq — same as lp_simple_pipeline.py)
# ============================================================

_parseq_model = None
_parseq_device = None
_parseq_transform = None


def _install_nltk_stub():
    # PARSeq's base.py does `from nltk import edit_distance`, which pulls in
    # nltk -> sklearn -> pandas. On this machine system pandas 1.3.5 has a
    # numpy ABI mismatch with numpy 2.x. edit_distance is only used at
    # train/eval time; for inference we can satisfy the import with a stub.
    # transformers also calls importlib.util.find_spec("nltk"), so the stub
    # needs a __spec__ to look like a real module.
    import sys
    import types
    from importlib.machinery import ModuleSpec
    if "nltk" in sys.modules:
        return
    stub = types.ModuleType("nltk")
    stub.edit_distance = lambda a, b: 0
    stub.__spec__ = ModuleSpec("nltk", loader=None)
    stub.__path__ = []
    sys.modules["nltk"] = stub


def init_parseq():
    global _parseq_model, _parseq_device, _parseq_transform
    if _parseq_model is not None:
        return
    _install_nltk_stub()
    import torch
    from torchvision import transforms

    _parseq_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] Loading PARSeq on {_parseq_device} ...", flush=True)
    _parseq_model = torch.hub.load("baudm/parseq", "parseq", pretrained=True)
    _parseq_model = _parseq_model.eval().to(_parseq_device)
    _parseq_transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    print("[init] PARSeq loaded.", flush=True)


def parseq_recognize(plate_bgr):
    import torch
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    inp = _parseq_transform(pil_img).unsqueeze(0).to(_parseq_device)
    with torch.no_grad():
        logits = _parseq_model(inp)
        probs = logits.softmax(-1)
    preds = _parseq_model.tokenizer.decode(probs)
    if isinstance(preds, (tuple, list)) and len(preds) == 2:
        texts, _ = preds
    else:
        texts = preds
    return texts[0] if isinstance(texts, (list, tuple)) else texts


# ============================================================
# Pipelines (each returns 4x2 corners in original-image coords, or None)
# ============================================================


def detect_twostep(img_bgr, ctx, conf_det):
    det_tensor = preprocess_plate_det(img_bgr)
    det_outputs = ctx["det_session"].run(
        ctx["det_out"], {ctx["det_in"]: det_tensor}
    )
    decoded = decode_plate_det(det_outputs, img_bgr.shape, conf_det)
    if decoded is None:
        return None
    x1, y1, x2, y2 = decoded["bbox"]
    plate_crop = img_bgr[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None
    rect_tensor = preprocess_rectifier(plate_crop)
    rect_outputs = ctx["rect_session"].run(
        ctx["rect_out"], {ctx["rect_in"]: rect_tensor}
    )
    return rectifier_corners_to_image(rect_outputs[0], decoded["bbox"])


def detect_yolo26(img_bgr, ctx, conf_best):
    tensor, gain, pad = preprocess_best(img_bgr, BEST_IMGSZ)
    outputs = ctx["session"].run(ctx["outputs"], {ctx["input"]: tensor})
    info = decode_best_detection(outputs[0], gain, pad, img_bgr.shape, conf_best)
    return None if info is None else info["keypoints"]


# ============================================================
# Helpers
# ============================================================


def clean_text(s):
    if not s:
        return ""
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def expected_from_filename(path):
    return Path(path).stem.upper()


def is_match(ocr_clean, expected):
    if not ocr_clean or not expected:
        return False
    return ocr_clean in expected


def corners_to_json(corners):
    if corners is None:
        return ""
    return json.dumps([[float(x), float(y)] for x, y in corners], separators=(",", ":"))


def list_images(root):
    root = Path(root)
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p.resolve()))
    paths.sort()
    return paths


def load_processed(csv_path):
    processed = set()
    if not csv_path.exists():
        return processed
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return processed
        if header != CSV_HEADER:
            raise RuntimeError(
                f"CSV header mismatch in {csv_path}\n"
                f"  expected: {CSV_HEADER}\n"
                f"  found:    {header}"
            )
        for row in reader:
            if row:
                processed.add(row[0])
    return processed


def open_csv_for_append(csv_path):
    new_file = not csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(CSV_HEADER)
        f.flush()
        os.fsync(f.fileno())
    return f, writer


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lpd-root", default="/home/itemhsu/amtk/lppart/lp/LPD"
    )
    parser.add_argument(
        "--csv-out",
        default=str(SCRIPT_DIR / "lpd_results.csv"),
    )
    parser.add_argument(
        "--nomatch-csv-out",
        default=str(SCRIPT_DIR / "lpd_nomatch.csv"),
        help="Subset CSV: rows where all 4 pipeline matches are 0",
    )
    parser.add_argument(
        "--plate-det", default="/home/itemhsu/amtk/lpr/0x1PlateDet/PlateDet.onnx"
    )
    parser.add_argument(
        "--rectifier", default="/home/itemhsu/amtk/lpr/PlateRectifier.onnx"
    )
    parser.add_argument(
        "--best-models",
        nargs="+",
        default=[
            "yolo26-train-1775509671/best.onnx",
            "yolo26-train-1773128236/best.onnx",
            "yolo26-train-1779339446/best.onnx",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--best-conf", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N pending images (0 = all)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    nomatch_path = Path(args.nomatch_csv_out)
    nomatch_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] Scanning {args.lpd_root} ...", flush=True)
    all_images = list_images(args.lpd_root)
    print(f"[init] Found {len(all_images)} images.", flush=True)

    print(f"[init] Reading existing CSV: {csv_path}", flush=True)
    processed = load_processed(csv_path)
    print(f"[init] Already processed: {len(processed)}", flush=True)

    pending = [p for p in all_images if p not in processed]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"[init] Pending: {len(pending)}", flush=True)
    if not pending:
        print("[done] Nothing to do.", flush=True)
        return

    print("[init] Loading ONNX sessions ...", flush=True)
    det_session = make_session(Path(args.plate_det), args.threads)
    rect_session = make_session(Path(args.rectifier), args.threads)
    twostep_ctx = {
        "det_session": det_session,
        "rect_session": rect_session,
        "det_in": det_session.get_inputs()[0].name,
        "rect_in": rect_session.get_inputs()[0].name,
        "det_out": [o.name for o in det_session.get_outputs()],
        "rect_out": [o.name for o in rect_session.get_outputs()],
    }

    yolo_ctxs = {}
    for best_path in args.best_models:
        key = YOLO26_LABEL_MAP.get(best_path)
        if key is None:
            print(f"[warn] Unknown YOLOv26 path (skipping): {best_path}", flush=True)
            continue
        sess = make_session(Path(best_path), args.threads)
        yolo_ctxs[key] = {
            "session": sess,
            "input": sess.get_inputs()[0].name,
            "outputs": [o.name for o in sess.get_outputs()],
        }
        print(f"[init] Loaded {key}: {best_path}", flush=True)

    init_parseq()

    pipelines = [("twostep", twostep_ctx, detect_twostep)]
    for key in ("yolo26m", "yolo26s", "yolo26n"):
        if key in yolo_ctxs:
            pipelines.append((key, yolo_ctxs[key], detect_yolo26))

    csv_file, writer = open_csv_for_append(csv_path)
    nomatch_file, nomatch_writer = open_csv_for_append(nomatch_path)
    try:
        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        for batch_idx in range(total_batches):
            t_batch = time.perf_counter()
            batch_paths = pending[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]

            # Step A: load all images for this batch into memory
            imgs = {}
            load_failed = []
            for p in batch_paths:
                img = cv2.imread(p)
                if img is None:
                    load_failed.append(p)
                else:
                    imgs[p] = img

            # Step B: run each pipeline over the whole batch, then switch
            results = defaultdict(dict)
            for key, ctx, fn in pipelines:
                t_pipe = time.perf_counter()
                detected_count = 0
                for p, img in imgs.items():
                    corners = None
                    try:
                        if key == "twostep":
                            corners = fn(img, ctx, args.det_conf)
                        else:
                            corners = fn(img, ctx, args.best_conf)
                    except Exception as exc:
                        print(f"[warn] {key} failed on {p}: {exc}", flush=True)
                        corners = None
                    if corners is None:
                        results[p][key] = ("0", "", "", "", "0")
                        continue
                    detected_count += 1
                    try:
                        plate = warp_plate(img, corners)
                        ocr_raw = parseq_recognize(plate)
                    except Exception as exc:
                        print(f"[warn] {key} OCR failed on {p}: {exc}", flush=True)
                        ocr_raw = ""
                    ocr_cl = clean_text(ocr_raw)
                    expected = expected_from_filename(p)
                    matched = "1" if is_match(ocr_cl, expected) else "0"
                    results[p][key] = (
                        "1",
                        corners_to_json(corners),
                        ocr_raw,
                        ocr_cl,
                        matched,
                    )
                dt = time.perf_counter() - t_pipe
                print(
                    f"[batch {batch_idx + 1}/{total_batches}] {key}: "
                    f"{detected_count}/{len(imgs)} detected, {dt:.2f}s",
                    flush=True,
                )

            # Step C: append rows for every batch path (including failed loads)
            nomatch_count = 0
            for p in batch_paths:
                expected = expected_from_filename(p)
                row = [p, expected]
                match_flags = []
                if p in load_failed:
                    for _ in pipelines:
                        row.extend(["0", "", "", "", "0"])
                        match_flags.append("0")
                else:
                    for key, _, _ in pipelines:
                        cols = results[p][key]
                        row.extend(cols)
                        match_flags.append(cols[4])
                writer.writerow(row)
                if all(flag == "0" for flag in match_flags):
                    nomatch_writer.writerow(row)
                    nomatch_count += 1
            csv_file.flush()
            os.fsync(csv_file.fileno())
            nomatch_file.flush()
            os.fsync(nomatch_file.fileno())
            dt_batch = time.perf_counter() - t_batch
            done = (batch_idx + 1) * args.batch_size
            done = min(done, len(pending))
            print(
                f"[batch {batch_idx + 1}/{total_batches}] flushed "
                f"({done}/{len(pending)}), nomatch +{nomatch_count}, "
                f"batch time {dt_batch:.2f}s",
                flush=True,
            )
    finally:
        csv_file.close()
        nomatch_file.close()

    print(f"[done] Results written to {csv_path}", flush=True)
    print(f"[done] No-match subset:    {nomatch_path}", flush=True)


if __name__ == "__main__":
    main()
