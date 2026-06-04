#!/usr/bin/env python3
"""Build YOLO-pose fine-tune dataset for YOLOv26-pose-n.

Selection:
  - yolo26n_no_detect (judges agree on corners, yolo26n missed): ~691
  - Cell 2 (plate-like / OCR differ between 2-step and yolo26n): ~5,906
  Total: ~6,597 candidate rows.

Ground truth: Two-step's corners (assumed to be reliable per user instruction).
Holdout: All 0721TW images kept out of train/val (pure test set).
Split:    80/20 train/val for the remaining (random with seed).

Output:
  yolo26n_ftn_dataset/
    dataset.yaml
    images/{train,val,holdout}/<filename>     (symlinks to originals)
    labels/{train,val,holdout}/<stem>.txt    (YOLO-pose format)
    manifest.csv                              (per-sample provenance)
"""

import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV = SCRIPT_DIR / "lpd_results.csv"
OUT_ROOT = SCRIPT_DIR.parent / "yolo26n_ftn_dataset"

JUDGES = ["twostep", "yolo26m", "yolo26s"]
TARGET = "yolo26n"
PLATE_RE = re.compile(r"^[0-9A-Z]{5,7}$")
SEED = 42
HOLDOUT_SUBDIR = "0721TW"


def quad_iou(c1, c2):
    try:
        p1, p2 = Polygon(c1), Polygon(c2)
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        if p1.area <= 0 or p2.area <= 0:
            return 0.0
        return p1.intersection(p2).area / p1.union(p2).area
    except Exception:
        return 0.0


def classify(r):
    jc = {}
    for j in JUDGES:
        if r[f"{j}_detected"] == "1":
            try:
                c = np.array(json.loads(r[f"{j}_corners"]), dtype=np.float32)
                if c.shape == (4, 2):
                    jc[j] = c
            except Exception:
                pass
    yn = None
    if r[f"{TARGET}_detected"] == "1":
        try:
            yn = np.array(json.loads(r[f"{TARGET}_corners"]), dtype=np.float32)
            if yn.shape != (4, 2):
                yn = None
        except Exception:
            yn = None
    if len(jc) < 2:
        return "no_judge_consensus", jc
    miou = float(np.mean([quad_iou(jc[a], jc[b]) for a, b in combinations(jc.keys(), 2)]))
    if miou < 0.8:
        return "weak_judge_consensus", jc
    if yn is None:
        return "yolo26n_no_detect", jc
    cons = jc.get("twostep", list(jc.values())[0])
    iou = quad_iou(yn, cons)
    if iou < 0.5:
        return "yolo26n_wrong_location", jc
    if iou < 0.8:
        return "yolo26n_low_iou", jc
    return "yolo26n_already_good", jc


def is_plate_like(s):
    return bool(s) and bool(PLATE_RE.match(s))


def sort_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(sums)]      # TL
    rect[2] = pts[np.argmax(sums)]      # BR
    rect[1] = pts[np.argmin(diffs)]     # TR
    rect[3] = pts[np.argmax(diffs)]     # BL
    return rect


def corners_to_yolo_label(corners, img_w, img_h):
    """Return YOLO-pose label line: class cx cy w h kpts (normalized)."""
    sorted_c = sort_corners(corners)
    xs, ys = sorted_c[:, 0], sorted_c[:, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    # Clip just in case
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))

    kpts = []
    for (kx, ky) in sorted_c:
        nx = max(0.0, min(1.0, float(kx) / img_w))
        ny = max(0.0, min(1.0, float(ky) / img_h))
        kpts.extend([f"{nx:.6f}", f"{ny:.6f}", "2"])

    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpts)


def main():
    print(f"[init] Reading {IN_CSV}", flush=True)
    with IN_CSV.open() as f:
        rows = list(csv.DictReader(f))
    print(f"[init] {len(rows):,} rows", flush=True)

    # Classify and select (collect candidates first, dedupe later)
    # Source A: yolo26n_no_detect (judges agree, yolo missed)
    # Source B: Cell 2 (loose) — 2-step plate-like + OCR diff
    #          where "diff" includes the case yolo has no OCR.
    # Allow same row to qualify under both; dedupe by image_path with priority.
    candidates_A = []   # (row, twostep_corners)
    candidates_B = []
    pre_total = {"A_no_detect": 0, "B_cell2": 0}
    skipped_no_2step = 0

    PROBLEM_BUCKETS = {
        "yolo26n_low_iou", "yolo26n_wrong_location",
        "yolo26n_no_detect", "weak_judge_consensus", "no_judge_consensus",
    }

    for r in rows:
        bucket, jc = classify(r)
        if bucket not in PROBLEM_BUCKETS:
            continue
        if "twostep" not in jc:
            # Without 2-step corners we have no GT — skip
            skipped_no_2step += 1
            continue
        ts_corners = jc["twostep"]

        # Source A: judges-consensus + yolo missed
        if bucket == "yolo26n_no_detect":
            candidates_A.append((r, ts_corners))
            pre_total["A_no_detect"] += 1

        # Source B: Cell 2 loose (2-step plate-like + yolo OCR != 2-step OCR;
        # empty yolo OCR counts as "different" per original 2x2 definition)
        ts_ocr = r["twostep_ocr_clean"]
        yn_ocr = r[f"{TARGET}_ocr_clean"]
        if ts_ocr and is_plate_like(ts_ocr) and ts_ocr != yn_ocr:
            candidates_B.append((r, ts_corners))
            pre_total["B_cell2"] += 1

    print(f"[pre-dedup] source A (no_detect):   {pre_total['A_no_detect']:,}", flush=True)
    print(f"[pre-dedup] source B (cell2 loose): {pre_total['B_cell2']:,}", flush=True)
    print(f"[pre-dedup] grand pool (with overlap): "
          f"{pre_total['A_no_detect'] + pre_total['B_cell2']:,}", flush=True)
    print(f"[pre-dedup] skipped (no 2-step corners): {skipped_no_2step:,}", flush=True)

    # Dedupe by image_path. Priority: keep "no_detect" source tag when in both
    # (because that's the more diagnostic / higher-value bucket for training).
    seen_paths = {}
    selected = []
    bucket_counter = Counter()
    overlap_count = 0
    for src, items in (("no_detect", candidates_A), ("cell2", candidates_B)):
        for r, ts_corners in items:
            path = r["image_path"]
            if path in seen_paths:
                if seen_paths[path] != src:
                    overlap_count += 1
                continue
            seen_paths[path] = src
            selected.append((r, src, ts_corners))
            bucket_counter[src] += 1

    print()
    print(f"[dedup] overlap (image in both sources, kept under 'no_detect'): {overlap_count:,}",
          flush=True)
    print(f"[final] no_detect: {bucket_counter['no_detect']:,}", flush=True)
    print(f"[final] cell2:     {bucket_counter['cell2']:,}", flush=True)
    print(f"[final] total:     {len(selected):,}", flush=True)

    # Holdout = 0721TW; remaining = train+val
    holdout, candidates = [], []
    for item in selected:
        r = item[0]
        try:
            parts = Path(r["image_path"]).parts
            sub = parts[parts.index("LPD") + 1]
        except (ValueError, IndexError):
            sub = "_other"
        item_tagged = item + (sub,)
        if sub == HOLDOUT_SUBDIR:
            holdout.append(item_tagged)
        else:
            candidates.append(item_tagged)

    random.seed(SEED)
    random.shuffle(candidates)
    n_val = int(len(candidates) * 0.2)
    val = candidates[:n_val]
    train = candidates[n_val:]

    print()
    print(f"[split] train:   {len(train):,}")
    print(f"[split] val:     {len(val):,}")
    print(f"[split] holdout: {len(holdout):,} (all 0721TW)")

    # Make output dirs
    if OUT_ROOT.exists():
        print(f"[clean] Removing existing {OUT_ROOT}", flush=True)
        shutil.rmtree(OUT_ROOT)
    for s in ("train", "val", "holdout"):
        (OUT_ROOT / "images" / s).mkdir(parents=True)
        (OUT_ROOT / "labels" / s).mkdir(parents=True)

    # Write labels + symlink images
    manifest_rows = []
    fail_count = 0
    for split_name, items in (("train", train), ("val", val), ("holdout", holdout)):
        print(f"[write] {split_name}: {len(items):,}", flush=True)
        for r, source, corners, sub in items:
            img_path = Path(r["image_path"])
            # Read image dimensions (use shape header, but cv2 is fine for jpg/png)
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    fail_count += 1
                    continue
                h, w = img.shape[:2]
            except Exception:
                fail_count += 1
                continue
            label_line = corners_to_yolo_label(corners, w, h)
            stem = img_path.stem
            # Avoid filename collisions: prefix with subdir + first 8 chars of orig name
            safe_name = f"{sub}__{img_path.name}"
            safe_stem = f"{sub}__{stem}"
            (OUT_ROOT / "labels" / split_name / f"{safe_stem}.txt").write_text(label_line)
            link = OUT_ROOT / "images" / split_name / safe_name
            if not link.exists():
                link.symlink_to(img_path.resolve())
            manifest_rows.append({
                "split": split_name,
                "source": source,
                "subdir": sub,
                "image_path": str(img_path),
                "image_w": w,
                "image_h": h,
                "filename_in_dataset": safe_name,
            })

    print(f"[done] manifest entries: {len(manifest_rows):,}", flush=True)
    print(f"[done] image read failures: {fail_count}", flush=True)

    # Manifest CSV
    manifest_path = OUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # dataset.yaml
    yaml = f"""# YOLOv26-pose-n fine-tune dataset
# Sources:
#   - yolo26n_no_detect bucket (judges agreed, yolo26n missed)
#   - Cell 2: plate-like / OCR differ between 2-step and yolo26n
# Ground truth: Two-step's corners (assumed correct per user instruction)
# Holdout: All 0721TW samples kept separate.

path: {OUT_ROOT.resolve()}
train: images/train
val: images/val
# test: images/holdout    # use yolo pose val data=...test for hold-out eval

kpt_shape: [4, 3]   # 4 keypoints, each (x, y, visibility)

names:
  0: license_plate

# Keypoint order (sort_corners):
#   0: TL (top-left)
#   1: TR (top-right)
#   2: BR (bottom-right)
#   3: BL (bottom-left)
"""
    (OUT_ROOT / "dataset.yaml").write_text(yaml)
    print(f"[done] dataset.yaml written")
    print(f"[done] Dataset root: {OUT_ROOT}")

    # Final stats
    print()
    print("=== Final dataset stats ===")
    by_split_source = Counter()
    by_split_sub = defaultdict(Counter)
    for m in manifest_rows:
        by_split_source[(m["split"], m["source"])] += 1
        by_split_sub[m["split"]][m["subdir"]] += 1
    for split in ("train", "val", "holdout"):
        nd = by_split_source[(split, "no_detect")]
        c2 = by_split_source[(split, "cell2")]
        total = nd + c2
        print(f"  {split:<8} total={total:>5,}  no_detect={nd:>5,}  cell2={c2:>5,}")
        for sub, n in by_split_sub[split].most_common():
            print(f"       {sub}: {n}")


if __name__ == "__main__":
    main()
