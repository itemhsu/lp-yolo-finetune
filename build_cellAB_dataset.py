#!/usr/bin/env python3
"""Build YOLO pose dataset from Cell AB of lpd_twostep_vs_merged.csv.

Cell AB: twostep_match=0, merged_match=0, twostep_ocr is plate-like,
         twostep_ocr != merged_ocr.

Uses twostep_corners as GT keypoints (TL/TR/BR/BL).
Output: cellAB_dataset/  (train/val/holdout splits)
"""

import csv
import json
import re
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH   = SCRIPT_DIR / "lpd_twostep_vs_merged.csv"
OUT_DIR    = SCRIPT_DIR / "cellAB_dataset"

TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
HOLDOUT_RATIO = 0.10

random.seed(42)


def plate_like(s):
    s = s.strip()
    if len(s) < 4 or len(s) > 8:
        return False
    return bool(re.search(r'[A-Z]', s)) and bool(re.search(r'[0-9]', s))


def corners_to_yolo(corners, img_w, img_h):
    """
    corners: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (TL TR BR BL, pixel)
    Returns YOLO pose label string:
      class cx cy w h  kp0x kp0y 2  kp1x kp1y 2  kp2x kp2y 2  kp3x kp3y 2
    All coords normalised 0-1.
    """
    pts = np.array(corners, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    kpts = []
    for x, y in corners:
        kpts.append(f"{x/img_w:.6f} {y/img_h:.6f} 2")
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpts)


def main():
    print(f"[load] {CSV_PATH}", flush=True)
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    print(f"[load] {len(rows):,} rows total", flush=True)

    # --- Filter Cell AB ---
    AB = []
    skip_no_corners = 0
    skip_bad_img    = 0
    for r in rows:
        if r["twostep_match"] != "0" or r["merged_match"] != "0":
            continue
        ts_ocr = r["twostep_ocr_clean"]
        mg_ocr = r["merged_ocr_clean"]
        if not plate_like(ts_ocr):
            continue
        if ts_ocr == mg_ocr:
            continue
        if not r["twostep_corners"]:
            skip_no_corners += 1
            continue
        AB.append(r)

    print(f"[filter] Cell AB: {len(AB):,}  (skipped no_corners={skip_no_corners})", flush=True)

    # --- Shuffle & split ---
    random.shuffle(AB)
    n = len(AB)
    n_train   = int(n * TRAIN_RATIO)
    n_val     = int(n * VAL_RATIO)
    splits = {
        "train":   AB[:n_train],
        "val":     AB[n_train:n_train + n_val],
        "holdout": AB[n_train + n_val:],
    }
    for sp, rows_ in splits.items():
        print(f"  {sp}: {len(rows_):,}", flush=True)

    # --- Build directories ---
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for sp in splits:
        (OUT_DIR / sp / "images").mkdir(parents=True)
        (OUT_DIR / sp / "labels").mkdir(parents=True)

    # --- Write labels (symlink images) ---
    fail_img  = 0
    fail_parse = 0
    written   = 0

    for sp, rows_ in splits.items():
        img_dir = OUT_DIR / sp / "images"
        lbl_dir = OUT_DIR / sp / "labels"
        for r in rows_:
            img_path = Path(r["image_path"])
            if not img_path.exists():
                fail_img += 1
                continue

            # Parse corners
            try:
                corners = json.loads(r["twostep_corners"])
                assert len(corners) == 4
            except Exception:
                fail_parse += 1
                continue

            # Get image size
            img = cv2.imread(str(img_path))
            if img is None:
                fail_img += 1
                continue
            h, w = img.shape[:2]

            label_str = corners_to_yolo(corners, w, h)

            stem = img_path.stem
            # Symlink image
            dst_img = img_dir / img_path.name
            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())
            else:
                # name collision: prefix with parent dir
                dst_img = img_dir / f"{img_path.parent.name}_{img_path.name}"
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                stem = dst_img.stem

            # Write label
            lbl_file = lbl_dir / f"{stem}.txt"
            lbl_file.write_text(label_str + "\n")
            written += 1

    print(f"[done] written={written:,}  fail_img={fail_img}  fail_parse={fail_parse}", flush=True)

    # --- data.yaml ---
    yaml_content = f"""path: {OUT_DIR}
train: train/images
val:   val/images

nc: 1
names: ['plate']
kpt_shape: [4, 3]
"""
    (OUT_DIR / "data.yaml").write_text(yaml_content)
    print(f"[done] data.yaml written", flush=True)
    print(f"[done] dataset: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
