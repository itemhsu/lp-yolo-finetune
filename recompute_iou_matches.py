#!/usr/bin/env python3
"""Recompute match columns using polygon IoU + minimum OCR length filter.

Two fixes applied on top of the raw bench CSV:
  1. Min OCR length (default 4): reject matches where OCR output is too short
     to be a real Taiwan plate (e.g. "10" matches "105615" as substring → false +ve)
  2. IoU fallback for v2/v3: if corners overlap two-step at IoU >= thresh,
     consider detection correct regardless of OCR result.

New columns added:
  twostep_match_valid  — twostep_match filtered by min OCR length
  v2_iou               — polygon IoU(v2_corners, twostep_corners)
  v2_match_iou         — (v2_match AND len>=min) OR iou>=thresh
  v3_iou               — polygon IoU(v3_corners, twostep_corners)
  v3_match_iou         — (v3_match AND len>=min) OR iou>=thresh
"""

import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def poly_iou(c1, c2):
    """IoU between two convex quadrilaterals given as list of [x, y].

    cv2.intersectConvexConvex is not symmetric for degenerate polygons
    (e.g. duplicate corner points), so we take min of both orderings.
    """
    if not c1 or not c2:
        return 0.0
    try:
        p1 = np.array(c1, dtype=np.float32)
        p2 = np.array(c2, dtype=np.float32)
        a12, _ = cv2.intersectConvexConvex(p1, p2)
        a21, _ = cv2.intersectConvexConvex(p2, p1)
        area_inter = min(a12, a21)
        if area_inter <= 0:
            return 0.0
        area1 = cv2.contourArea(p1)
        area2 = cv2.contourArea(p2)
        area_union = area1 + area2 - area_inter
        return float(area_inter / area_union) if area_union > 0 else 0.0
    except Exception:
        return 0.0


def parse_corners(s):
    """JSON corners string → [[x,y], ...] or None."""
    if not s or not s.strip():
        return None
    try:
        pts = json.loads(s)
        result = [[float(pt[0]), float(pt[1])] for pt in pts if len(pt) >= 2]
        return result if len(result) >= 3 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default=str(SCRIPT_DIR / "lpd_three_way.csv"))
    ap.add_argument("--out",       default=str(SCRIPT_DIR / "lpd_three_way_iou.csv"))
    ap.add_argument("--thresh",    type=float, default=0.5)
    ap.add_argument("--min-ocr",   type=int,   default=4,
                    help="Minimum OCR output length to count as a valid match")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        rows = list(reader)

    print(f"[load] {len(rows):,} rows  iou_thresh={args.thresh}  min_ocr={args.min_ocr}", flush=True)

    v2_upgrades = v3_upgrades = v4_upgrades = 0
    ts_invalidated = 0

    for r in rows:
        ts_c = parse_corners(r.get("twostep_corners", ""))

        # ── two-step: apply min OCR length filter ────────────────────────────
        ts_ocr = r.get("twostep_ocr_clean", "")
        if r.get("twostep_match", "0") == "1" and len(ts_ocr) < args.min_ocr:
            r["twostep_match_valid"] = "0"
            ts_invalidated += 1
        else:
            r["twostep_match_valid"] = r.get("twostep_match", "0")

        # IoU upgrade is only meaningful when two-step confirmed the right plate.
        # If two-step itself has wrong OCR, its corners are an unreliable reference.
        ts_valid = r["twostep_match_valid"] == "1"

        # ── v2 ──────────────────────────────────────────────────────────────
        v2_c = parse_corners(r.get("v2_corners", ""))
        v2_iou_val = poly_iou(v2_c, ts_c) if (v2_c and ts_c) else 0.0
        v2_ocr_valid = (r.get("v2_match", "0") == "1"
                        and len(r.get("v2_ocr_clean", "")) >= args.min_ocr)
        v2_iou_upgrade = ts_valid and v2_iou_val >= args.thresh
        v2_match_iou = "1" if (v2_ocr_valid or v2_iou_upgrade) else "0"
        if v2_match_iou == "1" and r.get("v2_match", "0") == "0":
            v2_upgrades += 1

        # ── v3 ──────────────────────────────────────────────────────────────
        v3_c = parse_corners(r.get("v3_corners", ""))
        v3_iou_val = poly_iou(v3_c, ts_c) if (v3_c and ts_c) else 0.0
        v3_ocr_valid = (r.get("v3_match", "0") == "1"
                        and len(r.get("v3_ocr_clean", "")) >= args.min_ocr)
        v3_iou_upgrade = ts_valid and v3_iou_val >= args.thresh
        v3_match_iou = "1" if (v3_ocr_valid or v3_iou_upgrade) else "0"
        if v3_match_iou == "1" and r.get("v3_match", "0") == "0":
            v3_upgrades += 1

        r["v2_iou"]       = f"{v2_iou_val:.4f}"
        r["v2_match_iou"] = v2_match_iou
        r["v3_iou"]       = f"{v3_iou_val:.4f}"
        r["v3_match_iou"] = v3_match_iou

        # ── v4 (optional — only if columns exist) ───────────────────────────
        if "v4_corners" in r:
            v4_c = parse_corners(r.get("v4_corners", ""))
            v4_iou_val = poly_iou(v4_c, ts_c) if (v4_c and ts_c) else 0.0
            v4_ocr_valid = (r.get("v4_match", "0") == "1"
                            and len(r.get("v4_ocr_clean", "")) >= args.min_ocr)
            v4_iou_upgrade = ts_valid and v4_iou_val >= args.thresh
            v4_match_iou = "1" if (v4_ocr_valid or v4_iou_upgrade) else "0"
            if v4_match_iou == "1" and r.get("v4_match", "0") == "0":
                v4_upgrades += 1
            r["v4_iou"]       = f"{v4_iou_val:.4f}"
            r["v4_match_iou"] = v4_match_iou

    has_v4 = "v4_corners" in rows[0] if rows else False
    extra = ["twostep_match_valid", "v2_iou", "v2_match_iou", "v3_iou", "v3_match_iou"]
    if has_v4:
        extra += ["v4_iou", "v4_match_iou"]
    new_fields = list(orig_fields) + extra
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    orig_ts = sum(1 for r in rows if r["twostep_match"] == "1")
    new_ts  = sum(1 for r in rows if r["twostep_match_valid"] == "1")
    orig_v2 = sum(1 for r in rows if r["v2_match"] == "1")
    orig_v3 = sum(1 for r in rows if r["v3_match"] == "1")
    new_v2  = sum(1 for r in rows if r["v2_match_iou"] == "1")
    new_v3  = sum(1 for r in rows if r["v3_match_iou"] == "1")

    print(f"[done] {args.out}")
    print(f"\n  Two-step OCR-only: {orig_ts:,} ({orig_ts/n*100:.2f}%)  →  min-len filtered: {new_ts:,}  (-{ts_invalidated:,})")
    print(f"  v2  OCR-only: {orig_v2:,} ({orig_v2/n*100:.2f}%)  →  IoU-corrected: {new_v2:,} ({new_v2/n*100:.2f}%)  +{v2_upgrades:,}")
    print(f"  v3  OCR-only: {orig_v3:,} ({orig_v3/n*100:.2f}%)  →  IoU-corrected: {new_v3:,} ({new_v3/n*100:.2f}%)  +{v3_upgrades:,}")
    if has_v4:
        orig_v4 = sum(1 for r in rows if r.get("v4_match","0") == "1")
        new_v4  = sum(1 for r in rows if r.get("v4_match_iou","0") == "1")
        print(f"  v4  OCR-only: {orig_v4:,} ({orig_v4/n*100:.2f}%)  →  IoU-corrected: {new_v4:,} ({new_v4/n*100:.2f}%)  +{v4_upgrades:,}")


if __name__ == "__main__":
    main()
