#!/usr/bin/env python3
"""Build merged YOLO-pose training dataset (Strategy E):
  - haug_base/: cleaned_dataset_haug base images (filter out _h120/_h140/_h160 augmentation variants)
  - new/: yolo26n_ftn_dataset (pseudo-labels from 2-step corners)
  - multi-path data.yaml, symlinks only — no file duplication

Output:
  merged_dataset/
    data.yaml
    haug_base/{train,val,test}/{images,labels}/
    new/{train,val,holdout}/{images,labels}/
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

LP_VIEWER = Path("/home/itemhsu/amtk/lppart/lp/lp_viewer_tool/lp_viewer_output")
OUT_ROOT = LP_VIEWER / "merged_dataset"
HAUG_ZIP = LP_VIEWER / "cleaned_dataset_haug.zip"
HAUG_EXTRACT = LP_VIEWER / "cleaned_dataset_haug_extracted"
NEW_DATASET = LP_VIEWER / "yolo26n_ftn_dataset"

# Roboflow augmentation suffixes: _h120, _h140, _h160 (hue variants)
AUG_PATTERN = re.compile(r"_h\d+$")


def is_aug_stem(stem):
    return bool(AUG_PATTERN.search(stem))


def extract_haug():
    if HAUG_EXTRACT.exists() and any(HAUG_EXTRACT.iterdir()):
        print(f"[haug] Already extracted at {HAUG_EXTRACT}", flush=True)
        return
    print(f"[haug] Extracting {HAUG_ZIP} ({HAUG_ZIP.stat().st_size/1e9:.1f} GB) ...", flush=True)
    HAUG_EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HAUG_ZIP) as z:
        z.extractall(HAUG_EXTRACT)
    print("[haug] Extraction done", flush=True)


def link_base_pair(src_imgs_dir, src_lbls_dir, dst_imgs_dir, dst_lbls_dir):
    """Symlink base images + labels; skip _h augmentation variants."""
    dst_imgs_dir.mkdir(parents=True, exist_ok=True)
    dst_lbls_dir.mkdir(parents=True, exist_ok=True)
    if not src_imgs_dir.exists():
        return (0, 0, 0)
    n_link = 0
    n_aug = 0
    n_no_lbl = 0
    for img_path in src_imgs_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if is_aug_stem(img_path.stem):
            n_aug += 1
            continue
        lbl_path = src_lbls_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            n_no_lbl += 1
            continue
        (dst_imgs_dir / img_path.name).symlink_to(img_path.resolve())
        (dst_lbls_dir / lbl_path.name).symlink_to(lbl_path.resolve())
        n_link += 1
    return (n_link, n_aug, n_no_lbl)


def link_all_pair(src_imgs_dir, src_lbls_dir, dst_imgs_dir, dst_lbls_dir):
    """Symlink ALL images+labels (no filtering); use for our new dataset."""
    dst_imgs_dir.mkdir(parents=True, exist_ok=True)
    dst_lbls_dir.mkdir(parents=True, exist_ok=True)
    if not src_imgs_dir.exists():
        return 0
    n = 0
    for img_path in src_imgs_dir.iterdir():
        if not (img_path.is_file() or img_path.is_symlink()):
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_path = src_lbls_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        # Resolve original target (in case src is already symlink)
        img_target = img_path.resolve()
        lbl_target = lbl_path.resolve()
        (dst_imgs_dir / img_path.name).symlink_to(img_target)
        (dst_lbls_dir / lbl_path.name).symlink_to(lbl_target)
        n += 1
    return n


def main():
    # 1) Clean previous output
    if OUT_ROOT.exists():
        print(f"[clean] Removing existing {OUT_ROOT}", flush=True)
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    # 2) Extract haug
    extract_haug()

    # 3) Link haug base images (Strategy B — no augmentation variants)
    print("\n=== Linking haug base images (no _h aug) ===", flush=True)
    haug_stats = {}
    # haug uses {train, valid, test}; we map valid -> val
    haug_to_our_split = {"train": "train", "valid": "val", "test": "test"}
    for haug_split, our_split in haug_to_our_split.items():
        src_imgs = HAUG_EXTRACT / haug_split / "images"
        src_lbls = HAUG_EXTRACT / haug_split / "labels"
        dst_imgs = OUT_ROOT / "haug_base" / our_split / "images"
        dst_lbls = OUT_ROOT / "haug_base" / our_split / "labels"
        n_link, n_aug, n_no_lbl = link_base_pair(src_imgs, src_lbls, dst_imgs, dst_lbls)
        haug_stats[our_split] = {"linked": n_link, "skipped_aug": n_aug, "no_label": n_no_lbl}
        print(f"  haug {haug_split:5s} -> {our_split:5s}: linked {n_link:>5,}, "
              f"skipped {n_aug:>5,} aug, {n_no_lbl:>3,} missing-label", flush=True)

    # 4) Link new dataset (already filtered, just relink)
    print("\n=== Linking new pseudo-label dataset ===", flush=True)
    new_stats = {}
    for split in ("train", "val", "holdout"):
        src_imgs = NEW_DATASET / "images" / split
        src_lbls = NEW_DATASET / "labels" / split
        dst_imgs = OUT_ROOT / "new" / split / "images"
        dst_lbls = OUT_ROOT / "new" / split / "labels"
        n = link_all_pair(src_imgs, src_lbls, dst_imgs, dst_lbls)
        new_stats[split] = n
        print(f"  new {split:8s}: linked {n:,}", flush=True)

    # 5) Generate multi-path data.yaml
    yaml_content = f"""# Merged YOLOv26-pose-n fine-tune dataset (Strategy E)
# Sources:
#   - haug_base: cleaned_dataset_haug base images (no _h120/_h140/_h160 augmentation variants)
#   - new:       yolo26n_ftn_dataset (pseudo-labels from 2-step corners; LPD)
# Ground truth assumption: 2-step corners are accurate (per user instruction).
# Holdout (0721TW, new dataset only) kept separate; not in train/val for honest eval.

path: {OUT_ROOT.resolve()}

train:
  - haug_base/train/images
  - new/train/images

val:
  - haug_base/val/images
  - new/val/images

# For final hold-out evaluation, use:
#   yolo pose val model=... data=... split=test  (after adding test below)
# test:
#   - haug_base/test/images
#   - new/holdout/images

kpt_shape: [4, 3]
flip_idx: [0, 1, 2, 3]    # identity (kept consistent with haug)

names:
  0: license_plate
"""
    (OUT_ROOT / "data.yaml").write_text(yaml_content)
    print(f"\n[done] data.yaml written: {OUT_ROOT / 'data.yaml'}", flush=True)

    # 6) Summary
    print("\n" + "=" * 60)
    print("FINAL DATASET STATS")
    print("=" * 60)
    for split in ("train", "val"):
        h = haug_stats[split]["linked"]
        n = new_stats[split]
        total = h + n
        print(f"  {split:5s}: haug={h:>6,}  +  new={n:>6,}  =  {total:>6,}")
    print(f"  test :  haug={haug_stats['test']['linked']:,}  (held aside)")
    print(f"  holdout (new only): {new_stats['holdout']:,}  (0721TW; held aside)")
    print()
    print(f"Merged dataset root: {OUT_ROOT}")
    print()
    print("Recommended training command (from ./model directory):")
    print(f"""
yolo pose train \\
  model=yolo26-train-1779339446/best.pt \\
  data={OUT_ROOT}/data.yaml \\
  epochs=30 imgsz=640 batch=16 \\
  lr0=0.001 lrf=0.01 \\
  freeze=10 \\
  project=yolo26_finetune \\
  name=merged_v1 \\
  patience=10 device=0
""")


if __name__ == "__main__":
    main()
