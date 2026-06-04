#!/usr/bin/env python3
"""Build merged_v3_dataset = merged_dataset + NEW Cell AB images only.

Dedup by resolved real path: AB images already in merged_dataset are excluded.
The new (non-overlap) AB images are symlinked into merged_v3_dataset/ab_new/.

data.yaml uses multi-path pointing to:
  - merged_dataset  (existing)
  - merged_v3_dataset/ab_new  (truly new AB images only)
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LP_VIEWER  = SCRIPT_DIR.parent
MERGED_DS  = LP_VIEWER / "merged_dataset"
CELLAB_DS  = SCRIPT_DIR / "cellAB_dataset"
OUT_DIR    = SCRIPT_DIR / "merged_v3_dataset"


def collect_real_paths(dirs):
    s = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            try:
                s.add(f.resolve())
            except Exception:
                pass
    return s


def link_new_only(ab_img_dir, ab_lbl_dir, dst_img_dir, dst_lbl_dir, exclude_real):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = 0
    for f in ab_img_dir.iterdir():
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        try:
            real = f.resolve()
        except Exception:
            continue
        if real in exclude_real:
            n_skip += 1
            continue
        lbl = ab_lbl_dir / f"{f.stem}.txt"
        if not lbl.exists():
            continue
        dst_img = dst_img_dir / f.name
        dst_lbl = dst_lbl_dir / lbl.name
        if not dst_img.exists():
            dst_img.symlink_to(real)
        if not dst_lbl.exists():
            dst_lbl.symlink_to(lbl.resolve())
        n_new += 1
    return n_new, n_skip


def count_images(dirs):
    total = 0
    for d in dirs:
        if d.exists():
            total += sum(1 for f in d.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    return total


def main():
    # Rebuild output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    merged_train_dirs = [MERGED_DS/"haug_base/train/images", MERGED_DS/"new/train/images"]
    merged_val_dirs   = [MERGED_DS/"haug_base/val/images",   MERGED_DS/"new/val/images"]
    merged_real = collect_real_paths(merged_train_dirs + merged_val_dirs)
    print(f"merged_dataset images (train+val): {len(merged_real):,}")

    # Link truly new AB images
    for split in ("train", "val"):
        n, s = link_new_only(
            CELLAB_DS/split/"images", CELLAB_DS/split/"labels",
            OUT_DIR/f"ab_new/{split}/images", OUT_DIR/f"ab_new/{split}/labels",
            merged_real,
        )
        print(f"  ab_new/{split}: new={n:,}  skipped(overlap)={s:,}")

    # Write data.yaml
    ab_new_train = count_images([OUT_DIR/"ab_new/train/images"])
    ab_new_val   = count_images([OUT_DIR/"ab_new/val/images"])
    merged_train = count_images(merged_train_dirs)
    merged_val   = count_images(merged_val_dirs)

    yaml_lines = [
        "# Merged-v3 dataset = merged_dataset + truly-new Cell AB images",
        f"# merged train: {merged_train:,}  ab_new train: {ab_new_train:,}  total: {merged_train+ab_new_train:,}",
        f"# merged val:   {merged_val:,}    ab_new val:   {ab_new_val:,}    total: {merged_val+ab_new_val:,}",
        "",
        f"path: {OUT_DIR}",
        "",
        "train:",
        f"  - {MERGED_DS}/haug_base/train/images",
        f"  - {MERGED_DS}/new/train/images",
        f"  - ab_new/train/images",
        "",
        "val:",
        f"  - {MERGED_DS}/haug_base/val/images",
        f"  - {MERGED_DS}/new/val/images",
        f"  - ab_new/val/images",
        "",
        "kpt_shape: [4, 3]",
        "flip_idx: [0, 1, 2, 3]",
        "",
        "names:",
        "  0: license_plate",
    ]
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"\n[done] {OUT_DIR}/data.yaml")
    print(f"  total train: {merged_train + ab_new_train:,}")
    print(f"  total val:   {merged_val   + ab_new_val:,}")


if __name__ == "__main__":
    main()
