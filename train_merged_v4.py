#!/usr/bin/env python3
"""Train merged-v4: 300 epochs from v3 best.pt on merged_v3_dataset."""

from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent

model = YOLO(str(SCRIPT_DIR / "artifacts/yolo26n-merged-v3-20260602/best.pt"))

model.train(
    data=str(SCRIPT_DIR / "merged_v3_dataset/data.yaml"),
    epochs=300,
    imgsz=640,
    batch=16,
    workers=4,
    freeze=0,
    lr0=0.0001,
    lrf=0.01,
    warmup_epochs=3,
    device=0,
    project=str(SCRIPT_DIR / "runs/merged-v4"),
    name="train",
    exist_ok=True,
    verbose=True,
)
