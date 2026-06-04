#!/usr/bin/env python3
"""Compare two-step LPR ONNX pipeline timing against one or more best.onnx models on one image.

Timed sections include:
- PlateDet.onnx session.run(...) + decode/NMS
- inter-stage crop + PlateRectifier preprocessing
- PlateRectifier.onnx session.run(...) + corner decode
- final warpPerspective
- full two-step pipeline
- best.onnx session.run(...)

Session initialization, image loading, PlateDet static preprocessing, and file
output are outside the timed measurements.
"""

import argparse
import html
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PLATE_DET_W = 416
PLATE_DET_H = 224
RECT_W = 128
RECT_H = 64
BEST_IMGSZ = 640
RAW_ANCHORS = [
    (8, 4), (15, 8), (23, 12),
    (32, 17), (46, 23), (68, 36),
    (98, 44), (149, 70), (250, 132),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms_center_boxes(boxes, iou_thresh):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    scores = boxes[:, 4]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def process_plate_det_output(output, anchors, stride, conf_thresh):
    _, _, grid_h, grid_w = output.shape
    output = output.reshape(1, 3, 6, grid_h, grid_w)
    output = output.transpose(0, 1, 3, 4, 2)[0]
    grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))

    boxes = []
    for anchor_idx in range(3):
        layer_data = output[anchor_idx]
        box_xy = sigmoid(layer_data[..., :2])
        box_wh = np.exp(np.clip(layer_data[..., 2:4], -20, 20)) * anchors[anchor_idx]
        scores = sigmoid(layer_data[..., 4]) * sigmoid(layer_data[..., 5])
        mask = scores > conf_thresh
        if not np.any(mask):
            continue

        pred_xy = (box_xy[mask] + np.stack((grid_x[mask], grid_y[mask]), axis=-1)) * stride
        for xy, wh, score in zip(pred_xy, box_wh[mask], scores[mask]):
            boxes.append([xy[0], xy[1], wh[0], wh[1], float(score)])
    return boxes


def decode_plate_det(outputs, orig_shape, conf_thresh=0.5, nms_thresh=0.4):
    all_boxes = []
    for output in outputs:
        stride = int(round(PLATE_DET_H / output.shape[2]))
        if stride == 8:
            anchors = RAW_ANCHORS[0:3]
        elif stride == 16:
            anchors = RAW_ANCHORS[3:6]
        elif stride == 32:
            anchors = RAW_ANCHORS[6:9]
        else:
            continue
        all_boxes.extend(process_plate_det_output(output, anchors, stride, conf_thresh))

    final_boxes = nms_center_boxes(all_boxes, nms_thresh)
    if not final_boxes:
        return None

    box = max(final_boxes, key=lambda b: b[4])
    img_h, img_w = orig_shape[:2]
    scale_x = img_w / PLATE_DET_W
    scale_y = img_h / PLATE_DET_H
    cx, cy = box[0] * scale_x, box[1] * scale_y
    w, h = box[2] * scale_x, box[3] * scale_y
    x1 = int(np.clip(cx - w / 2, 0, img_w - 1))
    y1 = int(np.clip(cy - h / 2, 0, img_h - 1))
    x2 = int(np.clip(cx + w / 2, 0, img_w - 1))
    y2 = int(np.clip(cy + h / 2, 0, img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"bbox": (x1, y1, x2, y2), "score": float(box[4])}


def preprocess_plate_det(img_bgr):
    resized = cv2.resize(img_bgr, (PLATE_DET_W, PLATE_DET_H), interpolation=cv2.INTER_LINEAR)
    tensor = resized[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(tensor[None], dtype=np.float32) / 255.0


def preprocess_rectifier(plate_bgr):
    resized = cv2.resize(plate_bgr, (RECT_W, RECT_H), interpolation=cv2.INTER_LINEAR)
    tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None], dtype=np.float32)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    gain = min(new_h / h, new_w / w)
    resized_w, resized_h = int(round(w * gain)), int(round(h * gain))
    pad_w = (new_w - resized_w) / 2
    pad_h = (new_h - resized_h) / 2
    if (w, h) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, gain, (left, top)


def preprocess_best(img_bgr, imgsz):
    padded, gain, pad = letterbox(img_bgr, (imgsz, imgsz))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return tensor[None], gain, pad


def sort_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def rectifier_corners_to_image(output, bbox):
    x1, y1, x2, y2 = bbox
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    corners = np.asarray(output).reshape(4, 2).astype(np.float32)
    corners[:, 0] = corners[:, 0] * crop_w + x1
    corners[:, 1] = corners[:, 1] * crop_h + y1
    return sort_corners(corners)


def decode_best_detection(output, gain, pad, orig_shape, conf_threshold):
    detections = np.asarray(output)
    if detections.ndim == 3:
        detections = detections[0]
    detections = detections[detections[:, 4] >= conf_threshold]
    if len(detections) == 0:
        return None
    row = detections[np.argmax(detections[:, 4])]
    kpts = row[6:].reshape(4, 3)[:, :2].astype(np.float32)
    kpts[:, 0] = (kpts[:, 0] - pad[0]) / gain
    kpts[:, 1] = (kpts[:, 1] - pad[1]) / gain
    h, w = orig_shape[:2]
    kpts[:, 0] = np.clip(kpts[:, 0], 0, w - 1)
    kpts[:, 1] = np.clip(kpts[:, 1], 0, h - 1)
    return {"conf": float(row[4]), "keypoints": sort_corners(kpts)}


def warp_plate(img_bgr, corners, target_w=400, target_h=120):
    corners = sort_corners(corners)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, matrix, (target_w, target_h), flags=cv2.INTER_LANCZOS4)


def draw_final(img_bgr, bbox, corners):
    out = img_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    pts = corners.astype(int)
    for idx, pt in enumerate(pts):
        cv2.circle(out, tuple(pt), 4, (0, 255, 255), -1)
        cv2.putText(out, str(idx), tuple(pt + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    return out


def make_session(model_path, threads):
    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_single(session, input_name, output_names, tensor):
    return session.run(output_names, {input_name: tensor})


def benchmark(callable_fn, warmup, runs):
    last_output = None
    for _ in range(warmup):
        last_output = callable_fn()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        last_output = callable_fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings, last_output


def benchmark_segments(callable_fn, warmup, runs):
    last_output = None
    for _ in range(warmup):
        last_output = callable_fn()

    segment_timings = {}
    last_output = None
    for _ in range(runs):
        result, timings = callable_fn()
        last_output = result
        for name, value in timings.items():
            segment_timings.setdefault(name, []).append(value)
    return segment_timings, last_output


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    avg = float(arr.mean())
    return {
        "avg": avg,
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fps": 1000.0 / avg,
    }


def format_stats(name, stats):
    return [
        f"{name}:",
        f"  avg: {stats['avg']:.3f} ms",
        f"  median: {stats['median']:.3f} ms",
        f"  p90: {stats['p90']:.3f} ms",
        f"  p95: {stats['p95']:.3f} ms",
        f"  min: {stats['min']:.3f} ms",
        f"  max: {stats['max']:.3f} ms",
        f"  fps: {stats['fps']:.2f}",
    ]


def model_label(path):
    path = Path(path)
    if path.name == "best.onnx":
        return f"{path.parent.name}/best.onnx"
    return path.name


def model_type(label):
    if "1775509671" in label:
        return "YOLOv26-pose-m"
    if "1773128236" in label:
        return "YOLOv26-pose-s"
    if "1779339446" in label:
        return "YOLOv26-pose-n"
    return "YOLO"


def display_model(label):
    return f"{model_type(label)} ({label})"


def format_report(args, provider, shape_info, det_info, best_infos, round_results, output_paths):
    lines = [
        "Two-step LPR ONNX vs YOLO best.onnx benchmark",
        f"Image: {args.image}",
        f"Provider: {provider}",
        f"Warmup before each measured phase: {args.warmup}",
        f"Rounds: {', '.join(str(x) for x in args.rounds)}",
        "",
        "Model types:",
        "  Two-step pipeline: YOLOv4 PlateDet + PlateRectifier corner model",
    ]
    for label in best_infos:
        lines.append(f"  {display_model(label)}")

    lines.extend([
        "",
        "Model inputs:",
        f"  YOLOv4 PlateDet: {shape_info['det']}",
        f"  PlateRectifier: {shape_info['rect']}",
    ])
    for label in best_infos:
        lines.append(f"  {display_model(label)}: {shape_info[label]}")

    lines.extend([
        "",
        "Detections:",
        f"  PlateDet score: {det_info['score']:.6f}",
        f"  PlateDet bbox: {det_info['bbox']}",
    ])
    for label, info in best_infos.items():
        lines.append(f"  {display_model(label)} confidence: {info['conf']:.6f}")

    best_stat_names = [f"{label} inference" for label in best_infos]
    for runs, stats_by_name in round_results:
        lines.extend(["", f"Measured runs: {runs}", "Segment timing:"])
        for name in ["Full two-step pipeline"]:
            lines.extend(format_stats(name, stats_by_name[name]))
        for name in best_stat_names:
            lines.extend(format_stats(name, stats_by_name[name]))
        two_step = stats_by_name["Full two-step pipeline"]["avg"]
        for name in best_stat_names:
            best = stats_by_name[name]["avg"]
            lines.append(f"Comparison avg: full two-step / {name} = {two_step / best:.3f}x")

    lines.extend(
        [
            "",
            f"Saved two-step annotated image: {output_paths['annotated']}",
            f"Saved two-step final plate image: {output_paths['plate']}",
        ]
    )
    for label, path in output_paths["best_plates"].items():
        lines.append(f"Saved {label} final plate image: {path}")
    return "\n".join(lines) + "\n"


def rel_path(path, base_dir):
    return Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix()


def fmt_ms(stats, key="avg"):
    return f"{stats[key]:.3f}"


def format_html_report(args, provider, shape_info, det_info, best_infos, round_results, output_paths, result_path, html_path):
    summary_runs, summary_stats = round_results[-1]
    two_step_fps = summary_stats["Full two-step pipeline"]["fps"]
    summary_lines = [
        f"YOLOv4 PlateDet + PlateRectifier: {two_step_fps:.2f} FPS over {summary_runs} measured runs."
    ]

    rows = []
    for runs, stats_by_name in round_results:
        names = ["Full two-step pipeline"]
        names.extend(f"{label} inference" for label in best_infos)
        for name in names:
            stats = stats_by_name[name]
            rows.append(
                "<tr>"
                f"<td class=\"runs\">{runs}</td>"
                f"<td class=\"section\">{html.escape(name)}</td>"
                f"<td>{fmt_ms(stats)} ms</td>"
                f"<td>{stats['fps']:.2f} FPS</td>"
                "</tr>"
            )

    image_cards = [
        ("Two-step annotated", output_paths["annotated"]),
        ("Two-step final plate", output_paths["plate"]),
    ]
    image_cards.extend((f"{display_model(label)} final plate", path) for label, path in output_paths["best_plates"].items())
    image_html = []
    for title, path in image_cards:
        image_html.append(
            "<figure>"
            f"<img src=\"{html.escape(rel_path(path, html_path.parent))}\" alt=\"{html.escape(title)}\">"
            f"<figcaption>{html.escape(title)}</figcaption>"
            "</figure>"
        )

    model_rows = [
        ("YOLOv4 PlateDet", shape_info["det"]),
        ("PlateRectifier", shape_info["rect"]),
    ]
    model_rows.extend((display_model(label), shape_info[label]) for label in best_infos)

    detection_items = [
        f"<li>PlateDet score: <code>{det_info['score']:.6f}</code></li>",
        f"<li>PlateDet bbox: <code>{det_info['bbox']}</code></li>",
    ]
    detection_items.extend(
        f"<li>{html.escape(display_model(label))} confidence: <code>{info['conf']:.6f}</code></li>"
        for label, info in best_infos.items()
    )

    model_items = "".join(
        f"<li>{html.escape(name)}: <code>{shape}</code></li>" for name, shape in model_rows
    )
    summary_items = "".join(f"<li>{html.escape(line)}</li>" for line in summary_lines)
    detection_html = "".join(detection_items)
    table_rows = "\n".join(rows)
    image_grid = "\n".join(image_html)
    result_rel = html.escape(rel_path(result_path, html_path.parent))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LPR ONNX Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 22px 0; }}
    table {{ border-collapse: collapse; width: 50%; min-width: 520px; table-layout: fixed; font-size: 13px; }}
    th, td {{ border: 1px solid #cfd7df; padding: 4px 6px; }}
    th {{ background: #edf2f7; text-align: left; }}
    td {{ vertical-align: top; }}
    th.runs, td.runs {{ width: 50px; text-align: left; }}
    th.section, td.section {{ width: 56%; text-align: left; word-break: break-word; }}
    th.ms, td.ms {{ width: 22%; text-align: left; white-space: nowrap; }}
    th.fps, td.fps {{ width: 18%; text-align: left; white-space: nowrap; }}
    code {{ background: #eef2f6; padding: 1px 4px; border-radius: 3px; }}
    .meta {{ line-height: 1.65; }}
    .summary {{ border: 2px solid #2563eb; background: #eff6ff; padding: 16px; }}
    .summary ul {{ margin: 8px 0 0; padding-left: 22px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }}
    .metric-card {{ border: 2px solid #94a3b8; background: #ffffff; padding: 14px; }}
    .metric-card.win {{ border-color: #15803d; background: #f0fdf4; }}
    .metric-card.near {{ border-color: #b45309; background: #fffbeb; }}
    .metric-title {{ font-weight: 700; margin-bottom: 8px; }}
    .metric-value {{ font-size: 28px; font-weight: 800; margin-bottom: 6px; }}
    .metric-detail, .metric-ratio {{ color: #3b4754; font-size: 14px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid #cfd7df; padding: 10px; background: #fff; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>LPR ONNX Benchmark Report</h1>

  <section>
    <h2>Timing</h2>
    <table>
      <thead>
        <tr><th class="runs">Runs</th><th class="section">Section</th><th class="ms">Avg ms</th><th class="fps">FPS</th></tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </section>

    <section class="summary">
    <h2>Summary</h2>
    <ul>{summary_items}</ul>
  </section>

  <section class="meta">
    <div>Image: <code>{html.escape(args.image)}</code></div>
    <div>Provider: <code>{html.escape(provider)}</code></div>
    <div>Warmup before each measured phase: <code>{args.warmup}</code></div>
    <div>Rounds: <code>{html.escape(', '.join(str(x) for x in args.rounds))}</code></div>
    <div>Text report: <a href="{result_rel}">{result_rel}</a></div>
  </section>

  <section>
    <h2>Model Inputs</h2>
    <ul>{model_items}</ul>
  </section>

  <section>
    <h2>Detections</h2>
    <ul>{detection_html}</ul>
  </section>

  <section>
    <h2>Images</h2>
    <div class="grid">{image_grid}</div>
  </section>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/home/itemhsu/amtk/lppart/lp/lp_viewer_tool/lp_viewer_output/testImg/000/AQY6066_0.jpg")
    parser.add_argument("--plate-det", default="/home/itemhsu/amtk/lpr/0x1PlateDet/PlateDet.onnx")
    parser.add_argument("--rectifier", default="/home/itemhsu/amtk/lpr/PlateRectifier.onnx")
    parser.add_argument(
        "--best-models",
        nargs="+",
        default=[
            "yolo26-train-1775509671/best.onnx",
            "yolo26-train-1773128236/best.onnx",
            "yolo26-train-1779339446/best.onnx",
        ],
    )
    parser.add_argument("--out-dir", default="/home/itemhsu/amtk/lppart/lp/lp_viewer_tool/lp_viewer_output/model")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, nargs="+", default=[20])
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--best-conf", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    image_path = Path(args.image)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    det_session = make_session(Path(args.plate_det), args.threads)
    rect_session = make_session(Path(args.rectifier), args.threads)
    best_sessions = {}
    for best_path in args.best_models:
        label = model_label(best_path)
        session = make_session(Path(best_path), args.threads)
        best_sessions[label] = {
            "path": Path(best_path),
            "session": session,
            "input": session.get_inputs()[0].name,
            "outputs": [o.name for o in session.get_outputs()],
        }

    det_in = det_session.get_inputs()[0].name
    rect_in = rect_session.get_inputs()[0].name
    det_out = [o.name for o in det_session.get_outputs()]
    rect_out = [o.name for o in rect_session.get_outputs()]

    det_tensor = preprocess_plate_det(img_bgr)
    det_outputs = run_single(det_session, det_in, det_out, det_tensor)
    det_info = decode_plate_det(det_outputs, img_bgr.shape, args.det_conf)
    if det_info is None:
        raise RuntimeError(f"PlateDet found no plate above confidence {args.det_conf}")

    x1, y1, x2, y2 = det_info["bbox"]
    plate_crop = img_bgr[y1:y2, x1:x2]
    rect_tensor = preprocess_rectifier(plate_crop)
    rect_outputs = run_single(rect_session, rect_in, rect_out, rect_tensor)
    two_step_corners = rectifier_corners_to_image(rect_outputs[0], det_info["bbox"])

    best_infos = {}
    for label, item in best_sessions.items():
        tensor, gain, pad = preprocess_best(img_bgr, BEST_IMGSZ)
        outputs = run_single(item["session"], item["input"], item["outputs"], tensor)
        info = decode_best_detection(outputs[0], gain, pad, img_bgr.shape, args.best_conf)
        if info is None:
            raise RuntimeError(f"{label} found no plate above confidence {args.best_conf}")
        item["tensor"] = tensor
        item["gain"] = gain
        item["pad"] = pad
        best_infos[label] = info

    def run_plate_det_with_nms():
        outputs = run_single(det_session, det_in, det_out, det_tensor)
        decoded = decode_plate_det(outputs, img_bgr.shape, args.det_conf)
        if decoded is None:
            raise RuntimeError(f"PlateDet found no plate above confidence {args.det_conf}")
        return decoded

    def run_rectifier_with_decode(rect_tensor_arg, bbox):
        rect_outputs = run_single(rect_session, rect_in, rect_out, rect_tensor_arg)
        corners = rectifier_corners_to_image(rect_outputs[0], bbox)
        return rect_outputs, corners

    def run_full_two_step_segments():
        total_start = time.perf_counter()

        start = time.perf_counter()
        outputs = run_single(det_session, det_in, det_out, det_tensor)
        decoded = decode_plate_det(outputs, img_bgr.shape, args.det_conf)
        if decoded is None:
            raise RuntimeError(f"PlateDet found no plate above confidence {args.det_conf}")
        det_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        crop_x1, crop_y1, crop_x2, crop_y2 = decoded["bbox"]
        crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        rect_tensor_arg = preprocess_rectifier(crop)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        _, corners = run_rectifier_with_decode(rect_tensor_arg, decoded["bbox"])
        rect_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        plate = warp_plate(img_bgr, corners)
        warp_ms = (time.perf_counter() - start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return (
            {"decoded": decoded, "corners": corners, "plate": plate},
            {
                "PlateDet + NMS": det_ms,
                "Crop + Rectifier preprocess": preprocess_ms,
                "PlateRectifier + corner decode": rect_ms,
                "Final warpPerspective": warp_ms,
                "Full two-step pipeline": total_ms,
            },
        )

    round_results = []
    for runs in args.rounds:
        segment_times, _ = benchmark_segments(run_full_two_step_segments, args.warmup, runs)
        stats = {name: summarize(times) for name, times in segment_times.items()}
        for label, item in best_sessions.items():
            best_times, _ = benchmark(
                lambda item=item: run_single(item["session"], item["input"], item["outputs"], item["tensor"]),
                args.warmup,
                runs,
            )
            stats[f"{label} inference"] = summarize(best_times)
        round_results.append(
            (
                runs,
                stats,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "lpr_two_step_AQY6066_0_annotated.jpg"
    plate_path = out_dir / "lpr_two_step_AQY6066_0_plate.jpg"
    result_path = out_dir / "lpr_two_step_vs_best_AQY6066_0_result.txt"
    html_path = out_dir / "lpr_two_step_vs_best_AQY6066_0_report.html"

    annotated = draw_final(img_bgr, det_info["bbox"], two_step_corners)
    two_step_plate = warp_plate(img_bgr, two_step_corners)
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(plate_path), two_step_plate)

    best_plate_paths = {}
    for label, info in best_infos.items():
        safe_label = label.replace("/", "_").replace(".", "_")
        best_plate_path = out_dir / f"{safe_label}_AQY6066_0_plate_compare.jpg"
        best_plate = warp_plate(img_bgr, info["keypoints"])
        cv2.imwrite(str(best_plate_path), best_plate)
        best_plate_paths[label] = best_plate_path

    shape_info = {
        "det": list(det_tensor.shape),
        "rect": list(rect_tensor.shape),
    }
    for label, item in best_sessions.items():
        shape_info[label] = list(item["tensor"].shape)
    output_paths = {
        "annotated": annotated_path,
        "plate": plate_path,
        "best_plates": best_plate_paths,
    }
    report = format_report(
        args=args,
        provider=det_session.get_providers()[0],
        shape_info=shape_info,
        det_info=det_info,
        best_infos=best_infos,
        round_results=round_results,
        output_paths=output_paths,
    )
    result_path.write_text(report, encoding="utf-8")
    html_report = format_html_report(
        args=args,
        provider=det_session.get_providers()[0],
        shape_info=shape_info,
        det_info=det_info,
        best_infos=best_infos,
        round_results=round_results,
        output_paths=output_paths,
        result_path=result_path,
        html_path=html_path,
    )
    html_path.write_text(html_report, encoding="utf-8")
    print(report, end="")
    print(f"Saved result text: {result_path}")
    print(f"Saved HTML report: {html_path}")


if __name__ == "__main__":
    main()
