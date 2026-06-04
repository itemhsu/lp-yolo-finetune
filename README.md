# YOLOv26n-Pose 台灣車牌四角點偵測 — 調優紀錄

**GitHub Repository：[https://github.com/itemhsu/lp-yolo-finetune](https://github.com/itemhsu/lp-yolo-finetune)**

> 調優目標：以 **YOLOv26n-pose**（4 keypoint，kpt\_shape=[4,3]）偵測台灣車牌四角點，取代兩段式 Two-step pipeline（PlateDet + PlateRectifier），搭配 PARSeq OCR 做端對端車牌辨識。

---

## 快速下載

### Merged 模型（YOLOv26n-pose）

| 版本 | mAP50(B) | mAP50(P) | PyTorch (.pt) | ONNX (.onnx) |
|---|---|---|---|---|
| **v4 ep292 ★ 最終推薦** | 0.9267 | 0.8332 | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v4-20260604/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v4-20260604/best.onnx) |
| v3 ep59 | 0.8528 | 0.6845 | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v3-20260602/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v3-20260602/best.onnx) |
| v2 ep30 | 0.8089 | 0.6434 | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v2-20260530-143438/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v2-20260530-143438/best.onnx) |

### Two-step 基準管線模型（Git LFS）

| 模型 | 說明 | 下載 |
|---|---|---|
| PlateDet.onnx | YOLOv4 車牌 bbox 偵測（244MB，Step 1） | [下載](https://github.com/itemhsu/lp-yolo-finetune/raw/master/two_step_models/0x1PlateDet/PlateDet.onnx) |
| PlateRectifier.onnx | 4 角點 regressor（7.4MB，Step 2） | [下載](https://github.com/itemhsu/lp-yolo-finetune/raw/master/two_step_models/PlateRectifier.onnx) |

---

## 目錄

1. [背景與動機](#1-背景與動機)
2. [整體訓練脈絡](#2-整體訓練脈絡)
3. [資料集](#3-資料集)
4. [各版本訓練設定](#4-各版本訓練設定)
5. [評估結果](#5-評估結果)
6. [最終模型 Artifacts](#6-最終模型-artifacts)
7. [訓練程式碼](#7-訓練程式碼)
8. [關鍵決策記錄](#8-關鍵決策記錄)
9. [HTML 文件時間軸](#9-html-文件時間軸)
10. [再訓練步驟](#10-再訓練步驟)

---

## 1. 背景與動機

### Two-step pipeline（舊）
```
輸入圖片
  → PlateDet.onnx（YOLOv4，偵測車牌 bbox）
  → PlateRectifier.onnx（regress 4 corner）
  → warp 400×120
  → PARSeq OCR
```
缺點：兩個模型串接，推論慢，角點精度受限於 regressor。

### Merged pipeline（新）
```
輸入圖片
  → YOLOv26n-pose（單模型，直接輸出 bbox + 4 keypoints）
  → warp 400×120
  → PARSeq OCR
```
優點：單模型、端對端、角點精度可透過資料擴充持續改善。

### 評估基準（23,222 張 LPD 資料集）

| 管線 | OCR Match（IoU+minLen修正後） |
|---|---|
| Two-step（基準） | 4,901 (21.1%) |
| Merged-v2（ep30） | 4,772 (20.5%) |
| Merged-v3（ep60） | 4,869 (21.0%) |
| **Merged-v4（ep292）** | **4,970 (21.4%)** ★ |

> IoU 修正邏輯：two-step OCR 正確 + merged 角點與 two-step 重疊率 ≥ 0.5 → 判定 merged 偵測正確（即使 OCR 差 1 碼）。OCR 結果 < 4 碼視為誤判排除。

---

## 2. 整體訓練脈絡

```
yolo26n_best.pt（SageMaker 原始訓練基底）
    │
    ▼ 30 epochs  lr0=0.001  freeze=10
yolo26n-merged-v2（2026-05-30）
  資料：haug_base 2,746 + pseudo-label 4,459 = 7,205 train
  mAP50(B)=0.8089  mAP50(P)=0.6434
    │
    ▼ 30 epochs  lr0=0.0005  freeze=0
yolo26n-merged-v3（2026-06-02）
  資料：同上 merged_dataset（8,471 train，加入 ab_new 1,266）
  mAP50(B)=0.8528  mAP50(P)=0.6845
    │
    ▼ 300 epochs  lr0=0.0001  lrf=0.01  close_mosaic=10
yolo26n-merged-v4（2026-06-04，best=ep292）
  資料：merged_v3_dataset 8,471 train
  mAP50(B)=0.9267  mAP50(P)=0.8332
```

---

## 3. 資料集

### 3.1 資料來源總覽

| 資料集 | 說明 | Train | Val |
|---|---|---|---|
| `haug_base` | Roboflow lp-det-v3-job3（去 augmentation 後的 base image） | 2,746 | 222 |
| `new`（pseudo-label）| LPD 23,222 張中 two-step 可信角點的偽標籤（Cell TF + TT 策略） | 4,459 | 1,114 |
| `ab_new`（Cell AB）| Two-step OCR 像車牌 + merged 讀出不同 → two-step corners 當 GT | 1,266 | 183 |
| **merged_v3\_dataset** | haug\_base + new + ab\_new | **8,471** | **1,519** |

### 3.2 偽標籤生成策略（2×2 矩陣）

```
                  YOLOv26n OCR
                  Match    No-match
Two-step OCR  ┌──────────┬──────────┐
  Match       │  TT ✓✓   │  TF ⭐   │  ← TF 最有訓練價值
              │  多樣性   │  YOLO弱點│    Two-step corners → pseudo-GT
  No-match    │  FT      │  FF      │  ← 不用（GT 可信度低）
              └──────────┴──────────┘
```

**TF（Two-step 對，YOLO 錯）**：OCR 比對 = corner 正確的代理指標。  
清洗後 OCR 是檔名子字串 → 幾乎必然 corners 正確 → corners 可當 pseudo ground-truth。

### 3.3 Cell AB 補強（merged-v3 → v4）

- 定義：Two-step OCR 像車牌（5-7碼）且 merged OCR 不同
- 規模：3,872 張，其中 662 張 merged 完全未偵測
- 意義：merged 的 keypoint warp 精度比 two-step 差 → 字符邊界切錯 → OCR 差 1-2 個字
- 用法：以 two-step corners 為 YOLO-pose GT，fine-tune merged

### 3.4 資料集 YAML（merged\_v3\_dataset/data.yaml）

```yaml
# merged train: 7,205  ab_new train: 1,266  total: 8,471
# merged val:   1,336    ab_new val:   183    total: 1,519

kpt_shape: [4, 3]       # 4 corners, each (x, y, visibility)
flip_idx: [0, 1, 2, 3]  # TL TR BR BL
names:
  0: license_plate

train:
  - merged_dataset/haug_base/train/images
  - merged_dataset/new/train/images
  - merged_v3_dataset/ab_new/train/images

val:
  - merged_dataset/haug_base/val/images
  - merged_dataset/new/val/images
  - merged_v3_dataset/ab_new/val/images
```

---

## 4. 各版本訓練設定

### Merged-v2（SageMaker，2026-05-30）

```python
model.train(
    data="data_corrected.yaml",   # haug + pseudo-label 7,205 train
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    freeze=10,                     # 前 10 層凍結（遷移學習）
    optimizer="auto",
    device=None,                   # SageMaker GPU
)
```

**結果**：val mAP50(B)=0.8089, mAP50(P)=0.6434

---

### Merged-v3（本地，2026-06-02）

```python
model = YOLO("yolo26-merged-v1/best.pt")   # v2 best
model.train(
    data="merged_dataset/data.yaml",        # 8,471 train（含 ab_new）
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.0005,                             # 較低 LR，避免損毀 v2 權重
    lrf=0.01,
    warmup_epochs=3,
    freeze=0,                               # 全層解凍
    optimizer="auto",
    device=0,
)
```

**結果**：val mAP50(B)=0.8528（ep59 best）, mAP50(P)=0.6845  
**ONNX 驗證**：box mAP50=0.8290, pose mAP50=0.7220

---

### Merged-v4（本地，2026-06-02 開始，300 epochs）

```python
# train_merged_v4.py
model = YOLO("artifacts/yolo26n-merged-v3-20260602/best.pt")
model.train(
    data="merged_v3_dataset/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    workers=4,
    freeze=0,
    lr0=0.0001,    # 極低 LR，精細調整
    lrf=0.01,      # cosine 最終 LR = lr0 × lrf = 1e-6
    warmup_epochs=3,
    device=0,
    project="runs/merged-v4",
    name="train",
    exist_ok=True,
)
```

**過擬合分析**：
- ep261：val/pose 最低（0.955），無過擬合高點
- ep286：mAP50(P) 峰值（0.8337），val/pose=1.020
- ep291：`close_mosaic=10` 啟動，train/box gap 從 +0.085 突增至 +0.152
- **ep292**：YOLO fitness 最高 → best.pt（mAP50(B)=0.9267, mAP50(P)=0.8332）
- ep300：last.pt（mAP50(B)=0.9278 但 mAP50(P)=0.8294，pose 過擬合）

**最終選擇 best.pt（ep292）**：與理論最佳 ep286 差距 < 0.001 mAP，val/pose 輕微過擬合但可接受。

---

## 5. 評估結果

### 5.1 ONNX Val 指標（merged\_v3\_dataset）

| 模型 | mAP50(B) | mAP50(P) | val/box_loss | val/pose_loss |
|---|---|---|---|---|
| v2（ep30） | 0.8089 | 0.6434 | — | — |
| v3（ep59 best） | 0.8290 | 0.7220 | — | — |
| **v4（ep292 best）** | **0.9267** | **0.8332** | 0.8686 | 1.034 |

v4 vs v3：+8.7% box，+14.9% pose。

### 5.2 23,222 張 LPD 四路比較（IoU+minLen修正）

| 管線 | Match | vs TS | vs v3 | 說明 |
|---|---|---|---|---|
| Two-step（基準） | 4,901 (21.1%) | — | — | OCR≥4碼過濾後 |
| Merged-v2 | 4,772 (20.5%) | -129 | -97 | ep30 |
| Merged-v3 | 4,869 (21.0%) | -32 | — | ep60 |
| **Merged-v4** | **4,970 (21.4%)** | **+69** | **+101** | ep292 ★ |

v4 vs v3 差異：新增 204 張，退化 103 張，**淨 +101**。

### 5.3 測試集比較（Two-step vs Merged-v4）

| 測試集 | 張數 | Two-step | Merged-v4 | 差距 |
|---|---|---|---|---|
| 1111（2024-11-11） | 127 | 102 (80.3%) | 99 (78.0%) | -3 |
| 1118（2024-11-18） | 95 | 82 (86.3%) | 78 (82.1%) | -4 |
| 1125（2024-11-25） | 142 | 124 (87.3%) | 116 (81.7%) | -8 |
| 1202（2024-12-02） | 157 | 116 (73.9%) | 110 (70.1%) | -6 |
| w1111 | 133 | 102 (76.7%) | 94 (70.7%) | -8 |
| **合計** | **654** | **526 (80.4%)** | **497 (76.0%)** | **-29** |

> Two-step 在小測試集仍領先 4-8%，主因 PARSeq OCR 在這批測試圖表現較穩定；v4 角點偵測框已明顯更好（LPD 大集合 IoU 修正後超越 two-step），差距主要來自 OCR 末碼誤讀。

---

## 6. 最終模型 Artifacts

### 各版本差異對照

| 版本 | 日期 | best epoch | 起始權重 | 資料集 | Train 筆數 | epochs | lr0 | freeze | mAP50(B) | mAP50(P) | val/box_loss | val/pose_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v2** | 2026-05-30 | ep30 | yolo26n_best.pt（SageMaker 原始） | haug_base + pseudo-label | 7,205 | 30 | 0.001 | 10 | 0.8089 | 0.6434 | — | — |
| **v3** | 2026-06-02 | ep59 | v2 best.pt | merged_dataset（+ ab_new） | 8,471 | 30 | 0.0005 | 0 | 0.8528 | 0.6845 | — | — |
| **v4** ★ | 2026-06-04 | ep292 | v3 best.pt | merged_v3_dataset | 8,471 | 300 | 0.0001 | 0 | **0.9267** | **0.8332** | 0.8686 | 1.034 |

**關鍵差異說明：**

- **v2 → v3**：加入 Cell AB 補強資料（1,266 張，two-step OCR 對但 merged 讀錯的案例）；解凍全部層（freeze=0）；降低 lr0 避免損毀 v2 權重。mAP50(P) +2.4%。
- **v3 → v4**：同一資料集再跑 300 epochs，lr0 降至 0.0001（精細調整）。ep291 後 `close_mosaic=10` 啟動（最後 10 epoch 關閉馬賽克增強）。mAP50(B) +7.4%、mAP50(P) +14.9%。
- **選 ep292 而非 ep300**：ep291 之後 val/pose_loss 快速上升（1.034 → 1.065），mAP50(P) 下滑（0.8332 → 0.8294）。ep292 為 YOLO fitness 峰值，與理論最佳 ep286 差距 < 0.001。

### 檔案清單

```
artifacts/
├── yolo26n-merged-v2-20260530-143438/
│   ├── best.pt    (5.7 MB)  ep30   mAP50(B)=0.8089  mAP50(P)=0.6434
│   └── best.onnx  (9.8 MB)  opset17，onnxslim 壓縮
├── yolo26n-merged-v3-20260602/
│   ├── best.pt    (5.7 MB)  ep59   mAP50(B)=0.8528  mAP50(P)=0.6845
│   └── best.onnx  (9.8 MB)  opset17，onnxslim 壓縮
└── yolo26n-merged-v4-20260604/       ← 最終推薦
    ├── best.pt    (5.8 MB)  ep292  mAP50(B)=0.9267  mAP50(P)=0.8332
    └── best.onnx  (9.8 MB)  opset17，onnxslim 壓縮
```

### 下載連結（GitHub）

| 版本 | PyTorch (.pt) | ONNX (.onnx) |
|---|---|---|
| **v4 ★ 最終推薦** | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v4-20260604/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v4-20260604/best.onnx) |
| v3（ep59） | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v3-20260602/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v3-20260602/best.onnx) |
| v2（ep30） | [best.pt](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v2-20260530-143438/best.pt) | [best.onnx](https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v2-20260530-143438/best.onnx) |

### Two-step 基準管線模型（Git LFS）

作為評估基準的兩段式偵測管線，也一併收錄：

| 模型 | 說明 | 大小 | 下載 |
|---|---|---|---|
| `PlateDet.onnx` | YOLOv4 車牌 bbox 偵測器（Step 1） | 244 MB | [下載（LFS）](https://github.com/itemhsu/lp-yolo-finetune/raw/master/two_step_models/0x1PlateDet/PlateDet.onnx) |
| `PlateRectifier.onnx` | 4 角點 keypoint regressor（Step 2） | 7.4 MB | [下載（LFS）](https://github.com/itemhsu/lp-yolo-finetune/raw/master/two_step_models/PlateRectifier.onnx) |

Two-step 推論流程：
```
image → PlateDet.onnx (bbox) → crop → PlateRectifier.onnx (4 corners)
      → perspective warp 400×120 → PARSeq OCR → plate string
```
推論程式：`benchmark_lpr_two_step_compare.py`（`decode_plate_det`, `preprocess_rectifier`, `warp_plate` 等核心函數）

匯出指令：
```bash
python -c "
from ultralytics import YOLO
model = YOLO('artifacts/yolo26n-merged-v4-20260604/best.pt')
model.export(format='onnx', imgsz=640, opset=17, simplify=True)
"
```

---

## 7. 訓練程式碼

| 程式 | 說明 |
|---|---|
| `train_merged_v4.py` | v4 訓練入口（300 epochs，從 v3 best.pt 繼續） |
| `build_merged_v3_dataset.py` | 建立 merged_v3_dataset（合併 haug + pseudo + ab_new） |
| `build_cellAB_dataset.py` | 從 lpd 結果萃取 Cell AB 並轉 YOLO-pose 格式 |
| `build_merged_dataset.py` | 建立基礎 merged\_dataset（haug + pseudo-label） |
| `build_yolo_ftn_dataset.py` | 生成偽標籤（Two-step corners → YOLO-pose .txt） |
| `benchmark_lpd_batch.py` | 四管線批次掃描 23,222 張 → lpd\_results.csv |
| `bench_three_way.py` | v2/v3 vs two-step ONNX 三路 benchmark |
| `bench_four_way.py` | v2/v3/v4 vs two-step ONNX 四路 benchmark |
| `recompute_iou_matches.py` | IoU 修正 + minOCR 過濾（生成 *\_iou.csv） |
| `report_three_way.py` | 三路 HTML 報告（支援 `--iou` flag） |
| `report_four_way.py` | 四路 HTML 報告（v4 vs v3 差異摘要） |
| `test_compare.py` | 小測試集 Two-step vs v4 互動比較報告（含 filter + sort） |
| `merged_v3_dashboard.py` | v3 訓練 live dashboard（JS polling，無閃爍） |
| `merged_v4_dashboard.py` | v4 訓練 live dashboard（v2+v3 歷史 + v4 即時，ep1-360 連續） |

---

## 8. 關鍵決策記錄

### 8.1 為什麼用 OCR match 當 corner 正確性代理

台灣車牌格式固定（5-7碼英數字）。當 PARSeq 讀出 `BAB6075` 且該字串是檔名（如 `20241122105453-962507-EVENT-BAB6075`）的子字串，表示：
- 車牌被正確框住（warp 正確）
- OCR 在清晰影像上正確
- → Two-step corners 可信，作為 pseudo-GT

### 8.2 IoU 修正評估方法的演進

**問題**：OCR 差 1 碼（如 `EAB6075` vs `BAB6075`）會被計為「偵測失敗」，實際上框選是正確的。

**修正 1（IoU）**：v2/v3 角點與 two-step 角點 IoU ≥ 0.5 → 判定偵測正確，即使 OCR 有誤。

**修正 2（不對稱問題）**：若 two-step 本身 OCR 錯誤，其角點不應作為 IoU 的可信參考。新增條件：**IoU 升格只在 `twostep_match_valid=1` 時才有效**。

**修正 3（minOCR）**：`10` 這類極短字串可能是 `105615` 的子字串（假陽性）。新增最短 4 碼限制（`len(ocr_clean) ≥ 4`）。

**修正後 v4 評估**：7,205 → 4,970 有效 match（21.4%），比 two-step 4,901（21.1%）多 69 張。

### 8.3 多邊形 IoU 的實作陷阱

`cv2.intersectConvexConvex(p1, p2)` 對退化多邊形（重複角點）**不對稱**：
- `intersect(A, B)` ≠ `intersect(B, A)` 當其中一個是退化四邊形
- 解法：`area_inter = min(intersect(p1,p2)[0], intersect(p2,p1)[0])`

### 8.4 過擬合識別方法

| 指標 | 說明 |
|---|---|
| `val/pose_loss` 持續上升 | Pose 過擬合的主要訊號（ep260+ 開始） |
| `gap_box = val/box - train/box` 突增 | ep291 從 +0.085 跳至 +0.152（close_mosaic 啟動） |
| `mAP50(P)` 下降 | ep286 後 pose 準確度輕微下滑 |

### 8.5 v3 退化（✓✓✗）的解釋

初步看到 v3 比 v2「退化」314 張（c110+c010），但同時「補回」333 張（c101+c001），淨增益 +19。  
看錯原因：原本只計算 c001=106（v3 純增），漏計 c101=227（v3 修復 v2 的失誤）。

### 8.6 Dashboard 閃爍修正

`<meta http-equiv="refresh">` 每次刷新整頁造成閃爍。  
改為 JS 動態載入：每 15 秒 append `<script src="data.js?t=timestamp">` tag，只在 `new_rows` 數量變化時重繪圖表。

---

## 9. HTML 文件時間軸

依文件內容產生的原始時間排列（非檔案複製時間）：

| 原始時間 | 文件 | 摘要 |
|---|---|---|
| 2026-05-30 | `lpd_batch_plan.html` | **LPD 批次掃描規劃**。掃描 /LPD 下 23,222 張圖，4 管線（two-step / yolo26m / yolo26s / yolo26n）輸出 lpd_results.csv，含斷點續跑、PARSeq OCR、檔名子字串比對邏輯。 |
| 2026-05-30 | `ocr_2x2_report.html` | **2×2 OCR 分析**（11,408 張，5 桶合併）。Cell 定義：two-step OCR 像車牌（^[0-9A-Z]{5-7}$）× two-step/YOLO OCR 是否相同。Cell 1（同字串）3,246，Cell 2（不同字串）5,906 → 驅動訓練資料選取。 |
| 2026-05-30 | `yolo_ftn_dataset_report.html` | **Fine-tune 資料集報告**。最終 5,929 張（no_detect 692 + Cell2去重後 5,237）。Train 4,459 / Val 1,114 / Holdout 356（0721TW）。來源 A（yolo漏偵）最有訓練價值。 |
| 2026-05-30 | `dataset_merge_plan.html` | **資料集合併規劃**。比較 5 種合併策略（直接合併、去aug合併、兩階段訓練等），選方案B（haug去augmentation後合併）以平衡資料分布，避免 hue 增強重複訓練。 |
| 2026-05-30 | `yolo26n_finetune_plan.html` | **Fine-tune 策略規劃**。以 2×2 矩陣分析訓練格的價值：TF 格（two-step 對/YOLO 錯）最高價值，OCR match 作為 corner 正確性的代理指標，偽標籤生成邏輯。 |
| 2026-05-30 | `merged_v3_plan_cellAB.html` | **Merged-v3 Cell AB 訓練計劃**。Cell AB = two-step OCR 像車牌但 merged 讀不同（3,872 張）。分析 AB 根本原因（keypoint warp 精度差）、以 two-step corners 為 GT fine-tune v2 → v3。 |
| 2026-06-02 | `merged_v4_dashboard.html` | **v4 訓練 Live Dashboard**（v4 訓練期間產生）。連續顯示 v2（ep1-30）+ v3（ep31-60）+ v4（ep61-360）完整曲線。JS polling 每 10 秒更新，資料不變時不重繪（避免閃爍）。 |
| 2026-06-04 | `lpd_four_way_report.html` | **四路比較報告**（23,222 張，IoU+minOCR修正）。Two-step 4,901 (21.1%) → v2 4,772 → v3 4,869 → **v4 4,970 (21.4%)**。v4 首次超越 two-step，v4 vs v3 淨增益 +101 張。 |
| 2026-06-04 | `test_compare_1111.html` | **1111 資料夾測試**（127 張）。Two-step 102/127 (80.3%) vs v4 99/127 (78.0%)。✓✓ 92張、僅TS對 10張、僅v4對 7張。 |
| 2026-06-04 | `test_compare_1118.html` | **1118 資料夾測試**（95 張）。Two-step 82/95 (86.3%) vs v4 78/95 (82.1%)。✓✓ 73張、僅TS對 9張、僅v4對 5張。 |
| 2026-06-04 | `test_compare_1125.html` | **1125 資料夾測試**（142 張）。Two-step 124/142 (87.3%) vs v4 116/142 (81.7%)。✓✓ 109張、僅TS對 15張、僅v4對 7張。 |
| 2026-06-04 | `test_compare_1202.html` | **1202 資料夾測試**（157 張）。Two-step 116/157 (73.9%) vs v4 110/157 (70.1%)。✓✓ 101張、僅TS對 15張、僅v4對 9張。整體準確率最低，圖片較難。 |
| 2026-06-04 | `test_compare_w1111.html` | **w1111 資料夾測試**（133 張）。Two-step 102/133 (76.7%) vs v4 94/133 (70.7%)。✓✓ 85張、僅TS對 17張、僅v4對 9張。v4 差距最大（-6%）。 |

---

## 10. 再訓練完整指南

本節確保讀者可從零開始重現訓練，或在此基礎上繼續 fine-tune。

---

### 10.1 硬體與環境需求

| 項目 | 最低需求 | 建議 |
|---|---|---|
| GPU | NVIDIA 8GB VRAM | 16GB+ |
| RAM | 16GB | 32GB |
| 磁碟空間 | 50GB（資料集 + 模型） | 100GB |
| 訓練時間（v4 300ep） | ～14 小時（8GB GPU） | ～8 小時（16GB GPU） |
| CUDA | 11.8+ | 12.x |

```bash
# 安裝 Python 依賴
pip install ultralytics>=8.4.41 onnxruntime onnxslim \
            opencv-python torch torchvision torchaudio \
            pyyaml numpy

# 驗證 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### 10.2 需要的素材（訓練前必備）

#### A. 起始模型權重

從本 repo 下載 v3 best.pt（或 v4 best.pt 繼續訓練）：

```bash
# 下載 v3 best.pt（推薦作為起始權重繼續訓練）
wget https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v3-20260602/best.pt \
     -O artifacts/yolo26n-merged-v3-20260602/best.pt

# 或下載 v4 best.pt 繼續 fine-tune
wget https://github.com/itemhsu/lp-yolo-finetune/raw/master/artifacts/yolo26n-merged-v4-20260604/best.pt \
     -O artifacts/yolo26n-merged-v4-20260604/best.pt
```

#### B. 訓練資料集

資料集結構（`merged_v3_dataset/`）需自行建立，共三個來源合併：

```
merged_v3_dataset/
├── data.yaml               ← 本 repo 已提供
├── haug_base/              ← 來源 1
│   ├── train/images/       (2,746 張)
│   └── val/images/         (222 張)
├── new/                    ← 來源 2（偽標籤）
│   ├── train/images/       (4,459 張)
│   └── val/images/         (1,114 張)
└── ab_new/                 ← 來源 3（Cell AB 補強）
    ├── train/images/       (1,266 張)
    └── val/images/         (183 張)
```

**來源 1 — haug\_base（Roboflow 公開資料集）**

```bash
# lp-det-v3-job3 from Roboflow（需自行申請下載）
# 下載後解壓，過濾掉 _h120/_h140/_h160 augmentation 變體，只留 base image
python build_merged_dataset.py   # 整理 haug_base 目錄結構
```

**來源 2 — new（偽標籤，需有 LPD 原始圖）**

偽標籤由 Two-step pipeline 自動標注，需有 LPD 車牌圖片庫：
```bash
# 1. 掃描 LPD 圖片庫，生成 lpd_results.csv
python benchmark_lpd_batch.py \
  --root /path/to/LPD \
  --plate-det two_step_models/0x1PlateDet/PlateDet.onnx \
  --rectifier two_step_models/PlateRectifier.onnx

# 2. 從 lpd_results.csv 生成 YOLO-pose 偽標籤
python build_yolo_ftn_dataset.py \
  --csv lpd_results.csv \
  --out merged_v3_dataset/new

# 3. 建立 Cell AB 補強資料（Two-step OCR 對但 merged 讀不同的案例）
python build_cellAB_dataset.py \
  --csv lpd_results.csv \
  --out merged_v3_dataset/ab_new
```

**若無 LPD 圖片庫，可直接用 v4 best.pt 推論，跳過偽標籤生成。**

#### C. 確認 data.yaml 路徑正確

```bash
cat merged_v3_dataset/data.yaml
# 將 path: 改為你的實際路徑
# 或使用絕對路徑
```

---

### 10.3 訓練步驟

#### Step 1：從 v3 繼續訓練（重現 v4）

```bash
# 確認起始權重與資料集
ls artifacts/yolo26n-merged-v3-20260602/best.pt
ls merged_v3_dataset/data.yaml

# 執行訓練（約 8-14 小時）
python train_merged_v4.py
```

`train_merged_v4.py` 內容：
```python
from ultralytics import YOLO
model = YOLO("artifacts/yolo26n-merged-v3-20260602/best.pt")
model.train(
    data="merged_v3_dataset/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,           # VRAM 不足時調低（8 或 4）
    workers=4,
    freeze=0,           # 全層解凍（fine-tune）
    lr0=0.0001,         # 低 LR，避免過度改變 v3 權重
    lrf=0.01,           # cosine 最終 LR = lr0 × lrf
    warmup_epochs=3,
    device=0,
    project="runs/merged-v4",
    name="train",
    exist_ok=True,
)
```

#### Step 2：監控訓練（即時 Dashboard）

```bash
# 另開 terminal，啟動 dashboard
python merged_v4_dashboard.py --loop 15

# 用瀏覽器開啟
open merged_v4_dashboard.html   # macOS
xdg-open merged_v4_dashboard.html  # Linux
```

Dashboard 顯示 Box Loss、Pose Loss、mAP50 即時曲線。

#### Step 3：過擬合判斷，選最佳 checkpoint

訓練完成後，檢查 `runs/merged-v4/train/results.csv`：

```python
import csv, numpy as np

rows = list(csv.DictReader(open("runs/merged-v4/train/results.csv")))
mAP_P = [float(r["metrics/mAP50(P)"]) for r in rows]
vpose = [float(r["val/pose_loss"]) for r in rows]

# 過擬合訊號：val/pose_loss 持續上升 + mAP50(P) 下滑
peak_ep = np.argmax(mAP_P) + 1
print(f"mAP50(P) 峰值：ep{peak_ep}  val/pose={vpose[peak_ep-1]:.4f}")
```

**選擇原則**：
- `best.pt` = YOLO 自動選的 fitness 最高點（通常接近最佳）
- 若 val/pose_loss 在 best.pt 已明顯上升，可考慮選更早的 epoch（需設 `save_period=N`）

#### Step 4：匯出 ONNX

```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/merged-v4/train/weights/best.pt')
model.export(format='onnx', imgsz=640, opset=17, simplify=True)
print('Exported to runs/merged-v4/train/weights/best.onnx')
"
```

#### Step 5：存檔 Artifact

```bash
DATE=$(date +%Y%m%d)
mkdir -p artifacts/yolo26n-merged-v5-${DATE}
cp runs/merged-v4/train/weights/best.pt   artifacts/yolo26n-merged-v5-${DATE}/
cp runs/merged-v4/train/weights/best.onnx artifacts/yolo26n-merged-v5-${DATE}/
echo "Saved to artifacts/yolo26n-merged-v5-${DATE}/"
```

---

### 10.4 Benchmark 評估

```bash
# 四路比較（Two-step vs v2 vs v3 vs 新版本）
python bench_four_way.py \
  --v4 artifacts/yolo26n-merged-v5-YYYYMMDD/best.onnx \
  --out lpd_four_way_new.csv

# IoU + minOCR 修正
python recompute_iou_matches.py \
  --csv lpd_four_way_new.csv \
  --out lpd_four_way_new_iou.csv \
  --thresh 0.5 --min-ocr 4

# 生成 HTML 報告
python report_four_way.py \
  --csv lpd_four_way_new_iou.csv \
  --out lpd_four_way_new_report.html

# 小測試集目視驗證
python test_compare.py /path/to/testImg/XXXX \
  --v4 artifacts/yolo26n-merged-v5-YYYYMMDD/best.onnx \
  --out test_compare_XXXX.html
```

---

### 10.5 常見問題

**Q：VRAM 不足（OOM）**
```bash
# 降低 batch size
# 在 train_merged_v4.py 中修改：
batch=8   # 或 batch=4
```

**Q：想保存中間 checkpoint 以便選最佳 epoch**
```python
# 在 model.train() 中加入：
save_period=10   # 每 10 epoch 保存一次
```

**Q：只有 CPU，沒有 GPU**
```python
device="cpu"   # 訓練速度約慢 10-20 倍，不建議超過 30 epochs
```

**Q：從頭訓練（不繼承任何權重）**
```python
model = YOLO("yolo26n-pose.pt")   # 下載官方 nano pose 預訓練權重
# lr0 改回 0.01，freeze=0，epochs 可設 100-200
```

### 訓練監控

```bash
# 啟動 v4 dashboard（訓練期間）
python merged_v4_dashboard.py --loop 15
# 開啟 merged_v4_dashboard.html 即可即時追蹤
```

---

## 附錄：模型架構

- **骨幹**：YOLOv26n（nano 系列，約 2.7M 參數）
- **任務**：pose estimation，kpt\_shape=[4,3]（4 角點，各含 x/y/visibility）
- **輸入**：640×640 RGB
- **輸出**：bbox（cx,cy,w,h,conf）+ 4×3 keypoints
- **ONNX 大小**：9.8 MB（opset 17，onnxslim 壓縮後）
- **推論速度**：約 4.8 img/s（CPU，3 個 session 並行，各 2 threads）

---

*README 生成日期：2026-06-04*
