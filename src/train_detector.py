"""Phần B – Phát hiện & khoanh vùng khuyết tật trên NEU-DET bằng YOLOv8 (bản nano và small).

Chạy: .venv/bin/python src/train_detector.py [--epochs 100] [--models yolov8n yolov8s]
Kết quả: results/detection_results.csv, results/detection_per_class.csv, results/models/yolo_*.pt,
         results/figures/det_*.jpg, thư mục huấn luyện results/yolo_runs/
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, FIG, MODELS, NEU_CLASSES, RES, neu_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
args = ap.parse_args()

rows, per_class = [], []
for name in args.models:
    t0 = time.time()
    model = YOLO(f"{name}.pt")  # khởi tạo từ trọng số COCO
    model.train(data=str(neu_yaml()), epochs=args.epochs, imgsz=args.imgsz, batch=32, device="mps",
                workers=4, project=str(RES / "yolo_runs"), name=name, exist_ok=True, seed=0,
                val=False, plots=True, verbose=False)  # val=False: bỏ kiểm định mỗi epoch cho nhanh (MPS chậm ở bước NMS)
    train_min = (time.time() - t0) / 60
    wdir = RES / "yolo_runs" / name / "weights"
    best = wdir / "best.pt" if (wdir / "best.pt").exists() else wdir / "last.pt"
    shutil.copy(best, MODELS / f"{name}_neu.pt")
    m = YOLO(str(best))
    val = m.val(data=str(neu_yaml()), imgsz=args.imgsz, device="cpu", split="val", plots=True,
                project=str(RES / "yolo_runs"), name=f"{name}_val", exist_ok=True, verbose=False)
    speed = val.speed  # ms/ảnh
    rows.append({"model": name, "mAP50": val.box.map50, "mAP50_95": val.box.map, "precision": val.box.mp,
                 "recall": val.box.mr, "ms_per_image": sum(speed.values()), "train_minutes": train_min,
                 "params_M": sum(p.numel() for p in m.model.parameters()) / 1e6})
    for i, c in enumerate(NEU_CLASSES):
        per_class.append({"model": name, "class": c, "AP50": val.box.ap50[i], "AP50_95": val.box.ap[i]})
    for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
        src = RES / "yolo_runs" / f"{name}_val" / f
        if src.exists():
            shutil.copy(src, FIG / f"det_{name}_{f}")
    print(rows[-1], flush=True)

    # Hình minh hoạ dự đoán trên vài ảnh test (mỗi lớp 1 ảnh)
    test_imgs = [sorted((DATA / "NEU-DET/test/images").glob(f"{c}_*.jpg"))[0] for c in NEU_CLASSES]
    for r in m.predict([str(p) for p in test_imgs], imgsz=args.imgsz, device="mps", conf=0.25, verbose=False):
        r.save(filename=str(FIG / f"det_{name}_pred_{Path(r.path).stem}.jpg"))

df = pd.DataFrame(rows)
old = RES / "detection_results.csv"
if old.exists():  # giữ kết quả của mô hình khác nếu chạy từng mô hình một
    prev = pd.read_csv(old)
    df = pd.concat([prev[~prev["model"].isin(df["model"])], df])
df.to_csv(old, index=False, float_format="%.3f")
pc = pd.DataFrame(per_class)
oldp = RES / "detection_per_class.csv"
if oldp.exists():
    prev = pd.read_csv(oldp)
    pc = pd.concat([prev[~prev["model"].isin(pc["model"])], pc])
pc.to_csv(oldp, index=False, float_format="%.3f")
print(df.to_string(index=False))
