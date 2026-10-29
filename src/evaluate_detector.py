"""Đánh giá lại các mô hình YOLO đã huấn luyện trên tập test NEU-DET (CPU – tránh lỗi NMS quá thời gian trên MPS).

Chạy: .venv/bin/python src/evaluate_detector.py
Cập nhật: results/detection_results.csv, results/detection_per_class.csv, hình dự đoán & ma trận nhầm lẫn.
"""
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, FIG, MODELS, NEU_CLASSES, RES, neu_yaml

old = pd.read_csv(RES / "detection_results.csv").set_index("model") if (RES / "detection_results.csv").exists() else None
rows, per_class = [], []
for name in ("yolov8n", "yolov8s"):
    w = MODELS / f"{name}_neu.pt"
    if not w.exists():
        continue
    m = YOLO(str(w))
    val = m.val(data=str(neu_yaml()), imgsz=320, device="cpu", batch=16, plots=True,
                project=str(RES / "yolo_runs"), name=f"{name}_val", exist_ok=True, verbose=False)
    # đo tốc độ suy luận thực tế từng ảnh (CPU và MPS)
    imgs = [str(p) for p in sorted((DATA / "NEU-DET/test/images").glob("*.jpg"))[:60]]
    speeds = {}
    for dev in ("cpu", "mps"):
        m.predict(imgs[:5], imgsz=320, device=dev, verbose=False)  # khởi động
        t = time.time(); m.predict(imgs, imgsz=320, device=dev, verbose=False); speeds[dev] = (time.time() - t) / len(imgs) * 1000
    rows.append({"model": name, "mAP50": val.box.map50, "mAP50_95": val.box.map, "precision": val.box.mp, "recall": val.box.mr,
                 "ms_per_image_cpu": speeds["cpu"], "ms_per_image_mps": speeds["mps"],
                 "train_minutes": old.loc[name, "train_minutes"] if old is not None and name in old.index else float("nan"),
                 "params_M": sum(p.numel() for p in m.model.parameters()) / 1e6})
    for i, c in enumerate(NEU_CLASSES):
        per_class.append({"model": name, "class": c, "AP50": val.box.ap50[i], "AP50_95": val.box.ap[i]})
    for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = RES / "yolo_runs" / f"{name}_val" / f
        if src.exists():
            shutil.copy(src, FIG / f"det_{name}_{f}")
    test_imgs = [sorted((DATA / "NEU-DET/test/images").glob(f"{c}_*.jpg"))[0] for c in NEU_CLASSES]
    for r in m.predict([str(p) for p in test_imgs], imgsz=320, device="cpu", conf=0.25, verbose=False):
        r.save(filename=str(FIG / f"det_{name}_pred_{Path(r.path).stem}.jpg"))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(RES / "detection_results.csv", index=False, float_format="%.3f")
pd.DataFrame(per_class).to_csv(RES / "detection_per_class.csv", index=False, float_format="%.3f")
print(pd.DataFrame(rows).round(3).to_string(index=False))
