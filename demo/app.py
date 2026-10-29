"""Demo kiểm tra bề mặt thép cán nóng (Gradio).

Chạy:  .venv/bin/python demo/app.py   → mở http://127.0.0.1:7861
Tải ảnh bề mặt thép (ảnh xám, ~200x200 như NEU) → loại khuyết tật (ResNet18) + khung khoanh vùng (YOLOv8).
Có thể làm xấu ảnh theo các điều kiện chụp của Phần C để so sánh ResNet18 thường và ResNet18 bền.
"""
import os
import sys
from pathlib import Path

os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import DATA, MODELS, NEU_CLASSES, NEU_VI  # noqa: E402
from robustness import PERTURB  # noqa: E402
from run_classification import EVAL_TF  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_classifier(name):
    path = MODELS / name
    if not path.exists():
        return None
    net = resnet18(); net.fc = nn.Linear(512, len(NEU_CLASSES))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval().to(DEVICE)


CLS_ROBUST = load_classifier("resnet18_NEU_robust.pt")
CLS_BASE = load_classifier("resnet18_NEU.pt")
DET_PATH = next((MODELS / f for f in ("yolov8n_neu.pt", "yolov8s_neu.pt") if (MODELS / f).exists()), None)
DET = YOLO(str(DET_PATH)) if DET_PATH else None


def classify(net, im):
    if net is None:
        return None
    with torch.no_grad():
        prob = torch.softmax(net(EVAL_TF(im)[None].to(DEVICE)), 1)[0].cpu().numpy()
    return {NEU_VI[c]: float(p) for c, p in zip(NEU_CLASSES, prob)}


def inspect(image, conf, cond):
    if image is None:
        return None, None, None, None, ""
    im = PERTURB[cond](Image.fromarray(image).convert("L"))
    label_robust, label_base = classify(CLS_ROBUST, im), classify(CLS_BASE, im)
    boxed, lines = None, []
    if DET is not None:
        r = DET.predict(np.stack([np.asarray(im)] * 3, -1), imgsz=320, conf=conf, verbose=False)[0]
        boxed = r.plot()[..., ::-1]
        for b in r.boxes:
            c = NEU_CLASSES[int(b.cls)]
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            lines.append(f"- {NEU_VI[c]}: độ tin cậy {float(b.conf):.2f}, vị trí ({x1},{y1})–({x2},{y2})")
    summary = (f"Phát hiện {len(lines)} vùng lỗi:\n" + "\n".join(lines)) if lines else "Không phát hiện vùng lỗi nào ở ngưỡng đã chọn."
    return np.asarray(im), label_robust, label_base, boxed, summary


examples = [[str(sorted((DATA / "NEU-DET/test/images").glob(f"{c}_*.jpg"))[1]), 0.25, "Ảnh gốc"] for c in NEU_CLASSES]

with gr.Blocks(title="Kiểm tra bề mặt thép cán nóng") as demo:
    gr.Markdown("# Kiểm tra khuyết tật bề mặt thép cán nóng\n"
                "Phân loại: ResNet18 (thường và bền) · "
                f"Khoanh vùng: {DET_PATH.name if DET_PATH else 'chưa có mô hình YOLO'}")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Ảnh bề mặt thép", type="numpy")
            cond = gr.Dropdown(list(PERTURB), value="Ảnh gốc", label="Điều kiện chụp (làm xấu ảnh như Phần C)")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Ngưỡng tin cậy khung lỗi")
            btn = gr.Button("Kiểm tra", variant="primary")
        with gr.Column():
            out_deg = gr.Image(label="Ảnh đưa vào mô hình (sau khi làm xấu)")
            out_img = gr.Image(label="Vùng lỗi phát hiện (YOLOv8)")
    with gr.Row():
        out_robust = gr.Label(label="ResNet18 + tăng cường mô phỏng điều kiện chụp", num_top_classes=3)
        out_base = gr.Label(label="ResNet18 tinh chỉnh (không tăng cường điều kiện chụp)", num_top_classes=3)
    out_txt = gr.Textbox(label="Chi tiết khoanh vùng", lines=5)
    outputs = [out_deg, out_robust, out_base, out_img, out_txt]
    btn.click(inspect, [inp, conf, cond], outputs)
    cond.change(inspect, [inp, conf, cond], outputs)
    gr.Examples(examples, [inp, conf, cond], label="Ảnh mẫu từ tập test NEU (mỗi loại lỗi một ảnh)")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
