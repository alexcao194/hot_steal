"""Thử nhận dạng ảnh thật (thư mục anhthat/) bằng các mô hình đã huấn luyện trên NEU.

Với mỗi ảnh:
  1. ResNet18 (thường và bền) phân loại cả ảnh (ảnh xám, 224×224).
  2. Cắt ảnh thành các ô vuông 320 px, phân loại từng ô → loại lỗi chiếm đa số.
  3. YOLOv8n khoanh vùng trên cả ảnh.
  4. Điểm "ngoài phân phối": khoảng cách cosine trung bình tới 5 ảnh NEU huấn luyện gần nhất (đặc trưng ResNet18 bền),
     so với ngưỡng = phân vị 95% của chính số đo này trên tập test NEU. Vượt ngưỡng → ảnh khác xa dữ liệu huấn luyện,
     kết quả phân loại không đáng tin.

Chạy: .venv/bin/python src/predict_real.py [thư_mục_ảnh]
Kết quả: results/anhthat/predictions.csv, results/anhthat/<tên>_ket_qua.jpg, results/anhthat/tong_hop.jpg
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models import resnet18
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, NEU_CLASSES, NEU_VI, RES, ROOT, neu_split
from run_classification import DEVICE, EVAL_TF

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "anhthat"
OUT = RES / "anhthat"
OUT.mkdir(parents=True, exist_ok=True)
try:  # font có dấu tiếng Việt (macOS); không có thì dùng font mặc định
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    FONT_S = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
except OSError:
    FONT = FONT_S = ImageFont.load_default()
SHORT = {"crazing": "Rạn nứt", "inclusion": "Tạp chất", "patches": "Mảng loang", "pitted_surface": "Rỗ bề mặt",
         "rolled-in_scale": "Vảy cán", "scratches": "Vết xước"}


def load(name):
    net = resnet18(); net.fc = nn.Linear(512, 6)
    net.load_state_dict(torch.load(MODELS / name, map_location="cpu"))
    return net.eval().to(DEVICE)


robust, base = load("resnet18_NEU_robust.pt"), load("resnet18_NEU.pt")
feat_net = nn.Sequential(*list(robust.children())[:-1])  # bỏ lớp cuối → vector 512 chiều
det = YOLO(str(MODELS / "yolov8n_neu.pt"))


@torch.no_grad()
def run(net, ims):
    x = torch.stack([EVAL_TF(im) for im in ims]).to(DEVICE)
    return torch.softmax(net(x), 1).cpu().numpy()


@torch.no_grad()
def feats(ims, bs=64):
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([EVAL_TF(im) for im in ims[i:i + bs]]).to(DEVICE)
        f = feat_net(x).flatten(1)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def knn_dist(q, bank, k=5):
    return 1 - np.sort(q @ bank.T, axis=1)[:, -k:].mean(1)


# ---- ngưỡng ngoài phân phối từ dữ liệu NEU ----
train, test = neu_split()
bank = feats([Image.open(p).convert("L") for p, _ in train])
thr = float(np.percentile(knn_dist(feats([Image.open(p).convert("L") for p, _ in test]), bank), 95))


def tiles(im, size=320, step=320):
    w, h = im.size
    for y in range(0, max(1, h - size + 1), step):
        for x in range(0, max(1, w - size + 1), step):
            yield (x, y), im.crop((x, y, x + size, y + size))


rows = []
files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for f in files:
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    w, h = im.size
    if w / h > 3 or h / w > 3:  # ảnh chụp màn hình / dải dài – không phải ảnh bề mặt
        rows.append({"file": f.name, "ghi_chú": "bỏ qua: không phải ảnh bề mặt (tỉ lệ khung bất thường)"})
        continue
    g = im.convert("L")
    pr, pb = run(robust, [g])[0], run(base, [g])[0]
    ood = float(knn_dist(feats([g]), bank)[0])

    tl = list(tiles(g))
    tp = run(robust, [t for _, t in tl])
    tile_cls = tp.argmax(1)
    counts = np.bincount(tile_cls, minlength=6)
    tile_ood = knn_dist(feats([t for _, t in tl]), bank)

    r = det.predict(np.asarray(im)[..., ::-1], imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    boxes = [(NEU_CLASSES[int(b.cls)], float(b.conf)) for b in r.boxes]

    # hình kết quả: ảnh + khung YOLO + ô lưới tô theo lớp dự đoán (ô "lạ" để trống)
    vis = Image.fromarray(r.plot()[..., ::-1]).convert("RGB")
    d = ImageDraw.Draw(vis)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ((x, y), _), c, o in zip(tl, tile_cls, tile_ood):
        if o <= thr:
            d.rectangle([x + 3, y + 3, x + 317, y + 317], outline=palette[c], width=6)
            d.text((x + 10, y + 10), SHORT[NEU_CLASSES[c]], fill=palette[c], font=FONT)
    vis.thumbnail((900, 900))
    vis.save(OUT / f"{f.stem[:24]}_ket_qua.jpg", quality=88)

    rows.append({
        "file": f.name,
        "resnet18_bền": NEU_VI[NEU_CLASSES[pr.argmax()]], "độ_tin_cậy_bền": round(float(pr.max()), 3),
        "resnet18_thường": NEU_VI[NEU_CLASSES[pb.argmax()]], "độ_tin_cậy_thường": round(float(pb.max()), 3),
        "hai_mô_hình_đồng_ý": bool(pr.argmax() == pb.argmax()),
        "điểm_lạ": round(ood, 3), "ngưỡng_lạ": round(thr, 3), "ngoài_phân_phối": ood > thr,
        "số_ô": len(tl), "ô_giống_NEU": int((tile_ood <= thr).sum()),
        "lớp_đa_số_trong_ô": NEU_VI[NEU_CLASSES[counts.argmax()]],
        "yolo_số_khung": len(boxes), "yolo_khung": "; ".join(f"{SHORT[c]} {p:.2f}" for c, p in boxes),
    })
    print(f.name[:28], rows[-1]["resnet18_bền"], rows[-1]["độ_tin_cậy_bền"], "lạ" if ood > thr else "giống NEU", f"YOLO {len(boxes)} khung", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

# bảng tổng hợp hình
imgs = sorted(OUT.glob("*_ket_qua.jpg"))
T = 300; cols = 6
sheet = Image.new("RGB", (cols * T, ((len(imgs) + cols - 1) // cols) * (T + 40)), "white")
for i, p in enumerate(imgs):
    im = Image.open(p); im.thumbnail((T, T)); x, y = (i % cols) * T, (i // cols) * (T + 40)
    sheet.paste(im, (x, y))
    row = df[df["file"].str.startswith(p.name.replace("_ket_qua.jpg", ""))].iloc[0]
    ImageDraw.Draw(sheet).text((x + 4, y + T + 4), f"#{i + 1} {SHORT[[k for k, v in NEU_VI.items() if v == row['resnet18_bền']][0]]} {row['độ_tin_cậy_bền']:.2f}"
                               f"{' (lạ)' if row['ngoài_phân_phối'] else ''}", fill="black", font=FONT_S)
sheet.save(OUT / "tong_hop.jpg", quality=85)
print(f"\nNgưỡng lạ (p95 trên NEU test) = {thr:.3f}")
print(df.drop(columns=["file", "ngưỡng_lạ"]).to_string())
