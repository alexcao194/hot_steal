"""Cắt ảnh thật thành các ô nhỏ, tìm những ô giống khuyết tật NEU (6 loại cũ).

Các bước cho mỗi ảnh:
  1. Ảnh xám, cắt ô vuông ở 3 kích cỡ (160, 224, 320 px), bước trượt = 1/2 kích cỡ.
  2. Chuẩn hoá độ sáng / tương phản từng ô về trung bình và độ lệch chuẩn của ảnh NEU
     (ảnh NEU chụp đều sáng, ảnh điện thoại thì không).
  3. Điểm lạ của ô = khoảng cách cosine trung bình tới 5 ảnh NEU huấn luyện gần nhất (đặc trưng ResNet18 bền).
     Chỉ giữ ô có điểm lạ ≤ ngưỡng (phân vị 99% trên tập test NEU) VÀ độ tin cậy phân loại ≥ 0.8.
  4. Trong mỗi ảnh, giữ các ô tốt nhất không chồng lên nhau (IoU < 0.3).

Lưu ý: NEU không có lớp "bề mặt tốt", nên một ô thép bình thường vẫn có thể bị gán một loại lỗi → cần người xem lại.

Chạy: .venv/bin/python src/crop_match.py [thư_mục_ảnh]
Kết quả: results/anhthat_crops/<lớp>/*.jpg, results/anhthat_crops/crops.csv, results/anhthat_crops/tong_hop_theo_lop.jpg,
         results/anhthat_crops/<ảnh>_vung.jpg (ảnh gốc có vẽ các ô được giữ)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, NEU_CLASSES, NEU_VI, RES, ROOT, neu_split
from run_classification import DEVICE, EVAL_TF

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "anhthat"
OUT = RES / "anhthat_crops"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
SIZES, CONF_MIN, IOU_MAX, MAX_PER_IMAGE = (160, 224, 320), 0.8, 0.3, 6
SHORT = {"crazing": "Rạn nứt", "inclusion": "Tạp chất", "patches": "Mảng loang", "pitted_surface": "Rỗ bề mặt",
         "rolled-in_scale": "Vảy cán", "scratches": "Vết xước"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    FONT_S = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
except OSError:
    FONT = FONT_S = ImageFont.load_default()

net = resnet18(); net.fc = nn.Linear(512, 6)
net.load_state_dict(torch.load(MODELS / "resnet18_NEU_robust.pt", map_location="cpu"))
net = net.eval().to(DEVICE)
body = nn.Sequential(*list(net.children())[:-1])


@torch.no_grad()
def embed_and_classify(ims, bs=128):
    F, P = [], []
    for i in range(0, len(ims), bs):
        x = torch.stack([EVAL_TF(im) for im in ims[i:i + bs]]).to(DEVICE)
        f = body(x).flatten(1)
        P.append(torch.softmax(net.fc(f), 1).cpu().numpy())
        F.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(F), np.concatenate(P)


def knn_dist(q, bank, k=5):
    return 1 - np.sort(q @ bank.T, axis=1)[:, -k:].mean(1)


# ---- thống kê NEU: ngân hàng đặc trưng, ngưỡng lạ, độ sáng chuẩn ----
train, test = neu_split()
tr_imgs = [Image.open(p).convert("L") for p, _ in train]
arr = np.stack([np.asarray(im, dtype=np.float32) for im in tr_imgs])
NEU_MEAN, NEU_STD = float(arr.mean()), float(arr.reshape(len(arr), -1).std(1).mean())
bank, _ = embed_and_classify(tr_imgs)
te_f, _ = embed_and_classify([Image.open(p).convert("L") for p, _ in test])
THR = float(np.percentile(knn_dist(te_f, bank), 99))
print(f"NEU: độ sáng TB {NEU_MEAN:.1f}, độ lệch chuẩn TB {NEU_STD:.1f}, ngưỡng lạ (p99) {THR:.3f}", flush=True)


def normalize(tile):
    a = np.asarray(tile, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6) * NEU_STD + NEU_MEAN
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


rows = []
files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for idx, f in enumerate(files, 1):
    rgb = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    w, h = rgb.size
    if w / h > 3 or h / w > 3:
        print(f"#{idx} bỏ qua (không phải ảnh bề mặt)"); continue
    g = rgb.convert("L")
    boxes, crops = [], []
    for s in SIZES:
        for y in range(0, h - s + 1, s // 2):
            for x in range(0, w - s + 1, s // 2):
                boxes.append((x, y, x + s, y + s)); crops.append(normalize(g.crop((x, y, x + s, y + s))))
    F, P = embed_and_classify(crops)
    d = knn_dist(F, bank)
    conf, cls = P.max(1), P.argmax(1)
    ok = np.where((d <= THR) & (conf >= CONF_MIN))[0]
    ok = ok[np.argsort(d[ok])]  # ô giống NEU nhất trước
    kept = []
    for i in ok:
        if all(iou(boxes[i], boxes[j]) < IOU_MAX for j in kept):
            kept.append(i)
        if len(kept) == MAX_PER_IMAGE:
            break
    vis = rgb.copy(); dr = ImageDraw.Draw(vis)
    for i in kept:
        c = NEU_CLASSES[cls[i]]
        name = f"anh{idx:02d}_{boxes[i][0]}_{boxes[i][1]}_{boxes[i][2] - boxes[i][0]}.jpg"
        (OUT / c).mkdir(exist_ok=True)
        rgb.crop(boxes[i]).save(OUT / c / name, quality=90)
        dr.rectangle(boxes[i], outline=COLORS[cls[i]], width=6)
        dr.text((boxes[i][0] + 8, boxes[i][1] + 6), f"{SHORT[c]} {conf[i]:.2f}", fill=COLORS[cls[i]], font=FONT)
        rows.append({"ảnh": f"#{idx}", "file": f.name, "x": boxes[i][0], "y": boxes[i][1], "cỡ": boxes[i][2] - boxes[i][0],
                     "lớp": NEU_VI[c], "độ_tin_cậy": round(float(conf[i]), 3), "điểm_lạ": round(float(d[i]), 3),
                     "crop": f"{c}/{name}", "người_xác_nhận": ""})
    vis.thumbnail((900, 900)); vis.save(OUT / f"anh{idx:02d}_vung.jpg", quality=85)
    print(f"#{idx}: {len(boxes)} ô → {int((d <= THR).sum())} ô giống NEU → giữ {len(kept)}"
          + (": " + ", ".join(SHORT[NEU_CLASSES[cls[i]]] for i in kept) if kept else ""), flush=True)

df = pd.DataFrame(rows, columns=["ảnh", "file", "x", "y", "cỡ", "lớp", "độ_tin_cậy", "điểm_lạ", "crop", "người_xác_nhận"])
df.to_csv(OUT / "crops.csv", index=False, encoding="utf-8-sig")

# tổng hợp: mỗi hàng một lớp, tối đa 10 ô giống NEU nhất
T, N = 150, 10
classes = [c for c in NEU_CLASSES if (OUT / c).exists()]
if classes:
    sheet = Image.new("RGB", (190 + N * (T + 6), len(classes) * (T + 30)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, c in enumerate(classes):
        sub = df[df["crop"].str.startswith(c + "/")].sort_values("điểm_lạ").head(N)
        dr.text((6, r * (T + 30) + T // 2 - 10), f"{SHORT[c]}\n({len(df[df['crop'].str.startswith(c + '/')])} ô)", fill="black", font=FONT_S)
        for k, (_, row) in enumerate(sub.iterrows()):
            im = Image.open(OUT / row["crop"]).resize((T, T))
            x, y = 190 + k * (T + 6), r * (T + 30)
            sheet.paste(im, (x, y))
            dr.text((x, y + T + 2), f"{row['ảnh']} tc {row['độ_tin_cậy']:.2f}", fill="black", font=FONT_S)
    sheet.save(OUT / "tong_hop_theo_lop.jpg", quality=88)
print("\nSố ô giữ lại theo lớp:", df["lớp"].value_counts().to_dict() if len(df) else {})
