"""Thử ghép nhãn NEU sang danh mục lỗi của xưởng và thêm lớp "Khác" trên các ô cắt từ ảnh vật tư thép.

Đầu vào: danh sách ô đã gán nhãn sơ bộ (csv có cột file, x, y, co, nhan_so_bo) và thư mục ảnh gốc.
Danh mục lỗi nguyên vật liệu đầu vào: Vảy, Vết loang, Rỗ, Trầy xước, Khác (mép tấm, gỉ, nét bút, lỗ khoan, bề mặt tốt...).
Ô có cột loai_bo khác rỗng (mối hàn, bắn toé do hàn – thuộc công đoạn sản xuất, không phải vật tư đầu vào) bị bỏ qua.

So sánh bốn phương án (và mốc M0 luôn trả lời "Khác"), đánh giá chừa-một-ảnh (các ô của cùng một ảnh không vừa học vừa kiểm tra):
  M1. Ghép nhãn: lớp NEU → lớp xưởng (rạn nứt, tạp chất → Khác), không từ chối.
  M2. Ghép nhãn + từ chối: ô có điểm lạ > ngưỡng p99 của NEU hoặc độ tin cậy < 0,8 → Khác.
  M3. Học lại trên ảnh xưởng: hồi quy logistic trên đặc trưng ResNet18 bền, có lớp Khác.
  M4. Như M3 nhưng thêm ảnh NEU (đã ghép nhãn) vào tập học.

Chạy: .venv/bin/python src/factory_mapping.py <csv_nhãn> <thư_mục_ảnh>
Kết quả: results/anhthat_mapping/{ket_qua.csv, theo_lop.csv, ma_tran_*.csv}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, NEU_CLASSES, RES, neu_split
from run_classification import DEVICE, EVAL_TF

LABELS, SRC = Path(sys.argv[1]), Path(sys.argv[2])
OUT = RES / "anhthat_mapping"
OUT.mkdir(parents=True, exist_ok=True)
FACTORY = ["Vảy", "Vết loang", "Rỗ", "Trầy xước", "Khác"]
OTHER = FACTORY.index("Khác")
NEU2F = {"rolled-in_scale": "Vảy", "patches": "Vết loang", "pitted_surface": "Rỗ",
         "scratches": "Trầy xước", "crazing": "Khác", "inclusion": "Khác"}
CONF_MIN = 0.8

net = resnet18(); net.fc = nn.Linear(512, 6)
net.load_state_dict(torch.load(MODELS / "resnet18_NEU_robust.pt", map_location="cpu"))
net = net.eval().to(DEVICE)
body = nn.Sequential(*list(net.children())[:-1])


@torch.no_grad()
def embed(ims, bs=128):
    F, P = [], []
    for i in range(0, len(ims), bs):
        f = body(torch.stack([EVAL_TF(im) for im in ims[i:i + bs]]).to(DEVICE)).flatten(1)
        P.append(torch.softmax(net.fc(f), 1).cpu().numpy())
        F.append(nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(F), np.concatenate(P)


def knn_dist(q, bank, k=5):
    return 1 - np.sort(q @ bank.T, axis=1)[:, -k:].mean(1)


# ---- NEU: ngân hàng đặc trưng, ngưỡng lạ, độ sáng chuẩn (như crop_match.py) ----
train, test = neu_split()
tr_imgs = [Image.open(p).convert("L") for p, _ in train]
arr = np.stack([np.asarray(im, dtype=np.float32) for im in tr_imgs])
NEU_MEAN, NEU_STD = float(arr.mean()), float(arr.reshape(len(arr), -1).std(1).mean())
bank, _ = embed(tr_imgs)
te_f, _ = embed([Image.open(p).convert("L") for p, _ in test])
THR = float(np.percentile(knn_dist(te_f, bank), 99))
neu_y = np.array([FACTORY.index(NEU2F[NEU_CLASSES[c]]) for _, c in train])
neu_keep = neu_y != OTHER  # rạn nứt, tạp chất NEU không phải "Khác" của xưởng → không dùng để học lớp Khác


def normalize(tile):
    a = np.asarray(tile, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6) * NEU_STD + NEU_MEAN
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---- ô xưởng ----
L = pd.read_csv(LABELS)
if "loai_bo" in L:
    L = L[L.loai_bo.isna()].reset_index(drop=True)
cache = {}
tiles = []
for r in L.itertuples():
    if r.file not in cache:
        cache[r.file] = ImageOps.exif_transpose(Image.open(SRC / r.file)).convert("L")
    tiles.append(normalize(cache[r.file].crop((r.x, r.y, r.x + r.co, r.y + r.co))))
F, P = embed(tiles)
d = knn_dist(F, bank)
y = np.array([FACTORY.index(v) for v in L.nhan_so_bo])
groups = L.anh.values
print(f"{len(L)} ô từ {len(set(groups))} ảnh; nhãn: {pd.Series(L.nhan_so_bo).value_counts().to_dict()}", flush=True)

pred = {"M0. Luôn trả lời Khác (mốc so sánh)": np.full_like(y, OTHER)}
neu_pred = np.array([FACTORY.index(NEU2F[NEU_CLASSES[c]]) for c in P.argmax(1)])
pred["M1. Ghép nhãn NEU → xưởng"] = neu_pred
pred["M2. Ghép nhãn + từ chối ảnh lạ"] = np.where((d > THR) | (P.max(1) < CONF_MIN), OTHER, neu_pred)


def loio(extra_X=None, extra_y=None):
    out = np.empty_like(y)
    for g in np.unique(groups):
        te = groups == g
        Xtr, ytr = F[~te], y[~te]
        if extra_X is not None:
            Xtr, ytr = np.vstack([Xtr, extra_X]), np.concatenate([ytr, extra_y])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        out[te] = clf.predict(F[te])
    return out


pred["M3. Học lại trên ô xưởng (có lớp Khác)"] = loio()
pred["M4. Học lại: ô xưởng + ảnh NEU đã ghép nhãn"] = loio(bank[neu_keep], neu_y[neu_keep])

rows, per = [], []
defect = y != OTHER
for name, p in pred.items():
    false_alarm = float(((p != OTHER) & ~defect).sum() / (~defect).sum())
    caught = float(((p != OTHER) & defect).sum() / defect.sum())
    rows.append({"phương_án": name, "độ_chính_xác": (p == y).mean(), "macro_F1": f1_score(y, p, average="macro", labels=range(5)),
                 "tỉ_lệ_bắt_được_lỗi": caught, "tỉ_lệ_báo_động_giả": false_alarm,
                 "đúng_loại_trong_các_ô_lỗi": float((p[defect] == y[defect]).mean())})
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    for k, c in enumerate(FACTORY):
        per.append({"phương_án": name, "lớp": c, "số_ô": int(sup[k]), "precision": pr[k], "recall": rc[k], "F1": f1[k]})
    pd.DataFrame(confusion_matrix(y, p, labels=range(5)), index=FACTORY, columns=FACTORY).to_csv(
        OUT / f"ma_tran_{name[:2]}.csv", encoding="utf-8-sig")
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "ket_qua.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(per).round(3).to_csv(OUT / "theo_lop.csv", index=False, encoding="utf-8-sig")
L.assign(diem_la=d.round(3), **{k[:2]: [FACTORY[i] for i in v] for k, v in pred.items()}).to_csv(
    OUT / "du_doan_tung_o.csv", index=False, encoding="utf-8-sig")
print(f"Ngưỡng lạ p99 = {THR:.3f}")
print(res.to_string(index=False))
