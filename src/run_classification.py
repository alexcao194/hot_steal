"""Phần A – Phân loại loại khuyết tật bề mặt thép cán nóng (NEU và X-SDD).

So sánh:
  1. GLCM + LBP + HOG + SVM (phương án cơ sở – đặc trưng kết cấu cổ điển)
  2. GLCM + LBP + HOG + kNN
  3. Đặc trưng CNN ResNet50 (ImageNet, không huấn luyện lại) + hồi quy logistic
  4. ResNet18 tinh chỉnh (fine-tune) toàn mạng, có tăng cường dữ liệu

Chạy: .venv/bin/python src/run_classification.py
Kết quả: results/classification_results.csv, results/cls_report_*.txt, results/figures/cls_*.jpg,
         results/models/resnet18_<dataset>.pt, results/models/svm_<dataset>.joblib
"""
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

sys.path.insert(0, str(Path(__file__).parent))
from common import DATASETS, FIG, MODELS, RES
from features import handcrafted

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TF = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.ToTensor(), NORM])
TRAIN_TF = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
                      T.ColorJitter(brightness=0.3, contrast=0.3), T.ToTensor(), NORM])


def load(items):
    return [Image.open(p).convert("L") for p, _ in items]


def timed_predict(fn, X, n):
    t = time.time(); out = fn(X); return out, (time.time() - t) / n * 1000


# ---------- 1–2: đặc trưng thủ công ----------
def classical(train_imgs, ytr, test_imgs):
    Xtr = np.stack([handcrafted(im) for im in train_imgs])
    t = time.time(); Xte = np.stack([handcrafted(im) for im in test_imgs]); feat_ms = (time.time() - t) / len(test_imgs) * 1000
    svm = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0)),
                       {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001]},
                       cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xtr, ytr)
    p_svm, ms_svm = timed_predict(svm.predict_proba, Xte, len(Xte))
    p_knn, ms_knn = timed_predict(knn.predict_proba, Xte, len(Xte))
    return {"1. GLCM+LBP+HOG + SVM (cơ sở)": (p_svm, feat_ms + ms_svm),
            "2. GLCM+LBP+HOG + kNN": (p_knn, feat_ms + ms_knn)}, svm.best_estimator_


# ---------- 3: đặc trưng CNN cố định ----------
@torch.no_grad()
def cnn_features(imgs, net):
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([EVAL_TF(im) for im in imgs[i:i + 64]]).to(DEVICE)
        out.append(net(x).cpu().numpy())
    return np.concatenate(out)


def frozen_cnn(train_imgs, ytr, test_imgs):
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); net.fc = nn.Identity(); net = net.eval().to(DEVICE)
    Ftr = cnn_features(train_imgs, net)
    t = time.time(); Fte = cnn_features(test_imgs, net); feat_ms = (time.time() - t) / len(test_imgs) * 1000
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Ftr, ytr)
    p, ms = timed_predict(clf.predict_proba, Fte, len(Fte))
    return p, feat_ms + ms


# ---------- 4: tinh chỉnh ResNet18 ----------
class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, y, tf): self.imgs, self.y, self.tf = imgs, y, tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.tf(self.imgs[i]), self.y[i]


def finetune(train_imgs, ytr, test_imgs, n_cls, epochs=20, tag=""):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = nn.Linear(512, n_cls); net = net.to(DEVICE)
    # trọng số lớp nghịch với tần suất để bù mất cân bằng (X-SDD)
    w = torch.tensor(len(ytr) / (n_cls * np.bincount(ytr, minlength=n_cls)), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = torch.utils.data.DataLoader(DS(train_imgs, ytr, TRAIN_TF), batch_size=32, shuffle=True, num_workers=0)
    for ep in range(epochs):
        net.train(); tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(net(x), y); loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()
        print(f"  [{tag}] epoch {ep + 1}/{epochs} loss={tot / len(ytr):.3f}", flush=True)
    net.eval()
    t = time.time()
    with torch.no_grad():
        prob = torch.cat([torch.softmax(net(torch.stack([EVAL_TF(im) for im in test_imgs[i:i + 64]]).to(DEVICE)), 1).cpu()
                          for i in range(0, len(test_imgs), 64)]).numpy()
    return prob, (time.time() - t) / len(test_imgs) * 1000, net


def save_confusion(y, pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=labels, ax=ax, colorbar=False, xticks_rotation=35)
    ax.set_title(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def save_errors(test, y, pred, classes, path, n=12):
    wrong = [i for i in range(len(y)) if y[i] != pred[i]][:n]
    if not wrong:
        return
    cols = 6; rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for a, i in zip(axes.ravel(), wrong):
        a.imshow(Image.open(test[i][0]).convert("L"), cmap="gray")
        a.set_title(f"thật: {classes[y[i]]}\nđoán: {classes[pred[i]]}", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    rows = []
    for ds_name, (split_fn, classes, vi) in DATASETS.items():
        train, test = split_fn()
        ytr = np.array([c for _, c in train]); yte = np.array([c for _, c in test])
        tr_imgs, te_imgs = load(train), load(test)
        print(f"== {ds_name}: train {len(train)}, test {len(test)}", flush=True)

        methods, svm = classical(tr_imgs, ytr, te_imgs)
        joblib.dump(svm, MODELS / f"svm_{ds_name}.joblib")
        methods["3. ResNet50 (ImageNet, cố định) + LogReg"] = frozen_cnn(tr_imgs, ytr, te_imgs)
        prob, ms, net = finetune(tr_imgs, ytr, te_imgs, len(classes), tag=ds_name)
        torch.save(net.state_dict(), MODELS / f"resnet18_{ds_name}.pt")
        methods["4. ResNet18 tinh chỉnh + tăng cường dữ liệu"] = (prob, ms)

        for name, (p, ms) in methods.items():
            pred = p.argmax(1)
            rows.append({"dataset": ds_name, "method": name, "accuracy": accuracy_score(yte, pred),
                         "macro_F1": f1_score(yte, pred, average="macro"), "ms_per_image": ms})
            print(f"  {name}: acc={rows[-1]['accuracy']:.3f} F1={rows[-1]['macro_F1']:.3f} {ms:.1f}ms", flush=True)
            tag = name.split(".")[0]
            (RES / f"cls_report_{ds_name}_m{tag}.txt").write_text(
                f"{ds_name} – {name}\n\n" + classification_report(yte, pred, target_names=[vi[c] for c in classes], digits=3),
                encoding="utf-8")
            if tag in ("1", "4"):
                short = [c.replace(" of ", "\nof ").replace("_", "\n") for c in classes]
                save_confusion(yte, pred, short, f"{ds_name} – {name}", FIG / f"cls_confusion_{ds_name}_m{tag}.jpg")
                save_errors(test, yte, pred, classes, FIG / f"cls_errors_{ds_name}_m{tag}.jpg")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "classification_results.csv", index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
