"""Phần C – Độ bền với điều kiện chụp khi kiểm tra vật tư thép (NEU).

Mô phỏng: mờ chuyển động (rung tay, vật tư đang di chuyển), mờ lấy nét, tối/sáng bất thường, tương phản thấp, nhiễu cảm biến, nén JPEG.
So sánh: SVM cơ sở, ResNet18 tinh chỉnh thường, và ResNet18 huấn luyện với tăng cường "mô phỏng điều kiện chụp" (cải tiến).

Chạy (sau run_classification.py): .venv/bin/python src/robustness.py
Kết quả: results/robustness.csv, results/figures/robustness.jpg, results/figures/robustness_examples.jpg,
         results/models/resnet18_NEU_robust.pt
"""
import io
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
import run_classification as rc
from common import FIG, MODELS, NEU_CLASSES, RES, neu_split
from features import handcrafted


def motion_blur(im, k=9):
    a = np.asarray(im, dtype=np.float32)
    ker = np.ones(k) / k
    out = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, a)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def uneven_light(im):
    """Đèn chiếu lệch: độ sáng giảm dần từ trái sang phải (×1.3 → ×0.4)."""
    a = np.asarray(im, dtype=np.float32)
    g = np.linspace(1.3, 0.4, a.shape[1])[None, :]
    return Image.fromarray(np.clip(a * g, 0, 255).astype(np.uint8))


def noise(im, s=20, seed=0):
    a = np.asarray(im, dtype=np.float32) + np.random.default_rng(seed).normal(0, s, im.size[::-1])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(im, q=15):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("L")


PERTURB = {
    "Ảnh gốc": lambda im: im,
    "Mờ chuyển động (9px)": motion_blur,
    "Mờ lấy nét (σ=1.5)": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "Tối (×0.5)": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "Sáng (×1.5)": lambda im: ImageEnhance.Brightness(im).enhance(1.5),
    "Tương phản thấp (×0.5)": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
    "Nhiễu cảm biến (σ=20)": noise,
    "Nén JPEG (q=15)": jpeg,
    # Ba điều kiện dưới KHÔNG có trong tăng cường lúc huấn luyện → kiểm tra khả năng tổng quát thật sự
    "[chưa gặp] Mờ chuyển động mạnh (15px)": lambda im: motion_blur(im, 15),
    "[chưa gặp] Chiếu sáng không đều": lambda im: uneven_light(im),
    "[chưa gặp] Giảm độ phân giải ×0.4": lambda im: im.resize((80, 80), Image.BILINEAR).resize(im.size, Image.BILINEAR),
}


class LineSim:
    """Tăng cường ngẫu nhiên mô phỏng điều kiện chụp xấu (dùng khi huấn luyện)."""
    def __call__(self, im):
        r = np.random.rand()
        if r < 0.15: im = motion_blur(im, np.random.choice([5, 7, 9, 11]))
        elif r < 0.30: im = im.filter(ImageFilter.GaussianBlur(np.random.uniform(0.5, 2)))
        if np.random.rand() < 0.3: im = noise(im, np.random.uniform(5, 25), seed=np.random.randint(1e6))
        if np.random.rand() < 0.2: im = jpeg(im, np.random.randint(10, 40))
        return im


def predict_cnn(net, imgs):
    with torch.no_grad():
        return torch.cat([torch.softmax(net(torch.stack([rc.EVAL_TF(im) for im in imgs[i:i + 64]]).to(rc.DEVICE)), 1).cpu()
                          for i in range(0, len(imgs), 64)]).numpy().argmax(1)


def main():
    train, test = neu_split()
    ytr = np.array([c for _, c in train]); yte = np.array([c for _, c in test])
    tr_imgs, te_imgs = rc.load(train), rc.load(test)

    svm = joblib.load(MODELS / "svm_NEU.joblib")
    base = resnet18(); base.fc = nn.Linear(512, 6)
    base.load_state_dict(torch.load(MODELS / "resnet18_NEU.pt", map_location="cpu")); base = base.eval().to(rc.DEVICE)

    # Cải tiến: huấn luyện lại với tăng cường mô phỏng điều kiện chụp (dùng lại mô hình đã lưu nếu có; xoá file để huấn luyện lại)
    robust_path = MODELS / "resnet18_NEU_robust.pt"
    if robust_path.exists():
        robust = resnet18(); robust.fc = nn.Linear(512, 6)
        robust.load_state_dict(torch.load(robust_path, map_location="cpu")); robust = robust.to(rc.DEVICE)
    else:
        rc.TRAIN_TF = T.Compose([LineSim(), T.ColorJitter(brightness=0.5, contrast=0.5), rc.TRAIN_TF])
        _, _, robust = rc.finetune(tr_imgs, ytr, te_imgs, 6, tag="NEU-robust")
        torch.save(robust.state_dict(), robust_path)
    robust.eval()

    rows = []
    for pname, fn in PERTURB.items():
        imgs = [fn(im) for im in te_imgs]
        preds = {
            "GLCM+LBP+HOG + SVM (cơ sở)": svm.predict(np.stack([handcrafted(im) for im in imgs])),
            "ResNet18 tinh chỉnh": predict_cnn(base, imgs),
            "ResNet18 + tăng cường mô phỏng điều kiện chụp": predict_cnn(robust, imgs),
        }
        for m, p in preds.items():
            rows.append({"điều_kiện": pname, "phương_án": m, "accuracy": float((p == yte).mean())})
        print(pname, {m: round(float((p == yte).mean()), 3) for m, p in preds.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "robustness.csv", index=False, float_format="%.3f")

    piv = df.pivot(index="điều_kiện", columns="phương_án", values="accuracy").loc[list(PERTURB)]
    ax = piv.plot.bar(figsize=(13, 5.5), width=0.8, ylim=(0, 1.05))
    ax.set_ylabel("Độ chính xác"); ax.set_xlabel(""); ax.set_title("Độ bền với điều kiện chụp (NEU, tập test)", pad=28)
    plt.xticks(rotation=20, ha="right"); plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8, frameon=False); plt.tight_layout()
    plt.savefig(FIG / "robustness.jpg", dpi=100); plt.close()

    ex = te_imgs[[c for _, c in test].index(NEU_CLASSES.index("scratches"))]
    fig, axes = plt.subplots(1, len(PERTURB), figsize=(1.9 * len(PERTURB), 2.6))
    for a, (pname, fn) in zip(axes, PERTURB.items()):
        a.imshow(fn(ex), cmap="gray", vmin=0, vmax=255); a.set_title(pname.replace("] ", "]\n"), fontsize=6.5); a.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "robustness_examples.jpg", dpi=100); plt.close()
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
