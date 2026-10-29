"""Đường dẫn, danh sách lớp và hàm đọc hai bộ dữ liệu thép cán nóng."""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES, FIG, MODELS = ROOT / "results", ROOT / "results/figures", ROOT / "results/models"
for d in (RES, FIG, MODELS):
    d.mkdir(parents=True, exist_ok=True)

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def neu_yaml():
    """Ghi cấu hình dữ liệu YOLO cho NEU-DET theo đường dẫn tuyệt đối của máy đang chạy, trả về đường dẫn file."""
    path = DATA / "neu_det.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(NEU_CLASSES))
    path.write_text(f"path: {DATA / 'NEU-DET'}\ntrain: train/images\nval: test/images\nnames:\n{names}\n", encoding="utf-8")
    return path


NEU_VI = {
    "crazing": "Rạn nứt mạng (crazing)",
    "inclusion": "Tạp chất (inclusion)",
    "patches": "Mảng loang (patches)",
    "pitted_surface": "Rỗ bề mặt (pitted surface)",
    "rolled-in_scale": "Vảy cán ép vào (rolled-in scale)",
    "scratches": "Vết xước (scratches)",
}
XSDD_CLASSES = ["finishing roll printing", "iron sheet ash", "oxide scale of plate system",
                "oxide scale of temperature system", "red iron", "slag inclusion", "surface scratch"]
XSDD_VI = {
    "finishing roll printing": "Vết in trục cán tinh",
    "iron sheet ash": "Tro vảy sắt",
    "oxide scale of plate system": "Vảy ôxit (hệ tấm)",
    "oxide scale of temperature system": "Vảy ôxit (do nhiệt độ)",
    "red iron": "Vảy sắt đỏ",
    "slag inclusion": "Tạp chất xỉ",
    "surface scratch": "Vết xước bề mặt",
}


def neu_split():
    """Chia train/test có sẵn của NEU-DET (1620/180). Nhãn phân loại lấy theo tên file."""
    out = {}
    for split in ("train", "test"):
        files = sorted((DATA / "NEU-DET" / split / "images").glob("*.jpg"))
        out[split] = [(f, NEU_CLASSES.index(f.stem.rsplit("_", 1)[0])) for f in files]
    return out["train"], out["test"]


def xsdd_split(test_ratio=0.2, seed=0):
    """X-SDD không có chia sẵn → chia phân tầng 80/20, cố định seed."""
    rng = random.Random(seed)
    train, test = [], []
    for ci, c in enumerate(XSDD_CLASSES):
        files = sorted(f for f in (DATA / "X-SDD" / "datas" / c).iterdir() if f.suffix.lower() in (".jpg", ".png"))
        rng.shuffle(files)
        k = max(1, round(len(files) * test_ratio))
        test += [(f, ci) for f in files[:k]]
        train += [(f, ci) for f in files[k:]]
    return train, test


DATASETS = {
    "NEU": (neu_split, NEU_CLASSES, NEU_VI),
    "X-SDD": (xsdd_split, XSDD_CLASSES, XSDD_VI),
}
