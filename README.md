# Nhận dạng khuyết tật bề mặt thép cán nóng

Bài tập nhóm cuối kỳ – Học phần Nhận dạng mẫu.

**Bối cảnh thực tế:** kiểm tra vật tư thép cán nóng đầu vào (IQC) trong sản xuất. Công cụ hỗ trợ kiểm tra (camera / điện thoại
+ máy tính) phải nhận ra **loại** khuyết tật, **vị trí** của nó và **chịu được** điều kiện chụp xấu tại kho (mờ do rung tay
hoặc vật tư đang di chuyển, thiếu sáng, nhiễu, nén ảnh). Đề tài học trên hai bộ dữ liệu công khai về khuyết tật thép cán nóng
(NEU, X-SDD) rồi kiểm chứng trên ảnh vật tư thép nhập về tại một xưởng (ảnh gốc không công khai).

| Phần | Bài toán | Dữ liệu | Phương án |
|---|---|---|---|
| A | Phân loại loại khuyết tật | NEU (6 lớp), X-SDD (7 lớp) | GLCM+LBP+HOG + SVM / kNN (cơ sở) · ResNet50 cố định + LogReg · ResNet18 tinh chỉnh |
| B | Phát hiện & khoanh vùng khuyết tật | NEU-DET (có khung) | YOLOv8n, YOLOv8s |
| C | Độ bền với điều kiện chụp | NEU | SVM · ResNet18 · ResNet18 + tăng cường mô phỏng điều kiện chụp |
| Kiểm chứng | Ảnh vật tư thép nhập về | 27 ảnh xưởng (không công khai) | Ghép nhãn NEU → danh mục lỗi đầu vào, từ chối ảnh lạ, học lại có lớp “Khác” |

## Cấu trúc thư mục

```
3_MaNguon/
├── data/                      ← tải bằng src/download_data.sh (không nộp kèm)
│   ├── NEU-DET/{train,test}/{images,labels}   ← 1.800 ảnh 200×200, nhãn khung YOLO
│   └── X-SDD/datas/<lớp>/*.jpg|png             ← 1.360 ảnh, 7 lớp
├── src/
│   ├── download_data.sh       ← tải hai bộ dữ liệu công khai
│   ├── common.py              ← đường dẫn, tên lớp (tiếng Việt), chia train/test, cấu hình YOLO
│   ├── features.py            ← đặc trưng thủ công GLCM, LBP, HOG
│   ├── run_classification.py  ← Phần A
│   ├── train_detector.py      ← Phần B (huấn luyện YOLOv8)
│   ├── evaluate_detector.py   ← Phần B (đánh giá lại trên CPU, đo tốc độ)
│   ├── robustness.py          ← Phần C
│   ├── predict_real.py        ← chạy mô hình NEU trên ảnh vật tư (cả ảnh + YOLO + điểm lạ)
│   ├── crop_match.py          ← cắt ô ảnh vật tư, giữ ô giống NEU
│   ├── factory_mapping.py     ← ghép nhãn sang danh mục lỗi đầu vào + lớp “Khác” (mục 7.4 báo cáo)
│   └── make_real_figure.py    ← hình minh hoạ ô cắt (chỉ ảnh cắt cận)
├── results/                   ← bảng kết quả (.csv), hình (figures/), mô hình dùng cho demo (models/)
├── demo/app.py                ← demo Gradio
└── requirements.txt
```

## Cài đặt và chạy lại

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash src/download_data.sh                    # tải NEU-DET và X-SDD vào data/ (~45 MB)
.venv/bin/python src/run_classification.py   # Phần A (~15 phút trên Apple M4)
.venv/bin/python src/train_detector.py       # Phần B (YOLOv8n + YOLOv8s, 50 epoch, ~1,5 giờ)
.venv/bin/python src/evaluate_detector.py    # đánh giá lại trên CPU (tránh lỗi NMS chậm trên MPS)
.venv/bin/python src/robustness.py           # Phần C (~10 phút, cần chạy Phần A trước)
# Kiểm chứng trên ảnh vật tư (cần ảnh xưởng và file nhãn – không công khai):
.venv/bin/python src/crop_match.py <thư_mục_ảnh>
.venv/bin/python src/factory_mapping.py <nhãn.csv> <thư_mục_ảnh>
.venv/bin/python demo/app.py                 # mở http://127.0.0.1:7861
```

Mô hình dùng GPU Apple (MPS) nếu có. Chỉ muốn chạy demo: cài đặt, tải dữ liệu rồi chạy `demo/app.py` – ba mô hình demo cần (`resnet18_NEU.pt`, `resnet18_NEU_robust.pt`, `yolov8n_neu.pt`) đã có sẵn trong `results/models/`. Các mô hình khác được tạo lại khi chạy các script trên.


## Nguồn dữ liệu và tài liệu tham khảo

- **NEU surface defect database** – K. Song, Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects", Applied Surface Science, 2013; bản có khung: Y. He et al., "An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features", IEEE TIM, 2020. Bản dùng trong đề tài lấy từ https://github.com/Marfbin/NEU-DET-with-yolov8 (thư mục `data/NEU-DET`).
- **X-SDD** – X. Feng, X. Gao, L. Luo, "X-SDD: A New Benchmark for Hot Rolled Steel Strip Surface Defects Detection", Symmetry, 2021. https://github.com/Fighter20092392/X-SDD-A-New-benchmark
- Haralick et al. 1973 (GLCM); Ojala et al. 2002 (LBP); Dalal & Triggs 2005 (HOG); He et al. 2016 (ResNet); Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics).
- Thư viện: scikit-learn, scikit-image, PyTorch/torchvision, Ultralytics, Gradio.
