#!/bin/bash
# Tải hai bộ dữ liệu công khai vào data/ (không đưa vào git vì là dữ liệu của tác giả khác).
#   - NEU-DET (định dạng YOLO): https://github.com/Marfbin/NEU-DET-with-yolov8 (thư mục data/NEU-DET)
#   - X-SDD: https://github.com/Fighter20092392/X-SDD-A-New-benchmark (thư mục X-SDD)
set -e
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

if [ ! -d NEU-DET ]; then
  curl -sL -o neu.zip https://codeload.github.com/Marfbin/NEU-DET-with-yolov8/zip/refs/heads/main
  unzip -q neu.zip "NEU-DET-with-yolov8-main/data/NEU-DET/*"
  mv NEU-DET-with-yolov8-main/data/NEU-DET ./NEU-DET
  rm -rf NEU-DET-with-yolov8-main neu.zip NEU-DET/*/labels.cache
fi

if [ ! -d X-SDD ]; then
  curl -sL -o xsdd.zip https://codeload.github.com/Fighter20092392/X-SDD-A-New-benchmark/zip/refs/heads/main
  unzip -q xsdd.zip "X-SDD-A-New-benchmark-main/X-SDD/*"
  mv X-SDD-A-New-benchmark-main/X-SDD ./X-SDD
  rm -rf X-SDD-A-New-benchmark-main xsdd.zip
fi

echo "NEU-DET: $(ls NEU-DET/train/images | wc -l) ảnh train, $(ls NEU-DET/test/images | wc -l) ảnh test"
echo "X-SDD:   $(find X-SDD/datas -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l) ảnh"
