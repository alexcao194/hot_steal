"""Hình minh hoạ ô cắt từ ảnh vật tư: hàng trên khớp khá tốt, hàng dưới nhận nhầm (chỉ dùng ảnh cắt cận).

Chạy: .venv/bin/python src/make_real_figure.py <csv_nhãn> <thư_mục_ảnh>  → results/figures/anhthat_o_cat.png
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from common import FIG

L = pd.read_csv(sys.argv[1]).set_index("id")
SRC = Path(sys.argv[2])
# (id ô, lớp NEU mô hình trả lời, thực tế)
GOOD = [(19, "Vảy cán", "vảy lấm tấm"), (31, "Vảy cán", "vảy bong tróc"), (37, "Mảng loang", "vết loang đen"),
        (126, "Rỗ bề mặt", "lỗ rỗ li ti"), (46, "Vết xước", "vết xước thật")]
BAD = [(2, "Vết xước", "mép tấm"), (24, "Mảng loang", "nét bút, vệt gỉ"), (62, "Vết xước", "vệt gỉ"),
       (145, "Rỗ bề mặt", "bề mặt tốt"), (202, "Vảy cán", "lỗ khoan")]
BLUE, ORANGE, DARK, GREY = "#3E6A8A", "#B8460A", "#16202A", "#4A5563"
F = "/System/Library/Fonts/Supplemental/Arial{}.ttf"
f_h, f_t, f_b = (ImageFont.truetype(F.format(" Bold"), 34), ImageFont.truetype(F.format(""), 28), ImageFont.truetype(F.format(" Bold"), 28))
T, GAP, W = 340, 36, 1854
sheet = Image.new("RGB", (W, 1000), "white")
dr = ImageDraw.Draw(sheet)
for row, (items, title, color) in enumerate(((GOOD, "Khớp khá tốt (5 ví dụ)", BLUE), (BAD, "Nhận nhầm (5 ví dụ)", ORANGE))):
    y0 = row * 512
    dr.text((0, y0), title, fill=color, font=f_h)
    for k, (i, model, real) in enumerate(items):
        r = L.loc[i]
        im = ImageOps.exif_transpose(Image.open(SRC / r.file)).convert("RGB").crop((r.x, r.y, r.x + r.co, r.y + r.co)).resize((T, T))
        x, y = k * (T + GAP), y0 + 56
        sheet.paste(im, (x, y))
        dr.rectangle((x, y, x + T - 1, y + T - 1), outline=color, width=5)
        dr.text((x, y + T + 12), "Mô hình: ", fill=GREY, font=f_t)
        dr.text((x + dr.textlength("Mô hình: ", font=f_t), y + T + 12), model, fill=DARK, font=f_b)
        dr.text((x, y + T + 52), "Thực tế: ", fill=GREY, font=f_t)
        dr.text((x + dr.textlength("Thực tế: ", font=f_t), y + T + 52), real, fill=color, font=f_b)
sheet.save(FIG / "anhthat_o_cat.png")
print("Đã lưu", FIG / "anhthat_o_cat.png")
