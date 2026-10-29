"""Đặc trưng thủ công cho ảnh bề mặt thép (ảnh xám): GLCM (Haralick), LBP, HOG, thống kê cường độ."""
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

SIZE = 128


def handcrafted(im: Image.Image):
    g = np.asarray(im.convert("L").resize((SIZE, SIZE)), dtype=np.uint8)

    # GLCM: 3 khoảng cách x 4 hướng, lượng tử 32 mức xám
    q = (g // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    glcm_f = np.concatenate([graycoprops(glcm, p).ravel() for p in
                             ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")])

    lbp_f = []
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(g, P, R, method="uniform")
        lbp_f.append(np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)[0])

    hog_f = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    gf = g.astype(np.float32) / 255
    stats = [gf.mean(), gf.std(), np.percentile(gf, 5), np.percentile(gf, 95),
             np.abs(np.diff(gf, axis=0)).mean(), np.abs(np.diff(gf, axis=1)).mean()]
    return np.concatenate([glcm_f, *lbp_f, hog_f, stats]).astype(np.float32)
