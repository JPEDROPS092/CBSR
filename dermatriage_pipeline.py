# -*- coding: utf-8 -*-
"""
dermatriage_pipeline.py
=======================
Shared feature-extraction pipeline for the DermaTriage system (SBESC 2026).

This module is imported by BOTH:
  * train_export.py  -> training / evaluation / model export
  * edge_infer.py    -> on-device inference on edge hardware

Keeping segmentation + feature extraction in a single module GUARANTEES that
features are computed identically in every environment. A mismatch between
training-time and inference-time features is the most common (and silent)
cause of accuracy collapse when a model is deployed, so it is worth enforcing.

Dependencies:
  * numpy, opencv-python            (always required)
  * scikit-image                    (ONLY imported when texture features are
                                     requested; imported lazily inside
                                     texture_features(). The recommended
                                     Color+Meta edge model does NOT use
                                     texture, so a Raspberry Pi deployment of
                                     that model does not need scikit-image.)
"""
from __future__ import annotations

import numpy as np
import cv2

# --------------------------------------------------------------------------
# Feature layout constants
# --------------------------------------------------------------------------
N_COLOR = 70        # HSV moments(12) + Lab moments(6) + HSV hist(48) + RGB ratios(4)
N_TEXTURE = 42      # LBP hist(18) + GLCM props(24)
N_IMAGE = N_COLOR + N_TEXTURE  # 112

# Which blocks each named configuration uses. Color is ALWAYS present.
FEATURE_SPECS = {
    "Color":         {"texture": False, "meta": False},
    "Color+Texture": {"texture": True,  "meta": False},
    "Color+Meta":    {"texture": False, "meta": True},
    "All":           {"texture": True,  "meta": True},
}


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def segment(img, blur=5, ero=3, dil=5, min_area_frac=0.01):
    """HSV dual-mask (Otsu on S and inverted V) segmentation.

    Returns (mask, bbox) where bbox is (x1, y1, x2, y2) with 10% padding,
    or None if no valid contour is found.
    """
    h, w = img.shape[:2]
    k = blur if blur % 2 == 1 else blur + 1
    hsv = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (k, k), 0)

    # Saturation: lesion tends to be more saturated -> normal Otsu
    _, ms = cv2.threshold(hsv[:, :, 1], 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Value: malignant lesions are usually darker -> inverted Otsu
    _, mv = cv2.threshold(hsv[:, :, 2], 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(ms, mv)

    if ero > 0:
        mask = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero)))
    if dil > 0:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)),
            iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > h * w * min_area_frac]
    if not cnts:
        return mask, None

    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    px, py = int(bw * 0.1), int(bh * 0.1)
    bbox = (max(0, x - px), max(0, y - py),
            min(w, x + bw + px), min(h, y + bh + py))
    return mask, bbox


def crop_roi(img, bbox, size=224):
    """Crop the ROI given a bbox (or the whole frame if bbox is None) and
    resize to a square of `size`x`size`."""
    if bbox is None:
        roi = img
    else:
        x1, y1, x2, y2 = bbox
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            roi = img
    return cv2.resize(roi, (size, size))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def color_features(img, bins=16):
    """70-dim color descriptor.

    HSV moments (mean/std/median/skew per channel) = 12
    Lab moments  (mean/std per channel)            = 6
    HSV quantized histogram (16 bins x 3)          = 48
    Normalized RGB ratios                          = 4
    """
    f = []

    # HSV statistical moments (12)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    for ch in range(3):
        c = hsv[:, :, ch].ravel()
        m, s = c.mean(), c.std()
        skew = float(np.mean(((c - m) / (s + 1e-7)) ** 3))
        f += [m, s, float(np.median(c)), skew]

    # CIELab moments (6)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float64)
    for ch in range(3):
        c = lab[:, :, ch].ravel()
        f += [c.mean(), c.std()]

    # HSV histogram (48)
    hsv_u8 = hsv.astype(np.uint8)
    for ch in range(3):
        mx = 180 if ch == 0 else 256
        hist = cv2.calcHist([hsv_u8], [ch], None, [bins], [0, mx]).ravel()
        f.extend((hist / (hist.sum() + 1e-7)).tolist())

    # RGB ratios (4)   (img is BGR)
    b, g, r = [img[:, :, i].mean() for i in range(3)]
    t = b + g + r + 1e-7
    f += [r / t, g / t, b / t, (r - g) / t]

    return np.asarray(f, dtype=np.float32)  # 70


def texture_features(img, lbp_p=16, lbp_r=2):
    """42-dim texture descriptor (LBP histogram + GLCM properties).

    scikit-image is imported lazily so that a Color+Meta deployment never
    pays for it.
    """
    from skimage.feature import (local_binary_pattern, graycomatrix,
                                 graycoprops)
    f = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Uniform LBP histogram (18)
    lbp = local_binary_pattern(gray, lbp_p, lbp_r, method="uniform")
    nb = lbp_p + 2
    lh, _ = np.histogram(lbp.ravel(), bins=nb, range=(0, nb), density=True)
    f.extend(lh.tolist())

    # GLCM properties, 4 angles (24)
    gq = (gray // 8).astype(np.uint8)  # 32 gray levels
    glcm = graycomatrix(gq, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]:
        f.extend(graycoprops(glcm, prop).ravel().tolist())

    return np.asarray(f, dtype=np.float32)  # 42


def extract_image_features(img, use_texture=True, roi_size=224,
                           feat_size=(128, 128)):
    """Full image branch: segment -> crop ROI -> color (+ texture).

    Returns a 70-dim vector when use_texture=False, else 112-dim.
    """
    _, bbox = segment(img)
    roi = crop_roi(img, bbox, size=roi_size)
    small = cv2.resize(roi, feat_size)
    col = color_features(small)
    if use_texture:
        tex = texture_features(small)
        return np.concatenate([col, tex])  # 112
    return col  # 70


# --------------------------------------------------------------------------
# Clinical metadata encoder
# --------------------------------------------------------------------------
def _to_float(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v):
    if isinstance(v, str):
        v = v.strip()
    if v in (True, 1, "1", "True", "true", "TRUE", "Yes", "yes", "YES", "T", "t"):
        return 1.0
    if v in (False, 0, "0", "False", "false", "FALSE", "No", "no", "NO",
             "F", "f", "", None):
        return 0.0
    try:
        return 1.0 if float(v) != 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


class MetadataEncoder:
    """Reproducible encoder for PAD-UFES-20 clinical metadata.

    fit(df)          learns which optional columns exist and the top-N
                     anatomical regions, and freezes the output column order.
    transform(df)    encodes a whole DataFrame  -> (n, d) matrix.
    transform_one(d) encodes a single patient given a plain dict -> (d,)
                     vector, in the SAME order learned at fit time. This is
                     what the edge questionnaire uses.

    Because the column order is frozen at fit time and stored inside the
    exported model bundle, the metadata vector produced on a Raspberry Pi is
    guaranteed to line up with the StandardScaler and classifier.
    """

    BOOL = ["itch", "grew", "hurt", "changed", "bleed", "elevation",
            "smoke", "drink", "pesticide", "skin_cancer_history",
            "cancer_history", "biopsed"]

    def __init__(self, n_regions=6):
        self.n_regions = n_regions
        self.regions_ = []
        self.bool_cols_ = []
        self.has_diam = False
        self.has_fitz = False
        self.has_gender = False
        self.feature_names_ = []

    def fit(self, df):
        cols = set(df.columns)
        self.has_diam = ("diameter_1" in cols) and ("diameter_2" in cols)
        self.has_fitz = "fitspatrick" in cols
        self.has_gender = "gender" in cols
        self.bool_cols_ = [c for c in self.BOOL if c in cols]
        if "region" in cols:
            self.regions_ = [str(r) for r in
                             df["region"].value_counts().head(self.n_regions).index]
        else:
            self.regions_ = []

        names = ["age"]
        if self.has_diam:
            names += ["diameter_1", "diameter_2", "area", "asym"]
        if self.has_fitz:
            names += ["fitz"]
        names += list(self.bool_cols_)
        if self.has_gender:
            names += ["male"]
        names += [f"r_{r[:8]}" for r in self.regions_]
        self.feature_names_ = names
        return self

    def _row_vector(self, row):
        def g(key, default=None):
            val = row.get(key, default)
            return val

        vec = [_to_float(g("age"), 60.0)]
        if self.has_diam:
            d1 = _to_float(g("diameter_1"), 0.0)
            d2 = _to_float(g("diameter_2"), 0.0)
            vec += [d1, d2, d1 * d2, d1 / (d2 + 1e-7)]
        if self.has_fitz:
            vec.append(_to_float(g("fitspatrick"), 0.0))
        for c in self.bool_cols_:
            vec.append(_to_bool(g(c)))
        if self.has_gender:
            vec.append(1.0 if str(g("gender", "")).upper() == "MALE" else 0.0)
        for r in self.regions_:
            vec.append(1.0 if str(g("region", "")) == r else 0.0)
        return np.asarray(vec, dtype=np.float32)

    def transform(self, df):
        records = df.to_dict("records")
        return np.vstack([self._row_vector(r) for r in records]).astype(np.float32)

    def transform_one(self, meta):
        return self._row_vector(dict(meta or {}))

    @property
    def n_features(self):
        return len(self.feature_names_)


# --------------------------------------------------------------------------
# Inference helper (used by edge_infer.py)
# --------------------------------------------------------------------------
def predict_from_image(bundle, img_bgr, meta=None):
    """Run a full prediction for one BGR image using a loaded model bundle.

    Returns (label, proba_or_None, class_names).
    """
    spec = bundle["feature_spec"]
    img_feats = extract_image_features(img_bgr, use_texture=spec["texture"])

    if spec["meta"]:
        enc = bundle["metadata_encoder"]
        meta_vec = enc.transform_one(meta or {})
        x = np.concatenate([img_feats, meta_vec])
    else:
        x = img_feats

    x = x.reshape(1, -1).astype(np.float32)
    x = bundle["scaler"].transform(x)

    model = bundle["model"]
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None

    le = bundle["label_encoder"]
    label = le.inverse_transform([pred])[0]
    classes = list(le.classes_)
    return label, proba, classes
