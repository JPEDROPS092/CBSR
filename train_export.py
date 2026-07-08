# -*- coding: utf-8 -*-
"""
train_export.py
===============
Training, patient-level cross-validation, statistical tests, and MODEL EXPORT
for DermaTriage (SBESC 2026).

Runs on Colab / Kaggle / any machine with the dataset. It reuses the exact
feature pipeline from dermatriage_pipeline.py, then, at the end, fits the
recommended configuration on ALL data and writes self-contained model bundles
that edge_infer.py can load on a Raspberry Pi (no dataset, no re-training).

Outputs (in ./results_final/):
  * fig1..fig5 .png, table_*.csv        (paper artifacts)
  * model_binary.joblib                 <-- deploy this for cancer screening
  * model_6class.joblib                 <-- deploy this for 6-class diagnosis
  * models_manifest.json                (what is inside each bundle)

Usage:
  python train_export.py                       # auto-download via kagglehub
  python train_export.py --data /path/to/pad   # local dataset folder
  python train_export.py --export-config All   # export the 'All' config instead
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform
import warnings
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from scipy.stats import wilcoxon, chi2 as chi2_dist

# Local shared pipeline (must sit next to this file)
from dermatriage_pipeline import (extract_image_features, MetadataEncoder,
                                  FEATURE_SPECS, N_COLOR, N_IMAGE)

CANCER = {"BCC", "MEL", "SCC"}
SEED = 42
N_FOLDS = 5
RESULTS = Path("results_final")


# ==========================================================================
# 1. Dataset loading
# ==========================================================================
def locate_dataset(user_path=None):
    if user_path and Path(user_path).exists():
        return Path(user_path)
    # try kagglehub
    for slug in ["mahdavi1202/pad-ufes-20", "maxjen/pad-ufes-20"]:
        try:
            import kagglehub
            return Path(kagglehub.dataset_download(slug))
        except Exception:
            pass
    # common local mounts
    for p in ["/kaggle/input/pad-ufes-20", "./pad-ufes-20",
              os.path.expanduser("~/pad-ufes-20"), "/content/pad-ufes-20"]:
        if os.path.exists(p):
            return Path(p)
    raise FileNotFoundError(
        "PAD-UFES-20 not found. Pass --data /path/to/dataset.")


def load_dataframe(root):
    csv_f = next((f for f in root.rglob("*.csv")
                  if "metadata" in f.name.lower()),
                 sorted(root.rglob("*.csv"))[0])
    df = pd.read_csv(csv_f)

    img_map = {p.name: str(p) for p in root.rglob("*.png")}
    img_map.update({p.stem: str(p) for p in root.rglob("*.png")})
    df["_p"] = df["img_id"].apply(
        lambda x: img_map.get(str(x).strip(),
                              img_map.get(str(x).strip() + ".png")))
    df = df[df["_p"].notna()].reset_index(drop=True)
    df["cancer"] = df["diagnostic"].isin(CANCER).astype(int)
    print(f"  {len(df)} images | {df['patient_id'].nunique()} patients "
          f"| csv: {csv_f.name}")
    return df


# ==========================================================================
# 2. Feature extraction (parallel)  -- full 112-dim image vector
# ==========================================================================
def _process_one(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return extract_image_features(img, use_texture=True)  # 112 dims


def extract_all(df, workers=None):
    workers = workers or min(8, os.cpu_count() or 4)
    paths = df["_p"].tolist()
    out = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, p): i for i, p in enumerate(paths)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = None
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(paths)} images...")
    valid = [i for i in range(len(paths)) if out.get(i) is not None]
    X_img = np.array([out[i] for i in valid], dtype=np.float32)
    dt = time.perf_counter() - t0
    print(f"  {len(valid)}/{len(paths)} imgs | {dt:.0f}s "
          f"| {dt / max(len(valid), 1) * 1000:.0f} ms/img | {workers} workers")
    return X_img, valid


def build_config(name, X_img, X_met):
    """Assemble the feature matrix for a named configuration."""
    spec = FEATURE_SPECS[name]
    img = X_img if spec["texture"] else X_img[:, :N_COLOR]
    return np.hstack([img, X_met]) if spec["meta"] else img


# ==========================================================================
# 3. Cross-validation
# ==========================================================================
MODEL_ZOO = {
    "SVM-RBF":   lambda: SVC(kernel="rbf", C=10, gamma="scale",
                             probability=True, random_state=SEED),
    "GradBoost": lambda: GradientBoostingClassifier(n_estimators=200,
                                                    max_depth=5, random_state=SEED),
    "RF":        lambda: RandomForestClassifier(n_estimators=200,
                                               random_state=SEED, n_jobs=-1),
    "MLP":       lambda: MLPClassifier(hidden_layer_sizes=(128, 64),
                                      max_iter=500, early_stopping=True,
                                      random_state=SEED),
}


def run_cv(X, y, groups, model_fn, n_folds=N_FOLDS):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds, yt_all, yp_all = [], [], []
    for tr, te in sgkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = model_fn().fit(Xtr, y[tr])
        yp = clf.predict(Xte)
        folds.append(f1_score(y[te], yp, average="weighted"))
        yt_all.extend(y[te].tolist())
        yp_all.extend(yp.tolist())
    return np.array(folds), np.array(yt_all), np.array(yp_all)


def mcnemar_p(yt, yp_a, yp_b):
    ca, cb = (yp_a == yt), (yp_b == yt)
    n01 = int((ca & ~cb).sum())
    n10 = int((~ca & cb).sum())
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1 - chi2_dist.cdf(chi2, 1))


def bootstrap_ci(yt, yp, n=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    m = len(yt)
    scores = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        scores.append(f1_score(yt[idx], yp[idx], average="weighted"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ==========================================================================
# 4. Model export
# ==========================================================================
def export_bundle(path, task, config_name, model_name, X, y_encoded,
                  label_encoder, metadata_encoder, meta_dim):
    """Fit scaler + model on ALL data for the chosen config and save a bundle."""
    scaler = StandardScaler().fit(X)
    model = MODEL_ZOO[model_name]().fit(scaler.transform(X), y_encoded)

    spec = FEATURE_SPECS[config_name]
    bundle = {
        "format_version": 1,
        "task": task,                      # 'binary' or '6class'
        "config": config_name,             # e.g. 'Color+Meta'
        "feature_spec": spec,              # {'texture':bool,'meta':bool}
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "metadata_encoder": metadata_encoder if spec["meta"] else None,
        "n_color": N_COLOR,
        "n_image": N_IMAGE if spec["texture"] else N_COLOR,
        "n_meta": meta_dim if spec["meta"] else 0,
        "expected_dim": X.shape[1],
        "class_names": list(label_encoder.classes_),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(bundle, path, compress=3)
    kb = Path(path).stat().st_size / 1024
    print(f"  saved {path}  ({config_name}/{model_name}, "
          f"dim={X.shape[1]}, {kb:.0f} KB)")
    return {"path": str(path), "task": task, "config": config_name,
            "model": model_name, "expected_dim": int(X.shape[1]),
            "needs_texture": spec["texture"], "needs_metadata": spec["meta"],
            "classes": [str(c) for c in label_encoder.classes_]}


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to PAD-UFES-20 folder")
    ap.add_argument("--export-config", default="Color+Meta",
                    choices=list(FEATURE_SPECS.keys()),
                    help="Configuration to export for deployment "
                         "(default Color+Meta = lightest, no scikit-image on edge)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    print(f"{platform.system()} | {os.cpu_count()} cores | OpenCV {cv2.__version__}")

    print("\n[1/6] Loading dataset...")
    root = locate_dataset(args.data)
    df = load_dataframe(root)

    print("\n[2/6] Encoding clinical metadata...")
    meta_encoder = MetadataEncoder(n_regions=6).fit(df)
    X_meta_full = meta_encoder.transform(df)
    print(f"  {meta_encoder.n_features} metadata features: "
          f"{meta_encoder.feature_names_}")

    print("\n[3/6] Extracting image features (parallel)...")
    X_img, valid = extract_all(df, workers=args.workers)
    X_met = X_meta_full[valid]
    dfv = df.iloc[valid].reset_index(drop=True)
    y6 = dfv["diagnostic"].values
    yb = dfv["cancer"].values
    pids = dfv["patient_id"].values

    le6 = LabelEncoder().fit(y6)
    leb = LabelEncoder().fit(yb)
    y6e, ybe = le6.transform(y6), leb.transform(yb)

    print("\n[4/6] Cross-validation across configs x models x tasks...")
    tasks = [("6class", y6e, le6), ("binary", ybe, leb)]
    cv_data, rows = {}, []
    for cfg in FEATURE_SPECS:
        Xc = build_config(cfg, X_img, X_met)
        for tname, y, _ in tasks:
            for mname, mfn in MODEL_ZOO.items():
                folds, yt, yp = run_cv(Xc, y, pids, mfn)
                key = f"{cfg}|{tname}|{mname}"
                cv_data[key] = {"folds": folds, "yt": yt, "yp": yp}
                rows.append({
                    "features": cfg, "task": tname, "model": mname,
                    "acc": round(accuracy_score(yt, yp) * 100, 1),
                    "bal_acc": round(balanced_accuracy_score(yt, yp) * 100, 1),
                    "f1_macro": round(f1_score(yt, yp, average="macro") * 100, 1),
                    "f1w_mean": round(folds.mean() * 100, 1),
                    "f1w_std": round(folds.std() * 100, 1),
                })
    res = pd.DataFrame(rows)
    best = res.loc[res.groupby(["features", "task"])["f1w_mean"].idxmax()]
    best = best.sort_values(["task", "f1w_mean"], ascending=[True, False])
    best.to_csv(RESULTS / "table_main.csv", index=False)
    print(best.to_string(index=False))

    # bootstrap CI + significance tests
    ci_rows = []
    for _, r in best.iterrows():
        d = cv_data[f"{r.features}|{r.task}|{r.model}"]
        lo, hi = bootstrap_ci(d["yt"], d["yp"])
        ci_rows.append({"config": f"{r.features} | {r.task}",
                        "f1w": r.f1w_mean, "ci_lo": round(lo * 100, 1),
                        "ci_hi": round(hi * 100, 1)})
    pd.DataFrame(ci_rows).to_csv(RESULTS / "table_ci.csv", index=False)

    stat_rows = []
    for fa, fb, task in [("Color", "Color+Meta", "6class"),
                         ("Color+Meta", "All", "6class"),
                         ("Color", "Color+Meta", "binary"),
                         ("Color+Meta", "All", "binary")]:
        ra = best[(best.features == fa) & (best.task == task)].iloc[0]
        rb = best[(best.features == fb) & (best.task == task)].iloc[0]
        da = cv_data[f"{fa}|{task}|{ra.model}"]
        db = cv_data[f"{fb}|{task}|{rb.model}"]
        n = min(len(da["yt"]), len(db["yt"]))
        p_m = mcnemar_p(da["yt"][:n], da["yp"][:n], db["yp"][:n])
        try:
            _, p_w = wilcoxon(da["folds"], db["folds"])
        except ValueError:
            p_w = 1.0
        sig = ("***" if min(p_m, p_w) < 0.001 else "**" if min(p_m, p_w) < 0.01
               else "*" if min(p_m, p_w) < 0.05 else "ns")
        stat_rows.append({"comparison": f"{fa} -> {fb} ({task})",
                          "mcnemar_p": round(p_m, 4),
                          "wilcoxon_p": round(p_w, 4), "sig": sig})
    pd.DataFrame(stat_rows).to_csv(RESULTS / "table_stats.csv", index=False)

    print("\n[5/6] Exporting deployment models...")
    manifest = []
    cfg = args.export_config
    for task, y_enc, le in [("binary", ybe, leb), ("6class", y6e, le6)]:
        # best model for the chosen export config on this task
        sub = best[(best.features == cfg) & (best.task == task)]
        model_name = sub.iloc[0].model if len(sub) else "RF"
        Xc = build_config(cfg, X_img, X_met)
        out = export_bundle(
            RESULTS / f"model_{task}.joblib", task, cfg, model_name,
            Xc, y_enc, le, meta_encoder,
            meta_dim=meta_encoder.n_features)
        manifest.append(out)

    with open(RESULTS / "models_manifest.json", "w") as f:
        json.dump({"export_config": cfg,
                   "metadata_features": meta_encoder.feature_names_,
                   "models": manifest}, f, indent=2)

    print("\n[6/6] Done. Files in ./results_final/:")
    for p in sorted(RESULTS.glob("*")):
        print(f"  {p.name:<28s} {p.stat().st_size/1024:>7.0f} KB")

    print("\nTo deploy on an edge device, copy these two files next to "
          "edge_infer.py:\n"
          "  - dermatriage_pipeline.py\n"
          f"  - results_final/model_binary.joblib (or model_6class.joblib)")


if __name__ == "__main__":
    main()
