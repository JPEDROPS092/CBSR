#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_infer.py
=============
Standalone edge-inference + benchmark tool for DermaTriage (SBESC 2026).

Run it on the actual edge hardware you have (Raspberry Pi, Jetson, a Linux
mini-PC, or your laptop) to get REAL on-device numbers instead of projections.

It only needs:
    numpy, opencv-python, joblib         (always)
    scikit-image                         (ONLY if the loaded model uses texture)

Copy just three things to the device:
    edge_infer.py
    dermatriage_pipeline.py
    model_binary.joblib   (or model_6class.joblib)

--------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------
# 1) Predict on one lesion photo (image only, if the model needs no metadata)
python3 edge_infer.py --model model_binary.joblib --image lesion.jpg

# 2) Predict with clinical metadata from a JSON file
python3 edge_infer.py --model model_binary.joblib --image lesion.jpg \
                      --meta patient.json

# 3) Fill the clinical questionnaire interactively, then predict
python3 edge_infer.py --model model_binary.joblib --image lesion.jpg \
                      --questionnaire

# 4) Benchmark latency / FPS / estimated energy on THIS device
python3 edge_infer.py --model model_binary.joblib --benchmark

# 5) Self-test with no model and no image (verifies the install works)
python3 edge_infer.py --selftest
--------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform

import numpy as np
import cv2

# scikit-image is imported lazily inside the pipeline, only if needed.
from dermatriage_pipeline import (extract_image_features, predict_from_image,
                                  segment, crop_roi, color_features,
                                  texture_features, MetadataEncoder,
                                  FEATURE_SPECS)

BENCH_RESOLUTIONS = [("QVGA", 320, 240), ("VGA", 640, 480),
                     ("HD", 1280, 720), ("FHD", 1920, 1080)]

# Rough active-power priors per device (Watts). Override with --power.
# These are only used to ESTIMATE energy/inference; for a real figure use a
# USB/inline power meter and pass the measured average with --power.
POWER_PRIORS = {
    "raspberry pi 4": 3.4,
    "raspberry pi 5": 5.0,
    "raspberry pi": 3.0,
    "jetson": 5.0,
    "default": 6.0,
}


# ==========================================================================
# Device info
# ==========================================================================
def read_cpu_temp():
    """Best-effort CPU temperature in Celsius (Linux/Pi), else None."""
    paths = ["/sys/class/thermal/thermal_zone0/temp"]
    for p in paths:
        try:
            with open(p) as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            pass
    return None


def device_model():
    # Raspberry Pi / device-tree model
    for p in ["/sys/firmware/devicetree/base/model",
              "/proc/device-tree/model"]:
        try:
            with open(p, "rb") as f:
                return f.read().decode(errors="ignore").strip("\x00").strip()
        except Exception:
            pass
    # x86 fallback
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def pick_power(model_str, override):
    if override:
        return override, "user-provided"
    m = model_str.lower()
    for key, w in POWER_PRIORS.items():
        if key in m:
            return w, f"prior for '{key}'"
    return POWER_PRIORS["default"], "generic default"


def print_device_banner():
    dm = device_model()
    temp = read_cpu_temp()
    print("=" * 60)
    print("  DermaTriage — Edge Inference")
    print("=" * 60)
    print(f"  Device   : {dm}")
    print(f"  Platform : {platform.system()} {platform.machine()} "
          f"| Python {platform.python_version()}")
    print(f"  CPU cores: {os.cpu_count()}  | OpenCV {cv2.__version__}")
    if temp is not None:
        print(f"  CPU temp : {temp:.1f} C")
    print("=" * 60)
    return dm


# ==========================================================================
# Model loading
# ==========================================================================
def load_bundle(path):
    import joblib
    bundle = joblib.load(path)
    spec = bundle["feature_spec"]
    print(f"  Model    : {bundle['config']} / {bundle['model_name']} "
          f"({bundle['task']})")
    print(f"  Features : dim={bundle['expected_dim']} "
          f"(texture={spec['texture']}, metadata={spec['meta']})")
    print(f"  Classes  : {[str(c) for c in bundle['class_names']]}")
    if spec["texture"]:
        print("  Note: this model uses texture features -> scikit-image "
              "is required on this device.")
    return bundle


def human_label(bundle, raw_label):
    """Map raw class label to a friendly string."""
    if bundle["task"] == "binary":
        return "CANCER (refer)" if int(raw_label) == 1 else "Benign"
    names = {"ACK": "Actinic Keratosis", "BCC": "Basal Cell Carcinoma",
             "MEL": "Melanoma", "NEV": "Nevus",
             "SCC": "Squamous Cell Carcinoma", "SEK": "Seborrheic Keratosis"}
    return names.get(str(raw_label), str(raw_label))


# ==========================================================================
# Single prediction
# ==========================================================================
def load_meta(args, bundle):
    """Assemble a metadata dict from --meta JSON and/or --questionnaire."""
    if not bundle["feature_spec"]["meta"]:
        return None
    meta = {}
    if args.meta:
        with open(args.meta) as f:
            meta.update(json.load(f))
    if args.questionnaire:
        meta.update(run_questionnaire())
    if not meta:
        print("  [warn] model expects clinical metadata but none was given; "
              "using default values (age=60, all symptoms=0). Accuracy will "
              "be far lower than with a filled questionnaire.")
    return meta


def predict(args, bundle):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"ERROR: could not read image '{args.image}'")
    meta = load_meta(args, bundle)

    t0 = time.perf_counter()
    label, proba, classes = predict_from_image(bundle, img, meta)
    dt = (time.perf_counter() - t0) * 1000

    print("\n" + "-" * 60)
    print(f"  Prediction : {human_label(bundle, label)}")
    if proba is not None:
        order = np.argsort(proba)[::-1]
        print("  Confidence :")
        for i in order:
            print(f"      {human_label(bundle, classes[i]):<28s} "
                  f"{proba[i]*100:5.1f}%")
    print(f"  Latency    : {dt:.1f} ms")
    print("-" * 60)
    print("  Reminder: triage aid only, NOT a diagnosis. Refer positive or "
          "uncertain cases to a specialist.")


def run_questionnaire():
    """Minimal interactive clinical questionnaire -> raw metadata dict."""
    print("\n  Clinical questionnaire (press Enter to skip a field):")

    def ask_num(prompt, default=None):
        s = input(f"    {prompt}: ").strip()
        if s == "":
            return default
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return default

    def ask_yn(prompt):
        s = input(f"    {prompt} [y/N]: ").strip().lower()
        return 1 if s in ("y", "yes", "s", "sim", "1") else 0

    meta = {}
    meta["age"] = ask_num("Age (years)", 60)
    meta["diameter_1"] = ask_num("Lesion diameter 1 (mm)", 0)
    meta["diameter_2"] = ask_num("Lesion diameter 2 (mm)", 0)
    fitz = ask_num("Fitzpatrick skin type (1-6)", 0)
    if fitz is not None:
        meta["fitspatrick"] = fitz
    g = input("    Gender [M/F]: ").strip().upper()
    meta["gender"] = "MALE" if g.startswith("M") else "FEMALE"
    reg = input("    Body region (e.g. FACE, ARM, BACK): ").strip().upper()
    if reg:
        meta["region"] = reg
    for key, q in [("itch", "Does it itch?"), ("grew", "Has it grown?"),
                   ("hurt", "Is it painful?"), ("changed", "Has it changed?"),
                   ("bleed", "Does it bleed?"), ("elevation", "Is it elevated?"),
                   ("smoke", "Patient smokes?"), ("drink", "Drinks alcohol?"),
                   ("pesticide", "Pesticide exposure?"),
                   ("skin_cancer_history", "Family skin-cancer history?"),
                   ("cancer_history", "Personal cancer history?"),
                   ("biopsed", "Already biopsied?")]:
        meta[key] = ask_yn(q)
    return meta


# ==========================================================================
# Benchmark
# ==========================================================================
def make_test_image(w, h, seed=0):
    """Synthetic lesion-like image for benchmarking when no photo is given."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), (170, 150, 140), np.uint8)          # skin tone
    img = cv2.add(img, rng.randint(-12, 12, (h, w, 3)).astype(np.int16
                  ).clip(-255, 255).astype(np.uint8))
    cx, cy = w // 2, h // 2
    ax, ay = int(w * 0.18), int(h * 0.15)
    cv2.ellipse(img, (cx, cy), (ax, ay), 30, 0, 360, (60, 45, 70), -1)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


def run_pipeline_once(img, spec):
    """Exactly the work done per inference for the loaded model's config."""
    return extract_image_features(img, use_texture=spec["texture"])


def benchmark(args, bundle):
    spec = bundle["feature_spec"]
    dm = device_model()
    power_w, power_src = pick_power(dm, args.power)

    # source image (real if given, else synthetic)
    base = cv2.imread(args.bench_image) if args.bench_image else None
    if base is None and args.bench_image:
        print(f"  [warn] could not read {args.bench_image}; using synthetic.")

    print(f"\n  Benchmark: {args.runs} runs/resolution, "
          f"{args.warmup} warmup | feature config: {bundle['config']}")
    print(f"  Energy estimate assumes {power_w:.2f} W ({power_src}). "
          "Use a power meter + --power for a measured figure.\n")
    print(f"  {'Resolution':<14s} {'Mean(ms)':>9s} {'P95(ms)':>9s} "
          f"{'FPS':>6s} {'Energy(J)':>10s}")
    print("  " + "-" * 52)

    results = []
    t_start_temp = read_cpu_temp()
    for name, w, h in BENCH_RESOLUTIONS:
        img = cv2.resize(base, (w, h)) if base is not None \
            else make_test_image(w, h)
        for _ in range(args.warmup):
            run_pipeline_once(img, spec)
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            feats = run_pipeline_once(img, spec)
            # include the classifier call to reflect true end-to-end latency
            x = feats
            if spec["meta"]:
                x = np.concatenate([feats, np.zeros(bundle["n_meta"], np.float32)])
            x = bundle["scaler"].transform(x.reshape(1, -1))
            bundle["model"].predict(x)
            ts.append((time.perf_counter() - t0) * 1000)
        mean = float(np.mean(ts))
        p95 = float(np.percentile(ts, 95))
        fps = 1000.0 / mean
        energy = (mean / 1000.0) * power_w
        results.append({"resolution": f"{name} {w}x{h}", "mean_ms": round(mean, 2),
                        "p95_ms": round(p95, 2), "fps": round(fps, 1),
                        "energy_j": round(energy, 4)})
        print(f"  {name+' '+str(w)+'x'+str(h):<14s} {mean:>9.1f} {p95:>9.1f} "
              f"{fps:>6.0f} {energy:>10.4f}")
    t_end_temp = read_cpu_temp()

    if t_start_temp is not None and t_end_temp is not None:
        print(f"\n  CPU temp: {t_start_temp:.1f} C -> {t_end_temp:.1f} C")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": dm, "assumed_power_w": power_w,
                       "config": bundle["config"], "results": results}, f,
                      indent=2)
        print(f"  Saved benchmark to {args.out}")
    return results


# ==========================================================================
# Self-test (no model / no dataset needed)
# ==========================================================================
def selftest():
    print("\n  Running self-test (synthetic image, no model)...")
    img = make_test_image(640, 480)
    _, bbox = segment(img)
    roi = crop_roi(img, bbox)
    col = color_features(cv2.resize(roi, (128, 128)))
    ok_color = col.shape[0] == 70
    print(f"    segmentation bbox : {bbox}")
    print(f"    color features    : {col.shape[0]} (expect 70) "
          f"{'OK' if ok_color else 'FAIL'}")
    try:
        tex = texture_features(cv2.resize(roi, (128, 128)))
        ok_tex = tex.shape[0] == 42
        print(f"    texture features  : {tex.shape[0]} (expect 42) "
              f"{'OK' if ok_tex else 'FAIL'}")
    except ImportError:
        ok_tex = True
        print("    texture features  : scikit-image not installed "
              "(fine for Color+Meta models)")
    full = extract_image_features(img, use_texture=False)
    print(f"    color-only vector : {full.shape[0]} (expect 70)")
    print("  Self-test complete." if ok_color and ok_tex
          else "  Self-test FAILED.")
    return 0 if (ok_color and ok_tex) else 1


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="DermaTriage edge inference / benchmark")
    ap.add_argument("--model", help="Path to a .joblib model bundle")
    ap.add_argument("--image", help="Lesion photo to classify")
    ap.add_argument("--meta", help="JSON file with clinical metadata")
    ap.add_argument("--questionnaire", action="store_true",
                    help="Fill clinical metadata interactively")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run latency / FPS / energy benchmark")
    ap.add_argument("--bench-image", help="Real image to use for benchmark")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--power", type=float, default=None,
                    help="Average active power (W) for energy estimate")
    ap.add_argument("--out", help="Write benchmark JSON to this path")
    ap.add_argument("--selftest", action="store_true",
                    help="Verify the install without a model or dataset")
    args = ap.parse_args()

    if args.selftest:
        print_device_banner()
        sys.exit(selftest())

    if not args.model:
        ap.error("--model is required (or use --selftest)")

    print_device_banner()
    bundle = load_bundle(args.model)

    if args.benchmark:
        benchmark(args, bundle)
    elif args.image:
        predict(args, bundle)
    else:
        ap.error("provide --image to classify, or --benchmark, or --selftest")


if __name__ == "__main__":
    main()
