#!/usr/bin/env python3
"""
train_classifier.py — Train a classifier on YAMNet feature vectors.

Loads the .npz produced by extract_features.py, trains an SVM with
cross-validation, and saves the trained model + label encoder to
/opt/noisemon/models/classifier.joblib

Usage:
    python3 train_classifier.py [--features /opt/noisemon/models/features.npz]
                                [--out     /opt/noisemon/models/classifier.joblib]
                                [--min-samples 5]
"""

import argparse, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",    default="/opt/noisemon/models/features.npz")
    ap.add_argument("--out",         default="/opt/noisemon/models/classifier.joblib")
    ap.add_argument("--min-samples", type=int, default=5,
                    help="Drop classes with fewer than this many samples")
    args = ap.parse_args()

    try:
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report
        import joblib
    except ImportError:
        print("scikit-learn not found. Install with:")
        print("  /opt/noisemon/venv/bin/pip install scikit-learn joblib")
        sys.exit(1)

    data = np.load(args.features, allow_pickle=True)
    X, y_raw = data["X"], data["y"]
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    # Explicitly ignored labels (ambiguous/multi-source clips — not useful for training)
    IGNORE_CLASSES = {"false_positive"}
    ignore_mask = np.array([lbl not in IGNORE_CLASSES for lbl in y_raw])
    if (~ignore_mask).any():
        ignored = set(y_raw[~ignore_mask])
        print(f"Ignoring classes (ambiguous): {ignored}")
    X, y_raw = X[ignore_mask], y_raw[ignore_mask]

    # Drop rare classes
    classes, counts = np.unique(y_raw, return_counts=True)
    keep_classes = set(c for c, n in zip(classes, counts) if n >= args.min_samples)
    dropped = set(classes) - keep_classes
    if dropped:
        print(f"Dropping classes with <{args.min_samples} samples: {dropped}")
    mask = np.array([lbl in keep_classes for lbl in y_raw])
    X, y_raw = X[mask], y_raw[mask]

    print("\nClass distribution:")
    for c, n in zip(*np.unique(y_raw, return_counts=True)):
        print(f"  {c:20s}  {n}")

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # Pipeline: scale → one-vs-rest binary SVMs (one per class).
    # Each class gets its own decision boundary; multiple classes can fire
    # simultaneously at inference time, which handles overlapping sources.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ovr",    OneVsRestClassifier(
                       SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, class_weight="balanced"))),
    ])

    # Cross-validation
    n_splits = min(5, min(counts[counts >= args.min_samples]))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"\nCross-val F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    else:
        print("\nNot enough samples for cross-validation — training on full dataset")

    # Train on all data
    pipe.fit(X, y)

    # Full-dataset report
    y_pred = pipe.predict(X)
    print("\nTraining set report (optimistic — use CV score for real accuracy):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # Save model + label encoder together
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"pipeline": pipe, "label_encoder": le}, args.out)
    print(f"\nModel saved → {args.out}")
    print(f"Classes: {list(le.classes_)}")

if __name__ == "__main__":
    main()
