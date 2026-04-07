#!/usr/bin/env python3
"""
analyze_training.py — Confusion matrix and per-class outlier report.

Runs cross-validated predictions on the training data to give an honest
picture of classifier performance and identify which clips are likely
mislabelled or atypical within their class.

Usage:
    python3 analyze_training.py [--features models/features.npz]
                                [--db /var/lib/noisemon/noise.db]
                                [--clips /var/lib/noisemon/clips]
                                [--min-samples 5]
                                [--outliers 10]
"""

import argparse, os, sqlite3, sys
import numpy as np

DB_PATH       = "/var/lib/noisemon/noise.db"
FEATURES_PATH = "/opt/noisemon/models/features.npz"
CLIPS_DIR     = "/var/lib/noisemon/clips"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",    default=FEATURES_PATH)
    ap.add_argument("--db",          default=DB_PATH)
    ap.add_argument("--clips",       default=CLIPS_DIR)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--outliers",    type=int, default=10,
                    help="How many outliers to show per class")
    args = ap.parse_args()

    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import confusion_matrix
        import joblib
    except ImportError:
        sys.exit("scikit-learn not found — run: pip install scikit-learn joblib")

    # ── Load features ──────────────────────────────────────────────────────────
    data = np.load(args.features, allow_pickle=True)
    X_all, y_all = data["X"], data["y"]
    print(f"Loaded {len(X_all)} feature vectors ({X_all.shape[1]} dims)")

    # ── Load segment metadata from DB (same query & order as extract_features) ─
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows_all = conn.execute(
        "SELECT clip_path, t_start, t_end, label FROM segments ORDER BY label"
    ).fetchall()
    conn.close()

    # ── Apply same filters as extract_features.py ─────────────────────────────
    # (skip missing clips, skip too-short segments)
    meta = []
    for r in rows_all:
        clip_file = os.path.join(args.clips, r["clip_path"])
        if not os.path.exists(clip_file):
            continue
        t_start = max(0.0, float(r["t_start"]))
        t_end   = float(r["t_end"])
        if t_end - t_start < 0.5:
            continue
        meta.append(dict(r))

    if len(meta) != len(X_all):
        print(f"WARNING: DB has {len(meta)} usable segments but features.npz has "
              f"{len(X_all)}. Re-run extract_features.py first for accurate outliers.")
        # Truncate to shorter to avoid crash; outlier filenames may be misaligned
        n = min(len(meta), len(X_all))
        meta, X_all, y_all = meta[:n], X_all[:n], y_all[:n]

    # ── Drop rare classes ──────────────────────────────────────────────────────
    classes, counts = np.unique(y_all, return_counts=True)
    keep = set(c for c, n in zip(classes, counts) if n >= args.min_samples)
    dropped = set(classes) - keep
    if dropped:
        print(f"Skipping classes with <{args.min_samples} samples: {sorted(dropped)}")
    mask = np.array([lbl in keep for lbl in y_all])
    X, y_raw, meta = X_all[mask], y_all[mask], [m for m, k in zip(meta, mask) if k]

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    class_names = le.classes_

    # ── Cross-validated predictions ───────────────────────────────────────────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, class_weight="balanced")),
    ])
    n_splits = min(5, int(min(counts[counts >= args.min_samples])))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\nRunning {n_splits}-fold cross-validation on {len(X)} samples…")
    y_pred   = cross_val_predict(pipe, X, y, cv=cv)
    y_proba  = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    print("Done.\n")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    col_w = max(len(n) for n in class_names) + 2

    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in class_names)
    print("=" * len(header))
    print("CONFUSION MATRIX  (rows = true, cols = predicted)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row_str = f"{name:>{col_w}}"
        for j in range(len(class_names)):
            val = cm[i][j]
            cell = f"{val:>{col_w}}" if i == j else (f"{val:>{col_w}}" if val == 0 else f"\033[31m{val:>{col_w}}\033[0m")
            row_str += cell
        total = cm[i].sum()
        correct = cm[i][i]
        row_str += f"   {correct}/{total} ({100*correct//total}%)"
        print(row_str)
    print("=" * len(header))

    # ── Overall accuracy ───────────────────────────────────────────────────────
    accuracy = (y_pred == y).mean()
    print(f"\nOverall CV accuracy: {accuracy*100:.1f}%\n")

    # ── Per-class outlier report ───────────────────────────────────────────────
    true_conf = y_proba[np.arange(len(y)), y]   # model's confidence in the true label
    pred_names = le.inverse_transform(y_pred)

    print("=" * 70)
    print(f"OUTLIERS PER CLASS  (lowest model confidence in true label)")
    print("=" * 70)

    for class_name in class_names:
        class_idx  = le.transform([class_name])[0]
        idx        = np.where(y == class_idx)[0]
        conf       = true_conf[idx]
        preds      = pred_names[idx]
        sort_order = np.argsort(conf)        # ascending: lowest confidence first

        n_show     = min(args.outliers, len(idx))
        n_wrong    = (y_pred[idx] != class_idx).sum()

        print(f"\n{'─'*70}")
        print(f"  {class_name.upper()}  ({len(idx)} samples, {n_wrong} misclassified by CV)")
        print(f"{'─'*70}")
        print(f"  {'Conf':>6}  {'Predicted':>15}  Clip  [segment]")
        print(f"  {'----':>6}  {'----------':>15}  ----")

        for rank, i in enumerate(sort_order[:n_show]):
            sample_idx = idx[i]
            m   = meta[sample_idx]
            ok  = "✓" if preds[i] == class_name else "✗"
            print(f"  {conf[i]*100:5.1f}%  {ok} {preds[i]:>15}  "
                  f"{m['clip_path']}  [{m['t_start']:.1f}s–{m['t_end']:.1f}s]")

    print(f"\n{'='*70}")
    print("Tip: ✗ = misclassified by cross-validation — likely mislabelled or")
    print("     atypical. Review these clips in the labelling UI and re-label")
    print("     or delete the segment, then re-run extract_features.py +")
    print("     train_classifier.py.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
