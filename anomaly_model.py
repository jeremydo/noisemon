#!/usr/bin/env python3
"""
anomaly_model.py — Train an Isolation Forest anomaly detector.

Uses YAMNet feature vectors from clips labelled as 'birds' or 'aircraft'
as the normal distribution.  Everything else is treated as anomalous.

Usage:
    python3 anomaly_model.py [--features /opt/noisemon/models/features.npz]
                             [--out     /opt/noisemon/models/anomaly.joblib]
                             [--normal  birds aircraft crows owl]
"""

import argparse, sys, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",  default="/opt/noisemon/models/features.npz")
    ap.add_argument("--out",       default="/opt/noisemon/models/anomaly.joblib")
    ap.add_argument("--normal",    nargs="+",
                    default=["birds", "aircraft", "crows", "owl"],
                    help="Classes treated as normal background")
    ap.add_argument("--contamination", type=float, default=0.05,
                    help="Expected fraction of outliers in the normal training set")
    args = ap.parse_args()

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError:
        print("scikit-learn not found.  Install with:")
        print("  /opt/noisemon/venv/bin/pip install scikit-learn joblib")
        sys.exit(1)

    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]

    normal_mask = np.isin(y, args.normal)
    X_normal    = X[normal_mask]

    print(f"Normal classes : {args.normal}")
    print(f"Normal samples : {len(X_normal)}")
    print(f"Total in file  : {len(X)}  ({len(X)-len(X_normal)} non-normal for eval)")

    if len(X_normal) < 10:
        print("ERROR: fewer than 10 normal samples — label more birds/aircraft clips first.")
        sys.exit(1)

    pipe = Pipeline([
        ("scaler",  StandardScaler()),
        ("iforest", IsolationForest(n_estimators=300,
                                    contamination=args.contamination,
                                    random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_normal)

    # Evaluate on all labelled data to check separation
    scores = pipe.decision_function(X)
    preds  = pipe.predict(X)   # 1 = normal, -1 = anomaly

    print("\nAnomaly detection on all labelled data:")
    print(f"  {'class':<20}  {'n':>4}  {'anomaly%':>8}  {'mean_score':>10}")
    for cls in sorted(set(y)):
        mask = y == cls
        rate = np.mean(preds[mask] == -1)
        msco = np.mean(scores[mask])
        flag = "← NORMAL" if cls in args.normal else ""
        print(f"  {cls:<20}  {mask.sum():>4}  {rate:>8.0%}  {msco:>+10.3f}  {flag}")

    print(f"\nScore distribution (normal training set):")
    ns = scores[normal_mask]
    print(f"  min={ns.min():+.3f}  p10={np.percentile(ns,10):+.3f}  "
          f"median={np.median(ns):+.3f}  p90={np.percentile(ns,90):+.3f}  max={ns.max():+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"pipeline": pipe, "normal_classes": args.normal,
                 "contamination": args.contamination}, args.out)
    print(f"\nModel saved → {args.out}")

if __name__ == "__main__":
    main()
