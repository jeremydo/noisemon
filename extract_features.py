#!/usr/bin/env python3
"""
extract_features.py — Extract YAMNet score vectors from labelled segments.

Reads segments from the DB, slices the corresponding WAV clips, runs each
1-second window through YAMNet, and saves the results to a .npz file
ready for train_classifier.py.

Usage:
    python3 extract_features.py [--out /opt/noisemon/models/features.npz]
"""

import argparse, sqlite3, os, sys
import numpy as np
import soundfile as sf
import scipy.signal

DB_PATH    = "/var/lib/noisemon/noise.db"
CLIPS_DIR  = "/var/lib/noisemon/clips"
MODEL_PATH = "/opt/noisemon/models/yamnet.tflite"
YAMNET_SR  = 16000
WIN_SAMPLES = 15600   # 0.975 s @ 16 kHz (YAMNet native window)

def resample(audio, orig_sr):
    if orig_sr == YAMNET_SR:
        return audio.astype(np.float32)
    target = int(len(audio) * YAMNET_SR / orig_sr)
    return scipy.signal.resample(audio, target).astype(np.float32)

def load_yamnet():
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]["index"]
    out  = interp.get_output_details()[0]["index"]
    return interp, inp, out

def score_windows(interp, inp_idx, out_idx, audio_16k):
    """Run YAMNet on every non-overlapping 1s window. Returns (N, 521) array."""
    windows, start = [], 0
    while start + WIN_SAMPLES <= len(audio_16k):
        windows.append(audio_16k[start:start + WIN_SAMPLES])
        start += WIN_SAMPLES
    if not windows:
        # pad short clip to one window
        padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
        padded[:len(audio_16k)] = audio_16k[:WIN_SAMPLES]
        windows.append(padded)

    scores = []
    for w in windows:
        interp.set_tensor(inp_idx, w)
        interp.invoke()
        scores.append(interp.get_tensor(out_idx).squeeze().copy())
    return np.array(scores)   # (N, 521)

def aggregate(scores):
    """
    Aggregate per-window scores into a single feature vector.
    Concatenate mean, max, std → 521*3 = 1563 dims.
    """
    return np.concatenate([
        scores.mean(axis=0),
        scores.max(axis=0),
        scores.std(axis=0),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/noisemon/models/features.npz")
    ap.add_argument("--db",  default=DB_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT clip_path, t_start, t_end, label FROM segments ORDER BY label"
    ).fetchall()
    conn.close()

    if not rows:
        print("No labelled segments found. Label some clips first via the UI.")
        sys.exit(1)

    print(f"Found {len(rows)} segments across "
          f"{len(set(r['label'] for r in rows))} classes")

    interp, inp_idx, out_idx = load_yamnet()

    features, labels, skipped = [], [], 0
    for i, row in enumerate(rows):
        clip_file = os.path.join(CLIPS_DIR, row["clip_path"])
        if not os.path.exists(clip_file):
            print(f"  SKIP (missing): {row['clip_path']}")
            skipped += 1
            continue

        try:
            audio, sr = sf.read(clip_file, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)   # stereo → mono

            # Slice to labelled region
            t_start = max(0.0, float(row["t_start"]))
            t_end   = min(len(audio) / sr, float(row["t_end"]))
            if t_end - t_start < 0.5:
                print(f"  SKIP (too short {t_end-t_start:.2f}s): {row['clip_path']}")
                skipped += 1
                continue

            seg = audio[int(t_start * sr):int(t_end * sr)]
            seg_16k = resample(seg, sr)

            # Normalise RMS so volume doesn't dominate features
            rms = np.sqrt(np.mean(seg_16k ** 2))
            if rms > 1e-6:
                seg_16k = seg_16k * (0.1 / rms)
            seg_16k = np.clip(seg_16k, -1.0, 1.0)

            scores = score_windows(interp, inp_idx, out_idx, seg_16k)
            feat   = aggregate(scores)
            features.append(feat)
            labels.append(row["label"])

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)} done...")

        except Exception as e:
            print(f"  ERROR {row['clip_path']}: {e}")
            skipped += 1

    if not features:
        print("No features extracted.")
        sys.exit(1)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, y=y)

    print(f"\nSaved {len(X)} feature vectors → {args.out}  (skipped {skipped})")
    print("Class distribution:")
    for lbl in sorted(set(y)):
        print(f"  {lbl:20s}  {np.sum(y==lbl)}")

if __name__ == "__main__":
    main()
