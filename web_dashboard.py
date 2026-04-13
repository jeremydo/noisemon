#!/usr/bin/env python3
"""NoiseMon Web Dashboard — Flask app serving charts and event history."""

from flask import Flask, render_template_string, jsonify, request, Response, \
                  session, redirect, url_for
import sqlite3, time, functools, os, threading
from datetime import datetime

DB_PATH     = "/var/lib/noisemon/noise.db"

try:
    from config import AUTH_USERS
except ImportError:
    raise SystemExit("config.py not found — copy config.example.py to config.py and set your credentials")
DB_CEILING  = 110.0    # dB — readings above this are excluded from all stats/charts
CONFIDENCE_MIN = 0.60  # minimum event confidence shown in charts and event table

app = Flask(__name__)
_key_path = "/var/lib/noisemon/flask_secret.key"
try:
    app.secret_key = open(_key_path, "rb").read()
except FileNotFoundError:
    app.secret_key = os.urandom(24)
    open(_key_path, "wb").write(app.secret_key)

def require_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("authed"):
            # API requests get 401; page requests get redirected to login
            if request.path.startswith("/api/") or request.path.startswith("/clips/"):
                return Response("Unauthorized", 401)
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

def get_conn():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

def ts_to_iso(ts):
    return datetime.fromtimestamp(ts).isoformat()

@app.route("/login", methods=["GET","POST"])
def login():
    error = ""
    if request.method == "POST":
        if AUTH_USERS.get(request.form.get("username")) == request.form.get("password"):
            session["authed"] = True
            return redirect(request.args.get("next") or "/")
        error = "Invalid credentials"
    return render_template_string(LOGIN_HTML, error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/bipyramid")
def bipyramid_page():
    from flask import send_from_directory
    return send_from_directory("/opt/noisemon/static", "bipyramid.html")

@app.route("/")
@require_auth
def index():
    return render_template_string(DASHBOARD_HTML)

_meas_cache = {}  # key: (period, agg) -> {"ts": int, "data": response}
_MEAS_TTLS  = {"1h": 15, "6h": 30, "24h": 30, "7d": 120, "30d": 300, "90d": 600, "180d": 600}

@app.route("/api/measurements")
@require_auth
def api_measurements():
    period = request.args.get("period", "24h")
    agg    = request.args.get("agg", "5m")
    cache_key = (period, agg)
    now = int(time.time())
    ttl = _MEAS_TTLS.get(period, 30)
    cached = _meas_cache.get(cache_key)
    if cached and now - cached["ts"] < ttl:
        from flask import Response
        return Response(cached["data"], mimetype="application/json")

    periods = {"1h":3600,"6h":21600,"24h":86400,"7d":604800,
               "30d":2592000,"90d":7776000,"180d":15552000}
    secs  = periods.get(period, 86400)
    since = int(time.time()) - secs

    buckets = {"1m":60,"5m":300,"15m":900,"1h":3600,"6h":21600,"1d":86400}
    bucket  = buckets.get(agg, 300)

    with get_conn() as c:
        rows = c.execute("""
            SELECT (ts/?)*? AS bucket,
                   AVG(db_avg)   AS db_avg,
                   MAX(db_peak)  AS db_peak,
                   MIN(db_min)   AS db_min,
                   AVG(freq_low)  AS fl,
                   AVG(freq_mid)  AS fm,
                   AVG(freq_high) AS fh,
                   COUNT(*) AS n
            FROM measurements
            WHERE ts >= ? AND db_avg < ? AND db_peak < ?
            GROUP BY bucket
            ORDER BY bucket
        """, (bucket, bucket, since, DB_CEILING, DB_CEILING)).fetchall()

    data = [{
        "t":       ts_to_iso(r["bucket"]),
        "db_avg":  round(r["db_avg"],1)  if r["db_avg"]  else None,
        "db_peak": round(r["db_peak"],1) if r["db_peak"] else None,
        "db_min":  round(r["db_min"],1)  if r["db_min"]  else None,
        "fl": round(r["fl"]*100,1) if r["fl"] else 0,
        "fm": round(r["fm"]*100,1) if r["fm"] else 0,
        "fh": round(r["fh"]*100,1) if r["fh"] else 0,
    } for r in rows]
    import json as _json
    payload = _json.dumps(data)
    _meas_cache[cache_key] = {"ts": now, "data": payload}
    from flask import Response
    return Response(payload, mimetype="application/json")

@app.route("/api/events")
@require_auth
def api_events():
    period = request.args.get("period", "24h")
    periods = {"1h":3600,"6h":21600,"24h":86400,"7d":604800,
               "30d":2592000,"90d":7776000,"180d":15552000}
    since = int(time.time()) - periods.get(period, 86400)

    import json as _json
    with get_conn() as c:
        suppress = ("birds", "crows", "owl")
        rows = c.execute("""
            SELECT ts_start, source, confidence, db_avg, clip_path, adsb_json
            FROM events WHERE ts_start >= ? AND confidence >= ?
              AND source NOT IN (?,?,?)
            ORDER BY ts_start DESC LIMIT 500
        """, (since, CONFIDENCE_MIN, *suppress)).fetchall()

    return jsonify([{
        "ts":     r["ts_start"],
        "t":      ts_to_iso(r["ts_start"]),
        "source": r["source"],
        "conf":   round(r["confidence"]*100),
        "db":     round(r["db_avg"],1),
        "clip":   r["clip_path"],
        "adsb":   _json.loads(r["adsb_json"]) if r["adsb_json"] else None,
    } for r in rows])

_summary_cache = {"ts": 0, "data": None}
_SUMMARY_TTL   = 60  # seconds

@app.route("/api/summary")
@require_auth
def api_summary():
    now = int(time.time())
    if now - _summary_cache["ts"] < _SUMMARY_TTL and _summary_cache["data"] is not None:
        from flask import Response
        return Response(_summary_cache["data"], mimetype="application/json")
    # Day = 07:00–22:00 local, Night = 22:00–07:00 (matches CA noise ordinance hours)
    DAY_START, DAY_END = 7, 22

    with get_conn() as c:
        last = c.execute(
            "SELECT ts, db_avg, db_peak FROM measurements ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        rows_24h = c.execute("""
            SELECT db_avg,
                   CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) as hour
            FROM measurements WHERE ts > ? AND db_avg IS NOT NULL AND db_avg < ?
        """, (now - 86400, DB_CEILING)).fetchall()
        week_max = c.execute(
            "SELECT MAX(db_peak) as m FROM measurements WHERE ts>? AND db_peak < ?",
            (now - 604800, DB_CEILING)
        ).fetchone()
        event_counts = c.execute("""
            SELECT source, COUNT(*) as cnt
            FROM events WHERE ts_start > ? AND confidence >= ?
              AND source NOT IN ('birds','crows','owl')
            GROUP BY source ORDER BY cnt DESC
        """, (now-86400, CONFIDENCE_MIN)).fetchall()

        # ── Same-weekday baseline (last 6 matching weekdays, excluding today) ──
        # Computes per-day L90/L10 for each reference day, then takes the median.
        # SQLite %w: 0=Sunday … 6=Saturday; Python weekday(): 0=Monday … 6=Sunday.
        today_dt       = datetime.now()
        today_start_ts = int(today_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        sqlite_dow     = str((today_dt.weekday() + 1) % 7)
        rows_baseline  = c.execute("""
            SELECT db_avg, ts,
                   CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) as hour
            FROM measurements
            WHERE ts >= ? AND ts < ?
              AND db_avg IS NOT NULL AND db_avg < ?
              AND strftime('%w', ts, 'unixepoch', 'localtime') = ?
        """, (today_start_ts - 49 * 86400, today_start_ts, DB_CEILING, sqlite_dow)).fetchall()

    day_vals   = sorted(r["db_avg"] for r in rows_24h
                        if DAY_START <= r["hour"] < DAY_END)
    night_vals = sorted(r["db_avg"] for r in rows_24h
                        if r["hour"] >= DAY_END or r["hour"] < DAY_START)

    def pct(vals, p):
        """Return p-th percentile of a pre-sorted list, or None if empty."""
        if not vals:
            return None
        return round(vals[min(int(len(vals) * p / 100), len(vals) - 1)], 1)

    # ── Per-day baseline L90/L10 from same-weekday reference days ──
    from collections import defaultdict
    ref_day  = defaultdict(list)   # date_str -> day-hour db_avg values
    ref_night= defaultdict(list)
    for r in rows_baseline:
        date_key = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
        if DAY_START <= r["hour"] < DAY_END:
            ref_day[date_key].append(r["db_avg"])
        else:
            ref_night[date_key].append(r["db_avg"])

    def daily_pct(readings_by_day, p):
        """Return median of per-day p-th percentiles across reference days."""
        daily = []
        for vals in readings_by_day.values():
            if len(vals) < 60:   # skip days with < 1 min of data
                continue
            s = sorted(vals)
            daily.append(s[min(int(len(s) * p / 100), len(s) - 1)])
        if not daily:
            return None
        daily.sort()
        return round(daily[len(daily) // 2], 1)

    def delta(today_val, base_val):
        if today_val is None or base_val is None:
            return None
        return round(today_val - base_val, 1)

    baseline_days_n   = len(ref_day)   # number of reference days found
    b_day_l90  = daily_pct(ref_day,   10)
    b_day_l10  = daily_pct(ref_day,   90)
    b_night_l90= daily_pct(ref_night, 10)
    b_night_l10= daily_pct(ref_night, 90)

    day_l90   = pct(day_vals,   10)
    day_l10   = pct(day_vals,   90)
    night_l90 = pct(night_vals, 10)
    night_l10 = pct(night_vals, 90)

    day_nc    = round(day_l10   - day_l90,   1) if day_l10   and day_l90   else None
    night_nc  = round(night_l10 - night_l90, 1) if night_l10 and night_l90 else None
    b_day_nc  = round(b_day_l10  - b_day_l90,  1) if b_day_l10  and b_day_l90  else None
    b_night_nc= round(b_night_l10- b_night_l90, 1) if b_night_l10 and b_night_l90 else None

    data = {
        "current_db":  round(last["db_avg"], 1)  if last and last["db_avg"]  else None,
        "peak_db":     round(last["db_peak"], 1) if last and last["db_peak"] else None,
        "last_ts":     ts_to_iso(last["ts"])     if last and last["ts"]      else None,
        "week_max":    round(week_max["m"], 1)   if week_max and week_max["m"] else None,
        # L90 = 10th percentile = background noise floor (exceeded 90% of the time)
        # L10 = 90th percentile = intrusive noise level  (exceeded 10% of the time)
        # NC  = noise climate = L10 − L90 (spread; wider = more intrusive events)
        "day_l90":          day_l90,
        "day_l10":          day_l10,
        "day_nc":           day_nc,
        "night_l90":        night_l90,
        "night_l10":        night_l10,
        "night_nc":         night_nc,
        "delta_day_l90":    delta(day_l90,   b_day_l90),
        "delta_day_l10":    delta(day_l10,   b_day_l10),
        "delta_day_nc":     delta(day_nc,    b_day_nc),
        "delta_night_l90":  delta(night_l90, b_night_l90),
        "delta_night_l10":  delta(night_l10, b_night_l10),
        "delta_night_nc":   delta(night_nc,  b_night_nc),
        "baseline_days_n":  baseline_days_n,
        "day_n":            len(day_vals),
        "night_n":          len(night_vals),
        "top_sources": [{"source": r["source"], "count": r["cnt"]} for r in event_counts],
    }
    import json as _json2
    payload = _json2.dumps(data)
    _summary_cache["data"] = payload
    _summary_cache["ts"]   = now
    from flask import Response
    return Response(payload, mimetype="application/json")

@app.route("/api/clips")
@require_auth
def api_clips():
    period = request.args.get("period", "24h")
    periods = {"1h":3600,"6h":21600,"24h":86400,"7d":604800,
               "30d":2592000}
    since = int(time.time()) - periods.get(period, 86400)
    with get_conn() as c:
        rows = c.execute("""
            SELECT ts_start, source, confidence, db_avg, clip_path
            FROM events
            WHERE ts_start >= ? AND clip_path IS NOT NULL
            ORDER BY ts_start DESC LIMIT 200
        """, (since,)).fetchall()
    return jsonify([{
        "t":      ts_to_iso(r["ts_start"]),
        "source": r["source"],
        "conf":   round(r["confidence"]*100),
        "db":     round(r["db_avg"],1),
        "clip":   r["clip_path"],
    } for r in rows])

@app.route("/clips/<filename>")
@require_auth
def serve_clip(filename):
    from flask import send_from_directory
    resp = send_from_directory("/var/lib/noisemon/clips", filename,
                               conditional=True)
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

@app.route("/label")
@require_auth
def label_page():
    resp = Response(render_template_string(LABEL_HTML), mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/api/clips/unlabelled")
@require_auth
def api_unlabelled_clips():
    """Return clips that have no segments labelled yet."""
    with get_conn() as c:
        rows = c.execute("""
            SELECT e.ts_start, e.source, e.confidence, e.db_avg, e.clip_path
            FROM events e
            WHERE e.clip_path IS NOT NULL
            AND e.clip_path NOT IN (
                SELECT DISTINCT clip_path FROM segments
            )
            ORDER BY e.ts_start DESC
            LIMIT 100
        """).fetchall()
    return jsonify([{
        "t":      ts_to_iso(r["ts_start"]),
        "source": r["source"],
        "conf":   round(r["confidence"]*100),
        "db":     round(r["db_avg"],1),
        "clip":   r["clip_path"],
    } for r in rows])

@app.route("/api/clips/labelled")
@require_auth
def api_labelled_clips():
    """Return clips that have been labelled with their segment counts."""
    with get_conn() as c:
        rows = c.execute("""
            SELECT s.clip_path,
                   COUNT(*) as segment_count,
                   GROUP_CONCAT(DISTINCT s.label) as labels,
                   e.source, e.confidence
            FROM segments s
            LEFT JOIN events e ON e.clip_path = s.clip_path
            GROUP BY s.clip_path
            ORDER BY MAX(s.ts_created) DESC
            LIMIT 100
        """).fetchall()
    return jsonify([{
        "clip":          r["clip_path"],
        "segment_count": r["segment_count"],
        "labels":        r["labels"],
        "source":        r["source"] or "",
        "conf":          round((r["confidence"] or 0) * 100),
    } for r in rows])

@app.route("/api/segments", methods=["POST"])
@require_auth
def api_save_segments():
    """Save labelled segments for a clip."""
    data = request.json
    clip_path = data.get("clip_path")
    segments  = data.get("segments", [])

    if not clip_path or not segments:
        return jsonify({"error": "missing clip_path or segments"}), 400

    with get_conn() as c:
        # Delete existing segments for this clip (allow re-labelling)
        c.execute("DELETE FROM segments WHERE clip_path=?", (clip_path,))
        for seg in segments:
            c.execute(
                "INSERT INTO segments (clip_path,t_start,t_end,label) "
                "VALUES (?,?,?,?)",
                (clip_path, seg["start"], seg["end"], seg["label"])
            )

    return jsonify({"saved": len(segments)})

@app.route("/api/correct_event", methods=["POST"])
@require_auth
def api_correct_event():
    """Quick-correct the source label of a detected event from the main dashboard.
    Replaces any existing segments for the clip with the single corrected label,
    and updates the displayed source on the event row immediately."""
    data      = request.json
    ts_start  = data.get("ts_start")
    clip_path = data.get("clip_path")
    label     = data.get("label")
    if not all([ts_start is not None, clip_path, label]):
        return jsonify({"error": "missing fields"}), 400
    with get_conn() as c:
        c.execute("DELETE FROM segments WHERE clip_path=?", (clip_path,))
        c.execute(
            "INSERT INTO segments (clip_path,t_start,t_end,label) VALUES (?,?,?,?)",
            (clip_path, 0.0, 120.0, label)
        )
        c.execute(
            "UPDATE events SET source=? WHERE ts_start=? AND clip_path=?",
            (label, ts_start, clip_path)
        )
    return jsonify({"ok": True})

@app.route("/api/segments/<path:clip_path>")
@require_auth
def api_get_segments(clip_path):
    """Get segments for a clip (own) plus any from temporally overlapping clips."""
    PRE, POST = 30, 90   # must match noise_monitor constants
    result = []
    with get_conn() as c:
        # Own segments
        own = c.execute(
            "SELECT t_start, t_end, label FROM segments WHERE clip_path=?",
            (clip_path,)
        ).fetchall()
        result.extend({
            "start": r["t_start"], "end": r["t_end"],
            "label": r["label"],   "own": True,
        } for r in own)

        # Find overlapping clips via event timestamps
        evt = c.execute(
            "SELECT ts_start FROM events WHERE clip_path=? LIMIT 1", (clip_path,)
        ).fetchone()
        if evt:
            ts_a = evt["ts_start"]
            others = c.execute("""
                SELECT clip_path, ts_start FROM events
                WHERE clip_path IS NOT NULL AND clip_path != ?
                  AND (ts_start - ?) < (? + ?)
                  AND (ts_start + ?) > (? - ?)
            """, (clip_path, PRE, ts_a, POST, POST, ts_a, PRE)).fetchall()
            for o in others:
                ts_b = o["ts_start"]
                offset = ts_b - ts_a   # positive = B starts later than A
                segs = c.execute(
                    "SELECT t_start, t_end, label FROM segments WHERE clip_path=?",
                    (o["clip_path"],)
                ).fetchall()
                for s in segs:
                    # Convert from clip B's time frame to clip A's:
                    # abs_time = (ts_b - PRE) + t_start_b
                    # in clip A's frame: abs_time - (ts_a - PRE) = t_start_b + (ts_b - ts_a) = t_start_b + offset
                    t0 = s["t_start"] + offset
                    t1 = s["t_end"]   + offset
                    if t1 > 0 and t0 < (PRE + POST):
                        result.append({
                            "start":     round(max(0.0, t0), 3),
                            "end":       round(min(float(PRE + POST), t1), 3),
                            "label":     s["label"],
                            "own":       False,
                            "from_clip": o["clip_path"],
                        })
    return jsonify(result)

# ── Review endpoints ───────────────────────────────────────────────────────────

@app.route("/review")
@require_auth
def review_page():
    resp = Response(render_template_string(REVIEW_HTML), mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/api/review/classes")
@require_auth
def api_review_classes():
    with get_conn() as c:
        rows = c.execute(
            "SELECT label, COUNT(*) as n FROM segments GROUP BY label ORDER BY label"
        ).fetchall()
    return jsonify([{"label": r["label"], "n": r["n"]} for r in rows])

@app.route("/api/review/segments/<label>")
@require_auth
def api_review_segments_by_label(label):
    with get_conn() as c:
        rows = c.execute(
            "SELECT rowid as id, clip_path, t_start, t_end FROM segments "
            "WHERE label=? ORDER BY clip_path, t_start",
            (label,)
        ).fetchall()
    return jsonify([{
        "id": r["id"], "clip": r["clip_path"],
        "start": r["t_start"], "end": r["t_end"],
    } for r in rows])

@app.route("/api/review/segment/<int:seg_id>", methods=["DELETE"])
@require_auth
def api_delete_review_segment(seg_id):
    with get_conn() as c:
        c.execute("DELETE FROM segments WHERE rowid=?", (seg_id,))
    return jsonify({"ok": True})

# ── Outlier analysis ───────────────────────────────────────────────────────────

_OUTLIER_FEATURES = "/opt/noisemon/models/features.npz"
_OUTLIER_CLIPS    = "/var/lib/noisemon/clips"
_OUTLIER_MIN_SAMPLES = 5

_outliers_cache = {"status": "idle", "computed_at": None, "items": [], "accuracy": None}
_outliers_lock  = threading.Lock()

def _run_outlier_analysis():
    import numpy as np
    try:
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
        except ImportError:
            raise RuntimeError("scikit-learn not installed on this system")

        data  = np.load(_OUTLIER_FEATURES, allow_pickle=True)
        X_all = data["X"]
        y_all = data["y"]

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows_all = conn.execute(
            "SELECT rowid, clip_path, t_start, t_end, label FROM segments ORDER BY label"
        ).fetchall()
        conn.close()

        meta = []
        for r in rows_all:
            clip_file = os.path.join(_OUTLIER_CLIPS, r["clip_path"])
            if not os.path.exists(clip_file):
                continue
            t_start = max(0.0, float(r["t_start"]))
            t_end   = float(r["t_end"])
            if t_end - t_start < 0.5:
                continue
            meta.append(dict(r))

        n = min(len(meta), len(X_all))
        meta, X_all, y_all = meta[:n], X_all[:n], y_all[:n]

        classes, counts = np.unique(y_all, return_counts=True)
        keep  = set(c for c, cnt in zip(classes, counts) if cnt >= _OUTLIER_MIN_SAMPLES)
        mask  = np.array([lbl in keep for lbl in y_all])
        X     = X_all[mask]
        y_raw = y_all[mask]
        meta_f = [m for m, k in zip(meta, mask) if k]

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        _, kept_counts = np.unique(y_raw, return_counts=True)
        n_splits = min(5, int(min(kept_counts)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, class_weight="balanced")),
        ])
        y_pred  = cross_val_predict(pipe, X, y, cv=cv)
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")

        accuracy   = float((y_pred == y).mean())
        true_conf  = y_proba[np.arange(len(y)), y]
        pred_names = le.inverse_transform(y_pred)

        items = []
        for i, m in enumerate(meta_f):
            items.append({
                "seg_id":     int(m.get("rowid") or 0),
                "clip":       m["clip_path"],
                "t_start":    float(m["t_start"]),
                "t_end":      float(m["t_end"]),
                "true_label": m["label"],
                "predicted":  str(pred_names[i]),
                "confidence": float(true_conf[i]),
            })
        items.sort(key=lambda x: x["confidence"])

        with _outliers_lock:
            _outliers_cache.update({
                "status":      "ready",
                "computed_at": int(time.time()),
                "items":       items,
                "accuracy":    round(accuracy * 100, 1),
                "error":       None,
            })
    except Exception as e:
        with _outliers_lock:
            _outliers_cache.update({
                "status":      "error",
                "error":       str(e),
                "computed_at": None,
                "items":       [],
                "accuracy":    None,
            })


@app.route("/api/outliers/compute", methods=["POST"])
@require_auth
def api_outliers_compute():
    with _outliers_lock:
        if _outliers_cache["status"] == "computing":
            return jsonify({"status": "computing"})
        _outliers_cache["status"] = "computing"
    t = threading.Thread(target=_run_outlier_analysis, daemon=True)
    t.start()
    return jsonify({"status": "computing"})


@app.route("/api/outliers")
@require_auth
def api_outliers():
    with _outliers_lock:
        return jsonify(dict(_outliers_cache))


# ── Retrain ────────────────────────────────────────────────────────────────────

_retrain_cache = {"status": "idle", "step": "", "error": None, "completed_at": None}
_retrain_lock  = threading.Lock()

def _run_retrain():
    import subprocess
    venv_python = "/opt/noisemon/venv/bin/python3"
    steps = [
        ("Extracting features…", "/opt/noisemon/extract_features.py"),
        ("Training classifier…", "/opt/noisemon/train_classifier.py"),
    ]
    try:
        for step_name, script in steps:
            with _retrain_lock:
                _retrain_cache["step"] = step_name
            result = subprocess.run(
                [venv_python, script], cwd="/opt/noisemon",
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                raise RuntimeError(f"{script} failed:\n{result.stderr[-500:]}")
        with _retrain_lock:
            _retrain_cache.update({
                "status": "done", "step": "", "error": None,
                "completed_at": int(time.time()),
            })
    except Exception as e:
        with _retrain_lock:
            _retrain_cache.update({"status": "error", "step": "", "error": str(e)})


@app.route("/api/retrain", methods=["POST"])
@require_auth
def api_retrain_start():
    with _retrain_lock:
        if _retrain_cache["status"] == "running":
            return jsonify({"status": "running"})
        _retrain_cache.update({"status": "running", "step": "Starting…", "error": None})
    t = threading.Thread(target=_run_retrain, daemon=True)
    t.start()
    return jsonify({"status": "running"})


@app.route("/api/retrain/status")
@require_auth
def api_retrain_status():
    with _retrain_lock:
        return jsonify(dict(_retrain_cache))


LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoiseMon — Login</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0f1117;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;
       display:flex;align-items:center;justify-content:center;min-height:100vh}
  .box{background:#1a1d27;border:1px solid #2a2d3e;border-radius:16px;
       padding:2.5rem;width:100%;max-width:360px}
  h1{font-size:1.4rem;margin-bottom:1.5rem;text-align:center}
  label{display:block;font-size:.85rem;color:#6b7280;margin-bottom:.4rem}
  input{width:100%;background:#252838;color:#e2e8f0;border:1px solid #2a2d3e;
        border-radius:8px;padding:.6rem .9rem;font-size:1rem;margin-bottom:1rem}
  button{width:100%;background:#6366f1;color:#fff;border:none;border-radius:8px;
         padding:.75rem;font-size:1rem;cursor:pointer;font-weight:600}
  button:hover{background:#4f52d4}
  .error{color:#ef4444;font-size:.85rem;margin-bottom:1rem;text-align:center}
</style>
</head>
<body>
<div class="box">
  <h1>🎙 NoiseMon</h1>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Username</label>
    <input type="text" name="username" autocomplete="username" autofocus>
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit">Sign in</button>
  </form>
</div>
</body>
</html>
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoiseMon</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg:#0f1117; --panel:#1a1d27; --border:#2a2d3e;
    --text:#e2e8f0; --muted:#6b7280; --accent:#6366f1;
    --green:#22c55e; --yellow:#f59e0b; --red:#ef4444;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}
  header{background:var(--panel);border-bottom:1px solid var(--border);
         padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.4rem;font-weight:700}
  header span{color:var(--muted);font-size:.9rem}
  .controls{padding:1rem 2rem;display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;
            border-bottom:1px solid var(--border);background:var(--panel)}
  .controls label{color:var(--muted);font-size:.85rem}
  select{background:#252838;color:var(--text);border:1px solid var(--border);
         border-radius:6px;padding:.35rem .7rem;font-size:.9rem;cursor:pointer}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:1rem;padding:1.5rem 2rem}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
        padding:1.2rem;text-align:center}
  .card .val{font-size:2rem;font-weight:700;margin:.3rem 0}
  .card .lbl{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
  .stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:.3rem .4rem;margin-top:.5rem}
  .stat-item .stat-val{font-size:1.2rem;font-weight:700}
  .stat-item .stat-lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:.1rem}
  .stat-delta{font-size:.68rem;font-weight:600;margin-left:.25rem;vertical-align:middle;white-space:nowrap}
  .delta-up{color:#f87171}
  .delta-dn{color:#4ade80}
  .delta-ok{color:var(--muted)}
  .stat-nc{grid-column:span 2;border-top:1px solid var(--border);margin-top:.25rem;padding-top:.35rem}
  .baseline-lbl{font-size:.62rem;color:var(--muted);margin-top:.5rem;letter-spacing:.03em}
  .card-sub-val{font-size:1.3rem;font-weight:700;margin:.15rem 0 0}
  .card-sub-lbl{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;margin-top:.6rem}
  .charts{padding:0 2rem 2rem;display:grid;gap:1.5rem}
  .chart-box{background:var(--panel);border:1px solid var(--border);
             border-radius:12px;padding:1.2rem}
  .chart-box h2{font-size:1rem;font-weight:600;margin-bottom:1rem}
  .events-table{width:100%;border-collapse:collapse;font-size:.88rem}
  .events-table th{color:var(--muted);text-align:left;padding:.5rem .75rem;
                   border-bottom:1px solid var(--border);font-weight:500}
  .events-table td{padding:.5rem .75rem;border-bottom:1px solid var(--border)}
  .badge{display:inline-block;padding:.2rem .6rem;border-radius:20px;
         font-size:.78rem;font-weight:600;text-transform:capitalize}
  #live-dot{width:8px;height:8px;background:var(--green);border-radius:50%;
            display:inline-block;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  .source-bars{display:flex;flex-direction:column;gap:.5rem}
  .src-row{display:flex;align-items:center;gap:.75rem;font-size:.88rem}
  .src-name{width:120px;text-transform:capitalize;color:var(--text)}
  .src-bar-bg{flex:1;background:#252838;border-radius:4px;height:10px;overflow:hidden}
  .src-bar{height:100%;border-radius:4px;transition:width .4s}
  .src-cnt{width:30px;text-align:right;color:var(--muted)}
  .play-btn{background:#6366f122;border:1px solid #6366f144;color:var(--text);
            border-radius:6px;padding:.2rem .55rem;cursor:pointer;font-size:.82rem;
            white-space:nowrap;transition:background .15s,color .15s}
  .play-btn:hover{background:#6366f144}
  .play-btn.active{background:#6366f1;color:#fff;border-color:#6366f1}
  #clip-player{position:fixed;bottom:0;left:0;right:0;background:var(--panel);
               border-top:1px solid var(--border);padding:.6rem 2rem;
               display:none;align-items:center;gap:1rem;z-index:100;
               box-shadow:0 -4px 16px #0006;transition:border-color .3s}
  #clip-player.is-playing{border-top-color:#6366f1}
  #clip-pulse{width:9px;height:9px;border-radius:50%;background:#6366f1;
              flex-shrink:0;display:none}
  #clip-player.is-playing #clip-pulse{display:block;animation:cpulse 1.2s ease-in-out infinite}
  @keyframes cpulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
  [data-tip]{position:relative;cursor:default}
  [data-tip]::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 8px);
    left:50%;transform:translateX(-50%);background:#12141f;color:var(--text);
    border:1px solid var(--border);border-radius:8px;padding:.45rem .65rem;
    font-size:.72rem;line-height:1.5;white-space:normal;width:210px;text-align:left;
    pointer-events:none;opacity:0;transition:opacity .15s;z-index:300;font-weight:400;
    text-transform:none;letter-spacing:0;box-shadow:0 4px 14px #0008}
  [data-tip]:hover::after{opacity:1}
  [data-tip]:has([data-tip]:hover)::after{opacity:0}
  #clip-player-info{font-size:.85rem;min-width:0;flex:0 0 auto;max-width:280px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #clip-player audio{flex:1;height:36px;min-width:0}
  #clip-player audio::-webkit-media-controls-panel{background:#252838}
  #clip-close{background:none;border:none;color:var(--muted);cursor:pointer;
              font-size:1.1rem;padding:.1rem .4rem;line-height:1}
  #clip-close:hover{color:var(--text)}
  body{padding-bottom:70px}
  .fix-btn{background:none;border:1px solid var(--border);color:var(--muted);
           border-radius:6px;padding:.18rem .4rem;cursor:pointer;font-size:.75rem;
           margin-left:.3rem;vertical-align:middle}
  .fix-btn:hover{background:#252838;color:var(--text)}
  #label-picker{position:fixed;background:var(--panel);border:1px solid var(--border);
                border-radius:10px;padding:.7rem .8rem;z-index:400;display:none;
                box-shadow:0 8px 28px #000a;min-width:260px}
  #label-picker-hdr{display:flex;justify-content:space-between;align-items:center;
                    font-size:.75rem;color:var(--muted);margin-bottom:.5rem}
  #label-picker-hdr button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:1rem;line-height:1;padding:0}
  #label-picker-hdr button:hover{color:var(--text)}
  #label-picker-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.3rem}
  .lp-btn{background:#1e2030;border:1px solid transparent;color:var(--text);border-radius:6px;
          padding:.35rem .4rem;cursor:pointer;font-size:.78rem;text-align:left;
          white-space:nowrap;transition:background .12s}
  .lp-btn:hover{background:#2d3148}
  .lp-btn.lp-current{border-color:#6366f166;background:#6366f122}
</style>
</head>
<body>
<header>
  <div id="live-dot"></div>
  <h1>🎙 NoiseMon</h1>
  <span>Neighborhood Noise Monitor</span>
  <span style="margin-left:auto;display:flex;align-items:center;gap:1.5rem">
    <a href="/label"  style="color:var(--accent);text-decoration:none;font-size:.9rem;font-weight:600">🏷 Label Clips</a>
    <a href="/review" style="color:var(--accent);text-decoration:none;font-size:.9rem;font-weight:600;margin-left:1rem">🔍 Review</a>
    <span style="color:var(--muted);font-size:.85rem" id="last-update"></span>
  </span>
</header>

<div class="controls">
  <label>Period:</label>
  <select id="period-sel" onchange="refresh()">
    <option value="1h">Last hour</option>
    <option value="6h">Last 6 hours</option>
    <option value="24h" selected>Last 24 hours</option>
    <option value="7d">Last 7 days</option>
    <option value="30d">Last 30 days</option>
    <option value="90d">Last 90 days</option>
    <option value="180d">Last 6 months</option>
  </select>
  <label>Resolution:</label>
  <select id="agg-sel" onchange="refresh()">
    <option value="1m">1 minute</option>
    <option value="5m" selected>5 minutes</option>
    <option value="15m">15 minutes</option>
    <option value="1h">1 hour</option>
    <option value="6h">6 hours</option>
    <option value="1d">1 day</option>
  </select>
</div>

<div class="cards">
  <div class="card">
    <div class="lbl" data-tip="A-weighted RMS level right now. A-weighting reduces low and high frequencies to match human hearing sensitivity.">Current dB(A)</div>
    <div class="val" id="c-cur">—</div>
    <div class="card-sub-lbl" data-tip="Highest flat (unweighted) instantaneous peak recorded in the last 7 days.">7-day Peak</div>
    <div class="card-sub-val" id="c-peak">—</div>
  </div>
  <div class="card">
    <div class="lbl">☀️ Day (7am–10pm)</div>
    <div class="stat-grid">
      <div class="stat-item" data-tip="L90 — background noise floor. The level exceeded 90% of the time: what the neighbourhood always sounds like at baseline.">
        <div><span class="stat-val" id="c-day-l90">—</span><span id="c-day-l90-d"></span></div>
        <div class="stat-lbl">L90 floor</div>
      </div>
      <div class="stat-item" data-tip="L10 — intrusive noise level. Exceeded only 10% of the time: represents significant events above the background.">
        <div><span class="stat-val" id="c-day-l10">—</span><span id="c-day-l10-d"></span></div>
        <div class="stat-lbl">L10 intrusive</div>
      </div>
      <div class="stat-item stat-nc" data-tip="Noise Climate (L10−L90). How variable the soundscape is. Narrow gap = steady background. Wide gap = frequent loud events above the floor.">
        <div><span class="stat-val" id="c-day-nc">—</span><span id="c-day-nc-d"></span></div>
        <div class="stat-lbl">NC (L10−L90)</div>
      </div>
    </div>
    <div class="baseline-lbl" data-tip="Compared to the median of the same day of week over the last 6 weeks. Red = noisier than typical, green = quieter." id="c-day-base-lbl"></div>
  </div>
  <div class="card">
    <div class="lbl">🌙 Night (10pm–7am)</div>
    <div class="stat-grid">
      <div class="stat-item" data-tip="L90 — background noise floor. The level exceeded 90% of the time: what the neighbourhood always sounds like at baseline.">
        <div><span class="stat-val" id="c-night-l90">—</span><span id="c-night-l90-d"></span></div>
        <div class="stat-lbl">L90 floor</div>
      </div>
      <div class="stat-item" data-tip="L10 — intrusive noise level. Exceeded only 10% of the time: represents significant events above the background.">
        <div><span class="stat-val" id="c-night-l10">—</span><span id="c-night-l10-d"></span></div>
        <div class="stat-lbl">L10 intrusive</div>
      </div>
      <div class="stat-item stat-nc" data-tip="Noise Climate (L10−L90). How variable the soundscape is. Narrow gap = steady background. Wide gap = frequent loud events above the floor.">
        <div><span class="stat-val" id="c-night-nc">—</span><span id="c-night-nc-d"></span></div>
        <div class="stat-lbl">NC (L10−L90)</div>
      </div>
    </div>
    <div class="baseline-lbl" data-tip="Compared to the median of the same day of week over the last 6 weeks. Red = noisier than typical, green = quieter." id="c-night-base-lbl"></div>
  </div>
  <div class="card" style="grid-column:span 2">
    <div class="lbl">Top Sources Today</div>
    <div id="top-sources" class="source-bars" style="margin-top:.75rem;text-align:left"></div>
  </div>
</div>

<div id="label-picker">
  <div id="label-picker-hdr">
    <span>Correct label</span>
    <button onclick="closePicker()">✕</button>
  </div>
  <div id="label-picker-grid"></div>
</div>

<div id="clip-player">
  <div id="clip-pulse"></div>
  <span id="clip-player-info"></span>
  <audio id="clip-audio" preload="auto"></audio>
  <button id="clip-close" title="Close" onclick="closePlayer()">✕</button>
</div>

<div class="charts">
  <div class="chart-box">
    <h2>📊 Noise Level Over Time
      <span style="color:var(--muted);font-weight:400;font-size:.85rem">(dB A-weighted)</span>
    </h2>
    <div id="chart-db" style="height:320px"></div>
  </div>
  <div class="chart-box">
    <h2>🔍 Detected Noise Events</h2>
    <div id="events-container"></div>
  </div>
</div>

<script>
const SOURCE_COLORS = {
  aircraft:    "#38bdf8",  // sky blue
  leaf_blower: "#fb923c",  // orange
  lawn_mower:  "#4ade80",  // green
  road_traffic:"#94a3b8",  // grey
  strimmer:    "#fde047",  // yellow
  voices:      "#f472b6",  // pink
  pickleball:  "#2dd4bf",  // teal
  dog_barking: "#ef4444",  // red
  music:       "#c084fc",  // purple
  birds:       "#86efac",  // mint (suppressed)
  crows:       "#a3e635",  // lime (suppressed)
  owl:         "#818cf8",  // indigo (suppressed)
};
const SOURCE_ICONS = {
  aircraft:"✈️", leaf_blower:"🍃", pickleball:"🏓",
  road_traffic:"🚗", lawn_mower:"🌿", dog_barking:"🐕",
  music:"🎵", voices:"💬", crows:"🐦‍⬛", birds:"🐦", strimmer:"🌀", owl:"🦉"
};

const layout_base = {
  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",
  font:{color:"#e2e8f0",size:12},
  margin:{t:10,r:20,b:50,l:55},
  xaxis:{gridcolor:"#2a2d3e",linecolor:"#2a2d3e"},
  yaxis:{gridcolor:"#2a2d3e",linecolor:"#2a2d3e"},
  legend:{bgcolor:"#252838",bordercolor:"#2a2d3e",borderwidth:1},
  hovermode:"x unified"
};
const config = {responsive:true, displayModeBar:false};

function formatDB(db) {
  if(db===null||db===undefined) return "—";
  const cls = db<55?"color:#22c55e":db<70?"color:#f59e0b":"color:#ef4444";
  return `<span style="${cls}">${db}</span>`;
}

function formatADSB(ac) {
  if(!ac || !ac.length) return "";
  return ac.slice(0,3).map(a => {
    const id  = a.flight || a.hex;
    const alt = a.alt_ft != null ? `${(a.alt_ft/1000).toFixed(1)}k ft` : "";
    const dst = `${a.dist_nm}nm`;
    const typ = a.type  || "";
    const tip = [a.reg, a.desc].filter(Boolean).join(" · ");
    return `<span title="${tip}" style="display:inline-block;margin-bottom:2px;
      padding:1px 5px;border-radius:4px;background:#1e2235;white-space:nowrap">
      ✈️ <b>${id}</b> ${typ} ${alt} ${dst}</span>`;
  }).join(" ");
}


async function refresh() {
  const period = document.getElementById("period-sel").value;
  const agg    = document.getElementById("agg-sel").value;

  const [meas, evts, summ] = await Promise.all([
    fetch(`/api/measurements?period=${period}&agg=${agg}`).then(r=>r.json()),
    fetch(`/api/events?period=${period}`).then(r=>r.json()),
    fetch("/api/summary").then(r=>r.json()),
  ]);

  // Cards
  document.getElementById("c-cur").innerHTML      = formatDB(summ.current_db);
  document.getElementById("c-peak").innerHTML     = formatDB(summ.week_max);

  const dayNames = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];

  function fmtDelta(d) {
    if (d === null || d === undefined) return "";
    const sign = d > 0 ? "+" : "";
    const cls  = d >  1.5 ? "delta-up" : d < -1.5 ? "delta-dn" : "delta-ok";
    const tip  = `${sign}${d.toFixed(1)} dB vs typical ${dayNames[new Date().getDay()]}. Red = noisier than usual, green = quieter.`;
    return `<span class="stat-delta ${cls}" data-tip="${tip}">${sign}${d.toFixed(1)}</span>`;
  }

  document.getElementById("c-day-l90").innerHTML   = formatDB(summ.day_l90);
  document.getElementById("c-day-l90-d").innerHTML = fmtDelta(summ.delta_day_l90);
  document.getElementById("c-day-l10").innerHTML   = formatDB(summ.day_l10);
  document.getElementById("c-day-l10-d").innerHTML = fmtDelta(summ.delta_day_l10);
  document.getElementById("c-day-nc").innerHTML    = formatDB(summ.day_nc);
  document.getElementById("c-day-nc-d").innerHTML  = fmtDelta(summ.delta_day_nc);

  document.getElementById("c-night-l90").innerHTML   = formatDB(summ.night_l90);
  document.getElementById("c-night-l90-d").innerHTML = fmtDelta(summ.delta_night_l90);
  document.getElementById("c-night-l10").innerHTML   = formatDB(summ.night_l10);
  document.getElementById("c-night-l10-d").innerHTML = fmtDelta(summ.delta_night_l10);
  document.getElementById("c-night-nc").innerHTML    = formatDB(summ.night_nc);
  document.getElementById("c-night-nc-d").innerHTML  = fmtDelta(summ.delta_night_nc);

  const todayName = dayNames[new Date().getDay()];
  const n = summ.baseline_days_n;
  const baseLbl = n >= 2 ? `vs typical ${todayName} · ${n} week${n>1?"s":""}` : "building baseline\u2026";
  document.getElementById("c-day-base-lbl").textContent   = baseLbl;
  document.getElementById("c-night-base-lbl").textContent = baseLbl;

  const maxCnt = Math.max(1, ...summ.top_sources.map(s=>s.count));
  document.getElementById("top-sources").innerHTML =
    summ.top_sources.slice(0,6).map(s=>`
      <div class="src-row">
        <div class="src-name">${SOURCE_ICONS[s.source]||"🔊"} ${s.source.replace(/_/g," ")}</div>
        <div class="src-bar-bg">
          <div class="src-bar" style="width:${s.count/maxCnt*100}%;background:${SOURCE_COLORS[s.source]||"#6366f1"}"></div>
        </div>
        <div class="src-cnt">${s.count}</div>
      </div>`).join("") || "<span style='color:var(--muted)'>No events yet</span>";

  // dB chart — with deduplicated event markers
  const times = meas.map(m=>m.t);
  // Server timestamps are naive local-time strings (no Z). Match that format so
  // Plotly treats them consistently — using toISOString() (UTC+Z) shifts the
  // x-axis right by the UTC offset, creating a blank gap at the chart edge.
  const _nd = new Date();
  const now = new Date(_nd.getTime() - _nd.getTimezoneOffset()*60000)
                .toISOString().slice(0,-1);  // local ISO without Z

  // Group events into chart buckets so dense periods get one marker, not dozens
  const aggMs = {"1m":60000,"5m":300000,"15m":900000,"1h":3600000,
                 "6h":21600000,"1d":86400000}[agg] || 300000;
  const evtMap = new Map();
  evts.forEach(e => {
    // Append "Z" so JS parses the naive local-time string as UTC, matching how
    // Plotly treats the measurement timestamps (also naive, treated as UTC).
    const bk = Math.floor(new Date(e.t + "Z").getTime() / aggMs) * aggMs;
    if (!evtMap.has(bk)) evtMap.set(bk, []);
    evtMap.get(bk).push(e);
  });
  const grouped = Array.from(evtMap.entries()).sort((a,b)=>a[0]-b[0]).map(([bk,evs])=>{
    evs.sort((a,b)=>b.db-a.db);  // highest dB first
    // Strip Z to keep naive format consistent with measurement timestamps
    return {t: new Date(bk).toISOString().slice(0,-1), primary: evs[0], all: evs};
  });

  const evtTrace = {
    x: grouped.map(g=>g.t),
    y: grouped.map(g=>g.primary.db),  // at the event's actual dB level
    mode:"markers",
    name:"Events",
    marker:{
      color: grouped.map(g=>SOURCE_COLORS[g.primary.source]||"#6366f1"),
      size:13, symbol:"triangle-up",
      line:{color:"#ffffff",width:1.5}
    },
    text: grouped.map(g=>{
      const time = new Date(g.primary.t).toLocaleTimeString();
      let tip = `<b>${time}</b><br>`;
      g.all.forEach(e => {
        tip += `${SOURCE_ICONS[e.source]||"🔊"} <b>${e.source.replace(/_/g," ")}</b> ` +
               `${e.conf}% · ${e.db} dB(A)<br>`;
      });
      const ac = g.all.find(e=>e.adsb && e.adsb.length);
      if (ac) {
        tip += "<br>";
        ac.adsb.slice(0,3).forEach(a=>{
          const alt = a.alt_ft!=null ? `${(a.alt_ft/1000).toFixed(1)}k ft` : "";
          tip += `✈️ ${a.flight||a.hex} ${a.type||""} ${alt} ${a.dist_nm}nm<br>`;
        });
      }
      return tip.trimEnd();
    }),
    hoverinfo:"none",
    showlegend:false
  };

  const xStart = times.length ? times[0] : now;
  Plotly.react("chart-db", [
    {x:times, y:meas.map(m=>m.db_peak), name:"Peak",
     mode:"lines", line:{color:"#f87171",width:1}, opacity:.7, hoverinfo:"none"},
    {x:times, y:meas.map(m=>m.db_avg),  name:"Average",
     mode:"lines", line:{color:"#818cf8",width:2}, hoverinfo:"none"},
    {x:times, y:meas.map(m=>m.db_min),  name:"Min",
     mode:"lines", line:{color:"#4ade80",width:1}, opacity:.6, hoverinfo:"none"},
    evtTrace,
  ], {
    ...layout_base,
    xaxis:{...layout_base.xaxis, range:[xStart, now]},
    yaxis:{...layout_base.yaxis, title:"dB(A)", range:[20,110]},
    shapes:[
      {type:"line",x0:xStart,x1:now,y0:55,y1:55,
       line:{color:"#f59e0b",width:1,dash:"dot"}},
      {type:"line",x0:xStart,x1:now,y0:70,y1:70,
       line:{color:"#ef4444",width:1,dash:"dot"}},
    ]
  }, config);

  // Custom positioned tooltip for event markers (set up once)
  if (!window._evtTipReady) {
    window._evtTipReady = true;
    const tip = document.createElement("div");
    tip.id = "evt-tip";
    tip.style.cssText = "position:fixed;background:#252838;border:1px solid #3a3d4e;" +
      "border-radius:6px;padding:8px 12px;font-size:12px;color:#e2e8f0;" +
      "pointer-events:none;z-index:9999;display:none;max-width:300px;line-height:1.6;" +
      "box-shadow:0 4px 12px rgba(0,0,0,.5)";
    document.body.appendChild(tip);
    const chart = document.getElementById("chart-db");
    chart.addEventListener("mousemove", e => {
      if (tip.style.display === "none") return;
      tip.style.left = (e.clientX + 16) + "px";
      tip.style.top  = Math.max(8, e.clientY - tip.offsetHeight - 10) + "px";
    });
    chart.on("plotly_hover", data => {
      const pt = data.points.find(p => p.data.name === "Events");
      if (!pt) { tip.style.display = "none"; return; }
      tip.innerHTML = pt.text || "";
      tip.style.display = "block";
    });
    chart.on("plotly_unhover", () => { tip.style.display = "none"; });
  }

  // Events table
  if(!evts.length) {
    document.getElementById("events-container").innerHTML =
      "<p style='color:var(--muted);padding:.5rem'>No events detected in this period.</p>";
    return;
  }
  document.getElementById("events-container").innerHTML = `
    <table class="events-table">
      <thead><tr>
        <th>Time</th><th>Source</th><th>Confidence</th><th>dB(A)</th><th>Aircraft / ADS-B</th><th>Clip</th>
      </tr></thead>
      <tbody>${evts.slice(0,100).map(e=>`
        <tr data-ts="${e.ts}">
          <td style="color:var(--muted)">${new Date(e.t).toLocaleString()}</td>
          <td><span class="badge" style="background:${SOURCE_COLORS[e.source]||"#6366f1"}22;
              color:${SOURCE_COLORS[e.source]||"#6366f1"}">
            ${SOURCE_ICONS[e.source]||"🔊"} ${e.source.replace(/_/g," ")}</span>
            ${e.clip ? `<button class="fix-btn" title="Correct label" onclick="openPicker(${e.ts},'${e.clip}','${e.source}',event)">✏</button>` : ""}
          </td>
          <td><span style="color:${e.conf>70?"#22c55e":e.conf>45?"#f59e0b":"#94a3b8"}">${e.conf}%</span></td>
          <td>${e.db}</td>
          <td style="font-size:.78rem;max-width:220px">${formatADSB(e.adsb)}</td>
          <td>${e.clip ? `<button class="play-btn" data-clip="${e.clip}" onclick="playClip('/clips/${e.clip}','${e.source}','${e.t}',this)">▶ Play</button>` : ""}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;

  document.getElementById("last-update").textContent =
    "Updated " + (summ.last_ts ? new Date(summ.last_ts).toLocaleTimeString() : new Date().toLocaleTimeString());

  // Re-apply any pending corrections — guards against stale in-flight refreshes
  // overwriting a label the user just corrected. Entry is cleared once the
  // server starts returning the new label itself.
  for (const [ts, label] of Object.entries(_corrections)) {
    const row = document.querySelector(`tr[data-ts="${ts}"]`);
    if (!row) continue;
    const evtData = evts.find(e => String(e.ts) === ts);
    if (evtData && evtData.source === label) {
      delete _corrections[ts];   // server confirmed — no longer need to override
      continue;
    }
    const badge = row.querySelector(".badge");
    if (badge) {
      badge.style.background = (SOURCE_COLORS[label]||"#6366f1") + "22";
      badge.style.color      = SOURCE_COLORS[label]||"#6366f1";
      badge.innerHTML        = `${SOURCE_ICONS[label]||"🔊"} ${label.replace(/_/g," ")}`;
    }
    const fixBtn = row.querySelector(".fix-btn");
    if (fixBtn) fixBtn.setAttribute("onclick", `openPicker(${ts},'${_corrections[ts+"_clip"]||""}','${label}',event)`);
  }

  // Re-sync play button state after table rebuild
  if (_clipAudio.dataset.src) {
    const clip = _clipAudio.dataset.src.replace("/clips/", "");
    const btn  = document.querySelector(`.play-btn[data-clip="${clip}"]`);
    if (btn) {
      _activeBtn = btn;
      const playing = !_clipAudio.paused;
      btn.classList.toggle("active", playing);
      btn.textContent = playing ? "⏸ Pause" : "▶ Play";
    }
  }
}

// ── Label correction picker ───────────────────────────────────────────────────
const CORRECT_LABELS = [
  ["aircraft","✈️"],        ["road_traffic","🚗"],    ["leaf_blower","🍃"],
  ["lawn_mower","🌿"],      ["strimmer","🌾"],         ["voices","🗣️"],
  ["birds","🐦"],           ["crows","🐦‍⬛"],            ["owl","🦉"],
  ["music","🎵"],           ["dog_barking","🐕"],      ["pool_pump","💧"],
  ["human_activity","🧑"],  ["false_positive","❌"],
];

const _picker      = document.getElementById("label-picker");
const _pickerGrid  = document.getElementById("label-picker-grid");
let   _pickerTs    = null;
let   _pickerClip  = null;
const _corrections = {};   // ts (string) → label; overrides stale refreshes until server confirms

function openPicker(ts, clip, currentSource, evt) {
  _pickerTs   = ts;
  _pickerClip = clip;
  _pickerGrid.innerHTML = CORRECT_LABELS.map(([label, icon]) =>
    `<button class="lp-btn${label===currentSource?" lp-current":""}"
             onclick="correctEvent('${label}')">${icon} ${label.replace(/_/g," ")}</button>`
  ).join("");
  // Position near the click, keeping within viewport
  const x = Math.min(evt.clientX, window.innerWidth  - 280);
  const y = Math.min(evt.clientY + 6, window.innerHeight - 220);
  _picker.style.left    = x + "px";
  _picker.style.top     = y + "px";
  _picker.style.display = "block";
  evt.stopPropagation();
}

function closePicker() { _picker.style.display = "none"; }
document.addEventListener("click", e => { if (!_picker.contains(e.target)) closePicker(); });

async function correctEvent(label) {
  closePicker();
  const res = await fetch("/api/correct_event", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ts_start: _pickerTs, clip_path: _pickerClip, label}),
  });
  if (!res.ok) return;
  // Record correction so stale in-flight refreshes can't overwrite it
  const ts = String(_pickerTs);
  _corrections[ts] = label;
  _corrections[ts + "_clip"] = _pickerClip;
  // Update the badge in-place immediately
  const row = document.querySelector(`tr[data-ts="${ts}"]`);
  if (row) {
    const badge = row.querySelector(".badge");
    if (badge) {
      badge.style.background = (SOURCE_COLORS[label]||"#6366f1") + "22";
      badge.style.color      = SOURCE_COLORS[label]||"#6366f1";
      badge.innerHTML        = `${SOURCE_ICONS[label]||"🔊"} ${label.replace(/_/g," ")}`;
    }
    const fixBtn = row.querySelector(".fix-btn");
    if (fixBtn) fixBtn.setAttribute("onclick", `openPicker(${ts},'${_pickerClip}','${label}',event)`);
  }
}

// ── Clip player ───────────────────────────────────────────────────────────────
// Clips are: PRE_ROLL(30s) of audio before detection + up to POST_ROLL(90s) after.
// The YAMNet inference window that triggered the event spans the last HISTORY(20s)
// of the pre-roll, i.e. t=10s–30s in the clip. Start playback at t=10s so the
// listener hears the build-up that drove the classification, then the event itself.
const CLIP_DETECT_OFFSET = 10;  // seconds into clip where detection window begins

const _clipAudio  = document.getElementById("clip-audio");
const _clipPlayer = document.getElementById("clip-player");
const _clipInfo   = document.getElementById("clip-player-info");
let   _activeBtn  = null;

function _setPlayingState(on) {
  _clipPlayer.classList.toggle("is-playing", on);
  if (_activeBtn) {
    _activeBtn.classList.toggle("active", on);
    _activeBtn.textContent = on ? "⏸ Pause" : "▶ Play";
  }
}

_clipAudio.addEventListener("play",  () => _setPlayingState(true));
_clipAudio.addEventListener("pause", () => _setPlayingState(false));
_clipAudio.addEventListener("ended", () => _setPlayingState(false));

function playClip(src, source, time, btn) {
  // Same clip already playing — toggle pause
  if (_clipAudio.dataset.src === src && !_clipAudio.paused) {
    _clipAudio.pause();
    return;
  }

  // Clear previous active button
  if (_activeBtn && _activeBtn !== btn) {
    _activeBtn.classList.remove("active");
    _activeBtn.textContent = "▶ Play";
  }
  _activeBtn = btn || null;

  const label = (SOURCE_ICONS[source] || "🔊") + " " +
                source.replace(/_/g, " ") + " · " +
                new Date(time).toLocaleString();
  _clipInfo.textContent = label;
  _clipPlayer.style.display = "flex";

  function seekAndPlay() {
    _clipAudio.currentTime = CLIP_DETECT_OFFSET;
    _clipAudio.play().catch(() => {});
  }

  if (_clipAudio.dataset.src === src) {
    // Same clip but was paused — resume from where it stopped rather than seeking back
    _clipAudio.play().catch(() => {});
  } else {
    _clipAudio.dataset.src = src;
    _clipAudio.src = src;
    _clipAudio.addEventListener("canplay", function onready() {
      _clipAudio.removeEventListener("canplay", onready);
      seekAndPlay();
    });
    _clipAudio.load();
  }
}

function closePlayer() {
  _clipAudio.pause();
  _clipPlayer.style.display = "none";
  _setPlayingState(false);
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""

LABEL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoiseMon — Label Clips</title>
<script src="https://unpkg.com/wavesurfer.js@7.8.0/dist/wavesurfer.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7.8.0/dist/plugins/regions.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7.8.0/dist/plugins/spectrogram.min.js"></script>
<style>
  :root {
    --bg:#0f1117; --panel:#1a1d27; --border:#2a2d3e;
    --text:#e2e8f0; --muted:#6b7280; --accent:#6366f1;
    --green:#22c55e; --yellow:#f59e0b; --red:#ef4444;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}
  header{background:var(--panel);border-bottom:1px solid var(--border);
         padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.4rem;font-weight:700}
  a.back{color:var(--muted);text-decoration:none;font-size:.9rem}
  a.back:hover{color:var(--text)}
  .layout{display:grid;grid-template-columns:320px 1fr;height:calc(100vh - 57px)}
  .sidebar{background:var(--panel);border-right:1px solid var(--border);
           overflow-y:auto;display:flex;flex-direction:column}
  .sidebar-header{padding:1rem;border-bottom:1px solid var(--border);
                  font-size:.85rem;color:var(--muted);text-transform:uppercase;
                  letter-spacing:.05em}
  .clip-item{padding:.75rem 1rem;border-bottom:1px solid var(--border);
             cursor:pointer;transition:background .15s}
  .clip-item:hover{background:#252838}
  .clip-item.active{background:#252838;border-left:3px solid var(--accent)}
  .clip-item.done{opacity:.5}
  .clip-name{font-size:.85rem;color:var(--text);word-break:break-all}
  .clip-meta{font-size:.75rem;color:var(--muted);margin-top:.2rem}
  .main{display:flex;flex-direction:column;padding:1.5rem;gap:1.5rem;overflow-y:auto}
  .waveform-box{background:var(--panel);border:1px solid var(--border);
                border-radius:12px;padding:1.2rem}
  .waveform-box h2{font-size:1rem;font-weight:600;margin-bottom:1rem}
  #waveform{border-radius:8px;overflow:hidden;background:#0a0d14;min-height:120px}
  #waveform div{background:#0a0d14 !important}
  #waveform canvas{display:block;background:#0a0d14 !important}
  #waveform-loading{display:none;padding:2.5rem 1rem;text-align:center}
  #waveform-loading .spinner{width:36px;height:36px;border:3px solid #2a2d3e;
    border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;
    margin:0 auto 1rem}
  @keyframes spin{to{transform:rotate(360deg)}}
  #waveform-progress{font-size:1.1rem;font-weight:700;color:var(--accent)}
  #waveform-status{font-size:.8rem;color:var(--muted);margin-top:.3rem}
  #waveform-error{display:none;padding:1.5rem;text-align:center;color:var(--red);font-size:.9rem}
  .controls{display:flex;gap:.75rem;margin-top:1rem;flex-wrap:wrap;align-items:center}
  button{background:#252838;color:var(--text);border:1px solid var(--border);
         border-radius:8px;padding:.5rem 1rem;font-size:.875rem;cursor:pointer;
         transition:background .15s}
  button:hover{background:#2d3148}
  button.primary{background:var(--accent);border-color:var(--accent)}
  button.primary:hover{background:#4f52d4}
  button.danger{background:#ef444422;border-color:var(--red);color:var(--red)}
  .label-box{background:var(--panel);border:1px solid var(--border);
             border-radius:12px;padding:1.2rem}
  .label-box h2{font-size:1rem;font-weight:600;margin-bottom:1rem}
  .label-buttons{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
  .label-btn{padding:.4rem .9rem;border-radius:20px;border:1px solid;
             font-size:.85rem;cursor:pointer;font-weight:600;
             text-transform:capitalize;transition:opacity .15s}
  .label-btn:hover{opacity:.8}
  .label-btn.selected{box-shadow:0 0 0 2px white}
  .segments-list{display:flex;flex-direction:column;gap:.5rem;margin-top:1rem}
  .seg-row{display:flex;align-items:center;gap:.75rem;font-size:.85rem;
           background:#252838;border-radius:8px;padding:.5rem .75rem;
           cursor:pointer;border:2px solid transparent;transition:border-color .15s}
  .seg-row.active{border-color:var(--accent)}
  .seg-row:hover{background:#2d3148}
  .seg-label{flex:1;text-transform:capitalize;font-weight:600}
  .seg-time{color:var(--muted);font-size:.8rem}
  .seg-del{background:none;border:none;color:var(--red);cursor:pointer;
           font-size:1rem;padding:0 .25rem}
  .save-bar{display:flex;gap:1rem;align-items:center}
  .status{color:var(--muted);font-size:.85rem}
  #no-clip{color:var(--muted);padding:2rem;text-align:center}
  .tabs{display:flex;gap:0;border-bottom:1px solid var(--border)}
  .tab{padding:.5rem 1rem;font-size:.85rem;cursor:pointer;color:var(--muted);
       border-bottom:2px solid transparent;margin-bottom:-1px}
  .tab.active{color:var(--text);border-bottom-color:var(--accent)}
</style>
</head>
<body>
<header>
  <a class="back" href="/">← Dashboard</a>
  <h1>🏷 Label Clips <span style="font-size:.6rem;color:var(--muted);font-weight:400;vertical-align:middle">v3</span></h1>
  <span id="progress" style="margin-left:auto;color:var(--muted);font-size:.85rem"></span>
</header>

<div class="layout">
  <!-- Sidebar: clip list -->
  <div class="sidebar">
    <div class="tabs">
      <div class="tab active" onclick="showTab('unlabelled')">Unlabelled</div>
      <div class="tab" onclick="showTab('labelled')">Done</div>
      <div class="tab" onclick="showTab('outliers')">Outliers</div>
    </div>
    <div id="clip-list"></div>
    <div id="outlier-panel" style="display:none;flex-direction:column;gap:.5rem;padding:.75rem">
      <div style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap">
        <button onclick="computeOutliers()" style="font-size:.8rem;padding:.35rem .75rem">Compute Outliers</button>
        <button onclick="retrainClassifier()" style="font-size:.8rem;padding:.35rem .75rem">Retrain Classifier</button>
      </div>
      <span id="outlier-status" style="font-size:.78rem;color:var(--muted)"></span>
      <span id="retrain-status" style="display:none;font-size:.78rem"></span>
      <div id="outlier-spinner" style="display:none;text-align:center;padding:.5rem">
        <div class="spinner" style="width:24px;height:24px;border-width:2px;margin:0 auto"></div>
      </div>
      <div id="outlier-list"></div>
    </div>
  </div>

  <!-- Main: waveform + labelling -->
  <div class="main">
    <div id="no-clip">← Select a clip from the list to begin labelling</div>

    <div id="editor" style="display:none">
      <div class="waveform-box">
        <h2 id="clip-title">—</h2>
        <div id="classifier-info" style="display:none;font-size:.8rem;color:var(--muted);
             margin-bottom:.75rem;padding:.3rem .5rem;border-left:2px solid;border-radius:2px"></div>
        <div id="waveform-loading">
          <div class="spinner"></div>
          <div id="waveform-progress">0%</div>
          <div id="waveform-status">Loading clip…</div>
        </div>
        <div id="waveform-error">⚠ Failed to load clip (timeout or network error).
          <button style="margin-left:.75rem" onclick="retryClip()">Retry</button>
        </div>
        <div id="waveform"></div>
        <div class="controls">
          <button onclick="ws && ws.playPause()">▶ Play / Pause</button>
          <button onclick="ws && ws.stop()">⏹ Stop</button>
          <span style="color:var(--muted);font-size:.85rem">
            Drag to mark a region · or press a label key to start, Space to end
          </span>
        </div>
        <div id="waveform-status-bar" style="margin-top:.5rem;font-size:.82rem;
             color:var(--accent);min-height:1.2em;font-weight:600"></div>
      </div>

      <div class="label-box">
        <h2>Labels <span id="label-hint" style="font-weight:400;color:var(--muted);font-size:.85rem">— drag waveform to add a region, then pick a label</span></h2>
        <div class="label-buttons" id="label-buttons"></div>
        <div class="save-bar" style="margin-top:.75rem;margin-bottom:.5rem">
          <button class="primary" onclick="nextClip()">Next →</button>
          <button class="danger" onclick="clearSegments()">Clear All</button>
          <span class="status" id="status"></span>
        </div>
        <div id="segments-list" class="segments-list"></div>
      </div>
    </div>
  </div>
</div>

<script>
const SOURCE_COLORS = {
  aircraft:"#818cf8", leaf_blower:"#fb923c", pickleball:"#34d399",
  road_traffic:"#94a3b8", lawn_mower:"#fbbf24", dog_barking:"#f87171",
  music:"#e879f9", voices:"#67e8f9", crows:"#86efac", birds:"#4ade80",
  owl:"#a5b4fc", strimmer:"#f59e0b", unknown:"#6b7280", other:"#a78bfa",
  raindrops:"#7dd3fc"
};
const LABELS = ["aircraft","leaf_blower","lawn_mower","strimmer","pickleball",
                "road_traffic","dog_barking","music","voices",
                "crows","birds","owl","raindrops","unknown","other"];

let ws = null;
let wsRegions = null;
let currentClip = null;
let currentClipSource = null;   // event source label (e.g. "music") for current clip
let currentClipConf  = 0;       // event confidence % for current clip
let classifierRegion = null;    // non-interactive classifier-window overlay region
let selectedLabel = LABELS[0];
let segments = [];      // own (editable) segments for this clip
let roSegments = [];    // read-only segments from overlapping clips
let activeRegion = null;
let clips = [];
let currentTab = "unlabelled";
let regionStartTime = null;   // set when a label key is pressed to mark start
let skipNextRegionCreated = false;  // set before programmatic addRegion() calls
let _newlyCreatedRegion = null;  // set on region-created; blocks spurious region-clicked on other regions
let pendingSeekTime = null;      // set by outlier row click; consumed on waveform ready
let outlierItems = [];           // current flat wrong-items list, for Next navigation
let currentOutlierKey = null;    // clip:tStart key of the currently loaded outlier row

const LABEL_SHORTCUTS = {
  'a': 'aircraft', 'b': 'birds', 'c': 'crows', 'd': 'dog_barking',
  'l': 'leaf_blower', 'm': 'lawn_mower', 'o': 'owl', 't': 'road_traffic',
  'p': 'pickleball', 'r': 'raindrops', 's': 'strimmer', 'v': 'voices',
  'u': 'pool_pump', 'h': 'human_activity'
};
const LABEL_DISPLAY = {
  aircraft:'(a)ircraft', birds:'(b)irds', crows:'(c)rows',
  dog_barking:'(d)og barking', leaf_blower:'(l)eaf blower',
  lawn_mower:'(m)ower', owl:'(o)wl', road_traffic:'(t)raffic',
  pickleball:'(p)ickleball', raindrops:'(r)aindrops', strimmer:'(s)trimmer', voices:'(v)oices',
  pool_pump:'pool p(u)mp', human_activity:'(h)uman activity',
  music:'music', unknown:'unknown', other:'other'
};

// ── Tabs ──────────────────────────────────────────────────────────────────────
function showTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".tab").forEach((t,i) => {
    t.classList.toggle("active",
      (i===0&&tab==="unlabelled")||(i===1&&tab==="labelled")||(i===2&&tab==="outliers"));
  });
  document.getElementById("clip-list").style.display     = tab === "outliers" ? "none"  : "block";
  document.getElementById("outlier-panel").style.display = tab === "outliers" ? "flex"  : "none";
  if(tab === "outliers") loadOutliers(); else loadClipList();
}

// ── Clip list ─────────────────────────────────────────────────────────────────
async function loadClipList() {
  const url = currentTab === "unlabelled"
    ? "/api/clips/unlabelled" : "/api/clips/labelled";
  const data = await fetch(url).then(r=>r.json());
  clips = data;

  const list = document.getElementById("clip-list");
  if(!data.length) {
    list.innerHTML = "<div style='padding:1rem;color:var(--muted)'>No clips yet.</div>";
    document.getElementById("progress").textContent = "";
    return;
  }

  if(currentTab === "unlabelled") {
    document.getElementById("progress").textContent =
      `${data.length} clips to label`;
    list.innerHTML = data.map((c,i) => `
      <div class="clip-item" id="ci-${i}" onclick="loadClip('${c.clip}', ${i}, '${c.source}', ${c.conf})">
        <div class="clip-name">${c.clip}</div>
        <div class="clip-meta">${new Date(c.t).toLocaleString()} · ${c.source} · ${c.db} dB</div>
      </div>`).join("");
  } else {
    list.innerHTML = data.map((c,i) => `
      <div class="clip-item done" onclick="loadClip('${c.clip}', ${i}, '${c.source||""}', ${c.conf||0})">
        <div class="clip-name">${c.clip}</div>
        <div class="clip-meta">${c.segment_count} segments · ${c.labels}</div>
      </div>`).join("");
  }
}

// ── Load clip into waveform ───────────────────────────────────────────────────
let loadTimeout = null;
let currentLoadPath = null;
const STALL_TIMEOUT = 60000;  // ms of no progress before giving up (clips can be ~11 MB)

function resetLoadWatchdog(clipPath) {
  if(loadTimeout) clearTimeout(loadTimeout);
  loadTimeout = setTimeout(() => {
    if(currentLoadPath !== clipPath) return;
    if(ws) { try { ws.destroy(); } catch(e){} ws = null; wsRegions = null; }
    setLoadingState("error");
    document.getElementById("status").textContent = "Stalled — no data for 10 s";
  }, STALL_TIMEOUT);
}

function setLoadingState(state, pct) {
  // state: "loading" | "ready" | "error"
  document.getElementById("waveform-loading").style.display = state === "loading" ? "block" : "none";
  document.getElementById("waveform-error").style.display   = state === "error"   ? "block" : "none";
  document.getElementById("waveform").style.display         = state === "ready"   ? "block" : "none";
  if(state === "loading" && pct !== undefined) {
    document.getElementById("waveform-progress").textContent = pct + "%";
    document.getElementById("waveform-status").textContent =
      pct < 100 ? "Loading clip…" : "Decoding audio…";
  }
}

function retryClip() {
  if(currentClip) loadClip(currentClip,
    clips.findIndex(c => (c.clip||c.clip_path) === currentClip));
}

async function loadClip(clipPath, idx, source='', conf=0) {
  currentClip = clipPath;
  currentClipSource = source || null;
  currentClipConf   = conf  || 0;
  classifierRegion  = null;
  currentLoadPath = clipPath;
  segments = [];
  roSegments = [];
  regionStartTime = null;

  // Highlight active
  document.querySelectorAll(".clip-item").forEach(el => el.classList.remove("active"));
  const el = document.getElementById(`ci-${idx}`);
  if(el) el.classList.add("active");

  document.getElementById("no-clip").style.display = "none";
  document.getElementById("editor").style.display  = "block";
  document.getElementById("clip-title").textContent = clipPath;
  document.getElementById("classifier-info").style.display = "none";
  document.getElementById("status").textContent = "";

  // Destroy previous wavesurfer
  if(ws) { ws.destroy(); ws = null; wsRegions = null; }
  if(loadTimeout) { clearTimeout(loadTimeout); loadTimeout = null; }

  setLoadingState("loading", 0);
  resetLoadWatchdog(clipPath);  // start watchdog — resets on each progress event

  // Create WaveSurfer with Regions plugin
  try {
    wsRegions = WaveSurfer.Regions.create();
    // Inferno-style colormap — values must be 0-1 (plugin multiplies by 255 internally)
    // Stops: black → dark purple → red → orange → yellow → cream
    const stops = [
      [0.00, [0.00, 0.00, 0.00]],
      [0.15, [0.08, 0.02, 0.15]],
      [0.35, [0.72, 0.06, 0.10]],
      [0.55, [0.85, 0.22, 0.02]],
      [0.75, [0.98, 0.60, 0.02]],
      [0.90, [0.99, 0.90, 0.35]],
      [1.00, [0.99, 0.99, 0.90]],
    ];
    function lerpStops(t) {
      t = Math.pow(t, 0.65);  // gamma: spread low-amplitude detail
      for (let s = 1; s < stops.length; s++) {
        if (t <= stops[s][0]) {
          const a = (t - stops[s-1][0]) / (stops[s][0] - stops[s-1][0]);
          return stops[s-1][1].map((c,i) => c + a * (stops[s][1][i] - c));
        }
      }
      return stops[stops.length-1][1];
    }
    const hotMap = Array.from({length:256}, (_,i) => [...lerpStops(i/255), 1]);
    const spectrogram = WaveSurfer.Spectrogram.create({
      height:       200,
      labels:       false,
      frequencyMax: 8000,
      colorMap:     hotMap,
    });
    ws = WaveSurfer.create({
      container:    "#waveform",
      waveColor:    "#4f52d4",
      progressColor:"#818cf8",
      height:       1,
      normalize:    true,
      plugins:      [wsRegions, spectrogram],
    });
  } catch(e) {
    clearTimeout(loadTimeout); loadTimeout = null;
    document.getElementById("waveform-loading").style.display = "none";
    document.getElementById("waveform").style.display = "none";
    document.getElementById("waveform-error").style.display = "block";
    document.getElementById("waveform-error").textContent = "Init error: " + e;
    return;
  }

  // Progress updates — each event resets the stall watchdog
  ws.on("loading", pct => {
    if(currentLoadPath !== clipPath) return;
    setLoadingState("loading", pct);
    resetLoadWatchdog(clipPath);
  });

  // Load existing segments if any (fetch in parallel with audio load)
  const existingPromise = fetch(`/api/segments/${clipPath}`).then(r=>r.json()).catch(()=>[]);

  ws.load(`/clips/${clipPath}`);

  ws.on("ready", async () => {
    if(currentLoadPath !== clipPath) return;
    clearTimeout(loadTimeout); loadTimeout = null;
    setLoadingState("ready");

    // WaveSurfer v7 renders inside a shadow DOM — inject a style element to override
    // any white backgrounds the spectrogram plugin adds (can't be reached by normal CSS).
    const waveHost = document.querySelector("#waveform");
    const shadow = waveHost && waveHost.shadowRoot;
    if (shadow) {
      let fixStyle = shadow.getElementById("noisemon-bg-fix");
      if (!fixStyle) {
        fixStyle = document.createElement("style");
        fixStyle.id = "noisemon-bg-fix";
        shadow.appendChild(fixStyle);
      }
      fixStyle.textContent = `
        div, canvas { background: #0a0d14 !important; background-color: #0a0d14 !important; }
      `;
      // Also patch any inline styles already present
      shadow.querySelectorAll("*").forEach(el => {
        const bg = el.style.background || el.style.backgroundColor;
        if(bg && (bg.includes("255, 255, 255") || bg === "white" || bg === "#fff" || bg === "#ffffff")) {
          el.style.background = "#0a0d14";
          el.style.backgroundColor = "#0a0d14";
        }
      });
    }

    wsRegions.enableDragSelection({ color: "rgba(99,102,241,0.3)" });

    // ── Classifier window overlay ─────────────────────────────────────────────
    // The clip starts PRE_ROLL_SECONDS (30s) before ts_start (the confirmation
    // moment).  YAMNet inferred on the HISTORY_SECONDS (20s) window ending at
    // ts_start, so in clip-relative time that window is always t=10s–30s.
    // Show it as a non-interactive shaded band so the labeller can see exactly
    // what audio drove the classification decision.
    const PRE_ROLL = 30, HISTORY = 20;
    const dur = ws.getDuration();
    if (currentClipSource && dur > 0) {
      const winStart = Math.max(0, PRE_ROLL - HISTORY);   // 10s
      const winEnd   = Math.min(dur, PRE_ROLL);            // 30s
      const col = SOURCE_COLORS[currentClipSource] || "#94a3b8";
      skipNextRegionCreated = true;
      classifierRegion = wsRegions.addRegion({
        start:  winStart,
        end:    winEnd,
        color:  hexToRgba(col, 0.18),
        drag:   false,
        resize: false,
      });
      const info = document.getElementById("classifier-info");
      info.textContent =
        `Classifier saw: ${currentClipSource.replace(/_/g," ")} ${currentClipConf}%`
        + `  ·  t=${winStart}–${winEnd}s (YAMNet window ending at trigger)`;
      info.style.display = "block";
      info.style.borderColor = col;
      info.style.color = col;
    }

    const existing = await existingPromise;
    if(currentLoadPath !== clipPath) return;  // clip changed while fetching

    existing.forEach(seg => {
      const col = SOURCE_COLORS[seg.label] || "#6366f1";
      skipNextRegionCreated = true;
      if(seg.own) {
        // Editable region for this clip's own segment
        const r = wsRegions.addRegion({
          start: seg.start, end: seg.end,
          color: hexToRgba(col, 0.35),
          drag: true, resize: true,
        });
        segments.push({ start: seg.start, end: seg.end, label: seg.label, region: r });
      } else {
        // Read-only region from an overlapping clip
        const shortName = seg.from_clip.replace(/^\d{8}_\d{4}(\d{2})_(.+)\.wav$/, "$2 :$1");
        const r = wsRegions.addRegion({
          start: seg.start, end: seg.end,
          color: hexToRgba(col, 0.15),
          drag: false, resize: false,
        });
        roSegments.push({ start: seg.start, end: seg.end, label: seg.label,
                          from_clip: seg.from_clip, shortName, region: r });
      }
    });
    activeSegmentIdx = -1;
    if(existing.length) renderSegments();

    // Auto-seek when clip was opened from the Outliers tab
    if(pendingSeekTime !== null) {
      ws.setTime(pendingSeekTime);
      pendingSeekTime = null;
    }
  });

  ws.on("error", err => {
    if(currentLoadPath !== clipPath) return;
    clearTimeout(loadTimeout); loadTimeout = null;
    setLoadingState("error");
    document.getElementById("status").textContent = "Error: " + err;
  });

  // New region created by drag — add as unlabeled, make it active
  wsRegions.on("region-created", region => {
    if(skipNextRegionCreated) { skipNextRegionCreated = false; return; }
    const seg = { start: region.start, end: region.end, label: selectedLabel, region };
    region.setOptions({ color: hexToRgba(SOURCE_COLORS[selectedLabel]||"#6366f1", 0.35) });
    segments.push(seg);
    setActiveSegment(segments.length - 1);
    renderSegments();
    _newlyCreatedRegion = region;
  });

  // Clicking a region in the waveform selects it
  wsRegions.on("region-clicked", (region, e) => {
    e.stopPropagation();
    if(_newlyCreatedRegion !== null) {
      const guard = _newlyCreatedRegion;
      _newlyCreatedRegion = null;
      if(region !== guard) return;  // spurious click on an old region — ignore
      return;
    }
    const idx = segments.findIndex(s => s.region === region);
    if(idx >= 0) setActiveSegment(idx);
  });

  renderLabelButtons();
}

// ── Active segment tracking ───────────────────────────────────────────────────
let activeSegmentIdx = -1;

function setActiveSegment(idx) {
  activeSegmentIdx = idx;
  // Scroll waveform to region
  if(idx >= 0 && segments[idx]) {
    const seg = segments[idx];
    if(ws) ws.setTime(seg.start);
  }
  renderSegments();
  updateHint();
}

function updateHint() {
  const hint = document.getElementById("label-hint");
  if(activeSegmentIdx >= 0 && segments[activeSegmentIdx]) {
    hint.textContent = `— click a label to assign it to the selected region`;
  } else {
    hint.textContent = `— drag waveform to add a region, then pick a label`;
  }
}

// ── Label buttons ─────────────────────────────────────────────────────────────
function renderLabelButtons() {
  document.getElementById("label-buttons").innerHTML = LABELS.map(l => `
    <button class="label-btn ${l===selectedLabel?"selected":""}"
      style="background:${SOURCE_COLORS[l]||"#6b7280"}22;
             border-color:${SOURCE_COLORS[l]||"#6b7280"};
             color:${SOURCE_COLORS[l]||"#6b7280"}"
      onclick="selectLabel('${l}')">
      ${LABEL_DISPLAY[l]||l.replace(/_/g," ")}
    </button>`).join("");
}

function selectLabel(label) {
  selectedLabel = label;
  // If a segment is active, relabel it
  if(activeSegmentIdx >= 0 && segments[activeSegmentIdx]) {
    const seg = segments[activeSegmentIdx];
    seg.label = label;
    seg.region.setOptions({ color: hexToRgba(SOURCE_COLORS[label]||"#6366f1", 0.35) });
    renderSegments();
  }
  renderLabelButtons();
}

// ── Segments list ─────────────────────────────────────────────────────────────
function makeRegionBadge(num, color) {
  const el = document.createElement("span");
  el.textContent = String(num);
  el.style.cssText = `position:absolute;top:2px;left:2px;font-size:10px;font-weight:700;
    color:#fff;background:${color||"rgba(0,0,0,0.6)"};border-radius:3px;
    padding:1px 4px;pointer-events:none;line-height:1.4;z-index:10`;
  return el;
}

function updateRegionNumbers() {
  segments.forEach((seg, i) => {
    const col = SOURCE_COLORS[seg.label] || "#6366f1";
    seg.region.setOptions({ content: makeRegionBadge(i + 1, col + "cc") });
  });
}

function renderSegments() {
  const el = document.getElementById("segments-list");
  let html = "";

  if(segments.length) {
    html += segments.map((s,i) => `
      <div class="seg-row ${i===activeSegmentIdx?"active":""}" onclick="setActiveSegment(${i})">
        <span style="font-size:.72rem;font-weight:700;color:${SOURCE_COLORS[s.label]||"#6b7280"};
                     min-width:1.2rem;text-align:right;margin-right:.3rem">${i+1}</span>
        <span class="seg-label" style="color:${SOURCE_COLORS[s.label]||"#6b7280"}">
          ${s.label.replace(/_/g," ")}
        </span>
        <span class="seg-time">${s.start.toFixed(1)}s → ${s.end.toFixed(1)}s
          (${(s.end-s.start).toFixed(1)}s)</span>
        <button class="seg-del" onclick="event.stopPropagation();deleteSegment(${i})">✕</button>
      </div>`).join("");
  }

  updateRegionNumbers();

  if(roSegments.length) {
    html += `<div style="font-size:.72rem;color:var(--muted);margin:.6rem 0 .3rem;
                         padding-top:.5rem;border-top:1px solid var(--border)">
               From overlapping clips (read-only)</div>`;
    html += roSegments.map(s => `
      <div class="seg-row" style="opacity:.6;cursor:default"
           onclick="if(ws)ws.setTime(${s.start})">
        <span class="seg-label" style="color:${SOURCE_COLORS[s.label]||"#6b7280"}">
          ${s.label.replace(/_/g," ")}
        </span>
        <span class="seg-time">${s.start.toFixed(1)}s → ${s.end.toFixed(1)}s</span>
        <span style="font-size:.7rem;color:var(--muted);margin-left:auto">${s.shortName}</span>
      </div>`).join("");
  }

  el.innerHTML = html;
  if(!segments.length && !roSegments.length) updateHint();
}

function deleteSegment(idx) {
  segments[idx].region.remove();
  segments.splice(idx, 1);
  if(activeSegmentIdx >= segments.length) activeSegmentIdx = segments.length - 1;
  renderSegments();
}

function clearSegments() {
  segments.forEach(s => s.region.remove());
  roSegments.forEach(s => s.region.remove());
  segments = [];
  roSegments = [];
  activeSegmentIdx = -1;
  renderSegments();
}

// ── Save ──────────────────────────────────────────────────────────────────────
async function saveSegments() {
  if(!currentClip) return;
  if(!segments.length) {
    document.getElementById("status").textContent = "No segments to save — add at least one region.";
    return;
  }

  const payload = {
    clip_path: currentClip,
    segments: segments.map(s => ({
      start: parseFloat(s.start.toFixed(3)),
      end:   parseFloat(s.end.toFixed(3)),
      label: s.label,
    }))
  };

  const resp = await fetch("/api/segments", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const result = await resp.json();

  document.getElementById("status").textContent =
    `✓ Saved ${result.saved} segments`;

  clearSegments();
  if(currentTab === "outliers") {
    const curIdx = outlierItems.findIndex(i => `${i.clip}:${i.t_start}` === currentOutlierKey);
    const next   = outlierItems[curIdx + 1];
    if(next) loadOutlierClip(next.clip, next.t_start);
  } else {
    const nextIdx = clips.findIndex(c => c.clip === currentClip) + 1;
    loadClipList();
    if(nextIdx < clips.length) loadClip(clips[nextIdx].clip, nextIdx);
  }
}

async function nextClip() {
  if(segments.length) {
    await saveSegments();  // saves and advances automatically
  } else if(currentTab === "outliers") {
    const curIdx = outlierItems.findIndex(i => `${i.clip}:${i.t_start}` === currentOutlierKey);
    const next   = outlierItems[curIdx + 1];
    if(next) loadOutlierClip(next.clip, next.t_start);
  } else {
    const idx = clips.findIndex(c => c.clip === currentClip);
    if(idx >= 0 && idx < clips.length - 1) loadClip(clips[idx+1].clip, idx+1);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  // Don't fire when typing in an input
  if(e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if(!ws) return;

  const key = e.key.toLowerCase();

  // Arrow keys: jump ±10s
  if(e.key === "ArrowLeft") {
    e.preventDefault();
    ws.setTime(Math.max(0, ws.getCurrentTime() - 5));
    return;
  }
  if(e.key === "ArrowRight") {
    e.preventDefault();
    ws.setTime(Math.min(ws.getDuration(), ws.getCurrentTime() + 5));
    return;
  }

  // Space: if a label region is in progress, close it and stop; otherwise play/pause
  if(e.key === " ") {
    e.preventDefault();
    if(regionStartTime !== null) {
      const start = regionStartTime;
      const end   = ws.getCurrentTime();
      regionStartTime = null;
      document.getElementById("waveform-status-bar").textContent = "";
      if(end > start + 0.1) {
        skipNextRegionCreated = true;
        const r = wsRegions.addRegion({
          start, end,
          color: hexToRgba(SOURCE_COLORS[selectedLabel]||"#6366f1", 0.35),
          drag: true, resize: true,
        });
        const seg = { start, end, label: selectedLabel, region: r };
        segments.push(seg);
        setActiveSegment(segments.length - 1);
        renderSegments();
      }
    } else {
      ws.playPause();   // space with no region in progress = play/pause
    }
    return;
  }

  // Label keys: mark start of a new region and begin playing.
  // Deactivate any current segment first — we're starting fresh, not relabelling.
  if(LABEL_SHORTCUTS[key]) {
    e.preventDefault();
    activeSegmentIdx = -1;
    selectLabel(LABEL_SHORTCUTS[key]);
    regionStartTime = ws.getCurrentTime();
    document.getElementById("waveform-status-bar").textContent =
      `Recording "${LABEL_DISPLAY[selectedLabel]||selectedLabel}" from ${regionStartTime.toFixed(1)}s — press Space to end`;
    if(!ws.isPlaying()) ws.play();
    return;
  }
});

// ── Outliers tab ──────────────────────────────────────────────────────────────
let outlierPollTimer = null;

async function computeOutliers() {
  await fetch("/api/outliers/compute", {method:"POST"});
  startOutlierPolling();
}

function startOutlierPolling() {
  if(outlierPollTimer) clearInterval(outlierPollTimer);
  outlierPollTimer = setInterval(pollOutliers, 2000);
  pollOutliers();
}

async function pollOutliers() {
  const data = await fetch("/api/outliers").then(r=>r.json());
  if(data.status === "ready" || data.status === "error" || data.status === "idle") {
    if(outlierPollTimer) { clearInterval(outlierPollTimer); outlierPollTimer = null; }
  }
  renderOutlierPanel(data);
}

async function loadOutliers() {
  const data = await fetch("/api/outliers").then(r=>r.json());
  renderOutlierPanel(data);
  if(data.status === "computing") startOutlierPolling();
}

function renderOutlierPanel(data) {
  const statusEl = document.getElementById("outlier-status");
  const spinEl   = document.getElementById("outlier-spinner");
  const listEl   = document.getElementById("outlier-list");
  if(!statusEl) return;

  spinEl.style.display = data.status === "computing" ? "block" : "none";

  if(data.status === "idle") {
    statusEl.textContent = "Not computed yet";
    listEl.innerHTML = "";
    return;
  }
  if(data.status === "computing") {
    statusEl.textContent = "Computing… (~30s)";
    return;
  }
  if(data.status === "error") {
    statusEl.textContent = "Error: " + (data.error || "unknown");
    listEl.innerHTML = "";
    return;
  }

  // ready — flat list, worst first. ✗ misclassified on top, ✓ low-confidence-but-correct dimmed below.
  const wrong   = data.items.filter(i => i.true_label !== i.predicted);  // already sorted asc by conf
  const correct = data.items.filter(i => i.true_label === i.predicted);
  outlierItems = wrong;  // store for Next → navigation

  const at = data.computed_at
    ? new Date(data.computed_at * 1000).toLocaleTimeString()
    : "";
  statusEl.textContent = `${wrong.length} misclassified · CV ${data.accuracy}% · ${at}`;

  if(!data.items.length) {
    listEl.innerHTML = "<div style='color:var(--green);font-size:.85rem;padding:.5rem 0'>No outliers found.</div>";
    return;
  }

  function renderRow(item, dimmed) {
    const confPct    = Math.round(item.confidence * 100);
    const isWrong    = item.true_label !== item.predicted;
    const badgeColor = dimmed ? "var(--muted)" : confPct < 30 ? "var(--red)" : "var(--yellow)";
    const clipShort  = item.clip.replace(/^\d{8}_\d{6}_(.+)\.wav$/, "$1").slice(0, 24);
    const meta       = isWrong
      ? `${item.true_label} · pred: ${item.predicted} · t=${item.t_start.toFixed(1)}–${item.t_end.toFixed(1)}s`
      : `${item.true_label} · low confidence · t=${item.t_start.toFixed(1)}–${item.t_end.toFixed(1)}s`;
    return `<div class="clip-item" data-okey="${item.clip}:${item.t_start}" style="${dimmed ? 'opacity:.4' : ''}"
                 onclick="loadOutlierClip('${item.clip}', ${item.t_start})">
      <div style="display:flex;align-items:center;gap:.4rem">
        <span style="color:${badgeColor};font-weight:700;font-size:.78rem;min-width:2.2rem">${confPct}%</span>
        <span style="color:${dimmed?'var(--muted)':'var(--red)'};font-size:.85rem">${isWrong?'✗':'✓'}</span>
        <span class="clip-name" style="font-size:.78rem">${clipShort}</span>
      </div>
      <div class="clip-meta">${meta}</div>
    </div>`;
  }

  let html = wrong.map(i => renderRow(i, false)).join("");
  if(correct.length) {
    html += `<div style="font-size:.72rem;color:var(--muted);padding:.6rem .25rem .2rem;
                         border-top:1px solid var(--border);margin-top:.4rem">
               Low confidence but correctly classified — review if time
             </div>`;
    html += correct.map(i => renderRow(i, true)).join("");
  }
  listEl.innerHTML = html;
}

function loadOutlierClip(clipPath, tStart) {
  pendingSeekTime = tStart;
  currentOutlierKey = `${clipPath}:${tStart}`;
  loadClip(clipPath, -1, '', 0);
  // loadClip clears all .clip-item active states synchronously — re-highlight this outlier row
  document.querySelectorAll("[data-okey]").forEach(el => {
    el.classList.toggle("active", el.dataset.okey === currentOutlierKey);
  });
}

// ── Retrain classifier ────────────────────────────────────────────────────────
let retrainPollTimer = null;

async function retrainClassifier() {
  if(!confirm("Retrain classifier? This re-extracts features and retrains the model (~3-5 min on RPi). Continue?")) return;
  const el = document.getElementById("retrain-status");
  el.style.display = "block";
  el.style.color = "var(--muted)";
  el.textContent = "Starting retrain…";
  await fetch("/api/retrain", {method: "POST"});
  startRetrainPolling();
}

function startRetrainPolling() {
  if(retrainPollTimer) clearInterval(retrainPollTimer);
  retrainPollTimer = setInterval(checkRetrainStatus, 3000);
  checkRetrainStatus();
}

async function checkRetrainStatus() {
  const data = await fetch("/api/retrain/status").then(r => r.json());
  const el = document.getElementById("retrain-status");
  if(!el) return;
  el.style.display = "block";
  if(data.status === "idle") {
    el.style.display = "none";
  } else if(data.status === "running") {
    el.style.color = "var(--muted)";
    el.textContent = data.step || "Running…";
  } else if(data.status === "done") {
    el.style.color = "var(--green)";
    const at = data.completed_at ? new Date(data.completed_at * 1000).toLocaleTimeString() : "";
    el.textContent = `Retrain complete ${at} — click Compute Outliers to refresh`;
    if(retrainPollTimer) { clearInterval(retrainPollTimer); retrainPollTimer = null; }
  } else if(data.status === "error") {
    el.style.color = "var(--red)";
    el.textContent = "Retrain failed: " + (data.error || "unknown error");
    if(retrainPollTimer) { clearInterval(retrainPollTimer); retrainPollTimer = null; }
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadClipList();
</script>
</body>
</html>
"""

REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoiseMon — Review Samples</title>
<style>
  :root{--bg:#0f1117;--panel:#1a1d27;--border:#2a2d3e;
        --text:#e2e8f0;--muted:#6b7280;--accent:#6366f1;
        --green:#22c55e;--red:#ef4444}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}
  header{background:var(--panel);border-bottom:1px solid var(--border);
         padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.4rem;font-weight:700}
  a.back{color:var(--muted);text-decoration:none;font-size:.9rem}
  a.back:hover{color:var(--text)}
  .layout{display:grid;grid-template-columns:220px 1fr;height:calc(100vh - 57px)}
  .sidebar{background:var(--panel);border-right:1px solid var(--border);overflow-y:auto}
  .sidebar-hdr{padding:.75rem 1rem;border-bottom:1px solid var(--border);
               font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
  .cls-item{padding:.65rem 1rem;border-bottom:1px solid var(--border);cursor:pointer;
            display:flex;justify-content:space-between;align-items:center;transition:background .15s}
  .cls-item:hover{background:#252838}
  .cls-item.active{background:#252838;border-left:3px solid var(--accent)}
  .cls-name{font-size:.9rem;text-transform:capitalize}
  .cls-cnt{font-size:.78rem;color:var(--muted);background:#1a1d27;padding:.1rem .45rem;border-radius:10px}
  .main{display:flex;flex-direction:column;padding:1.5rem;gap:1.25rem;overflow-y:auto;max-width:700px}
  #no-class{color:var(--muted);padding:2rem;text-align:center}
  .prog-bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden}
  .prog-fill{height:100%;background:var(--accent);transition:width .3s}
  .seg-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1.25rem;
            display:flex;flex-direction:column;gap:1rem}
  .seg-header{display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
  .seg-clip{font-size:.85rem;word-break:break-all;color:var(--text)}
  .seg-time{font-size:.8rem;color:var(--muted);white-space:nowrap}
  .audio-row{display:flex;gap:.75rem;align-items:center}
  audio{flex:1;height:36px;min-width:0}
  audio::-webkit-media-controls-panel{background:#252838}
  .seg-links{font-size:.78rem;color:var(--muted)}
  .seg-links a{color:var(--accent);text-decoration:none}
  .seg-links a:hover{text-decoration:underline}
  .vote-row{display:flex;gap:1rem;justify-content:center}
  .vbtn{padding:.75rem 2.5rem;border-radius:12px;border:2px solid;font-size:1rem;
        font-weight:700;cursor:pointer;transition:all .15s;background:transparent}
  .vbtn.keep{border-color:var(--green);color:var(--green)}
  .vbtn.keep:hover{background:#22c55e22}
  .vbtn.remove{border-color:var(--red);color:var(--red)}
  .vbtn.remove:hover{background:#ef444422}
  .vbtn:active{transform:scale(.97)}
  button.plain{background:#252838;color:var(--text);border:1px solid var(--border);
               border-radius:8px;padding:.4rem .85rem;font-size:.85rem;cursor:pointer}
  button.plain:hover{background:#2d3148}
  .hint{font-size:.78rem;color:var(--muted);text-align:center}
  .done-box{text-align:center;padding:2rem;color:var(--muted)}
  .done-box .big{font-size:2.5rem;margin-bottom:.5rem}
</style>
</head>
<body>
<header>
  <a class="back" href="/">← Dashboard</a>
  <h1>🔍 Review Samples</h1>
  <span id="session-stats" style="margin-left:auto;color:var(--muted);font-size:.85rem"></span>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-hdr">Classes</div>
    <div id="cls-list"></div>
  </div>
  <div class="main">
    <div id="no-class">← Pick a class to review its training samples</div>
    <div id="reviewer" style="display:none">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span id="prog-label" style="font-size:.85rem;color:var(--muted)"></span>
        <button class="plain" onclick="skipSegment()">Skip →</button>
      </div>
      <div class="prog-bar"><div class="prog-fill" id="prog-fill"></div></div>

      <div class="seg-card">
        <div class="seg-header">
          <span class="seg-clip" id="seg-clip">—</span>
          <span class="seg-time" id="seg-time"></span>
        </div>
        <div class="audio-row">
          <audio id="rev-audio" controls preload="auto"></audio>
        </div>
        <div class="seg-links">
          <a id="label-link" href="#" target="_blank">Open in label UI →</a>
        </div>
      </div>

      <div class="vote-row">
        <button class="vbtn keep"   onclick="keepSegment()">✓ Keep <span style="font-size:.75rem;opacity:.7">(Y)</span></button>
        <button class="vbtn remove" onclick="removeSegment()">✕ Remove <span style="font-size:.75rem;opacity:.7">(N)</span></button>
      </div>
      <div class="hint">Y = keep · N = remove · Space = replay · → = skip</div>

      <div id="done-box" style="display:none" class="done-box">
        <div class="big">✓</div>
        <div id="done-text"></div>
        <button class="plain" style="margin-top:1rem" onclick="selectClass(currentClass)">Reload</button>
      </div>
    </div>
  </div>
</div>

<script>
const SOURCE_COLORS = {
  aircraft:"#818cf8", leaf_blower:"#fb923c", pickleball:"#34d399",
  road_traffic:"#94a3b8", lawn_mower:"#fbbf24", dog_barking:"#f87171",
  music:"#e879f9", voices:"#67e8f9", crows:"#86efac", birds:"#4ade80",
  owl:"#a5b4fc", strimmer:"#f59e0b", unknown:"#6b7280", other:"#a78bfa"
};

let currentClass = null;
let segments     = [];
let idx          = 0;
let sessionKept  = 0, sessionRemoved = 0;
let stopAtEnd    = null;  // timeupdate handler reference

const audio = document.getElementById("rev-audio");

// ── Audio playback ────────────────────────────────────────────────────────────
function playSegment(seg) {
  if(stopAtEnd) { audio.removeEventListener("timeupdate", stopAtEnd); stopAtEnd = null; }

  const newSrc = `/clips/${seg.clip}`;
  if(audio.src !== location.origin + newSrc) {
    audio.src = newSrc;
  }

  audio.addEventListener("canplay", function seek() {
    audio.removeEventListener("canplay", seek);
    audio.currentTime = seg.start;
    audio.play().catch(()=>{});
  }, {once: true});

  stopAtEnd = () => {
    if(audio.currentTime >= seg.end) {
      audio.pause();
      audio.removeEventListener("timeupdate", stopAtEnd);
      stopAtEnd = null;
    }
  };
  audio.addEventListener("timeupdate", stopAtEnd);

  // If audio was already loaded (same clip), seek immediately
  if(audio.readyState >= 2) {
    audio.currentTime = seg.start;
    audio.play().catch(()=>{});
  }
}

function replaySegment() {
  if(idx >= segments.length) return;
  const seg = segments[idx];
  if(stopAtEnd) { audio.removeEventListener("timeupdate", stopAtEnd); stopAtEnd = null; }
  audio.currentTime = seg.start;
  audio.play().catch(()=>{});
  stopAtEnd = () => {
    if(audio.currentTime >= seg.end) {
      audio.pause();
      audio.removeEventListener("timeupdate", stopAtEnd);
      stopAtEnd = null;
    }
  };
  audio.addEventListener("timeupdate", stopAtEnd);
}

// ── Classes ───────────────────────────────────────────────────────────────────
async function loadClasses() {
  const data = await fetch("/api/review/classes").then(r=>r.json());
  const el = document.getElementById("cls-list");
  if(!data.length) { el.innerHTML="<div style='padding:1rem;color:var(--muted)'>No labelled segments.</div>"; return; }
  el.innerHTML = data.map(c=>`
    <div class="cls-item" id="cls-${c.label}" onclick="selectClass('${c.label}')">
      <span class="cls-name" style="color:${SOURCE_COLORS[c.label]||'var(--text)'}">
        ${c.label.replace(/_/g," ")}
      </span>
      <span class="cls-cnt" id="cnt-${c.label}">${c.n}</span>
    </div>`).join("");
}

async function selectClass(label) {
  if(stopAtEnd) { audio.removeEventListener("timeupdate", stopAtEnd); stopAtEnd = null; }
  audio.pause();

  currentClass = label;
  sessionKept = 0; sessionRemoved = 0;
  document.querySelectorAll(".cls-item").forEach(e=>e.classList.remove("active"));
  document.getElementById("cls-"+label)?.classList.add("active");
  document.getElementById("done-box").style.display = "none";
  document.getElementById("no-class").style.display = "none";
  document.getElementById("reviewer").style.display = "block";

  segments = await fetch(`/api/review/segments/${encodeURIComponent(label)}`).then(r=>r.json());
  idx = 0;
  showSegment();
}

// ── Segment display ───────────────────────────────────────────────────────────
function showSegment() {
  if(idx >= segments.length) { showDone(); return; }
  const seg = segments[idx];
  updateProgress();
  document.getElementById("seg-clip").textContent = seg.clip;
  document.getElementById("seg-time").textContent =
    `t=${seg.start.toFixed(1)}–${seg.end.toFixed(1)}s  (${(seg.end-seg.start).toFixed(1)}s)`;
  document.getElementById("label-link").href = `/label`;
  playSegment(seg);
}

function updateProgress() {
  document.getElementById("prog-fill").style.width =
    segments.length ? (idx / segments.length * 100) + "%" : "0%";
  document.getElementById("prog-label").textContent =
    `${idx+1} / ${segments.length}  ·  ${currentClass.replace(/_/g," ")}`;
  document.getElementById("session-stats").textContent =
    (sessionKept||sessionRemoved) ? `kept ${sessionKept} · removed ${sessionRemoved}` : "";
}

// ── Vote actions ──────────────────────────────────────────────────────────────
function keepSegment()   { sessionKept++;   idx++; showSegment(); }
function skipSegment()   {                  idx++; showSegment(); }

async function removeSegment() {
  const seg = segments[idx];
  await fetch(`/api/review/segment/${seg.id}`, {method:"DELETE"});
  sessionRemoved++;
  const cnt = document.getElementById(`cnt-${currentClass}`);
  if(cnt) cnt.textContent = Math.max(0, parseInt(cnt.textContent)-1);
  segments.splice(idx, 1);
  showSegment();
}

function showDone() {
  audio.pause();
  document.getElementById("done-box").style.display = "block";
  document.getElementById("done-text").textContent =
    `All ${currentClass.replace(/_/g," ")} samples reviewed — ` +
    `kept ${sessionKept}, removed ${sessionRemoved}.`;
  document.getElementById("prog-fill").style.width  = "100%";
  document.getElementById("prog-label").textContent = "Done";
}

// ── Keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA") return;
  if(document.getElementById("reviewer").style.display==="none") return;
  if(e.key.toLowerCase()==="y")      { e.preventDefault(); keepSegment(); }
  else if(e.key.toLowerCase()==="n") { e.preventDefault(); removeSegment(); }
  else if(e.key===" ")               { e.preventDefault(); replaySegment(); }
  else if(e.key==="ArrowRight")      { e.preventDefault(); skipSegment(); }
});

loadClasses();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
