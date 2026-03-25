#!/usr/bin/env python3
"""
NoiseMon Daemon — YAMNet edition with audio clip saving
Records audio, measures dB(A), classifies noise sources using
Google's YAMNet model (521 AudioSet classes) via ai-edge-litert.
Saves WAV clips of confirmed events for later investigation.
"""

import numpy as np
import sounddevice as sd
import sqlite3, time, threading, logging, os, sys, csv, json
import soundfile as sf
from collections import deque, Counter
from datetime import datetime, timedelta
from pathlib import Path
import scipy.signal, scipy.fft

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH           = "/var/lib/noisemon/noise.db"
LOG_PATH          = "/var/log/noisemon/noisemon.log"
CLIPS_DIR         = "/var/lib/noisemon/clips"
SAMPLE_RATE       = 48000
YAMNET_SR         = 16000
BLOCK_DURATION    = 1.0
CHANNELS          = 1
DEVICE_INDEX      = 1              # UMIK-1
CALIBRATION_DB    = 125.5
CAL_FILE          = "/opt/noisemon/models/umik1_cal_90.txt"
CLASSIFY_INTERVAL = 5
HISTORY_SECONDS   = 20             # YAMNet inference window
PRE_ROLL_SECONDS  = 30             # audio saved BEFORE the event
POST_ROLL_SECONDS = 90             # audio saved AFTER the event
CLIP_RETAIN_DAYS         = 30      # delete WAV clips older than this
MEASUREMENTS_RETAIN_DAYS = 90      # delete measurement rows older than this
PATTERN_WINDOW    = 60
PATTERN_MIN_HITS  = 2
CONFIDENCE_THRESH = 0.20
CATEGORY_THRESH   = {"voices": 0.25, "aircraft": 0.04, "leaf_blower": 0.08, "strimmer": 0.08}  # per-category overrides
YAMNET_GAIN       = 32.0               # fixed pre-gain; ~75 dB(A) → ~0.1 RMS input
MODEL_PATH        = "/opt/noisemon/models/yamnet.tflite"
LABELS_PATH       = "/opt/noisemon/models/yamnet_class_map.csv"
CLASSIFIER_PATH   = "/opt/noisemon/models/classifier.joblib"
TARGET_RMS        = 0.05               # target RMS for saved clips
MAX_GAIN_CLIP     = 100.0              # max gain multiplier for clip normalisation
DEDUP_WINDOW      = {                  # min seconds between events per source
    "aircraft":    300,                # 5 min — one event per real flyover; limits constant-background loops
    "leaf_blower": 300,
    "lawn_mower":  300,
    "strimmer":    300,
}
DEDUP_DEFAULT     = 60                 # fallback for sources not in DEDUP_WINDOW
CATEGORY_MIN_DB   = {                  # minimum db_avg to log an event (filters quiet false positives)
    "aircraft":    44.0,
}
SUPPRESS_CATEGORIES = {"birds", "crows", "owl", "pool_pump"}  # detected internally but not logged as events
SUSTAINED_BASELINE_SECS = 300         # 5-min rolling ambient baseline
SUSTAINED_ENERGY_ABOVE  = 3.0         # dB above ambient to count as elevated
# ── ADS-B ─────────────────────────────────────────────────────────────────────
try:
    from config import ADSB_LAT, ADSB_LON
except ImportError:
    raise SystemExit("config.py not found — copy config.example.py to config.py and set your location")
ADSB_RADIUS_NM     = 10               # search radius in nautical miles
ADSB_POLL_INTERVAL = 30               # seconds between API polls
ADSB_BOOST_HITS    = 2                # extra detection_hist hits when ADS-B confirms aircraft
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_MAP = [
    ("aircraft",     ["aircraft", "airplane", "jet", "helicopter", "propeller",
                      "fixed-wing", "aircraft engine", "foghorn",
                      "boat, water vehicle", "motorboat", "motorboat, speedboat"]),
    ("leaf_blower",  ["leaf blower", "blower", "vacuum cleaner", "hair dryer",
                      "mechanical fan"]),
    ("lawn_mower",   ["lawn mower", "mower", "chainsaw", "power tool",
                      "electric motor", "engine", "lawn", "mechanical"]),
    ("pickleball",   ["tennis", "racquet", "ball", "bouncing", "ping",
                      "whack", "crack", "pop"]),
    ("road_traffic", ["traffic", "car", "vehicle", "truck", "bus", "motorcycle",
                      "motor vehicle", "accelerating", "skidding", "horn",
                      "beep", "road", "highway"]),
    ("dog_barking",  ["dog", "bark", "howl", "whimper", "yelp",
                      "domestic animals"]),
    ("music",        ["music", "singing", "song", "guitar", "drum", "bass",
                      "beat", "piano", "instrument"]),
    ("voices",       ["speech", "conversation", "talk", "shout", "yell",
                      "crowd", "cheer", "laughter", "children",
                      "narration", "monologue"]),
    ("strimmer",     ["grass cutting", "trimmer", "string trimmer", "weed",
                      "whirring", "buzzing"]),
    ("owl",          ["owl", "hoot"]),
    ("crows",        ["crow", "caw", "raven", "jackdaw", "magpie"]),
    ("birds",        ["bird", "bird vocalization", "bird call", "bird song",
                      "chirp", "tweet", "duck", "quack",
                      "goose", "honk", "fowl", "animal", "wild animals"]),
]

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger("noisemon")


# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS measurements (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        INTEGER NOT NULL,
            db_avg    REAL NOT NULL,
            db_peak   REAL NOT NULL,
            db_min    REAL NOT NULL,
            freq_low  REAL,
            freq_mid  REAL,
            freq_high REAL
        );
        CREATE INDEX IF NOT EXISTS idx_ts ON measurements(ts);

        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start   INTEGER NOT NULL,
            ts_end     INTEGER,
            source     TEXT NOT NULL,
            confidence REAL NOT NULL,
            db_avg     REAL NOT NULL,
            clip_path  TEXT,
            adsb_json  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_evt ON events(ts_start);
    """)
    # Migration: add adsb_json to existing DBs
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    if "adsb_json" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN adsb_json TEXT")
        log.info("DB migration: added adsb_json column to events")
    conn.commit(); conn.close()
    log.info("DB ready: %s", DB_PATH)

def get_conn():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c


# ── Audio helpers ─────────────────────────────────────────────────────────────
def rms_db(rms):
    return 20.0 * np.log10(max(rms, 1e-12)) + CALIBRATION_DB

def a_weight(freqs):
    f2 = freqs**2; f4 = freqs**4
    num = 12194.0**2 * f4
    den = ((f2+20.6**2)*np.sqrt((f2+107.7**2)*(f2+737.9**2))*(f2+12194.0**2))
    return np.where(den>0, num/den, 0.0)

def load_cal_curve(path: str):
    """
    Load UMIK-1 calibration file.  Returns (interp_fn, sens_factor_dB) where
    interp_fn gives the frequency-response dB correction for any array of Hz,
    and sens_factor_dB is the unit-specific overall sensitivity offset.
    """
    freqs, offsets = [], []
    sens_factor = 0.0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip().strip('"')
                if not line or line.startswith("Auto"):
                    continue
                if line.startswith("Sens"):
                    try:
                        sens_factor = float(line.split("=")[1].split("d")[0].strip())
                        log.info("Cal Sens Factor: %.3f dB", sens_factor)
                    except Exception:
                        pass
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        freqs.append(float(parts[0]))
                        offsets.append(float(parts[1]))
                    except ValueError:
                        continue
        log.info("Loaded calibration curve: %d points, %.1f–%.1f Hz",
                 len(freqs), freqs[0], freqs[-1])
    except Exception as e:
        log.warning("Could not load cal file %s: %s — using flat response", path, e)
        return lambda f: np.zeros_like(f, dtype=np.float64), 0.0

    freqs   = np.array(freqs)
    offsets = np.array(offsets)

    def interp(f):
        """Return dB correction for frequency array f."""
        return np.interp(f, freqs, offsets, left=offsets[0], right=offsets[-1])

    return interp, sens_factor

# Load calibration curve at module level; apply Sens Factor to reference offset
cal_curve, _cal_sens = load_cal_curve(CAL_FILE)
CALIBRATION_DB -= _cal_sens
log.info("Effective CALIBRATION_DB after Sens Factor (%.3f dB): %.3f", _cal_sens, CALIBRATION_DB)

def band_frac(fft_mag, freqs, lo, hi):
    m = (freqs>=lo)&(freqs<=hi)
    t = np.sum(fft_mag**2)
    return float(np.sum(fft_mag[m]**2)/t) if t>0 else 0.0

def resample_to_16k(audio: np.ndarray) -> np.ndarray:
    target_len = int(len(audio) * YAMNET_SR / SAMPLE_RATE)
    return scipy.signal.resample(audio, target_len).astype(np.float32)


# ── Clip saving ───────────────────────────────────────────────────────────────

def save_clip(audio: np.ndarray, ts: int, source: str) -> str:
    """Save a normalised WAV clip and return the filename."""
    dt = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
    filename = f"{dt}_{source}.wav"
    filepath = os.path.join(CLIPS_DIR, filename)

    # Normalise to target RMS, capped at MAX_GAIN_CLIP to avoid amplifying
    # near-silent clips into pure noise.
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        gain = min(TARGET_RMS / rms, MAX_GAIN_CLIP)
        audio = audio * gain
        # Hard-limit to ±1 after gain in case of brief transient peaks
        audio = np.clip(audio, -1.0, 1.0)

    sf.write(filepath, audio, SAMPLE_RATE, subtype="PCM_16")
    log.info("Saved clip: %s (%.1fs)", filename, len(audio)/SAMPLE_RATE)
    return filename

def purge_old_clips():
    """Delete WAV clips older than CLIP_RETAIN_DAYS."""
    cutoff = time.time() - (CLIP_RETAIN_DAYS * 86400)
    count = 0
    for f in Path(CLIPS_DIR).glob("*.wav"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            count += 1
    if count:
        log.info("Purged %d old clips", count)

def purge_old_measurements():
    """Delete measurement rows older than MEASUREMENTS_RETAIN_DAYS and VACUUM."""
    cutoff = int(time.time() - (MEASUREMENTS_RETAIN_DAYS * 86400))
    with get_conn() as c:
        deleted = c.execute(
            "DELETE FROM measurements WHERE ts < ?", (cutoff,)
        ).rowcount
    if deleted:
        # VACUUM reclaims the freed pages; run outside a transaction
        conn = sqlite3.connect(DB_PATH)
        conn.execute("VACUUM")
        conn.close()
        log.info("Purged %d measurement rows older than %d days; DB vacuumed",
                 deleted, MEASUREMENTS_RETAIN_DAYS)

def _housekeeping_loop():
    """Run daily purges at midnight-ish."""
    while True:
        # Sleep until next midnight + a few minutes to avoid exact midnight spikes
        now = time.time()
        tomorrow = (now // 86400 + 1) * 86400 + 300   # 00:05 next day
        time.sleep(tomorrow - now)
        try:
            purge_old_clips()
            purge_old_measurements()
        except Exception as e:
            log.warning("Housekeeping error: %s", e)


# ── YAMNet classifier ─────────────────────────────────────────────────────────
class YAMNetClassifier:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        log.info("Loading YAMNet model from %s", MODEL_PATH)
        self.interpreter = Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = self._load_labels()
        self.class_to_category = self._build_class_map()
        log.info("YAMNet ready — %d classes mapped to %d categories",
                 len(self.labels), len(CATEGORY_MAP))

    def _load_labels(self):
        labels = {}
        with open(LABELS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                labels[int(row["index"])] = row["display_name"].lower()
        return labels

    def _build_class_map(self):
        mapping = {}
        for category, keywords in CATEGORY_MAP:
            for idx, name in self.labels.items():
                if any(kw.lower() in name for kw in keywords):
                    mapping[idx] = category
        return mapping

    def infer(self, audio_16k: np.ndarray):
        inp    = audio_16k.flatten().astype(np.float32)
        target = 15600

        # Run inference on up to 4 non-overlapping 1s windows from the tail of the
        # buffer and take the max score per class.  This catches transient events
        # (e.g. aircraft flyovers) that may only dominate one window.
        windows = []
        start = max(0, len(inp) - target * 4)
        while start + target <= len(inp):
            windows.append(inp[start:start + target])
            start += target
        if not windows:
            windows.append(np.pad(inp, (0, target - len(inp))) if len(inp) < target else inp[-target:])

        all_scores = []
        for w in windows:
            self.interpreter.set_tensor(self.input_details[0]["index"], w)
            self.interpreter.invoke()
            all_scores.append(
                self.interpreter.get_tensor(self.output_details[0]["index"]).squeeze().copy()
            )
        all_scores_arr = np.array(all_scores)   # (N_windows, 521)
        scores = np.max(all_scores_arr, axis=0)

        category_scores = {}
        for idx, score in enumerate(scores):
            cat = self.class_to_category.get(idx)
            if cat:
                if cat not in category_scores or score > category_scores[cat]:
                    category_scores[cat] = float(score)

        # Debug: log top 10 raw YAMNet classes
        top10_idx = np.argsort(scores)[-10:][::-1]
        top10 = [(self.labels[i], float(scores[i])) for i in top10_idx]
        log.debug("RAW top10: %s", ", ".join(f"{n}={s:.3f}" for n,s in top10))

        return sorted(category_scores.items(), key=lambda x: x[1], reverse=True), all_scores_arr


# ── Sustained-source detector ─────────────────────────────────────────────────
class SustainedSourceDetector:
    """
    Signal-processing detector for long-duration sources (aircraft, leaf blower,
    lawn mower) that YAMNet misclassifies because its 1-second windows miss the
    gradual onset / extended duration.

    Feeds on every 1-second block; call detect() every CLASSIFY_INTERVAL seconds.
    Returns a list of (category, confidence) for any sustained event found.
    """

    # Minimum consecutive elevated seconds to confirm a source
    MIN_DUR = {"aircraft": 15, "leaf_blower": 20, "lawn_mower": 20}

    def __init__(self):
        self._lock     = threading.Lock()
        self._baseline = deque(maxlen=SUSTAINED_BASELINE_SECS)
        self._recent   = deque(maxlen=120)   # last 2 minutes: (ts, db, fl, fm, fh)

    def feed(self, ts: int, db: float, fl: float, fm: float, fh: float):
        with self._lock:
            self._baseline.append(db)
            self._recent.append((ts, db, fl, fm, fh))

    def get_ambient(self) -> float | None:
        """Return current ambient estimate (10th-percentile of rolling baseline).
        Returns None if fewer than 60 seconds of data have accumulated."""
        with self._lock:
            if len(self._baseline) < 60:
                return None
            return float(np.percentile(list(self._baseline), 10))

    def detect(self, window_seconds: int = 30):
        """
        Analyse the last `window_seconds` of data.
        Returns list of (category, confidence 0-1).
        """
        with self._lock:
            if len(self._baseline) < 60:   # need ≥1 min before baseline is reliable
                return []

            ambient = float(np.percentile(list(self._baseline), 10))
            now     = time.time()
            cutoff  = now - window_seconds
            recent  = [(ts, db, fl, fm, fh)
                       for ts, db, fl, fm, fh in self._recent if ts >= cutoff]

        if len(recent) < 10:
            return []

        # Samples where energy is meaningfully above ambient
        elevated = [(ts, db, fl, fm, fh) for ts, db, fl, fm, fh in recent
                    if db >= ambient + SUSTAINED_ENERGY_ABOVE]

        if len(elevated) < min(self.MIN_DUR.values()):
            return []

        n_elev  = len(elevated)
        avg_db  = float(np.mean([db for _, db, *_ in elevated]))
        avg_fl  = float(np.mean([fl for _, _, fl, fm, fh in elevated]))
        avg_fm  = float(np.mean([fm for _, _, fl, fm, fh in elevated]))
        avg_fh  = float(np.mean([fh for _, _, fl, fm, fh in elevated]))
        db_excess = avg_db - ambient

        results = []

        # ── Aircraft ──────────────────────────────────────────────────────────
        # Engine drone is low-frequency; after A-weighting it raises freq_low
        # and shifts the spectral centroid toward the low-mid range.
        # We look for: elevated duration ≥ 15 s, freq_low pulled up above ambient,
        # and freq_mid dominant (engine harmonics sit in 200-2000 Hz A-weighted).
        # Also check for amplitude ramp-up (approach), computed over all recent
        # samples regardless of elevation threshold.
        if n_elev >= self.MIN_DUR["aircraft"]:
            # Spectral centroid score: aircraft engine boosts low-mid vs high
            centroid_score = avg_fm / max(avg_fh, 0.01)   # higher → more low-mid energy
            # Duration score: 0 at 15 s, 1 at 60 s
            dur_score = min(1.0, (n_elev - self.MIN_DUR["aircraft"]) / 45.0)

            # Amplitude slope over the elevated window (dB/s) —
            # positive slope = approaching aircraft
            if len(elevated) >= 4:
                times_e = np.array([ts for ts, *_ in elevated], dtype=float)
                dbs_e   = np.array([db for _, db, *_ in elevated])
                slope   = float(np.polyfit(times_e - times_e[0], dbs_e, 1)[0])
            else:
                slope = 0.0

            # Aircraft confidence: needs centroid ≥ 1.5 (more mid than high) or
            # a clear positive slope + sustained elevation
            if centroid_score >= 1.5 or (slope > 0.05 and n_elev >= self.MIN_DUR["aircraft"]):
                conf = 0.4 * dur_score + 0.3 * min(1.0, centroid_score / 3.0) + \
                       0.3 * min(1.0, db_excess / 10.0)
                if conf >= 0.20:
                    results.append(("aircraft", round(conf, 3)))
                    log.debug("SustainedDet aircraft  n_elev=%d  slope=%.3f  centroid=%.2f  conf=%.2f",
                              n_elev, slope, centroid_score, conf)

        # ── Lawn mower / leaf blower ───────────────────────────────────────────
        # Both are broadband sources; lawn mower is louder and peaks lower.
        # Spectral evenness: a uniform noise floor across bands gives all three
        # fractions near 1/3.  We measure this with the coefficient of variation
        # (low CV = even = broadband).
        if n_elev >= self.MIN_DUR["leaf_blower"]:
            fracs = np.array([avg_fl, avg_fm, avg_fh])
            cv    = float(np.std(fracs) / (np.mean(fracs) + 1e-6))  # low = broadband
            flat_score = max(0.0, 1.0 - cv / 0.5)  # 1 at cv=0, 0 at cv≥0.5
            dur_score  = min(1.0, (n_elev - self.MIN_DUR["leaf_blower"]) / 40.0)

            if flat_score >= 0.3:
                conf = 0.4 * flat_score + 0.3 * dur_score + 0.3 * min(1.0, db_excess / 12.0)
                if conf >= 0.20:
                    # Lawn mower: louder (>8 dB above ambient) and lower centroid
                    if db_excess >= 8.0 and avg_fl > avg_fh:
                        cat = "lawn_mower"
                    else:
                        cat = "leaf_blower"
                    results.append((cat, round(conf, 3)))
                    log.debug("SustainedDet %s  n_elev=%d  cv=%.3f  db_excess=%.1f  conf=%.2f",
                              cat, n_elev, cv, db_excess, conf)

        return results


# ── Trained classifier (optional) ─────────────────────────────────────────────
class TrainedClassifier:
    """
    Wraps the scikit-learn SVM trained by train_classifier.py.
    Returns None if no model file exists yet (graceful degradation).
    """
    WIN_SAMPLES = 15600   # 1s @ 16 kHz

    def __init__(self):
        self._pipe = None
        self._le   = None
        self._load()

    def _load(self):
        if not os.path.exists(CLASSIFIER_PATH):
            log.info("No trained classifier found at %s — using YAMNet only",
                     CLASSIFIER_PATH)
            return
        try:
            import joblib
            obj = joblib.load(CLASSIFIER_PATH)
            self._pipe = obj["pipeline"]
            self._le   = obj["label_encoder"]
            log.info("Trained classifier loaded: classes=%s", list(self._le.classes_))
        except Exception as e:
            log.warning("Could not load classifier: %s", e)

    def reload(self):
        """Hot-reload after retraining without restarting the service."""
        self._pipe = None
        self._le   = None
        self._load()

    def predict(self, yamnet_scores_per_window: np.ndarray):
        """
        scores_per_window: (N, 521) array from YAMNet inference.
        Returns (category, probability) or None if no model loaded.
        """
        if self._pipe is None:
            return None
        try:
            feat = np.concatenate([
                yamnet_scores_per_window.mean(axis=0),
                yamnet_scores_per_window.max(axis=0),
                yamnet_scores_per_window.std(axis=0),
            ]).reshape(1, -1).astype(np.float32)

            proba  = self._pipe.predict_proba(feat)[0]
            best   = int(np.argmax(proba))
            label  = self._le.classes_[best]
            conf   = float(proba[best])
            log.debug("TrainedCls: %s=%.2f  (all: %s)", label, conf,
                      " ".join(f"{c}={p:.2f}"
                               for c, p in zip(self._le.classes_, proba)))
            return label, conf
        except Exception as e:
            log.warning("Trained classifier predict failed: %s", e)
            return None


# ── ADS-B tracker ─────────────────────────────────────────────────────────────
class ADSBTracker:
    """
    Polls api.adsb.fi every ADSB_POLL_INTERVAL seconds for aircraft within
    ADSB_RADIUS_NM of the monitor location.  Results are cached and served
    to the audio classifier thread without blocking it.
    """
    _APIS = [
        "https://api.adsb.fi/v1/point/{lat}/{lon}/{radius}",
        "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius}",
    ]

    def __init__(self):
        self._lock     = threading.Lock()
        self._aircraft = []   # list of dicts, refreshed in background
        self._last_ok  = 0.0
        t = threading.Thread(target=self._loop, daemon=True, name="adsb")
        t.start()
        log.info("ADS-B tracker started  lat=%.4f lon=%.4f radius=%d nm",
                 ADSB_LAT, ADSB_LON, ADSB_RADIUS_NM)

    def _fetch(self):
        import urllib.request
        last_err = None
        for tmpl in self._APIS:
            url = tmpl.format(lat=ADSB_LAT, lon=ADSB_LON, radius=ADSB_RADIUS_NM)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "noisemon/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err
        ac = []
        for a in data.get("ac", []):
            alt = a.get("alt_baro")
            ac.append({
                "hex":     a.get("hex", ""),
                "flight":  (a.get("flight") or "").strip(),
                "alt_ft":  int(alt) if isinstance(alt, (int, float)) else None,
                "type":    a.get("t", ""),
                "reg":     a.get("r", ""),
                "desc":    a.get("desc", ""),
                "lat":     a.get("lat"),
                "lon":     a.get("lon"),
                "dist_nm": round(float(a.get("dst", 0)), 1),
            })
        ac.sort(key=lambda x: x["dist_nm"])
        return ac

    def _loop(self):
        while True:
            try:
                ac = self._fetch()
                with self._lock:
                    self._aircraft = ac
                    self._last_ok  = time.time()
                if ac:
                    log.debug("ADS-B: %d aircraft within %d nm — closest: %s %.0fnm %s",
                              len(ac), ADSB_RADIUS_NM,
                              ac[0]["flight"] or ac[0]["hex"],
                              ac[0]["dist_nm"],
                              f"{ac[0]['alt_ft']}ft" if ac[0]["alt_ft"] else "")
                else:
                    log.debug("ADS-B: no aircraft within %d nm", ADSB_RADIUS_NM)
            except Exception as e:
                log.debug("ADS-B fetch error: %s", e)
            time.sleep(ADSB_POLL_INTERVAL)

    def get_nearby(self):
        """Return current aircraft list (thread-safe copy)."""
        with self._lock:
            return list(self._aircraft)

    def is_fresh(self):
        """True if a successful poll completed within the last 2× poll interval."""
        return (time.time() - self._last_ok) < ADSB_POLL_INTERVAL * 2


# ── Main audio processor ──────────────────────────────────────────────────────
class AudioProcessor:
    def __init__(self):
        self._lock           = threading.Lock()
        self._hist           = deque(maxlen=int(HISTORY_SECONDS / BLOCK_DURATION))
        self._hist_16k       = deque(maxlen=int(HISTORY_SECONDS / BLOCK_DURATION))
        self._last_classify  = 0.0
        self._source_last_ts = {}        # {category: last_event_ts} for per-source dedup
        self._yamnet         = YAMNetClassifier()
        self._trained        = TrainedClassifier()
        self._detection_hist = deque(maxlen=500)
        self._sustained      = SustainedSourceDetector()
        self._adsb           = ADSBTracker()

        # Full-quality pre-roll buffer for clip saving
        pre_roll_blocks = int(PRE_ROLL_SECONDS / BLOCK_DURATION)
        self._audio_buffer = deque(maxlen=pre_roll_blocks)

        # Active clip captures: list of dicts, one per in-flight clip
        # Each dict: {buf, blocks_left, ts, source, conf, db}
        self._active_captures = []

        # Purge on startup, then daily thereafter
        purge_old_clips()
        purge_old_measurements()
        t = threading.Thread(target=_housekeeping_loop, daemon=True, name="housekeeping")
        t.start()

    def process(self, block, ts):
        block = block.flatten().astype(np.float64)
        n     = len(block)

        # ── Frequency analysis: A-weighted + mic calibration ──
        win   = scipy.signal.windows.hann(n)
        fft   = np.abs(scipy.fft.rfft(block * win))
        frq   = scipy.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        fft_a = fft * a_weight(np.maximum(frq, 1.0)) * (10.0 ** (cal_curve(frq) / 20.0))

        fl = band_frac(fft_a, frq,   20,  200)
        fm = band_frac(fft_a, frq,  200, 2000)
        fh = band_frac(fft_a, frq, 2000, 20000)

        # ── dB measurements ──
        # db_avg: true A-weighted RMS via Parseval on the A-weighted+calibrated FFT.
        # Dividing by sqrt(mean(win²)) corrects for the Hann window's power reduction.
        power_w = (fft_a[0]**2 + 2*np.sum(fft_a[1:-1]**2) + fft_a[-1]**2) / n**2
        db_avg  = rms_db(np.sqrt(power_w / np.mean(win**2)))

        # db_peak / db_min: flat (unweighted) instantaneous levels
        db_peak = rms_db(np.max(np.abs(block)))
        db_min  = rms_db(np.percentile(np.abs(block) + 1e-12, 10))

        # ── Store measurement ──
        with get_conn() as c:
            c.execute(
                "INSERT INTO measurements "
                "(ts,db_avg,db_peak,db_min,freq_low,freq_mid,freq_high) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, round(db_avg,2), round(db_peak,2), round(db_min,2),
                 round(fl,4), round(fm,4), round(fh,4))
            )

        # ── Feed sustained-source detector ──
        self._sustained.feed(ts, db_avg, fl, fm, fh)

        # ── Buffer audio ──
        block_32 = block.astype(np.float32)
        block_16k = resample_to_16k(block_32)

        with self._lock:
            self._hist.append({"ts": ts, "db": db_avg})
            self._hist_16k.append(block_16k)
            self._audio_buffer.append(block_32)

            # Feed every active clip capture with this block
            still_active = []
            for cap in self._active_captures:
                cap["buf"].append(block_32)
                cap["blocks_left"] -= 1
                if cap["blocks_left"] <= 0:
                    threading.Thread(
                        target=self._write_clip,
                        args=(cap["pre_roll"] + list(cap["buf"]),
                              cap["ts"], cap["source"], cap["conf"], cap["db"]),
                        daemon=True
                    ).start()
                else:
                    still_active.append(cap)
            self._active_captures = still_active

        # ── Classify periodically ──
        if time.time() - self._last_classify >= CLASSIFY_INTERVAL:
            self._last_classify = time.time()
            self._classify(ts, db_avg)

    def _write_clip(self, blocks, ts, source, confidence, db_mean):
        """Write WAV file and update the event record — runs in background thread."""
        try:
            audio = np.concatenate(blocks)
            filename = save_clip(audio, ts, source)
            with get_conn() as c:
                c.execute(
                    "UPDATE events SET clip_path=? "
                    "WHERE ts_start=? AND source=?",
                    (filename, ts, source)
                )
        except Exception as e:
            log.error("Clip save failed: %s", e)

    def _trigger_clip(self, ts, source, confidence, db_mean):
        """Start capturing post-roll audio for a clip."""
        with self._lock:
            self._active_captures.append({
                "pre_roll":   list(self._audio_buffer),  # snapshot of ring buffer now
                "buf":        [],
                "blocks_left": int(POST_ROLL_SECONDS / BLOCK_DURATION),
                "ts":         ts,
                "source":     source,
                "conf":       confidence,
                "db":         db_mean,
            })
            log.debug("Clip capture started for %s (active=%d)",
                      source, len(self._active_captures))

    def _classify(self, ts, current_db):
        with self._lock:
            if len(self._hist_16k) < 2:
                return
            audio_buf = np.concatenate(list(self._hist_16k))
            db_vals   = [h["db"] for h in self._hist]

        # Apply fixed reference gain for YAMNet.  Quiet background stays at
        # low input levels (low confidence); loud events reach useful levels.
        # Clip to [-1, 1] to prevent overflow on loud transients.
        buf_peak = np.max(np.abs(audio_buf))
        if buf_peak < 1e-6:
            return
        audio_buf = np.clip(audio_buf * YAMNET_GAIN, -1.0, 1.0)

        buf_rms = np.sqrt(np.mean(audio_buf**2))
        log.debug("YAMNet input RMS=%.6f (gained)  len=%d  dur=%.1fs",
                  buf_rms, len(audio_buf), len(audio_buf)/YAMNET_SR)

        db_mean = float(np.mean(db_vals))

        if db_mean < 25:
            return

        results, all_scores_arr = self._yamnet.infer(audio_buf)
        if not results:
            return

        top3 = ", ".join(f"{c}={s:.2f}" for c, s in results[:3])
        log.info("YAMNet top3: %s  dB=%.1f", top3, db_mean)

        # Trained classifier — if confident (≥60%), add as a strong detection hit
        aircraft_hit_this_cycle = False
        trained_result = self._trained.predict(all_scores_arr)
        if trained_result:
            tc_label, tc_conf = trained_result
            if tc_conf >= 0.60:
                self._detection_hist.append((ts, tc_label, tc_conf))
                log.info("TrainedCls: %-14s  conf=%.0f%%", tc_label, tc_conf * 100)
                if tc_label == "aircraft":
                    aircraft_hit_this_cycle = True

        # Record every category that meets its threshold so background sources
        # (leaf_blower, aircraft) can accumulate hits even when a louder source dominates.
        for cat, score in results:
            if score >= CATEGORY_THRESH.get(cat, CONFIDENCE_THRESH):
                self._detection_hist.append((ts, cat, score))
                if cat == "aircraft":
                    aircraft_hit_this_cycle = True

        # ADS-B boost: if audio hints at aircraft AND ADS-B sees aircraft nearby,
        # inject extra detection hits to help confirm the event sooner.
        if aircraft_hit_this_cycle and self._adsb.is_fresh():
            nearby = self._adsb.get_nearby()
            if nearby:
                for _ in range(ADSB_BOOST_HITS):
                    self._detection_hist.append((ts, "aircraft", 0.7))
                log.debug("ADS-B boost: %d aircraft nearby → +%d hits  (closest: %s %.0fnm)",
                          len(nearby), ADSB_BOOST_HITS,
                          nearby[0]["flight"] or nearby[0]["hex"], nearby[0]["dist_nm"])

        # ── YAMNet confirmation ────────────────────────────────────────────────
        yamnet_confirmed = {}   # {category: score}
        cutoff = ts - PATTERN_WINDOW
        recent = [(t, c, s) for t, c, s in self._detection_hist if t >= cutoff]

        if recent:
            cat_hits = Counter(c for _, c, _ in recent)
            cat_scores = {}
            for _, c, s in recent:
                if c not in cat_scores or s > cat_scores[c]:
                    cat_scores[c] = s

            # Aircraft gets priority: flyover drone frequently co-scores with
            # "music" in YAMNet, so a single aircraft hit wins over music hits.
            aircraft_hits = [(t, c, s) for t, c, s in recent if c == "aircraft"]
            if aircraft_hits:
                yamnet_confirmed["aircraft"] = max(s for _, _, s in aircraft_hits)

            for cat, hits in cat_hits.most_common():
                score = cat_scores[cat]
                if cat == "pickleball":
                    if hits >= PATTERN_MIN_HITS and score >= CATEGORY_THRESH.get(cat, CONFIDENCE_THRESH):
                        hit_times = sorted(t for t, c, _ in recent if c == "pickleball")
                        if len(hit_times) >= PATTERN_MIN_HITS:
                            gaps = [hit_times[i+1]-hit_times[i]
                                    for i in range(len(hit_times)-1)]
                            avg_gap = np.mean(gaps)
                            gap_std = np.std(gaps)
                            if 0.5 <= avg_gap <= 4.0 and gap_std < avg_gap * 0.8:
                                yamnet_confirmed.setdefault("pickleball", score)
                                log.info("Pickleball pattern confirmed: "
                                         "%d hits, avg_gap=%.1fs", hits, avg_gap)
                            elif hits >= 2 and score >= CATEGORY_THRESH.get(cat, CONFIDENCE_THRESH):
                                yamnet_confirmed.setdefault("pickleball", score)
                            elif score >= 0.4:
                                yamnet_confirmed.setdefault("pickleball", score)
                elif cat == "aircraft":
                    pass  # already handled above
                else:
                    if hits >= PATTERN_MIN_HITS and score >= CATEGORY_THRESH.get(cat, CONFIDENCE_THRESH):
                        # Suppress voices if birds scored comparably in this window —
                        # crow/owl vocalizations often score high on "speech" in YAMNet.
                        if cat == "voices":
                            bird_best = max((s for _, c, s in recent if c == "birds"), default=0.0)
                            if bird_best >= score * 0.3:
                                log.debug("Suppressing voices (bird_best=%.2f >= voices=%.2f * 0.3)",
                                          bird_best, score)
                                continue
                        yamnet_confirmed.setdefault(cat, score)

        # ── Sustained-source detector results ─────────────────────────────────
        sustained_results = self._sustained.detect(window_seconds=30)
        for cat, conf in sustained_results:
            # Merge: if category not already in yamnet_confirmed, add it;
            # if already there, take the max confidence.
            if cat not in yamnet_confirmed or conf > yamnet_confirmed[cat]:
                yamnet_confirmed[cat] = conf
                log.info("SustainedDet confirmed %-14s  conf=%.0f%%", cat, conf * 100)

        # ── Log each newly-confirmed source with per-source dedup ─────────────
        # Gate events against the rolling ambient: a source that has become the
        # sustained background (e.g. pool heater running for hours) will be
        # absorbed into the ambient baseline after ~5 min and will no longer
        # exceed ambient + SUSTAINED_ENERGY_ABOVE, suppressing false events.
        current_ambient = self._sustained.get_ambient()

        for cat, score in yamnet_confirmed.items():
            if cat in SUPPRESS_CATEGORIES:
                continue  # suppressed — still used internally for voice/bird logic
            dedup_secs = DEDUP_WINDOW.get(cat, DEDUP_DEFAULT)
            last_ts    = self._source_last_ts.get(cat, 0)
            if ts - last_ts < dedup_secs:
                continue  # too soon since last event for this source
            min_db = CATEGORY_MIN_DB.get(cat)
            if min_db is not None and db_mean < min_db:
                log.debug("EVENT suppressed %-14s  dB=%.1f < min %.1f", cat, db_mean, min_db)
                continue
            if current_ambient is not None and db_mean < current_ambient + SUSTAINED_ENERGY_ABOVE:
                log.debug("EVENT suppressed %-14s  dB=%.1f not above ambient %.1f + %.1f",
                          cat, db_mean, current_ambient, SUSTAINED_ENERGY_ABOVE)
                continue

            # Snapshot ADS-B aircraft for aircraft events
            adsb_json = None
            if cat == "aircraft":
                nearby = self._adsb.get_nearby()
                if nearby:
                    adsb_json = json.dumps(nearby)
                    log.info("ADS-B snapshot: %s",
                             " | ".join(
                                 f"{a['flight'] or a['hex']} "
                                 f"{a['type']} "
                                 f"{a['alt_ft']}ft "
                                 f"{a['dist_nm']}nm"
                                 for a in nearby[:5]
                             ))

            with get_conn() as c:
                c.execute(
                    "INSERT INTO events (ts_start,source,confidence,db_avg,adsb_json) "
                    "VALUES (?,?,?,?,?)",
                    (ts, cat, round(score, 3), round(db_mean, 1), adsb_json)
                )
            log.info("EVENT %-14s  conf=%.0f%%  dB=%.1f",
                     cat, score * 100, db_mean)
            self._source_last_ts[cat] = ts
            self._trigger_clip(ts, cat, score, db_mean)


# ── Recording loop ────────────────────────────────────────────────────────────
processor = None

def callback(indata, frames, time_info, status):
    if status:
        log.warning("Stream: %s", status)
    try:
        processor.process(indata.copy(), int(time.time()))
    except Exception as e:
        log.error("callback error (skipping block): %s", e)

def run():
    global processor
    init_db()
    processor = AudioProcessor()
    bs = int(SAMPLE_RATE * BLOCK_DURATION)
    log.info("NoiseMon/YAMNet starting — device=%s", DEVICE_INDEX)
    with sd.InputStream(device=DEVICE_INDEX, channels=CHANNELS,
                        samplerate=SAMPLE_RATE, blocksize=bs,
                        dtype="float32", callback=callback):
        log.info("Recording… Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Stopped.")

if __name__ == "__main__":
    run()
