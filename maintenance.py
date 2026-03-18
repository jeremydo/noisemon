#!/usr/bin/env python3
"""
NoiseMon maintenance — runs nightly via cron.
Prunes database records and clip files older than 6 months,
and removes any orphaned WAV files not referenced in the database.
"""
import sqlite3, time, os
from pathlib import Path

DB          = "/var/lib/noisemon/noise.db"
CLIPS_DIR   = "/var/lib/noisemon/clips"
KEEP_DAYS   = 180   # 6 months
KEEP_SECS   = KEEP_DAYS * 86400

conn   = sqlite3.connect(DB)
cutoff = int(time.time()) - KEEP_SECS

# ── 1. Prune old measurements ─────────────────────────────────────────────────
deleted_m = conn.execute(
    "DELETE FROM measurements WHERE ts < ?", (cutoff,)
).rowcount

# ── 2. Get clip filenames we're about to delete from DB ───────────────────────
old_clips = [
    row[0] for row in conn.execute(
        "SELECT clip_path FROM events WHERE ts_start < ? AND clip_path IS NOT NULL",
        (cutoff,)
    ).fetchall()
]

# ── 3. Prune old events ───────────────────────────────────────────────────────
deleted_e = conn.execute(
    "DELETE FROM events WHERE ts_start < ?", (cutoff,)
).rowcount

conn.commit()
conn.execute("VACUUM")

# ── 4. Delete WAV files for pruned events ────────────────────────────────────
deleted_wav = 0
for filename in old_clips:
    if filename:
        path = Path(CLIPS_DIR) / filename
        if path.exists():
            path.unlink()
            deleted_wav += 1

# ── 5. Delete orphaned WAV files (on disk but not in database) ───────────────
known_clips = {
    row[0] for row in conn.execute(
        "SELECT clip_path FROM events WHERE clip_path IS NOT NULL"
    ).fetchall()
}

orphaned = 0
for wav in Path(CLIPS_DIR).glob("*.wav"):
    if wav.name not in known_clips:
        wav.unlink()
        orphaned += 1

conn.close()

print(
    f"Pruned {deleted_m} measurements, {deleted_e} events "
    f"older than {KEEP_DAYS} days"
)
print(f"Deleted {deleted_wav} clip WAVs for pruned events")
print(f"Removed {orphaned} orphaned WAV files")
