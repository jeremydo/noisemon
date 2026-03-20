# NoiseMon configuration — copy this to config.py and fill in your values.
# config.py is gitignored so your secrets and location stay private.

# ── Web dashboard auth ────────────────────────────────────────────────────────
# Map of username → password. Add as many users as needed.
AUTH_USERS = {
    "noisemon": "change_me",
}

# ── ADS-B location ───────────────────────────────────────────────────────────
# Used to correlate aircraft detections with live ADS-B traffic overhead.
# Set to your latitude/longitude (decimal degrees).
ADSB_LAT = 0.0
ADSB_LON = 0.0
