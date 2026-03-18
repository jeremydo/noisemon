# NoiseMon configuration — copy this to config.py and fill in your values.
# config.py is gitignored so your secrets and location stay private.

# ── Web dashboard auth ────────────────────────────────────────────────────────
AUTH_USER = "noisemon"          # username for the web UI login
AUTH_PASS = "change_me"         # password for the web UI login

# ── ADS-B location ───────────────────────────────────────────────────────────
# Used to correlate aircraft detections with live ADS-B traffic overhead.
# Set to your latitude/longitude (decimal degrees).
ADSB_LAT = 0.0
ADSB_LON = 0.0
