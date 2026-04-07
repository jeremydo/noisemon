# NoiseMon

A neighborhood noise monitor running on a Raspberry Pi 4. Records audio continuously, measures dB(A), classifies noise events using Google's YAMNet model, and serves a web dashboard for reviewing events and labelling clips to train a custom classifier.

## Features

- **Real-time dB(A) measurement** — calibrated A-weighted RMS via Parseval's theorem
- **YAMNet classification** — 521 AudioSet classes mapped to categories (aircraft, lawn mower, leaf blower, voices, etc.)
- **Custom SVM classifier** — trained on your own labelled clips, fires alongside YAMNet
- **ADS-B integration** — correlates aircraft detections with live overhead traffic via adsb.fi
- **Audio clip capture** — 30s pre-roll, 90s post-roll WAV clips for every confirmed event
- **Web dashboard** — live dB chart, event history, spectrogram viewer
- **Labelling UI** — drag regions on spectrograms to label training samples
- **Review UI** — per-class yes/no workflow to audit and prune training data

## Hardware

- Raspberry Pi 4
- [MiniDSP UMIK-1](https://www.minidsp.com/products/acoustic-measurement/umik-1) USB measurement microphone (or any USB mic)
- Calibration file from MiniDSP for your specific mic serial number

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/jeremydo/noisemon.git /opt/noisemon
cd /opt/noisemon
cp config.example.py config.py
# Edit config.py — set AUTH_PASS and your ADSB_LAT/LON
```

### 2. Download YAMNet

```bash
mkdir -p models
wget -O models/yamnet.tflite \
  https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
wget -O models/yamnet_class_map.csv \
  https://raw.githubusercontent.com/google-research/audioset_tagging_cnn/master/metadata/class_labels_indices.csv
```

### 3. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy sounddevice soundfile scipy scikit-learn joblib ai-edge-litert flask requests
```

### 4. Create directories and database

```bash
sudo mkdir -p /var/lib/noisemon/clips /var/log/noisemon
sudo chown -R $USER /var/lib/noisemon /var/log/noisemon
```

The database schema is created automatically on first run.

### 5. Microphone calibration

Place your mic's calibration `.txt` file (MiniDSP format) at `models/umik1_cal_90.txt` (or update `CAL_FILE` in `noise_monitor.py`). The included `models/umik1_cal_90.txt` is for a specific UMIK-1 and will not be accurate for other units.

### 6. Run as a systemd service

```bash
sudo cp noisemon.service /etc/systemd/system/
sudo cp noisemon-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now noisemon noisemon-web
```

## Classifier training

Once you have labelled clips via the web UI:

```bash
source venv/bin/activate
python extract_features.py
python train_classifier.py
sudo systemctl restart noisemon
```

## Configuration

Key settings at the top of `noise_monitor.py`:

| Variable | Default | Description |
|---|---|---|
| `CALIBRATION_DB` | 125.5 | Mic sensitivity offset (dB) |
| `YAMNET_GAIN` | 32.0 | Pre-gain before YAMNet inference |
| `CONFIDENCE_THRESH` | 0.20 | Min YAMNet score to accumulate a detection |
| `PATTERN_MIN_HITS` | 2 | Detections within 60s needed to confirm an event |
| `PRE_ROLL_SECONDS` | 30 | Seconds of audio before event trigger saved to clip |
| `POST_ROLL_SECONDS` | 90 | Seconds of audio after trigger saved to clip |
| `ADSB_RADIUS_NM` | 10 | ADS-B search radius (nautical miles) |

## License

MIT
