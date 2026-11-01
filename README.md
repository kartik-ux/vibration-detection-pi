
# Vibration Detection Pi — Edge AI Predictive Maintenance

[![CI](https://github.com/YOUR-USERNAME/vibration-detection-pi/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/vibration-detection-pi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An open-source Edge AI prototype for local vibration monitoring, machine-fault pattern recognition and predictive-maintenance research on Raspberry Pi.**


An end-to-end Edge AI reference implementation for monitoring rotating machinery using three-axis accelerometer data. It includes a reproducible vibration simulator, signal feature extraction, supervised fault classification, unsupervised anomaly detection, detection logs and inference benchmarking. It can be developed on a laptop and adapted to a Raspberry Pi with an MPU6050 accelerometer.

> **Important:** the included simulator produces physically-inspired synthetic signals. Its metrics demonstrate that the software pipeline works on simulated data; they are not evidence of real-world machine-fault accuracy. Validate with labelled measurements from the target machine before operational use.

## Capabilities

- Three-axis vibration acquisition interface, with simulator and basic MPU6050 I²C adapter.
- Window-based signal processing and 48 time/frequency-domain features per window.
- Random Forest multi-class classifier: healthy, imbalance, misalignment, bearing fault and looseness.
- Isolation Forest anomaly detector trained on healthy-only windows.
- Confidence, anomaly score and per-window inference latency.
- CSV collection/detection logs; JSON training and benchmark reports.
- Reproducible CLI and unit tests.

## Architecture

`Accelerometer → windowing → signal features → Random Forest classification + Isolation Forest anomaly score → local logs / alerts`

The classifier answers *which known pattern is most likely?* The anomaly detector answers *does this window look unlike the healthy training envelope?* They are complementary signals, not interchangeable fault diagnoses.

## Repository layout

```text
vibration-detection-pi/
├── src/vibration_pi/
│   ├── cli.py          # collect/train/detect/benchmark commands
│   ├── config.py       # sample rate, window and paths
│   ├── features.py     # time/frequency-domain feature extraction
│   ├── pipeline.py     # training, inference, evaluation, benchmark
│   ├── sensors.py      # simulator and MPU6050 adapter
│   └── simulator.py    # synthetic machine vibration generator
├── tests/              # unit tests
├── data/               # generated feature dataset
├── models/             # trained model artifact
└── reports/            # metrics, logs and benchmark output
```

## Quick start — macOS / Linux

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/vibration-detection-pi.git
cd vibration-detection-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test]'
```

Run the complete simulator workflow:

```bash
vibration-pi all --windows-per-class 100 --windows 25 --iterations 300
```

Or run each stage independently:

```bash
vibration-pi collect --windows-per-class 100
vibration-pi train
vibration-pi detect --windows 25
vibration-pi benchmark --iterations 300
pytest -q
```

### Pipeline stages

1. **Collect:** creates labelled synthetic windows, extracts 48 features and writes `data/vibration_features.csv`.
2. **Train:** stratified train/test split; trains a Random Forest classifier and a healthy-only Isolation Forest; saves `models/vibration_model.joblib` and `reports/training_report.json`.
3. **Detect:** streams simulated windows through the saved model and writes `reports/detection_log.csv`.
4. **Benchmark:** measures host-side inference latency, p95 latency, throughput and model artifact size; writes `reports/benchmark.json`.

## Raspberry Pi hardware

Suggested prototype bill of materials:

| Component | Purpose |
|---|---|
| Raspberry Pi 4 or 5 | Local edge compute |
| MPU6050 breakout (I²C) | 3-axis accelerometer; gyro is not used here |
| Jumper wires | I²C and power connections |
| Stable 3.3 V-compatible breakout | Sensor power and logic compatibility |
| Rigid mounting fixture | Repeatable mechanical coupling to the machine |

### MPU6050 wiring

| MPU6050 | Raspberry Pi physical pin |
|---|---|
| VCC | 3V3 (pin 1), subject to breakout-board specification |
| GND | GND (pin 6) |
| SDA | GPIO 2 / SDA1 (pin 3) |
| SCL | GPIO 3 / SCL1 (pin 5) |

Enable I²C using `sudo raspi-config` → Interface Options → I2C. Check the device address with `i2cdetect -y 1` (commonly `0x68`). Install the optional driver:

```bash
sudo apt update
sudo apt install -y i2c-tools python3-venv
pip install -e '.[pi]'
```

The `MPU6050Sensor` adapter is a simple polling example, configured for ±2g and nominal 200 Hz. Its polling loop is not hard real-time and may have timing jitter. The current training defaults (800 Hz) are for the simulator, not a claim about this adapter. Before using physical data, configure a consistent sample rate, use FIFO/data-ready acquisition for better timing, calibrate offsets, verify axis orientation, and collect representative labelled examples. Do not run the current trained synthetic model as a machine safety system.

Example sensor integration in Python:

```python
from vibration_pi.sensors import MPU6050Sensor

sensor = MPU6050Sensor(bus=1, address=0x68, sample_rate_hz=200)
try:
    samples = sensor.read_window(n_samples=200)  # shape: (200, 3), units approximately g
finally:
    sensor.close()
```

To train and infer on real measurements, record windows with trustworthy machine-state labels, extract features using `extract_features(window, sample_rate_hz=200)`, and retrain using the same feature schema and sample-rate configuration. The example CLI deliberately uses the simulator until a real acquisition and labelling procedure is established.

## Understanding the outputs

- `prediction`: predicted known class; not a certified diagnosis.
- `confidence`: maximum Random Forest class probability. It may be overconfident and should be calibrated on held-out real data.
- `anomaly`: Isolation Forest healthy-envelope flag (`true` means unusual relative to healthy training examples).
- `anomaly_score`: higher generally means more in-family; lower values are more anomalous.
- `latency_ms`: feature extraction plus model inference time measured on the host.

A fault alert policy should combine thresholds, persistence across multiple windows, machine operating context and human review. Avoid triggering maintenance actions from a single model output.

## Evaluation and next engineering steps

For a credible industrial evaluation, collect data across multiple machines, speeds, loads, sensor mountings and operating temperatures. Split train/test by machine or acquisition session to prevent leakage. Report per-class precision/recall/F1, confusion matrix, false alarms per operating hour, missed-fault rate, detection delay, calibration, power draw and latency on the actual Pi. Add a temporal alert state machine, model versioning, watchdog/restart handling, secure remote updates and a dashboard only after the edge measurement path is reliable.

## Tests

```bash
pytest -q
```

## Engineering quality and limitations

- Continuous integration runs the test suite on Python 3.10, 3.11 and 3.12.
- Contributions and bug reports are guided by `CONTRIBUTING.md` and the GitHub issue templates.
- The default simulator is intended for reproducible software development, not certification or proof of industrial performance.
- Validate sampling, sensor mounting, calibration, model generalisation and alert thresholds on representative labelled equipment data before operational use.
- The current hardware adapter is for the MPU6050. Other accelerometers require a compatible driver and calibration procedure.

## GitHub repository topics

`edge-ai` `raspberry-pi` `predictive-maintenance` `vibration-analysis` `anomaly-detection` `condition-monitoring` `industrial-iot` `machine-learning` `python` `signal-processing`

## License

Released under the MIT License. See [LICENSE](LICENSE).
