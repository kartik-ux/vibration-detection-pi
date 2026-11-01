"""Signal conditioning and compact time/frequency-domain feature extraction."""
import numpy as np

FEATURE_NAMES = []
for axis in ("x", "y", "z", "magnitude"):
    FEATURE_NAMES += [f"{axis}_{name}" for name in
        ("mean", "std", "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis", "skewness")]
    FEATURE_NAMES += [f"{axis}_fft_band_{i}" for i in range(4)]

def _moments(x):
    mean = float(np.mean(x)); centered = x - mean
    std = float(np.std(x)) + 1e-12
    skew = float(np.mean(centered**3) / std**3)
    kurt = float(np.mean(centered**4) / std**4)
    rms = float(np.sqrt(np.mean(x*x)))
    peak = float(np.max(np.abs(x)))
    return [mean, std, rms, peak, float(np.ptp(x)), peak/(rms+1e-12), kurt, skew]

def extract_features(signal: np.ndarray, sample_rate_hz: int = 800) -> dict:
    a = np.asarray(signal, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 32:
        raise ValueError("signal must be an N x 3 array with at least 32 samples")
    a = a - np.mean(a, axis=0, keepdims=True)
    channels = [a[:, 0], a[:, 1], a[:, 2], np.linalg.norm(a, axis=1)]
    out = {}
    for name, x in zip(("x", "y", "z", "magnitude"), channels):
        vals = _moments(x)
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))**2
        freqs = np.fft.rfftfreq(len(x), d=1/sample_rate_hz)
        edges = np.linspace(0, sample_rate_hz/2, 5)
        total = float(spectrum.sum()) + 1e-12
        bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            bands.append(float(spectrum[mask].sum()/total))
        for key, value in zip(("mean", "std", "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis", "skewness"), vals):
            out[f"{name}_{key}"] = value
        for i, value in enumerate(bands): out[f"{name}_fft_band_{i}"] = value
    return out
