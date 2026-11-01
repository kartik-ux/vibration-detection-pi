"""Deterministic, physically-inspired three-axis vibration signal simulator."""
import numpy as np
from .config import CLASSES

class VibrationSimulator:
    def __init__(self, sample_rate_hz: int = 800, seed: int = 42):
        self.sample_rate_hz = sample_rate_hz
        self.rng = np.random.default_rng(seed)

    def window(self, label: str, n_samples: int = 800) -> np.ndarray:
        if label not in CLASSES:
            raise ValueError(f"Unknown class {label!r}; choose from {CLASSES}")
        t = np.arange(n_samples) / self.sample_rate_hz
        phase = self.rng.uniform(0, 2*np.pi)
        rpm_hz = self.rng.uniform(24, 31)
        noise = self.rng.normal(0, 0.055, (n_samples, 3))
        x = np.sin(2*np.pi*rpm_hz*t + phase)
        y = 0.62*np.sin(2*np.pi*rpm_hz*t + phase + 0.7)
        z = 0.38*np.sin(2*np.pi*rpm_hz*t + phase + 1.4)
        sig = np.column_stack([x, y, z])
        if label == "imbalance":
            sig[:, 0] *= 2.4
            sig[:, 1] *= 1.7
        elif label == "misalignment":
            harmonic = 0.75*np.sin(4*np.pi*rpm_hz*t + phase)
            sig += harmonic[:, None] * np.array([1.0, 0.8, 0.5])
        elif label == "bearing_fault":
            # Repeated damped impulses with broadband high-frequency content.
            impulses = np.zeros(n_samples)
            interval = max(8, int(self.sample_rate_hz / self.rng.uniform(75, 105)))
            for i in range(self.rng.integers(0, interval), n_samples, interval):
                end = min(n_samples, i + 18)
                impulses[i:end] += 2.6*np.exp(-np.arange(end-i)/4.0)
            sig += impulses[:, None] * np.array([1.0, 0.65, 0.4])
        elif label == "looseness":
            sig += 0.55*np.sign(np.sin(2*np.pi*rpm_hz*t/2 + phase))[:, None]
            sig += 0.18*np.sin(6*np.pi*rpm_hz*t)[:, None]
        return sig + noise

    def dataset(self, windows_per_class: int = 100, n_samples: int = 800):
        rows = []
        for label in CLASSES:
            for _ in range(windows_per_class):
                rows.append((self.window(label, n_samples), label))
        self.rng.shuffle(rows)
        return rows
