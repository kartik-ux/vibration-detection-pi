"""Sensor interfaces. Simulator is always available; MPU6050 needs smbus2 on Pi."""
import time
import numpy as np
from .simulator import VibrationSimulator

class SimulatedSensor:
    def __init__(self, sample_rate_hz=800, seed=123):
        self.sim = VibrationSimulator(sample_rate_hz, seed)
    def read_window(self, label="healthy", n_samples=800):
        return self.sim.window(label, n_samples)

class MPU6050Sensor:
    """Basic MPU6050 accelerometer reader (±2g, I2C address 0x68).

    This is a straightforward polling driver, not a hard real-time sampler.
    For production/high-rate acquisition use FIFO/data-ready interrupts and calibrate.
    """
    def __init__(self, bus=1, address=0x68, sample_rate_hz=200):
        try: from smbus2 import SMBus
        except ImportError as exc: raise RuntimeError("Install Raspberry Pi support: pip install -e '.[pi]'") from exc
        self.bus = SMBus(bus); self.address = address; self.sample_rate_hz = sample_rate_hz
        self.bus.write_byte_data(address, 0x6B, 0)
        self.bus.write_byte_data(address, 0x1C, 0x00)
    def _read_axis(self, register):
        hi = self.bus.read_byte_data(self.address, register)
        lo = self.bus.read_byte_data(self.address, register+1)
        value = (hi << 8) | lo
        return value - 65536 if value >= 32768 else value
    def read_window(self, label=None, n_samples=200):
        data=[]; delay=1/self.sample_rate_hz
        for _ in range(n_samples):
            start=time.perf_counter()
            xyz=[self._read_axis(r)/16384.0 for r in (0x3B,0x3D,0x3F)]
            data.append(xyz)
            remaining=delay-(time.perf_counter()-start)
            if remaining>0: time.sleep(remaining)
        return np.asarray(data)
    def close(self): self.bus.close()
