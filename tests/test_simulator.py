from vibration_pi.simulator import VibrationSimulator
from vibration_pi.config import CLASSES

def test_simulator_shapes():
    sim=VibrationSimulator()
    for label in CLASSES:
        assert sim.window(label,256).shape==(256,3)
