from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    sample_rate_hz: int = 800
    window_seconds: float = 1.0
    windows_per_class: int = 100
    random_seed: int = 42
    model_path: str = "models/vibration_model.joblib"
    data_path: str = "data/vibration_features.csv"
    reports_dir: str = "reports"

    @property
    def samples_per_window(self) -> int:
        return int(self.sample_rate_hz * self.window_seconds)

CLASSES = ("healthy", "imbalance", "misalignment", "bearing_fault", "looseness")
