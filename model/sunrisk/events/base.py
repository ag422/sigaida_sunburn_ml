"""Detector interface. Rules today; a trained classifier later implements the same protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .features import Windows


class EventDetector(Protocol):
    def predict(self, w: Windows) -> tuple[np.ndarray, np.ndarray]:
        """Return (environment codes, activity codes), one per window, indexing
        contracts.ENVIRONMENTS and contracts.ACTIVITIES."""
        ...


class ActivityClassifier(Protocol):
    """Activity-only model, e.g. trained on PAMAP2 IMU windows (no UV)."""

    def predict_activity(self, imu_features: np.ndarray) -> np.ndarray:
        ...
