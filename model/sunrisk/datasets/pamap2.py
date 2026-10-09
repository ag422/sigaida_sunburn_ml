"""PAMAP2 wrist IMU -> our device format (20 Hz, g, deg/s, left-wrist axes) with our labels.

Axis alignment (checked against gravity in the data, see docs/activity_classifier.md):
- PAMAP2 wrist x points along the forearm toward the hand (reads ~ -1 g with the arm hanging),
  the same as our sensor x.
- z points roughly out of the back of the wrist (positive when sitting with palms down).
- The IMU is on the DOMINANT wrist. Subject 108 is left-handed and his y axis has the opposite
  sign to everyone else, as a mirror image predicts. We mirror right-wrist data to the
  left-wrist convention of our device: y -> -y for acceleration and magnetometer;
  gyroscope is a pseudovector, so (x, y, z) -> (-x, y, -z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import ACTIVITIES

G = 9.80665
SOURCE_HZ = 100
TARGET_HZ = 20

# PAMAP2 activityID -> our activity class (None = excluded). See docs/datasets.md.
LABEL_MAP = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "vigorous",
    6: "vigorous",            # cycling: sweaty, but the wrist is fairly still on the handlebars
    7: "walking", 9: "sitting", 10: "sitting", 11: None,             # driving excluded
    12: "walking", 13: "walking",
    16: "light_activity", 17: "light_activity", 18: "light_activity", 19: "light_activity",
    20: "vigorous", 24: "vigorous",
}
LEFT_HANDED = {108}

# 0-based column indices in the .dat files
COL_ACTIVITY = 1
WRIST_ACC16 = slice(4, 7)
WRIST_GYRO = slice(10, 13)
WRIST_MAG = slice(13, 16)


@dataclass
class Run:
    """One contiguous stretch of a single activity for one subject, resampled to 20 Hz."""
    subject: int
    pamap_activity: int
    activity: str
    acc_g: np.ndarray       # (n, 3)
    gyro_dps: np.ndarray    # (n, 3)
    mag_ut: np.ndarray      # (n, 3)


def load_dat(path: str | Path) -> np.ndarray:
    """Activity id + wrist acc/gyro/mag columns, (N, 10). The only I/O in this module."""
    cols = (COL_ACTIVITY, *range(WRIST_ACC16.start, WRIST_ACC16.stop),
            *range(WRIST_GYRO.start, WRIST_GYRO.stop), *range(WRIST_MAG.start, WRIST_MAG.stop))
    return np.loadtxt(path, usecols=cols)


def _fill_nan(x):
    """Linear interpolation over NaN rows (wireless dropouts, < 1% of rows)."""
    x = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.all():
            return None
        if bad.any():
            x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x


def mirror_to_left(acc, gyro, mag):
    """Right-wrist frame -> left-wrist frame (reflection y -> -y)."""
    flip = np.array([1.0, -1.0, 1.0])
    return acc * flip, gyro * -flip, mag * flip


def runs_from_array(arr: np.ndarray, subject: int, min_seconds: float = 10.0) -> list[Run]:
    """Split into labelled runs, convert units/axes, and downsample 100 -> 20 Hz (block mean)."""
    act = arr[:, 0].astype(int)
    edges = np.flatnonzero(np.diff(act)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(act)]])
    k = SOURCE_HZ // TARGET_HZ
    runs = []
    for a, b in zip(starts, stops):
        label = LABEL_MAP.get(int(act[a]))
        if label is None or (b - a) < min_seconds * SOURCE_HZ:
            continue
        x = _fill_nan(arr[a:b, 1:])
        if x is None:
            continue
        n = (len(x) // k) * k
        x = x[:n].reshape(-1, k, x.shape[1]).mean(axis=1)
        acc, gyro, mag = x[:, 0:3] / G, np.rad2deg(x[:, 3:6]), x[:, 6:9]
        if subject not in LEFT_HANDED:
            acc, gyro, mag = mirror_to_left(acc, gyro, mag)
        runs.append(Run(subject, int(act[a]), label, acc, gyro, mag))
    return runs


def load_subjects(root: str | Path, sessions=("Protocol", "Optional")) -> list[Run]:
    root = Path(root)
    runs = []
    for session in sessions:
        for f in sorted((root / session).glob("subject*.dat")):
            runs.extend(runs_from_array(load_dat(f), int(f.stem[-3:])))
    return runs


def activity_code(name: str) -> int:
    return ACTIVITIES.index(name)
