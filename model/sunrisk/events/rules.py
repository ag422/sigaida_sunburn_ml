"""Rules-based event detector: thresholds with a physical reason, no training.

Activity (per 5 s window, from the wrist IMU):
  swimming  - whole-arm rotation: gravity averages out over the window (|mean acc| low)
              while rotating fast
  vigorous  - large impacts and fast rotation
  walking   - |acc| repeats at the step rate (autocorrelation peak at 1.4-2.6 Hz)
  still     - little rotation; then forearm hanging -> standing, forearm level -> sitting or
              lying. Lying vs sitting uses "no hand gestures for minutes" (people who sit
              still still lift a drink or phone); this is the weakest rule.
  otherwise - light_activity

Environment (from the UV level = measured / (expected sensor gain x clear sky)):
  level high        -> sun
  level very low    -> indoor, but only once sustained for INDOOR_CONFIRM_S. Until then the
                       previous label is kept, so a sleeve covering the sensor for a minute
                       or two does not silently zero the dose (the conservative choice).
  in between        -> diffuse-only: shade or cloud, decided per episode:
                       forecast says clear sky -> shade;
                       wearer moved when the episode started -> shade (walked under cover);
                       otherwise -> cloud (the sky changed, not the wearer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import ACTIVITIES, ENVIRONMENTS
from .features import WINDOW_S, Windows

A = {a: i for i, a in enumerate(ACTIVITIES)}
E = {e: i for i, e in enumerate(ENVIRONMENTS)}


@dataclass
class RuleDetector:
    still_gyro_dps: float = 15.0
    swim_mean_acc_g: float = 0.75
    swim_gyro_dps: float = 60.0
    vigorous_acc_std_g: float = 0.25
    vigorous_gyro_dps: float = 150.0
    walk_acf: float = 0.5
    walk_acc_std_g: float = 0.08
    hanging_acc_x_g: float = -0.7
    gesture_gyro_max_dps: float = 30.0
    lying_no_gesture_s: float = 180.0
    smooth_windows: int = 5

    sun_level: float = 0.5
    indoor_level: float = 0.10
    indoor_confirm_s: float = 300.0
    clear_forecast_cmf: float = 0.9
    entry_motion_s: float = 60.0

    # -- activity ----------------------------------------------------------------------
    def predict_activity(self, w: Windows) -> np.ndarray:
        gyro, acc_std = w.col("gyro_rms"), w.col("acc_mag_std")
        mean_norm, acf, ax = w.col("acc_mean_norm"), w.col("acf_walk"), w.col("acc_mean_x")
        n = len(gyro)
        out = np.full(n, A["light_activity"])
        still = gyro < self.still_gyro_dps
        swim = (mean_norm < self.swim_mean_acc_g) & (gyro > self.swim_gyro_dps)
        vig = ~swim & (acc_std > self.vigorous_acc_std_g) & (gyro > self.vigorous_gyro_dps)
        walk = ~swim & ~vig & (acf > self.walk_acf) & (acc_std > self.walk_acc_std_g)
        out[walk] = A["walking"]
        out[vig] = A["vigorous"]
        out[swim] = A["swimming"]

        hanging = ax < self.hanging_acc_x_g
        out[still & hanging] = A["standing"]
        level = still & ~hanging
        # lying: level forearm with no gesture in the surrounding window
        gest = w.col("gyro_max") > self.gesture_gyro_max_dps
        k = max(1, int(self.lying_no_gesture_s / WINDOW_S / 2))
        near_gesture = np.convolve(gest.astype(float), np.ones(2 * k + 1), mode="same") > 0
        out[level] = np.where(near_gesture[level], A["sitting"], A["lying"])
        return _mode_filter(out, self.smooth_windows, len(ACTIVITIES))

    # -- environment -------------------------------------------------------------------
    def predict_environment(self, w: Windows, activity: np.ndarray) -> np.ndarray:
        n = len(w.t_start)
        if w.uv_level is None:
            return np.full(n, E["sun"])
        lvl = _nan_smooth(w.uv_level, self.smooth_windows)
        raw = np.where(lvl >= self.sun_level, 0, np.where(lvl < self.indoor_level, 2, 1))  # 0 sun, 1 diffuse, 2 dark
        raw = np.where(np.isnan(lvl), -1, raw)
        raw = _absorb_transitions(raw, self.smooth_windows)

        self._pending = np.zeros(n, dtype=bool)
        moving = np.isin(activity, [A["walking"], A["vigorous"], A["light_activity"], A["swimming"]])
        out = np.full(n, E["sun"])
        prev = E["sun"]
        confirm = int(self.indoor_confirm_s / WINDOW_S)
        entry_k = int(self.entry_motion_s / WINDOW_S)
        i = 0
        while i < n:
            j = i
            while j < n and raw[j] == raw[i]:
                j += 1
            kind = raw[i]
            if kind == -1:
                out[i:j] = prev
            elif kind == 0:
                out[i:j] = prev = E["sun"]
            elif kind == 2:
                out[i:min(j, i + confirm)] = prev
                self._pending[i:min(j, i + confirm)] = True
                if j - i > confirm:
                    out[i + confirm:j] = prev = E["indoor"]
            else:
                clear_fc = w.forecast_cmf is not None and np.nanmean(w.forecast_cmf[i:j]) > self.clear_forecast_cmf
                entered_moving = moving[max(0, i - entry_k):i + entry_k + 1].any()
                label = E["shade"] if (clear_fc or entered_moving or prev == E["indoor"]) else E["cloud"]
                out[i:j] = prev = label
            i = j
        return out

    def predict(self, w: Windows):
        act = self.predict_activity(w)
        return self.predict_environment(w, act), act

    def hold_mask(self) -> np.ndarray:
        """Windows where the sensor went dark but indoor is not yet confirmed (from the last
        predict call), widened by the smoothing width to cover the ramps in and out. The
        pipeline ignores UV there: a covered sensor must not look like zero UV."""
        k = self.smooth_windows
        return np.convolve(self._pending.astype(float), np.ones(2 * k + 1), mode="same") > 0


def _absorb_transitions(raw, k):
    """A short 'diffuse' run directly next to a 'dark' run is the smoothing ramp of a sharp
    drop (sensor covered / walked indoors), not a real shade or cloud episode."""
    out = raw.copy()
    n = len(raw)
    i = 0
    while i < n:
        j = i
        while j < n and raw[j] == raw[i]:
            j += 1
        if raw[i] == 1 and j - i < k:
            left = raw[i - 1] if i > 0 else None
            right = raw[j] if j < n else None
            if left == 2 or right == 2:
                out[i:j] = 2
        i = j
    return out


def _mode_filter(x, k, n_classes):
    if k <= 1:
        return x
    onehot = np.eye(n_classes)[x]
    kernel = np.ones(k)
    counts = np.stack([np.convolve(onehot[:, c], kernel, mode="same") for c in range(n_classes)], axis=1)
    return counts.argmax(axis=1)


def _nan_smooth(x, k):
    v = np.nan_to_num(x)
    m = (~np.isnan(x)).astype(float)
    num = np.convolve(v, np.ones(k), mode="same")
    den = np.convolve(m, np.ones(k), mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
