"""Window features for event detection. Pure numpy.

IMU features use only accelerometer + gyroscope at 20 Hz, so the same function runs on PAMAP2
(wrist IMU downsampled to 20 Hz) and on our device. UV features need the UV stream plus sun
position and a forecast.

PORT NOTE: per-window statistics only (mean, std, autocorrelation at a few lags); the
autocorrelation can be computed directly without an FFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import IMU_RATE_HZ, IMU_SAMPLES_PER_PACKET

WINDOW_S = 5.0
WINDOW_N = int(WINDOW_S * IMU_RATE_HZ)          # 100 IMU samples
UV_PER_WINDOW = WINDOW_N // IMU_SAMPLES_PER_PACKET  # 12 UV samples (+ remainder)

IMU_FEATURES = ("acc_mean_x", "acc_mean_y", "acc_mean_z", "acc_mean_norm", "acc_mag_std",
                "gyro_rms", "gyro_max", "acf_walk", "tilt_nz")


@dataclass
class Windows:
    t_start: np.ndarray              # (W,)
    imu: np.ndarray                  # (W, len(IMU_FEATURES))
    uv_level: np.ndarray | None = None   # (W,) measured / (expected gain x clear sky)
    forecast_cmf: np.ndarray | None = None  # (W,) forecast all-sky / clear-sky for that hour
    clear_uvi: np.ndarray | None = None     # (W,) clear-sky UVI (for reporting when UV matters)

    def col(self, name):
        return self.imu[:, IMU_FEATURES.index(name)]


def _acf(x, lag):
    x = x - x.mean(axis=1, keepdims=True)
    den = np.sum(x * x, axis=1) + 1e-9
    return np.sum(x[:, :-lag] * x[:, lag:], axis=1) / den


def _max_acf(x, lags):
    return np.max(np.stack([_acf(x, l) for l in lags]), axis=0)


def imu_features(acc_g, gyro_dps, rate_hz=IMU_RATE_HZ):
    """acc/gyro (N, 3) -> (W, F) features over non-overlapping 5 s windows."""
    n = int(WINDOW_S * rate_hz)
    W = len(acc_g) // n
    a = np.asarray(acc_g[: W * n], dtype=float).reshape(W, n, 3)
    g = np.asarray(gyro_dps[: W * n], dtype=float).reshape(W, n, 3)
    am = np.linalg.norm(a, axis=2)
    gm = np.linalg.norm(g, axis=2)
    mean = a.mean(axis=1)
    mean_norm = np.linalg.norm(mean, axis=1)
    # step rhythm: |acc| repeats at the step rate, ~1.4-2.6 Hz for walking and running
    walk_lags = range(int(rate_hz / 2.6), int(rate_hz / 1.4) + 1)
    acf_walk = _max_acf(am, walk_lags)
    tilt = mean[:, 2] / np.maximum(mean_norm, 1e-9)
    return np.column_stack([mean, mean_norm, am.std(axis=1), np.sqrt((gm**2).mean(axis=1)),
                            gm.max(axis=1), acf_walk, tilt])


def window_labels(codes, n=WINDOW_N):
    """Majority label per window from per-sample integer codes."""
    W = len(codes) // n
    c = np.asarray(codes[: W * n]).reshape(W, n)
    return np.array([np.bincount(row).argmax() for row in c])


def uv_level_per_window(uvi_measured, sensor_gain, clear_sky_uvi, n_windows):
    """Mean of measured/(gain x clear-sky) per window; NaN when the sun is too low (< 0.5 UVI clear)."""
    k = UV_PER_WINDOW + (1 if WINDOW_N % IMU_SAMPLES_PER_PACKET else 0)
    # map UV samples to windows by position: UV sample j ends at IMU index 8j+7
    win = ((np.arange(len(uvi_measured)) * IMU_SAMPLES_PER_PACKET + IMU_SAMPLES_PER_PACKET - 1) // WINDOW_N)
    keep = win < n_windows
    expected = np.maximum(sensor_gain, 0.05) * clear_sky_uvi
    num = np.bincount(win[keep], weights=uvi_measured[keep], minlength=n_windows)
    den = np.bincount(win[keep], weights=expected[keep], minlength=n_windows)
    lvl = np.where(den > 0.5 * k, num / np.maximum(den, 1e-9), np.nan)
    return lvl


def build_windows(t_unix_s, acc_g, gyro_dps, uv_idx, uv_counts, uv_gain, uv_res_bits, lat, lon,
                  day_start_utc, forecast_uvi_hourly=None, site_scale=1.0, albedo=0.05,
                  ref_counts_per_uvi=None):
    """All features for a session in the device format.

    forecast_uvi_hourly: 24 hourly all-sky UVI values for the local day (e.g. Open-Meteo); in
    synthetic evaluation we pass NASA POWER for that day (a perfect forecast, so optimistic).
    """
    from ..bodydose import sensor_gain, sensor_tilt_nz
    from ..clearsky import clear_sky_uvi
    from ..geometry import diffuse_fraction
    from ..solar import solar_position, sun_vector
    from ..synth.generate import cmf_from_power
    from ..uv import counts_per_uvi

    imu = imu_features(acc_g, gyro_dps)
    W = imu.shape[0]
    t = np.asarray(t_unix_s)
    t_start = t[np.arange(W) * WINDOW_N]

    kw = {} if ref_counts_per_uvi is None else {"ref_counts_per_uvi": ref_counts_per_uvi}
    uvi_meas = np.asarray(uv_counts, dtype=float) / counts_per_uvi(uv_gain, uv_res_bits, **kw)
    tu = t[uv_idx]
    elev, az = solar_position(tu, lat, lon)
    P = IMU_SAMPLES_PER_PACKET
    acc_w = np.asarray(acc_g)[np.asarray(uv_idx)[:, None] - np.arange(P)[::-1][None, :]]
    # gain assuming open sky: the detector does not know the environment yet
    gain = sensor_gain(sun_vector(elev, az), diffuse_fraction(elev), albedo, np.ones(len(tu)),
                       nz=sensor_tilt_nz(acc_w))
    clear = clear_sky_uvi(elev) * site_scale
    level = uv_level_per_window(uvi_meas, gain, clear, W)

    cmf = None
    if forecast_uvi_hourly is not None:
        hourly = cmf_from_power(forecast_uvi_hourly, day_start_utc, lat, lon, site_scale)
        hour = np.clip(((t_start - day_start_utc) // 3600).astype(int), 0, 23)
        cmf = np.clip(hourly[hour], 0.0, 1.0)
    win = (np.arange(len(tu)) * P + P - 1) // WINDOW_N
    keep = win < W
    clear_w = (np.bincount(win[keep], weights=clear[keep], minlength=W)
               / np.maximum(np.bincount(win[keep], minlength=W), 1))
    return Windows(t_start, imu, level, cmf, clear_w)


# --- Extended features for learned classifiers ------------------------------------------
# Same 5 s windows. Sign of acc y is dropped (wrist roll and left/right mounting differ
# between people and datasets); everything else is either a magnitude or along the forearm
# (x) / out of the wrist (z), which align between PAMAP2 and our device.

ML_FEATURES = ("acc_mean_x", "acc_mean_y_abs", "acc_mean_z", "acc_mean_norm", "acc_mag_std",
               "gyro_rms", "gyro_max", "acf_walk", "tilt_nz",
               "acc_std_x", "acc_std_y", "acc_std_z", "gyro_std_x", "gyro_std_y", "gyro_std_z",
               "acc_mag_p10", "acc_mag_p90",
               "acc_band_0_1hz", "acc_band_1_3hz", "acc_band_3_8hz", "gyro_peak_hz")


def _band_fractions(x, rate_hz, bands):
    """Share of (de-meaned) spectral energy in each band, per window. x: (W, n)."""
    x = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1.0 / rate_hz)
    total = spec[:, 1:].sum(axis=1) + 1e-12
    return [spec[:, (f >= lo) & (f < hi)].sum(axis=1) / total for lo, hi in bands], spec, f


def imu_features_ml(acc_g, gyro_dps, rate_hz=IMU_RATE_HZ):
    """(W, len(ML_FEATURES)) features over non-overlapping 5 s windows.

    PORT NOTE: uses an FFT of 100 samples; a direct DFT is fine in TypeScript.
    """
    base = imu_features(acc_g, gyro_dps, rate_hz)
    n = int(WINDOW_S * rate_hz)
    W = base.shape[0]
    a = np.asarray(acc_g[: W * n], dtype=float).reshape(W, n, 3)
    g = np.asarray(gyro_dps[: W * n], dtype=float).reshape(W, n, 3)
    am = np.linalg.norm(a, axis=2)
    gm = np.linalg.norm(g, axis=2)
    bands, _, _ = _band_fractions(am, rate_hz, ((0.3, 1.0), (1.0, 3.0), (3.0, 8.0)))
    _, gspec, f = _band_fractions(gm, rate_hz, ())
    peak = f[1:][np.argmax(gspec[:, 1:], axis=1)]
    i = {name: k for k, name in enumerate(IMU_FEATURES)}
    return np.column_stack([
        base[:, i["acc_mean_x"]], np.abs(base[:, i["acc_mean_y"]]), base[:, i["acc_mean_z"]],
        base[:, i["acc_mean_norm"]], base[:, i["acc_mag_std"]], base[:, i["gyro_rms"]],
        base[:, i["gyro_max"]], base[:, i["acf_walk"]], base[:, i["tilt_nz"]],
        a.std(axis=1), g.std(axis=1),
        np.percentile(am, 10, axis=1), np.percentile(am, 90, axis=1),
        *bands, peak,
    ])
