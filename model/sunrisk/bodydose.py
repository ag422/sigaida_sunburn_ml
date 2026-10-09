"""Estimate ambient UV and per-body-part exposure from the wrist sensor.

All ratios here are relative to the LOCAL horizontal irradiance: what a flat, upward-facing
sensor at the wearer's position would read (so in shade it is already the shaded value).
This is the `uvi_ambient` the risk engine expects.

Steps per UV sample (2.5 Hz):
1. Sensor tilt from the accelerometer (mean of the 8 IMU samples): n_z = cos(angle from up).
2. Expected sensor gain = surface ratio of the sensor normal relative to local horizontal.
   Without a magnetometer the sensor's compass direction is unknown, so the gain is
   averaged over all azimuths. With a magnetometer, the normal's azimuth is computed
   (tilt-compensated compass; ignores magnetic declination and assumes calibrated hard/soft iron).
3. Ambient = sum(measured) / sum(gain) over a sliding window, using only samples where the
   sensor faces the sky well enough (gain > min_gain). Ratio-of-sums is robust to arm swing.
4. Body-part ratios from posture + sun position, averaged over unknown torso heading.
   The wrist cannot observe torso heading, so this averaging is permanent, not a stopgap.

PORT NOTE: the sliding window uses cumulative sums; in TypeScript keep running sums.
"""

from __future__ import annotations

import numpy as np

from .contracts import BODY_PARTS
from .geometry import body_part_ratios, diffuse_fraction, surface_ratio

DEFAULT_WINDOW_S = 60.0
DEFAULT_MIN_GAIN = 0.25


def local_horizontal_factor(fd, sun_up, direct_visible):
    """Local horizontal irradiance relative to unobstructed horizontal, per unit sky_view."""
    return np.where(sun_up, (1.0 - fd) * direct_visible, 0.0) + fd


def sensor_tilt_nz(acc_window_g):
    """acc (M, 8, 3) -> z-component (up) of the sensor normal, (M,)."""
    a = np.asarray(acc_window_g).mean(axis=1)
    return np.clip(a[:, 2] / np.maximum(np.linalg.norm(a, axis=1), 1e-9), -1.0, 1.0)


def sensor_normal_from_acc_mag(acc_window_g, mag_window_ut):
    """Tilt-compensated compass: sensor normal in ENU, (M, 3)."""
    up = np.asarray(acc_window_g).mean(axis=1)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    m = np.asarray(mag_window_ut).mean(axis=1)
    east = np.cross(m, up)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(up, east)
    # world components of the sensor z axis = z components of the world axes in sensor frame
    return np.stack([east[:, 2], north[:, 2], up[:, 2]], axis=1)


def sensor_gain(sun_enu, fd, albedo, direct_visible, nz=None, normal_enu=None, n_az=36):
    """Expected sensor reading / local horizontal. Give nz (azimuth unknown) or normal_enu."""
    sun_enu = np.asarray(sun_enu, dtype=float)
    up = sun_enu[:, 2] > 0
    denom = local_horizontal_factor(fd, up, direct_visible)
    if normal_enu is not None:
        r = surface_ratio(normal_enu, sun_enu, fd, albedo, 1.0, direct_visible)
    else:
        psi = np.arange(n_az) * 2 * np.pi / n_az
        h = np.sqrt(1.0 - nz**2)[:, None]
        normals = np.stack([h * np.sin(psi), h * np.cos(psi), np.repeat(nz[:, None], n_az, axis=1)], axis=-1)
        r = surface_ratio(normals, sun_enu[:, None, :], np.asarray(fd)[:, None], albedo, 1.0,
                          np.asarray(direct_visible)[:, None]).mean(axis=1)
    return r / np.maximum(denom, 1e-9)


def estimate_ambient(uvi_measured, gain, window_samples, min_gain=DEFAULT_MIN_GAIN, exclude=None):
    """Sliding ratio-of-sums estimate of local horizontal UVI, (M,).

    Samples whose sensor faces away (gain <= min_gain) or that are excluded (e.g. sensor
    covered) are skipped. If a whole window has none, the last estimate is carried forward
    (NaN at the very start).
    """
    use = gain > min_gain
    if exclude is not None:
        use &= ~np.asarray(exclude, dtype=bool)
    num = np.cumsum(np.where(use, uvi_measured, 0.0))
    den = np.cumsum(np.where(use, gain, 0.0))
    w = int(window_samples)
    num_w = num - np.concatenate([np.zeros(w), num[:-w]])[: num.size]
    den_w = den - np.concatenate([np.zeros(w), den[:-w]])[: den.size]
    valid = den_w > 1e-6
    est = np.full(num.size, np.nan)
    est[valid] = num_w[valid] / den_w[valid]
    idx = np.maximum.accumulate(np.where(valid, np.arange(num.size), -1))   # forward fill
    return np.where(idx >= 0, est[np.maximum(idx, 0)], np.nan)


def body_ratios_local(posture, sun_enu, fd, albedo, direct_visible, heading_deg=None, parts=BODY_PARTS):
    """Per-body-part irradiance relative to LOCAL horizontal for one sample, (P,)."""
    r = body_part_ratios(posture, sun_enu, fd, albedo, heading_deg=heading_deg, sky_view=1.0,
                         direct_visible=direct_visible, parts=parts)
    return r / max(float(local_horizontal_factor(fd, sun_enu[2] > 0, direct_visible)), 1e-9)


def environment_geometry(env_names, elev_deg):
    """(fd, direct_visible) per sample from environment labels (sun/shade/cloud/indoor)."""
    env = np.asarray(env_names)
    fd = diffuse_fraction(elev_deg, cloud=(env != "sun"))
    return fd, (env == "sun").astype(float)
