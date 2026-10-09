"""Device stream -> risk-engine inputs: ambient UV estimate and per-body-part ratios.

Pure function over arrays in the session format. Environment and posture labels come from
event detection (or ground truth when evaluating the geometry alone).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bodydose, config
from .contracts import BODY_PARTS, IMU_RATE_HZ, IMU_SAMPLES_PER_PACKET
from .geometry import (ACTIVITY_POSTURE, SUBMERGED_WHEN_SWIMMING, posture_normals,
                       rotate_heading, surface_ratio)
from .solar import solar_position, sun_vector
from .uv import counts_per_uvi

RATIO_UPDATE_S = 60.0   # body-part ratios change slowly; recompute once a minute


@dataclass
class ExposureEstimate:
    t_unix_s: np.ndarray        # (M,)
    uvi_measured: np.ndarray    # (M,) sensor reading converted to UVI (nominal calibration)
    sensor_gain: np.ndarray     # (M,) expected sensor / local-horizontal ratio
    uvi_local: np.ndarray       # (M,) estimated local horizontal UVI
    ratios: np.ndarray          # (M, P) per body part, relative to uvi_local


def postures_from_activity(activity_names, lying_posture="supine"):
    return np.array([lying_posture if a == "lying" else ACTIVITY_POSTURE[a] for a in activity_names])


def estimate_exposure(t_unix_s, uv_idx, uv_counts, uv_gain, uv_res_bits, acc_g, lat, lon,
                      environment, posture, albedo, mag_ut=None, ref_counts_per_uvi=None,
                      window_s=bodydose.DEFAULT_WINDOW_S, parts=BODY_PARTS,
                      hold=None) -> ExposureEstimate:
    """environment / posture: names per UV sample (M,).

    hold: optional bool per UV sample; where True the previous ambient estimate is kept
    (sensor dark but not confirmed indoors, e.g. a sleeve over it). Conservative.
    """
    P = IMU_SAMPLES_PER_PACKET
    M = len(uv_idx)
    tu = np.asarray(t_unix_s)[uv_idx]
    kw = {} if ref_counts_per_uvi is None else {"ref_counts_per_uvi": ref_counts_per_uvi}
    uvi_meas = np.asarray(uv_counts, dtype=float) / counts_per_uvi(uv_gain, uv_res_bits, **kw)

    elev, az = solar_position(tu, lat, lon)
    sun = sun_vector(elev, az)
    env = np.asarray(environment)
    fd, direct = bodydose.environment_geometry(env, elev)

    acc_w = np.asarray(acc_g)[uv_idx[:, None] - np.arange(P)[::-1][None, :]]          # (M, 8, 3)
    if mag_ut is not None:
        mag_w = np.asarray(mag_ut)[uv_idx[:, None] - np.arange(P)[::-1][None, :]]
        normal = bodydose.sensor_normal_from_acc_mag(acc_w, mag_w)
        gain = bodydose.sensor_gain(sun, fd, albedo, direct, normal_enu=normal)
    else:
        gain = bodydose.sensor_gain(sun, fd, albedo, direct, nz=bodydose.sensor_tilt_nz(acc_w))

    # while swimming the sensor is under water part of each stroke
    u = config.SWIM_SENSOR_SUBMERGED_FRACTION.value
    gain = gain * np.where(np.asarray(posture) == "swimming", 1.0 - u + u * config.WATER_UV_FACTOR.value, 1.0)

    uv_rate = IMU_RATE_HZ / P
    w = max(1, int(window_s * uv_rate))
    # held samples are left out, so the window keeps (forward-fills) the pre-darkness estimate
    local = bodydose.estimate_ambient(uvi_meas, gain, w, exclude=hold)
    local = np.nan_to_num(local, nan=0.0)

    # body parts: recompute once a minute, heading unknown -> averaged
    ratios = np.ones((M, len(parts)))
    step = max(1, int(RATIO_UPDATE_S * uv_rate))
    keys = np.arange(0, M, step)
    headings = np.arange(36) * 10.0
    post = np.asarray(posture)
    for k0 in keys:
        sl = slice(k0, min(k0 + step, M))
        normals = rotate_heading(posture_normals(post[k0], parts)[None, :, :], headings[:, None])  # (36,P,3)
        r = surface_ratio(normals, sun[k0][None, None, :], fd[k0], albedo, 1.0, direct[k0]).mean(axis=0)
        if post[k0] == "swimming":
            sub = np.array([SUBMERGED_WHEN_SWIMMING.get(p, 0.0) for p in parts])
            r = r * (1.0 - sub + sub * config.WATER_UV_FACTOR.value)
        denom = float(bodydose.local_horizontal_factor(fd[k0], sun[k0][2] > 0, direct[k0]))
        ratios[sl] = r / max(denom, 1e-9)

    # forearm = the sensor site: measured / estimated local, smoothed over the same window
    if "forearm" in parts:
        c = np.cumsum(uvi_meas)
        mean_meas = (c - np.concatenate([np.zeros(w), c[:-w]])[:M]) / np.minimum(np.arange(1, M + 1), w)
        ratios[:, list(parts).index("forearm")] = np.where(local > 1e-3, mean_meas / np.maximum(local, 1e-3), 0.0)

    return ExposureEstimate(tu, uvi_meas, gain, local, ratios)
