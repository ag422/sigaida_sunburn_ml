"""Generate a synthetic session: device-format sensor data plus ground truth.

UV chain for each 400 ms UV sample:
    clear-sky UVI(sun elevation) x site scale            (clearsky.py, fitted to NASA POWER)
    x minute cloud transmission                          (clouds.py, matched to hourly POWER CMF)
    = ambient UVI on an unobstructed horizontal surface  -> truth.uvi_ambient
    x surface ratio for the sensor normal, shade/indoor, water, sleeve
    = UVI at the sensor -> LTR390 counts (with true calibration error and shot noise)
Body-part truth uses the same geometry with the scripted posture and true heading.

The model later estimates ambient UVI and body-part ratios from the sensor + IMU alone, using
the SAME geometric model. Evaluating against this truth therefore measures what is lost from
noise, unknown heading and arm motion, NOT whether the geometry itself is right; that needs
real data (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..clearsky import clear_sky_uvi
from ..contracts import BODY_PARTS, IMU_RATE_HZ, IMU_SAMPLES_PER_PACKET
from ..geometry import (ACTIVITY_POSTURE, SUBMERGED_WHEN_SWIMMING, diffuse_fraction,
                        posture_normals, rotate_heading, surface_ratio)
from ..solar import solar_position, sun_vector
from . import motion
from .clouds import minute_clouds
from .scenario import ACT_CODE, ENV_CODE, Scenario

COUNTS_PER_UVI = config.LTR390_COUNTS_PER_UVI.value
MAX_COUNTS = 2**20 - 1


@dataclass
class SyntheticSession:
    # device-format data
    t_unix_s: np.ndarray          # (N,) IMU sample times, 20 Hz
    acc_g: np.ndarray             # (N, 3)
    gyro_dps: np.ndarray          # (N, 3)
    mag_ut: np.ndarray            # (N, 3)
    uv_idx: np.ndarray            # (M,) index into N of each UV conversion (last IMU sample of a packet)
    uv_counts: np.ndarray         # (M,) int
    uv_gain: int
    uv_res_bits: int
    # ground truth
    environment: np.ndarray       # (N,) code into contracts.ENVIRONMENTS (cloud included)
    activity: np.ndarray          # (N,) code into contracts.ACTIVITIES
    posture: np.ndarray           # (N,) str
    heading_deg: np.ndarray       # (N,) torso facing direction
    occluded: np.ndarray          # (N,) sensor covered by sleeve/hand
    uvi_ambient: np.ndarray       # (M,) unobstructed horizontal UVI (incl. clouds)
    uvi_sensor: np.ndarray        # (M,) UVI actually at the sensor
    body_part_uvi: np.ndarray     # (M, P) UVI on each body part (BODY_PARTS order)
    sun_elev_deg: np.ndarray      # (M,)
    sun_az_deg: np.ndarray        # (M,)
    sunscreen_events: list        # [(t_unix_s, spf, amount, parts)]
    cal_true: float               # true counts multiplier vs nominal (calibration error)
    lat: float
    lon: float


def cmf_from_power(uvi_hourly_power, day_start_utc, lat, lon, site_scale=1.0, ozone_du=None):
    """Hourly cloud modification factor = POWER all-sky / our clear-sky (hour averages)."""
    t = day_start_utc + np.arange(24)[:, None] * 3600.0 + (np.arange(6)[None, :] * 10 + 5) * 60.0
    elev, _ = solar_position(t, lat, lon)
    kw = {} if ozone_du is None else {"ozone_du": ozone_du}
    clear = clear_sky_uvi(elev, **kw).mean(axis=1) * site_scale
    p = np.asarray(uvi_hourly_power, dtype=float)
    return np.where(clear > 0.3, p / np.maximum(clear, 1e-9), 1.0)


def generate_session(scenario: Scenario, lat: float, lon: float, day_start_utc: float,
                     rng: np.random.Generator, uvi_hourly_power=None, site_scale: float = 1.0,
                     cal_log_sd: float = config.SENSOR_CAL_LOG_SD.value,
                     occlusions_per_hour: float = 0.5) -> SyntheticSession:
    """day_start_utc: unix time of local (standard-time) midnight for the scenario's date."""
    dt = 1.0 / IMU_RATE_HZ
    seg_n = [int(round(s.minutes * 60 / dt)) for s in scenario.segments]
    N = sum(seg_n)
    t0 = day_start_utc + scenario.start_local_min * 60.0
    t = t0 + np.arange(N) * dt

    # --- per-sample scripted truth and motion ---
    f_enu = np.zeros((N, 3)); n_enu = np.zeros((N, 3)); lin_enu = np.zeros((N, 3))
    underwater = np.zeros(N, dtype=bool)
    env = np.empty(N, dtype=object); act = np.empty(N, dtype=object)
    posture = np.empty(N, dtype=object); heading = np.zeros(N)
    sunscreen_events = []
    k = 0
    for seg, n in zip(scenario.segments, seg_n):
        sl = slice(k, k + n)
        f, nz, lin, uw = motion.segment_motion(seg.activity, seg.posture, n, dt, rng)
        h = motion.heading_track(seg.activity, n, dt, rng)
        f_enu[sl] = rotate_heading(f, h)
        n_enu[sl] = rotate_heading(nz, h)
        lin_enu[sl] = rotate_heading(lin, h)
        underwater[sl] = uw
        env[sl], act[sl], heading[sl] = seg.environment, seg.activity, h
        posture[sl] = seg.posture if seg.activity == "lying" and seg.posture else ACTIVITY_POSTURE[seg.activity]
        if seg.sunscreen:
            sunscreen_events.append((t[k], *seg.sunscreen))
        k += n

    acc, gyro, mag = motion.imu_from_orientation(f_enu, n_enu, lin_enu, dt, rng)

    occluded = np.zeros(N, dtype=bool)
    n_occ = rng.poisson(occlusions_per_hour * N * dt / 3600.0)
    for start in rng.integers(0, N, n_occ):
        occluded[start:start + int(rng.uniform(60, 300) / dt)] = True
    occluded &= env != "indoor"

    # --- sky: sun position, clear sky, clouds (minute resolution) ---
    minute = ((t - day_start_utc) // 60).astype(int)
    if uvi_hourly_power is not None:
        cmf = cmf_from_power(uvi_hourly_power, day_start_utc, lat, lon, site_scale)
        trans_min, cloudy_min = minute_clouds(cmf, rng)
    else:
        trans_min, cloudy_min = np.ones(1440), np.zeros(1440, dtype=bool)
    minute = np.clip(minute, 0, trans_min.size - 1)
    cloudy = cloudy_min[minute]

    env_truth = env.copy()
    env_truth[(env == "sun") & cloudy] = "cloud"

    # --- UV at 2.5 Hz: one conversion per packet of 8 IMU samples ---
    P = IMU_SAMPLES_PER_PACKET
    M = N // P
    uv_idx = np.arange(M) * P + (P - 1)
    tu = t[uv_idx]
    elev, az = solar_position(tu, lat, lon)
    sun = sun_vector(elev, az)
    cl = cloudy[uv_idx]
    fd = diffuse_fraction(elev, cl)
    amb = clear_sky_uvi(elev) * site_scale * trans_min[minute[uv_idx]]

    e_u = env[uv_idx]
    sky_view = np.select([e_u == "shade", e_u == "indoor"],
                         [config.SHADE_SKY_VIEW.value, config.INDOOR_UV_FACTOR.value], 1.0)
    direct_vis = (e_u == "sun").astype(float)
    albedo = config.ALBEDO[scenario.surface].value

    # sensor: average the surface ratio over the 8 IMU samples the conversion integrates
    nn = n_enu[: M * P].reshape(M, P, 3)
    r_sensor = surface_ratio(nn, sun[:, None, :], fd[:, None], albedo, sky_view[:, None], direct_vis[:, None])
    wet = underwater[: M * P].reshape(M, P)
    r_sensor = np.where(wet, r_sensor * config.WATER_UV_FACTOR.value, r_sensor).mean(axis=1)
    occ = occluded[: M * P].reshape(M, P).mean(axis=1)
    r_sensor *= 1.0 - occ * (1.0 - config.SLEEVE_OCCLUSION_FACTOR.value)
    uvi_sensor = amb * r_sensor

    cal_true = float(np.exp(rng.normal(0.0, cal_log_sd)))
    expected = uvi_sensor * COUNTS_PER_UVI * cal_true
    counts = np.clip(np.round(expected + rng.normal(0, 1, M) * np.sqrt(expected + 4.0)), 0, MAX_COUNTS).astype(np.int64)

    # body parts: scripted posture + true heading
    body = np.zeros((M, len(BODY_PARTS)))
    post_u = posture[uv_idx]
    for p in np.unique(post_u):
        m = post_u == p
        normals = rotate_heading(posture_normals(p)[None, :, :], heading[uv_idx][m][:, None])   # (m,P,3)
        r = surface_ratio(normals, sun[m][:, None, :], fd[m][:, None], albedo,
                          sky_view[m][:, None], direct_vis[m][:, None])
        if p == "swimming":
            sub = np.array([SUBMERGED_WHEN_SWIMMING.get(b, 0.0) for b in BODY_PARTS])
            r = r * (1.0 - sub + sub * config.WATER_UV_FACTOR.value)
        body[m] = r
    body *= amb[:, None]
    body[:, BODY_PARTS.index("forearm")] = uvi_sensor      # the skin right next to the sensor

    return SyntheticSession(
        t_unix_s=t, acc_g=acc, gyro_dps=gyro, mag_ut=mag,
        uv_idx=uv_idx, uv_counts=counts, uv_gain=18, uv_res_bits=20,
        environment=np.array([ENV_CODE[e] for e in env_truth]),
        activity=np.array([ACT_CODE[a] for a in act]),
        posture=posture.astype(str), heading_deg=heading, occluded=occluded,
        uvi_ambient=amb, uvi_sensor=uvi_sensor, body_part_uvi=body,
        sun_elev_deg=elev, sun_az_deg=az, sunscreen_events=sunscreen_events,
        cal_true=cal_true, lat=lat, lon=lon,
    )
