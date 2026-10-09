"""Synthetic wrist IMU: sensor orientation and accel/gyro/mag for each activity.

This is a stand-in for real data, not a biomechanics model. Frequencies and amplitudes below
are rough, chosen so the activities differ in the ways a classifier would use (arm swing
rhythm, impact magnitude, wrist tilt). They are NOT model assumptions and do not feed the risk
engine.

Sensor frame (device on the LEFT wrist, face out from the back of the wrist):
    x = along the forearm toward the hand, z = out of the sensor face, y = z cross x.
Heading frame: x = facing direction, y = left, z = up (see geometry.py).
"""

from __future__ import annotations

import numpy as np

from ..geometry import rotate_heading

G = 9.80665
EARTH_FIELD_ENU_UT = np.array([0.0, 20.0, -45.0])   # roughly mid-latitude northern hemisphere

# per-activity motion parameters (synthetic only)
WALK_SWING_DEG, WALK_STEP_HZ = 25.0, 0.9
VIGOROUS_SWING_DEG, VIGOROUS_HZ = 50.0, 1.4
SWIM_STROKE_HZ, SWIM_ROLL_DEG = 0.45, 30.0
GESTURE_RATE_PER_S, GESTURE_LIFT_DEG = 1 / 120, 70.0
ACC_NOISE_G, GYRO_NOISE_DPS, MAG_NOISE_UT = 0.01, 0.5, 0.5


def rodrigues(v, axis, angle_rad):
    """Rotate vectors v (N,3) about unit axes (N,3 or 3) by angles (N,)."""
    k = np.broadcast_to(np.asarray(axis, dtype=float), v.shape)
    a = np.asarray(angle_rad, dtype=float)[:, None]
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * np.sum(k * v, axis=1, keepdims=True) * (1 - np.cos(a))


def _smooth_noise(rng, n, dt, amp_deg, freqs=(0.2, 0.45)):
    """Sum of random-phase sinusoids, (n,) radians."""
    t = np.arange(n) * dt
    out = np.zeros(n)
    for f in freqs:
        out += np.sin(2 * np.pi * f * rng.uniform(0.8, 1.2) * t + rng.uniform(0, 2 * np.pi))
    return np.deg2rad(amp_deg) * out / len(freqs)


def segment_motion(activity: str, posture: str | None, n: int, dt: float, rng: np.random.Generator):
    """Sensor axes in the heading frame for one segment.

    Returns (fx (n,3), nz (n,3), lin_acc_g (n,3) in heading frame, underwater (n,) bool).
    """
    t = np.arange(n) * dt
    ones = np.ones((n, 1))
    ex, ey, ez = np.eye(3)
    lin = np.zeros((n, 3))
    underwater = np.zeros(n, dtype=bool)

    if activity in ("standing", "walking", "light_activity", "vigorous"):
        f = ones * -ez
        nz = ones * ey
        if activity == "standing":
            ang = _smooth_noise(rng, n, dt, 3.0, (0.1, 0.25))
        elif activity == "walking":
            hz = WALK_STEP_HZ * rng.uniform(0.85, 1.15)
            ph = 2 * np.pi * hz * t
            ang = np.deg2rad(WALK_SWING_DEG) * np.sin(ph)
            lin[:, 2] = 0.25 * np.sin(2 * ph)
            lin[:, 0] = 0.10 * np.sin(2 * ph + np.pi / 2)
        elif activity == "light_activity":
            f = ones * (ex - ez) / np.sqrt(2)
            ang = _smooth_noise(rng, n, dt, 35.0)
            twist = _smooth_noise(rng, n, dt, 30.0, (0.15, 0.35))
            nz = rodrigues(nz, f[0], twist)
        else:  # vigorous
            hz = VIGOROUS_HZ * rng.uniform(0.85, 1.15)
            ph = 2 * np.pi * hz * t
            ang = np.deg2rad(VIGOROUS_SWING_DEG) * np.sin(ph) + _smooth_noise(rng, n, dt, 30.0, (0.3, 0.7))
            lin[:, 2] = 1.2 * np.maximum(np.sin(2 * ph), 0.0) ** 4
            lin[:, 0] = 0.4 * np.sin(ph)
        f, nz = rodrigues(f, ey, ang), rodrigues(nz, ey, ang)

    elif activity == "sitting":
        f = ones * ex
        nz = rodrigues(ones * ez, ex, np.full(n, np.deg2rad(20.0)))
        ang = _smooth_noise(rng, n, dt, 2.0, (0.05, 0.2))
        # occasional gestures (drink, phone): forearm lifts toward the face for 4-8 s
        lift = np.zeros(n)
        k = 0
        while True:
            k += int(rng.exponential(1 / GESTURE_RATE_PER_S) / dt)
            if k >= n:
                break
            dur = int(rng.uniform(4, 8) / dt)
            w = np.sin(np.linspace(0, np.pi, dur)) ** 2
            seg = slice(k, min(k + dur, n))
            lift[seg] = w[: seg.stop - seg.start]
            k += dur
        ang = ang - np.deg2rad(GESTURE_LIFT_DEG) * lift      # negative about y raises the hand
        f, nz = rodrigues(f, ey, ang), rodrigues(nz, ey, ang)

    elif activity == "lying":
        f = ones * (-ex if posture != "prone" else ex)
        nz = ones * ez
        ang = _smooth_noise(rng, n, dt, 2.0, (0.03, 0.1))
        f, nz = rodrigues(f, ey, ang), rodrigues(nz, ey, ang)

    elif activity == "swimming":
        hz = SWIM_STROKE_HZ * rng.uniform(0.85, 1.15)
        ph = 2 * np.pi * hz * t
        f = rodrigues(ones * ex, ey, ph)          # arm circles forward-down-back-up
        nz = rodrigues(ones * ez, ey, ph)
        roll = np.deg2rad(SWIM_ROLL_DEG) * np.sin(ph)
        f, nz = rodrigues(f, ex, roll), rodrigues(nz, ex, roll)
        lin = -0.5 * f                              # centripetal, toward the shoulder (~0.5 g)
        underwater = f[:, 2] < -0.2
    else:
        raise ValueError(f"unknown activity {activity!r}")

    return f, nz, lin, underwater


def heading_track(activity: str, n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Facing direction over a segment, degrees clockwise from north."""
    h0 = rng.uniform(0, 360)
    if activity in ("lying", "sitting"):
        sigma_per_min = 1.0
    elif activity == "standing":
        sigma_per_min = 10.0
    else:
        sigma_per_min = 25.0
    steps = rng.normal(0.0, sigma_per_min * np.sqrt(dt / 60.0), n)
    return np.mod(h0 + np.cumsum(steps), 360.0)


def imu_from_orientation(f_enu, n_enu, lin_acc_enu_g, dt, rng):
    """Accel (g), gyro (deg/s), mag (uT) in the sensor frame from ENU sensor axes."""
    x = f_enu / np.linalg.norm(f_enu, axis=1, keepdims=True)
    z = n_enu - np.sum(n_enu * x, axis=1, keepdims=True) * x
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=2)                 # columns = sensor axes in ENU

    up = np.array([0.0, 0.0, 1.0])
    acc = np.einsum("nji,nj->ni", R, up + lin_acc_enu_g)        # R^T (g_up + a)
    mag = np.einsum("nji,j->ni", R, EARTH_FIELD_ENU_UT)

    dR = np.einsum("nji,njk->nik", R[:-1], R[1:])               # R_k^T R_{k+1}
    w = 0.5 * np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], axis=1) / dt
    gyro = np.rad2deg(np.vstack([w, w[-1:]]))

    n = len(f_enu)
    acc += rng.normal(0, ACC_NOISE_G, (n, 3))
    gyro += rng.normal(0, GYRO_NOISE_DPS, (n, 3))
    mag += rng.normal(0, MAG_NOISE_UT, (n, 3))
    return acc, gyro, mag


def to_enu(v_heading, heading_deg):
    return rotate_heading(v_heading, heading_deg)
