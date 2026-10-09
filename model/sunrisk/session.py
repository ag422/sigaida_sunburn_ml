"""End-to-end analysis of a recorded (or synthetic) session: the same chain the app runs live.

    device arrays -> event detection -> ambient UV + body-part ratios -> Monte Carlo tracker
    -> forecast-aware time-left per body part

Pure: arrays in, arrays out. The tracker is stepped once per 5 s detection window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .contracts import (ACTIVITIES, BODY_PARTS, ENVIRONMENTS, RiskStep, SunscreenApplication,
                        UserProfile)
from .events.features import WINDOW_N, build_windows
from .events.rules import RuleDetector
from .forecast import expected_uvi_curve
from .pipeline import estimate_exposure, postures_from_activity
from .tracker import ExposureTracker


@dataclass
class SessionResult:
    t_window: np.ndarray          # (W,) window start times
    environment: np.ndarray       # (W,) detected names
    activity: np.ndarray          # (W,) detected names
    uvi_local: np.ndarray         # (W,) estimated local horizontal UVI
    ratios: np.ndarray            # (W, P)
    t_out: np.ndarray             # (K,) times of risk outputs
    fraction_med: np.ndarray      # (K, P, 3) p10/p50/p90 fraction of MED used
    minutes_left: np.ndarray      # (K, P, 3) p10/p50/p90, forecast-aware
    minutes_left_constant: np.ndarray  # (K, P) p10 with the constant-UV projection (for comparison)
    alert: np.ndarray             # (K,) bool
    first_to_burn: np.ndarray     # (K,) names


def analyze_session(t_unix_s, acc_g, gyro_dps, uv_idx, uv_counts, uv_gain, uv_res_bits,
                    lat, lon, day_start_utc, profile: UserProfile,
                    sunscreen: list[SunscreenApplication] = (), forecast_uvi_hourly=None,
                    site_scale=1.0, albedo=config.ALBEDO["sand"].value, mag_ut=None,
                    rng=None, n_particles=config.N_PARTICLES, output_every_s=120.0,
                    detector=None) -> SessionResult:
    detector = detector or RuleDetector()
    w = build_windows(t_unix_s, acc_g, gyro_dps, uv_idx, uv_counts, uv_gain, uv_res_bits, lat, lon,
                      day_start_utc, forecast_uvi_hourly, site_scale, albedo)
    if hasattr(detector, "predict_session"):          # learned activity model needs raw IMU
        env_w, act_w = detector.predict_session(acc_g, gyro_dps, w)
    else:
        env_w, act_w = detector.predict(w)
    hold_w = detector.hold_mask() if hasattr(detector, "hold_mask") else np.zeros(len(env_w), bool)
    W = len(env_w)

    # per-UV-sample labels from the window they fall in
    uv_win = np.minimum(np.asarray(uv_idx) // WINDOW_N, W - 1)
    env_names = np.array(ENVIRONMENTS)[env_w]
    act_names = np.array(ACTIVITIES)[act_w]
    est = estimate_exposure(t_unix_s, uv_idx, uv_counts, uv_gain, uv_res_bits, acc_g, lat, lon,
                            env_names[uv_win], postures_from_activity(act_names)[uv_win], albedo,
                            mag_ut=mag_ut, hold=hold_w[uv_win])

    # window means for the tracker
    cnt = np.maximum(np.bincount(uv_win, minlength=W), 1)
    local_w = np.bincount(uv_win, weights=est.uvi_local, minlength=W) / cnt
    ratios_w = np.stack([np.bincount(uv_win, weights=est.ratios[:, j], minlength=W) / cnt
                         for j in range(len(BODY_PARTS))], axis=1)

    fc_t = None if forecast_uvi_hourly is None else day_start_utc + np.arange(24) * 3600.0
    tracker = ExposureTracker(profile, rng=rng or np.random.default_rng(0), n_particles=n_particles)
    pending = sorted(sunscreen, key=lambda s: s.t_unix_s)
    every = max(1, int(output_every_s / (WINDOW_N / 20.0)))
    outs = {k: [] for k in ("t", "frac", "mins", "const", "alert", "first")}
    for k in range(W):
        t = w.t_start[k]
        while pending and pending[0].t_unix_s <= t:
            tracker.apply_sunscreen(pending.pop(0))
        tracker.step(RiskStep(t, local_w[k], env_names[k], act_names[k],
                              dict(zip(BODY_PARTS, ratios_w[k]))))
        if k % every == 0:
            curve = expected_uvi_curve(t, lat, lon, uvi_now=local_w[k], environment=env_names[k],
                                       forecast_t=fc_t, forecast_uvi=forecast_uvi_hourly,
                                       site_scale=site_scale)
            o = tracker.output(uvi_curve=curve)
            c = tracker.output()
            outs["t"].append(t)
            outs["frac"].append([[p.fraction_med_used.p10, p.fraction_med_used.p50, p.fraction_med_used.p90] for p in o.parts])
            outs["mins"].append([[p.minutes_left.p10, p.minutes_left.p50, p.minutes_left.p90] for p in o.parts])
            outs["const"].append([p.minutes_left.p10 for p in c.parts])
            outs["alert"].append(o.alert)
            outs["first"].append(o.first_to_burn)

    return SessionResult(w.t_start, env_names, act_names, local_w, ratios_w,
                         np.array(outs["t"]), np.array(outs["frac"]), np.array(outs["mins"]),
                         np.array(outs["const"]), np.array(outs["alert"]), np.array(outs["first"]))

