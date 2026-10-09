from datetime import datetime, timezone

import numpy as np

from sunrisk.contracts import BODY_PARTS, SunscreenApplication, UserProfile
from sunrisk.session import analyze_session
from sunrisk.synth.generate import generate_session
from sunrisk.synth.scenario import Scenario, Segment

DAY = datetime(2024, 7, 15, 5, 0, tzinfo=timezone.utc).timestamp()
LAT, LON = 25.76, -80.19


def run(segments, sunscreen=()):
    ss = generate_session(Scenario(10 * 60, segments), LAT, LON, DAY, np.random.default_rng(0),
                          cal_log_sd=0.0, occlusions_per_hour=0.0)
    return analyze_session(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, ss.uv_counts, 18, 20, LAT, LON,
                           DAY, UserProfile(2), list(sunscreen), n_particles=100, output_every_s=60)


def test_end_to_end_outputs_have_consistent_shapes():
    r = run([Segment(30, "sun", "standing")])
    K, P = len(r.t_out), len(BODY_PARTS)
    assert r.fraction_med.shape == (K, P, 3) and r.minutes_left.shape == (K, P, 3)
    assert np.all(np.diff(r.fraction_med[:, :, 1], axis=0) >= -1e-9)    # dose only accumulates
    assert np.all(r.fraction_med[:, :, 0] <= r.fraction_med[:, :, 2])


def test_sun_burns_faster_than_shade_and_sunscreen_helps():
    sun = run([Segment(40, "sun", "sitting")])
    shade = run([Segment(40, "shade", "sitting")])
    protected = run([Segment(40, "sun", "sitting")],
                    [SunscreenApplication(DAY + 10 * 3600 - 1, 30, 2.0)])
    j = BODY_PARTS.index("shoulders")
    assert shade.fraction_med[-1, j, 1] < 0.5 * sun.fraction_med[-1, j, 1]
    assert protected.fraction_med[-1, j, 1] < 0.2 * sun.fraction_med[-1, j, 1]


def test_alert_is_immediate_unprotected_but_delayed_by_sunscreen():
    bare = run([Segment(60, "sun", "lying", "supine")])
    assert bare.alert[0]          # UVI ~9 at 10:00 in Miami, type II: p10 < 15 min straight away
    spf = run([Segment(60, "sun", "lying", "supine")], [SunscreenApplication(DAY + 10 * 3600 - 1, 30, 2.0)])
    assert not spf.alert[0]
