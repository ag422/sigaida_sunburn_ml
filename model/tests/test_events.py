from datetime import datetime, timezone

import numpy as np
import pytest

from sunrisk.contracts import ACTIVITIES, ENVIRONMENTS
from sunrisk.events.evaluate import confusion_matrix, per_class_scores
from sunrisk.events.features import IMU_FEATURES, WINDOW_N, Windows, build_windows, imu_features, window_labels
from sunrisk.events.rules import RuleDetector
from sunrisk.pipeline import estimate_exposure
from sunrisk.synth.generate import generate_session
from sunrisk.synth.scenario import Scenario, Segment

DAY = datetime(2024, 7, 15, 5, 0, tzinfo=timezone.utc).timestamp()
LAT, LON = 25.76, -80.19
CLEAR = np.array([0, 0, 0, 0, 0, 0, 0.2, 1, 3, 5.5, 8, 9.8, 10.5, 10, 8.5, 6, 3.5, 1.5, 0.3, 0, 0, 0, 0, 0])


def windows_for(sc, seed=0, hourly=None, occl=0.0):
    ss = generate_session(sc, LAT, LON, DAY, np.random.default_rng(seed), hourly, occlusions_per_hour=occl)
    w = build_windows(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, ss.uv_counts, ss.uv_gain,
                      ss.uv_res_bits, LAT, LON, DAY, hourly)
    return ss, w


def test_confusion_matrix_and_scores():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    p, r, f, s = per_class_scores(cm)
    assert r[1] == 1.0 and p[1] == pytest.approx(2 / 3) and s.tolist() == [2, 2, 1]


def test_window_labels_majority():
    codes = np.array([1] * 60 + [2] * 40 + [3] * 100)
    assert window_labels(codes).tolist() == [1, 3]


def test_imu_features_shape_and_rest_values():
    n = WINDOW_N * 3
    acc = np.tile([0.0, 0.0, 1.0], (n, 1))
    f = imu_features(acc, np.zeros((n, 3)))
    assert f.shape == (3, len(IMU_FEATURES))
    assert f[0, IMU_FEATURES.index("acc_mean_norm")] == pytest.approx(1.0)
    assert f[0, IMU_FEATURES.index("tilt_nz")] == pytest.approx(1.0)


@pytest.mark.parametrize("activity", ["standing", "walking", "vigorous", "swimming"])
def test_rules_detect_distinct_activities(activity):
    _, w = windows_for(Scenario(11 * 60, [Segment(10, "sun", activity)]))
    _, act = RuleDetector().predict(w)
    assert np.mean(act == ACTIVITIES.index(activity)) > 0.9


def test_sun_shade_indoor_sequence():
    sc = Scenario(11 * 60, [Segment(15, "sun", "standing"), Segment(1, "sun", "walking"),
                            Segment(15, "shade", "sitting"), Segment(20, "indoor", "sitting")])
    ss, w = windows_for(sc, hourly=CLEAR)
    env, _ = RuleDetector().predict(w)
    truth = window_labels(ss.environment)
    for name in ("sun", "shade"):
        m = truth == ENVIRONMENTS.index(name)
        assert np.mean(env[m] == ENVIRONMENTS.index(name)) > 0.85, name
    ind = np.where(truth == ENVIRONMENTS.index("indoor"))[0]
    late = ind[ind > ind[0] + 300 / 5 + 6]          # after the 5-minute confirmation
    assert np.mean(env[late] == ENVIRONMENTS.index("indoor")) > 0.9


def test_brief_darkness_is_held_not_zeroed():
    # sensor covered for 2 minutes in full sun: detector must not call it indoor,
    # and the pipeline must keep the previous ambient estimate
    sc = Scenario(11 * 60, [Segment(20, "sun", "sitting")])
    ss = generate_session(sc, LAT, LON, DAY, np.random.default_rng(0), CLEAR, occlusions_per_hour=0)
    cover = slice(len(ss.t_unix_s) // 2, len(ss.t_unix_s) // 2 + 2 * 60 * 20)
    uv_cov = (ss.uv_idx >= cover.start) & (ss.uv_idx < cover.stop)
    counts = ss.uv_counts.copy()
    counts[uv_cov] = (counts[uv_cov] * 0.05).astype(int)
    w = build_windows(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, counts, 18, 20, LAT, LON, DAY, CLEAR)
    det = RuleDetector()
    env, _ = det.predict(w)
    assert not np.any(env == ENVIRONMENTS.index("indoor"))
    hold_w = det.hold_mask()
    assert hold_w.any()
    hold_uv = hold_w[np.minimum(ss.uv_idx // WINDOW_N, len(hold_w) - 1)]
    est = estimate_exposure(ss.t_unix_s, ss.uv_idx, counts, 18, 20, ss.acc_g, LAT, LON,
                            np.array(["sun"] * len(counts)), ss.posture[ss.uv_idx], 0.15, hold=hold_uv)
    before = est.uvi_local[np.where(uv_cov)[0][0] - 150: np.where(uv_cov)[0][0] - 25].mean()
    assert est.uvi_local[uv_cov].mean() == pytest.approx(before, rel=0.05)
    # without the hold, the covered minutes would read close to zero
    raw = estimate_exposure(ss.t_unix_s, ss.uv_idx, counts, 18, 20, ss.acc_g, LAT, LON,
                            np.array(["sun"] * len(counts)), ss.posture[ss.uv_idx], 0.15)
    assert raw.uvi_local[uv_cov][-50:].mean() < 0.2 * before


def test_detector_without_uv_defaults_to_sun():
    w = Windows(np.zeros(4), np.zeros((4, len(IMU_FEATURES))))
    env = RuleDetector().predict_environment(w, np.zeros(4, dtype=int))
    assert (env == ENVIRONMENTS.index("sun")).all()
