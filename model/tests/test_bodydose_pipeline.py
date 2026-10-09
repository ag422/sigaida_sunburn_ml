from datetime import datetime, timezone

import numpy as np
import pytest

from sunrisk import bodydose, config
from sunrisk.contracts import BODY_PARTS, ENVIRONMENTS
from sunrisk.pipeline import estimate_exposure
from sunrisk.solar import sun_vector
from sunrisk.synth import motion
from sunrisk.synth.generate import generate_session
from sunrisk.synth.scenario import Scenario, Segment, beach_day
from sunrisk.uv import counts_per_uvi

DAY = datetime(2024, 7, 15, 5, 0, tzinfo=timezone.utc).timestamp()
LAT, LON = 25.76, -80.19


def test_counts_per_uvi_scales_with_gain_and_time():
    assert counts_per_uvi(18, 20) == config.LTR390_COUNTS_PER_UVI.value
    assert counts_per_uvi(9, 20) == pytest.approx(counts_per_uvi(18, 20) / 2)
    assert counts_per_uvi(18, 18) == pytest.approx(counts_per_uvi(18, 20) / 4)


def test_upward_sensor_gain_is_one():
    s = sun_vector(np.array([50.0]), np.array([90.0]))
    g = bodydose.sensor_gain(s, np.array([0.5]), 0.0, np.array([1.0]), nz=np.array([1.0]))
    assert g == pytest.approx([1.0])


def test_shade_gain_independent_of_sun_direction():
    # no direct beam -> only tilt matters
    a = bodydose.sensor_gain(sun_vector(np.array([20.0]), np.array([0.0])), np.array([1.0]), 0.0,
                             np.array([0.0]), nz=np.array([0.0]))
    assert a == pytest.approx([0.5])


def test_compass_recovers_sensor_normal():
    rng = np.random.default_rng(0)
    n = np.tile([0.6, -0.8, 0.0], (8, 1))
    f = np.tile([0.0, 0.0, -1.0], (8, 1))
    acc, _, mag = motion.imu_from_orientation(f, n, np.zeros((8, 3)), 0.05, rng)
    est = bodydose.sensor_normal_from_acc_mag(acc[None], mag[None])[0]
    assert est == pytest.approx([0.6, -0.8, 0.0], abs=0.03)


def test_estimate_ambient_ratio_of_sums_and_carry_forward():
    meas = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    gain = np.array([1.0, 0.5, 0.1, 0.1, 0.1])     # last three face away -> skipped
    est = bodydose.estimate_ambient(meas, gain, window_samples=2)
    assert est[1] == pytest.approx(10.0 / 1.5)
    assert est[4] == pytest.approx(est[2]) and not np.isnan(est[4])


def _estimate(ss, mag=False):
    env = np.array(ENVIRONMENTS)[ss.environment[ss.uv_idx]]
    return env, estimate_exposure(ss.t_unix_s, ss.uv_idx, ss.uv_counts, ss.uv_gain, ss.uv_res_bits,
                                  ss.acc_g, LAT, LON, env, ss.posture[ss.uv_idx],
                                  config.ALBEDO["sand"].value, mag_ut=ss.mag_ut if mag else None)


@pytest.mark.parametrize("mag", [False, True])
def test_ambient_estimate_unbiased_on_beach_day(mag):
    ss = generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(1), cal_log_sd=0.0,
                          occlusions_per_hour=0.0)
    env, est = _estimate(ss, mag)
    out = env != "indoor"
    assert est.uvi_local[out].sum() / ss.uvi_local[out].sum() == pytest.approx(1.0, abs=0.04)


def test_body_part_doses_within_15_percent():
    ss = generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(1), cal_log_sd=0.0,
                          occlusions_per_hour=0.0)
    _, est = _estimate(ss)
    dose_est = (est.uvi_local[:, None] * est.ratios).sum(axis=0)
    dose_true = ss.body_part_uvi.sum(axis=0)
    assert dose_est / dose_true == pytest.approx(np.ones(len(BODY_PARTS)), abs=0.15)


def test_magnetometer_helps_while_walking():
    sc = Scenario(11 * 60, [Segment(20, "sun", "walking")])
    ss = generate_session(sc, LAT, LON, DAY, np.random.default_rng(3), cal_log_sd=0.0, occlusions_per_hour=0.0)
    err = {}
    for mag in (False, True):
        _, est = _estimate(ss, mag)
        err[mag] = abs(est.uvi_local[200:].sum() / ss.uvi_local[200:].sum() - 1)
    assert err[True] < err[False]
