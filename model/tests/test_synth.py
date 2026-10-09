from datetime import datetime, timezone

import numpy as np
import pytest

from sunrisk import config
from sunrisk.contracts import ACTIVITIES, BODY_PARTS, ENVIRONMENTS
from sunrisk.synth.clouds import minute_clouds
from sunrisk.synth.generate import generate_session
from sunrisk.synth.scenario import Scenario, Segment, beach_day, random_scenario

DAY = datetime(2024, 7, 15, 5, 0, tzinfo=timezone.utc).timestamp()   # Miami local midnight (UTC-5)
LAT, LON = 25.76, -80.19


def short(seg, minutes=10):
    return Scenario(11 * 60, [Segment(minutes, *seg)])


@pytest.fixture(scope="module")
def beach():
    return generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(0))


def test_shapes_and_rates(beach):
    n, m = beach.t_unix_s.size, beach.uv_counts.size
    assert n == beach_day().total_minutes * 60 * 20
    assert m == n // 8
    assert np.allclose(np.diff(beach.t_unix_s), 0.05)
    assert beach.body_part_uvi.shape == (m, len(BODY_PARTS))


def test_deterministic_given_seed():
    a = generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(5))
    b = generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(5))
    assert np.array_equal(a.uv_counts, b.uv_counts) and np.array_equal(a.acc_g, b.acc_g)


def test_accelerometer_reads_one_g_at_rest():
    s = generate_session(short(("sun", "lying", "supine")), LAT, LON, DAY, np.random.default_rng(1))
    assert np.linalg.norm(s.acc_g, axis=1).mean() == pytest.approx(1.0, abs=0.02)
    assert s.acc_g[:, 2].mean() > 0.95          # sensor face up when lying with palm down


def test_activity_intensity_ordering():
    def gyro_rms(act):
        s = generate_session(short(("sun", act)), LAT, LON, DAY, np.random.default_rng(2))
        return np.sqrt((s.gyro_dps**2).mean())
    assert gyro_rms("standing") < gyro_rms("walking") < gyro_rms("vigorous")


def test_indoor_and_shade_reduce_uv():
    rng = lambda: np.random.default_rng(3)   # noqa: E731
    sun = generate_session(short(("sun", "sitting")), LAT, LON, DAY, rng(), occlusions_per_hour=0)
    shade = generate_session(short(("shade", "sitting")), LAT, LON, DAY, rng(), occlusions_per_hour=0)
    indoor = generate_session(short(("indoor", "sitting")), LAT, LON, DAY, rng(), occlusions_per_hour=0)
    assert indoor.uvi_sensor.mean() < 0.05 * sun.uvi_sensor.mean()
    assert indoor.uvi_sensor.mean() < shade.uvi_sensor.mean() < 0.7 * sun.uvi_sensor.mean()
    # ambient (unobstructed) UV is the same in all three
    assert shade.uvi_ambient.mean() == pytest.approx(sun.uvi_ambient.mean())


def test_counts_follow_sensor_uvi(beach):
    expected = beach.uvi_sensor * config.LTR390_COUNTS_PER_UVI.value * beach.cal_true
    bright = expected > 1000
    assert np.median(beach.uv_counts[bright] / expected[bright]) == pytest.approx(1.0, abs=0.01)


def test_cloud_label_only_when_power_cmf_given(beach):
    assert not np.any(beach.environment == ENVIRONMENTS.index("cloud"))
    hourly = np.full(24, 3.0)     # much lower than clear sky at midday -> lots of cloud
    s = generate_session(beach_day(), LAT, LON, DAY, np.random.default_rng(0), uvi_hourly_power=hourly)
    assert np.any(s.environment == ENVIRONMENTS.index("cloud"))


def test_minute_clouds_match_hourly_mean():
    rng = np.random.default_rng(0)
    cmf = np.array([1.0, 0.8, 0.6, 0.2] * 50)
    trans, cloudy = minute_clouds(cmf, rng)
    hourly = trans.reshape(-1, 60).mean(axis=1).reshape(50, 4).mean(axis=0)
    assert hourly == pytest.approx([1.0, 0.8, 0.6, 0.2], abs=0.05)
    assert cloudy.reshape(-1, 60)[3::4].all()        # overcast hours fully cloudy


def test_swimming_wets_sensor_and_body():
    s = generate_session(short(("sun", "swimming")), LAT, LON, DAY, np.random.default_rng(4), occlusions_per_hour=0)
    i_back, i_chest = BODY_PARTS.index("upper_back"), BODY_PARTS.index("chest")
    assert s.body_part_uvi[:, i_back].mean() > 5 * s.body_part_uvi[:, i_chest].mean()


def test_random_scenarios_are_valid():
    rng = np.random.default_rng(7)
    for _ in range(20):
        sc = random_scenario(rng, hours=4)
        assert sc.total_minutes >= 240
        assert all(seg.activity in ACTIVITIES for seg in sc.segments)


def test_segment_validation():
    with pytest.raises(ValueError):
        Segment(5, "cloud", "sitting")       # cloud is not scriptable
    with pytest.raises(ValueError):
        Segment(5, "sun", "dancing")
