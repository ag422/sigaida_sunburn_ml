import numpy as np
import pytest

from sunrisk import config
from sunrisk.contracts import MedObservation, RiskStep, SunscreenApplication, UserProfile
from sunrisk.tracker import ExposureTracker


def det(fitz=2, parts=("forearm",)):
    return ExposureTracker(UserProfile(fitz), uncertain=False, body_parts=parts)


def test_uvi8_type2_no_sunscreen_is_20_8_minutes():
    # 250 J/m^2 / (8 * 0.025 W/m^2) = 1250 s = 20.83 min
    t = det()
    t.step(RiskStep(0.0, 8.0))
    assert t.minutes_left()[0, 0] == pytest.approx(1250 / 60, abs=0.05)


def test_time_left_drops_as_dose_accumulates():
    t = det()
    t.step(RiskStep(0.0, 8.0))
    t.step(RiskStep(600.0, 8.0))   # 10 minutes at UVI 8
    assert t.minutes_left()[0, 0] == pytest.approx(1250 / 60 - 10, abs=0.05)
    assert t.dose[0, 0] == pytest.approx(120.0)


def test_already_burned_is_zero():
    t = det()
    t.step(RiskStep(0.0, 10.0))
    t.step(RiskStep(3600.0, 10.0))
    assert t.minutes_left()[0, 0] == 0.0


def test_zero_uv_is_beyond_horizon():
    t = det()
    t.step(RiskStep(0.0, 0.0))
    assert np.isinf(t.minutes_left()[0, 0])
    assert t.output().parts[0].beyond_horizon


def test_sunscreen_extends_time_left_but_wears_off():
    plain, protected = det(), det()
    protected.apply_sunscreen(SunscreenApplication(0.0, 30, 2.0))
    for t in (plain, protected):
        t.step(RiskStep(0.0, 8.0))
    # with wear-off, time-left is longer than no sunscreen but shorter than SPF x 20.8 min
    m = protected.minutes_left()[0, 0]
    assert plain.minutes_left()[0, 0] * 2 < m < 30 * 1250 / 60


def test_sunscreen_only_on_listed_parts():
    t = det(parts=("forearm", "face"))
    t.apply_sunscreen(SunscreenApplication(0.0, 30, 2.0, body_parts=("face",)))
    t.step(RiskStep(0.0, 8.0))
    m = t.minutes_left()[0]
    assert m[1] > m[0]
    assert t.output().first_to_burn == "forearm"


def test_swim_reduces_protection():
    t = det()
    t.apply_sunscreen(SunscreenApplication(0.0, 30, 2.0))
    t.step(RiskStep(0.0, 5.0, activity="swimming"))
    before = t.protect.copy()
    t.step(RiskStep(1.0, 5.0, activity="standing"))
    assert t.protect[0, 0] == pytest.approx(before[0, 0] * config.SWIM_RETENTION.value, rel=1e-3)


def test_exposure_ratio_scales_dose():
    t = det(parts=("forearm", "shoulders"))
    t.step(RiskStep(0.0, 8.0, exposure_ratios={"shoulders": 0.5}))
    t.step(RiskStep(600.0, 8.0))
    assert t.dose[0, 1] == pytest.approx(0.5 * t.dose[0, 0])


def test_monte_carlo_range_brackets_point_estimate_and_alerts_on_p10():
    t = ExposureTracker(UserProfile(2), rng=np.random.default_rng(42), body_parts=("forearm",))
    t.step(RiskStep(0.0, 8.0))
    q = t.output().parts[0].minutes_left
    assert q.p10 < 1250 / 60 < q.p90
    assert q.p10 < q.p50 < q.p90
    t.step(RiskStep(1200.0, 8.0))   # 20 min in: p10 should be under the alert lead
    assert t.output().alert


def test_med_observations_feed_the_posterior():
    burned_early = UserProfile(3, [MedObservation(150.0, 1.0)] * 3)
    t1 = ExposureTracker(burned_early, uncertain=False, body_parts=("forearm",))
    t2 = ExposureTracker(UserProfile(3), uncertain=False, body_parts=("forearm",))
    assert t1.med[0] < t2.med[0]


def test_forecast_curve_is_used():
    t = det()
    t.step(RiskStep(0.0, 8.0))
    halved = t.minutes_left(uvi_curve=np.full(720, 4.0))[0, 0]
    assert halved == pytest.approx(2 * 1250 / 60, abs=0.1)
