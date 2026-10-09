from datetime import datetime, timezone

import numpy as np
import pytest

from sunrisk import config
from sunrisk.contracts import RiskStep, UserProfile
from sunrisk.forecast import blend, expected_uvi_curve, forecast_cmf, minutes_to_dose
from sunrisk.tracker import ExposureTracker

LAT, LON = 25.76, -80.19
NOON_ISH = datetime(2024, 7, 15, 16, 0, tzinfo=timezone.utc).timestamp()   # 11:00 local standard


def test_minutes_to_dose():
    assert minutes_to_dose(np.full(100, 8.0), 250.0) == pytest.approx(1250 / 60, abs=0.01)
    assert minutes_to_dose(np.zeros(100), 250.0) == np.inf


def test_curve_follows_the_sun():
    c = expected_uvi_curve(NOON_ISH, LAT, LON)
    peak = int(np.argmax(c))
    assert 60 <= peak <= 110            # Miami solar noon ~12:20 local standard time, ~80 min ahead
    assert c[-1] == 0.0                 # 12 h later it's night


def test_brief_cloud_relaxes_to_forecast():
    clear = np.full(120, 10.0)
    k_fc = np.ones(120)
    curve = blend(clear, k_fc, uvi_now=3.5, tau_min=config.PERSIST_TAU_MIN["cloud"].value)
    assert curve[0] == pytest.approx(3.5)
    assert curve[60] > 9.9              # an hour later the cloud has (expectedly) passed


def test_indoor_projection_ignores_current_reading():
    a = expected_uvi_curve(NOON_ISH, LAT, LON, uvi_now=0.1, environment="indoor")
    b = expected_uvi_curve(NOON_ISH, LAT, LON)
    assert np.allclose(a, b)


def test_forecast_cmf_scales_curve():
    hours = NOON_ISH + np.arange(-2, 10) * 3600.0
    from sunrisk.clearsky import clear_sky_uvi
    from sunrisk.solar import solar_position
    half = 0.5 * clear_sky_uvi(solar_position(hours + 1800, LAT, LON)[0])
    c_full = expected_uvi_curve(NOON_ISH, LAT, LON)
    c_half = expected_uvi_curve(NOON_ISH, LAT, LON, forecast_t=hours, forecast_uvi=half)
    assert c_half[60:240].mean() == pytest.approx(0.5 * c_full[60:240].mean(), rel=0.05)


def test_no_forecast_means_clear_sky():
    t = np.array([0.0, 60.0])
    assert forecast_cmf(t, np.ones(2)).tolist() == [1.0, 1.0]


def test_tracker_with_forecast_curve_is_more_cautious_in_a_passing_cloud():
    tr = ExposureTracker(UserProfile(2), uncertain=False, body_parts=("forearm",))
    tr.step(RiskStep(NOON_ISH, 3.0, environment="cloud"))
    constant = tr.output()
    fc = tr.output(uvi_curve=expected_uvi_curve(NOON_ISH, LAT, LON, uvi_now=3.0, environment="cloud"))
    assert fc.projection == "forecast" and constant.projection == "constant"
    assert fc.parts[0].minutes_left.p50 < constant.parts[0].minutes_left.p50
