import json
from pathlib import Path

import numpy as np
import pytest

from sunrisk.datasets import nasa_power as npw

FIXTURE = Path(__file__).parent / "fixtures" / "nasa_power_la_20240701.json"


@pytest.fixture
def series():
    return npw.parse_hourly(json.loads(FIXTURE.read_text()))


def test_parse(series):
    assert series.lat == pytest.approx(34.05) and series.lon == pytest.approx(-118.24)
    assert series.t_unix_s.size == 24
    uv = series.values[npw.UV_PARAM]
    # Los Angeles, 1 July: solar noon ~20:00 UTC, UVI ~10
    assert 19 <= np.nanargmax(uv) <= 20 and 9 < np.nanmax(uv) < 11
    assert np.isnan(series.values["ALLSKY_KT"]).any()   # -999 at night -> NaN


def test_daily_curves_need_complete_local_days(series):
    starts, rows = npw.daily_uv_curves(series, utc_offset_h=0)
    assert rows.shape == (1, 24)
    _, rows_pdt = npw.daily_uv_curves(series, utc_offset_h=-7)  # spans two partial local days
    assert rows_pdt.shape == (0, 24)


def test_hourly_to_minutes_preserves_level():
    m = npw.hourly_to_minutes(np.array([0.0, 6.0, 6.0, 0.0]))
    assert m.size == 240 and m[90] == pytest.approx(6.0)


def test_error_payload_raises():
    with pytest.raises(ValueError):
        npw.parse_hourly({"header": "failed", "messages": ["bad parameter"]})
