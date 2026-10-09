from datetime import datetime, timezone

import numpy as np
import pytest

from sunrisk import geometry as g
from sunrisk.clearsky import clear_sky_uvi
from sunrisk.solar import solar_position, sun_vector


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def test_greenwich_june_solstice_noon():
    elev, az = solar_position(ts("2024-06-21T12:00"), 51.4769, 0.0)
    assert elev == pytest.approx(90 - 51.4769 + 23.44, abs=0.1)
    assert az == pytest.approx(180, abs=1.5)


@pytest.mark.parametrize("day,minutes", [("2024-11-03", 720 - 16.4), ("2024-02-11", 720 + 14.2)])
def test_equation_of_time_extremes(day, minutes):
    t = ts(day + "T00:00") + np.arange(0, 86400, 10.0)
    elev, _ = solar_position(t, 0.0, 0.0)
    assert (t[elev.argmax()] - ts(day + "T00:00")) / 60 == pytest.approx(minutes, abs=0.5)


def test_southern_hemisphere_sun_is_north_at_noon():
    # Sydney local solar noon in December ~01:55 UTC
    _, az = solar_position(ts("2024-12-21T01:55"), -33.87, 151.21)
    assert min(az, 360 - az) < 10


def test_sun_below_horizon_at_night():
    elev, _ = solar_position(ts("2024-06-21T00:00"), 51.48, 0.0)
    assert elev < 0


def test_sun_vector_unit_and_up():
    v = sun_vector(90.0, 0.0)
    assert v == pytest.approx([0, 0, 1], abs=1e-9)
    assert np.linalg.norm(sun_vector(30, 123)) == pytest.approx(1)


def test_clear_sky_uvi():
    assert clear_sky_uvi(90.0) == pytest.approx(12.5)
    assert clear_sky_uvi(-5.0) == 0.0
    assert clear_sky_uvi(60.0, ozone_du=350) < clear_sky_uvi(60.0)
    assert clear_sky_uvi(60.0, altitude_km=2) == pytest.approx(clear_sky_uvi(60.0) * 1.2)


def test_horizontal_surface_ratio_is_one_without_ground():
    s = sun_vector(50, 200)
    assert g.surface_ratio(np.array([0, 0, 1.0]), s, g.diffuse_fraction(50), albedo=0.0) == pytest.approx(1.0)


def test_shade_and_cloud_remove_direct_beam():
    s = sun_vector(60, 180)
    fd = g.diffuse_fraction(60)
    toward_sun = s
    sunny = g.surface_ratio(toward_sun, s, fd, 0.0)
    shaded = g.surface_ratio(toward_sun, s, fd, 0.0, sky_view=0.5, direct_visible=0.0)
    assert shaded < 0.5 * fd + 1e-9 < sunny
    cloudy = g.surface_ratio(np.array([0, 0, 1.0]), s, g.diffuse_fraction(60, cloud=True), 0.0)
    assert cloudy == pytest.approx(1.0)        # all-diffuse, horizontal: ratio stays 1 (attenuation is in E_h)


def test_diffuse_fraction_increases_toward_horizon():
    assert g.diffuse_fraction(80) < g.diffuse_fraction(30) < g.diffuse_fraction(5) < 1


def test_rotate_heading():
    fwd = np.array([1.0, 0, 0])
    assert g.rotate_heading(fwd, 0) == pytest.approx([0, 1, 0], abs=1e-9)    # facing north
    assert g.rotate_heading(fwd, 90) == pytest.approx([1, 0, 0], abs=1e-9)   # facing east
    assert g.rotate_heading(np.array([0, 1.0, 0]), 0) == pytest.approx([-1, 0, 0], abs=1e-9)  # left = west


def test_body_part_ratios_make_sense():
    s = sun_vector(70, 180)
    fd = g.diffuse_fraction(70)
    up = dict(zip(g.BODY_PARTS, g.body_part_ratios("upright", s, fd, 0.02)))
    sup = dict(zip(g.BODY_PARTS, g.body_part_ratios("supine", s, fd, 0.02)))
    prone = dict(zip(g.BODY_PARTS, g.body_part_ratios("prone", s, fd, 0.02)))
    assert up["shoulders"] > up["face"] > 0.2          # high sun: horizontal beats vertical
    assert sup["chest"] > 0.9 and sup["upper_back"] < 0.1
    assert prone["upper_back"] > 0.9 and prone["chest"] < 0.1
    swim = dict(zip(g.BODY_PARTS, g.body_part_ratios("swimming", s, fd, 0.08)))
    assert swim["upper_back"] > swim["chest"]


def test_known_heading_differs_from_average_only_for_vertical_parts():
    s = sun_vector(40, 180)
    fd = g.diffuse_fraction(40)
    face_sun = g.body_part_ratios("upright", s, fd, 0.0, heading_deg=180.0)
    face_away = g.body_part_ratios("upright", s, fd, 0.0, heading_deg=0.0)
    i_face, i_sh = g.BODY_PARTS.index("face"), g.BODY_PARTS.index("shoulders")
    assert face_sun[i_face] > 2 * face_away[i_face]
    assert face_sun[i_sh] == pytest.approx(face_away[i_sh])
