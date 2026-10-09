"""Solar position from latitude, longitude and UTC time. numpy only.

Implements the NOAA Solar Calculator equations (after Meeus, *Astronomical Algorithms*).
Accuracy is about 0.01 deg for 1800-2100, far better than this project needs.

Conventions:
- elevation in degrees above the horizon, with an atmospheric-refraction correction
- azimuth in degrees clockwise from true north (90 = east, 180 = south)
- local frame is ENU: x = east, y = north, z = up

PORT NOTE: straight scalar trig; ports to TypeScript line by line.
"""

from __future__ import annotations

import numpy as np

_R = np.pi / 180.0


def solar_position(t_unix_s, lat_deg, lon_deg):
    """Return (elevation_deg, azimuth_deg); inputs broadcast."""
    t = np.asarray(t_unix_s, dtype=float)
    lat = np.asarray(lat_deg, dtype=float) * _R

    jd = t / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0

    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)   # mean longitude
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)                      # mean anomaly
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = m * _R
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * mr) * 0.000289)
    omega = (125.04 - 1934.136 * jc) * _R
    app_long = (l0 + c - 0.00569 - 0.00478 * np.sin(omega)) * _R
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = (mean_obliq + 0.00256 * np.cos(omega)) * _R
    decl = np.arcsin(np.sin(obliq) * np.sin(app_long))

    y = np.tan(obliq / 2.0) ** 2
    l0r = l0 * _R
    eq_time_min = 4.0 / _R * (y * np.sin(2 * l0r) - 2 * ecc * np.sin(mr)
                              + 4 * ecc * y * np.sin(mr) * np.cos(2 * l0r)
                              - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * ecc * ecc * np.sin(2 * mr))

    minutes_utc = np.mod(t, 86400.0) / 60.0
    true_solar_min = np.mod(minutes_utc + eq_time_min + 4.0 * np.asarray(lon_deg, dtype=float), 1440.0)
    hour_angle = (true_solar_min / 4.0 - 180.0) * _R

    cos_zen = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle), -1.0, 1.0)
    zen = np.arccos(cos_zen)
    elev = 90.0 - zen / _R

    # azimuth (NOAA form), guarded at the zenith/poles
    denom = np.cos(lat) * np.sin(zen)
    cos_az = np.clip((np.sin(lat) * cos_zen - np.sin(decl)) / np.where(np.abs(denom) < 1e-12, 1e-12, denom), -1.0, 1.0)
    a = np.arccos(cos_az) / _R
    az = np.where(hour_angle > 0, np.mod(a + 180.0, 360.0), np.mod(540.0 - a, 360.0))

    return elev + _refraction_deg(elev), az


def _refraction_deg(elev):
    """NOAA approximation of atmospheric refraction (degrees)."""
    e = np.asarray(elev, dtype=float)
    te = np.tan(np.clip(e, -89.0, 89.9) * _R)
    arcsec = np.select(
        [e > 85.0, e > 5.0, e > -0.575],
        [0.0,
         58.1 / te - 0.07 / te**3 + 0.000086 / te**5,
         1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))],
        default=-20.772 / te,
    )
    return arcsec / 3600.0


def sun_vector(elev_deg, az_deg):
    """Unit vector toward the sun in ENU, shape (..., 3)."""
    e = np.asarray(elev_deg, dtype=float) * _R
    a = np.asarray(az_deg, dtype=float) * _R
    return np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], axis=-1)
