"""NASA POWER hourly point data: parsing and day extraction. Pure (dict in, arrays out).

Downloading lives in scripts/fetch_nasa_power.py. API docs: https://power.larc.nasa.gov/docs/

What the data is (and is not):
- ALLSKY_SFC_UV_INDEX is satellite-derived (CERES SYN1deg, ~1 deg grid, ~100 km), hourly.
  It shows realistic daily UV shapes, seasons and cloudy days at a location.
- It is NOT a point measurement: a single passing cloud or local shade is averaged away.
  Minute-scale variability comes from our synthetic generator, not from POWER.
- Timestamps are hour-starting in the requested time standard (we request UTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

UV_PARAM = "ALLSKY_SFC_UV_INDEX"
PARAMETERS = (UV_PARAM, "CLOUD_AMT", "ALLSKY_KT", "T2M")
FILL = -999.0


@dataclass
class PowerSeries:
    lat: float
    lon: float
    t_unix_s: np.ndarray            # hour start, UTC
    values: dict[str, np.ndarray]   # parameter -> values, NaN where missing


def parse_hourly(payload: dict) -> PowerSeries:
    """Parse a POWER hourly point JSON response."""
    if "properties" not in payload:
        raise ValueError(f"not a POWER data response: {payload.get('messages') or payload.get('header')}")
    lon, lat = payload["geometry"]["coordinates"][:2]
    params = payload["properties"]["parameter"]
    fill = payload.get("header", {}).get("fill_value", FILL)
    keys = sorted(next(iter(params.values())).keys())     # "YYYYMMDDHH"
    t = np.array([datetime.strptime(k, "%Y%m%d%H").replace(tzinfo=timezone.utc).timestamp() for k in keys])
    values = {}
    for name, series in params.items():
        v = np.array([series[k] for k in keys], dtype=float)
        v[v == fill] = np.nan
        values[name] = v
    return PowerSeries(float(lat), float(lon), t, values)


def merge(series: list[PowerSeries]) -> PowerSeries:
    """Concatenate several requests (e.g. one per year) for the same point."""
    order = np.argsort(np.concatenate([s.t_unix_s for s in series]))
    names = set.intersection(*(set(s.values) for s in series))
    return PowerSeries(
        series[0].lat, series[0].lon,
        np.concatenate([s.t_unix_s for s in series])[order],
        {n: np.concatenate([s.values[n] for s in series])[order] for n in names},
    )


def daily_uv_curves(s: PowerSeries, utc_offset_h: float = 0.0):
    """Split into local calendar days with all 24 hourly UV values present.

    Returns (day_start_unix_s[n_days], uvi[n_days, 24]).
    """
    local = s.t_unix_s + utc_offset_h * 3600
    day = np.floor(local / 86400).astype(np.int64)
    uv = s.values[UV_PARAM]
    starts, rows = [], []
    for d in np.unique(day):
        m = day == d
        if m.sum() == 24 and not np.isnan(uv[m]).any():
            starts.append(d * 86400 - utc_offset_h * 3600)
            rows.append(uv[m])
    return np.array(starts), np.array(rows).reshape(-1, 24)


def hourly_to_minutes(uvi_hourly: np.ndarray) -> np.ndarray:
    """Hour-average values -> 1-minute series by linear interpolation between hour centres."""
    n = uvi_hourly.size
    centres = np.arange(n) * 60 + 30
    return np.interp(np.arange(n * 60), centres, uvi_hourly)


def fit_site_scale(s: PowerSeries, max_cloud_pct: float = 5.0, min_elev_deg: float = 20.0) -> float:
    """Median POWER / clear-sky-formula ratio over clear hours (absorbs aerosols, ozone, bias)."""
    from ..clearsky import clear_sky_uvi
    from ..solar import solar_position

    sub = s.t_unix_s[:, None] + (np.arange(6) * 10 + 5) * 60.0
    elev, _ = solar_position(sub, s.lat, s.lon)
    model = clear_sky_uvi(elev).mean(axis=1)
    uv = s.values[UV_PARAM]
    clear = (s.values["CLOUD_AMT"] < max_cloud_pct) & (elev[:, 3] > min_elev_deg) & ~np.isnan(uv)
    if clear.sum() < 10:
        return 1.0
    return float(np.median(uv[clear] / model[clear]))
