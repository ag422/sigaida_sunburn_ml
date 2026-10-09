"""Forecast-aware expected UV curve for the time-left projection.

expected(t) = clear_sky(t) x k(t),
k(t) = w(t) x k_now + (1 - w(t)) x k_forecast(t),   w(t) = exp(-(t - now) / tau_env)

- k_now: what the sensor sees now relative to clear sky (captures current cloud/shade).
- k_forecast: forecast all-sky / clear-sky for that hour (1.0 if no forecast, i.e. assume
  clear sky: the conservative default).
- tau_env: how long the current condition is expected to last (config.PERSIST_TAU_MIN).

The shape over the day comes from the sun's position, so the countdown accounts for UV
rising toward noon and falling after it, instead of holding the current reading constant.

PORT NOTE: per-minute loop of scalar maths; solar_position is the only heavy part.
"""

from __future__ import annotations

import numpy as np

from . import config
from .clearsky import clear_sky_uvi
from .solar import solar_position

MIN_CLEAR_UVI = 0.3


def clear_sky_curve(t_now, lat, lon, n_min, site_scale=1.0, step_min=1.0):
    t = t_now + np.arange(n_min) * step_min * 60.0
    elev, _ = solar_position(t, lat, lon)
    return t, clear_sky_uvi(elev) * site_scale


def forecast_cmf(t, clear, forecast_t=None, forecast_uvi=None, lat=None, lon=None, site_scale=1.0):
    """Forecast/clear-sky ratio at times t. Forecast values are hour averages starting at forecast_t."""
    if forecast_t is None or len(forecast_t) == 0:
        return np.ones_like(t)
    ft = np.asarray(forecast_t, dtype=float) + 1800.0              # hour centres
    fu = np.asarray(forecast_uvi, dtype=float)
    elev, _ = solar_position(ft, lat, lon)
    fc_clear = clear_sky_uvi(elev) * site_scale
    cmf = np.where(fc_clear > MIN_CLEAR_UVI, fu / np.maximum(fc_clear, 1e-9), 1.0)
    return np.clip(np.interp(t, ft, np.clip(cmf, 0.0, 1.2)), 0.0, 1.2)


def expected_uvi_curve(t_now, lat, lon, uvi_now=None, environment="sun", forecast_t=None,
                       forecast_uvi=None, horizon_min=config.PROJECTION_HORIZON_MIN,
                       site_scale=1.0):
    """Expected local-horizontal UVI at 1-minute steps from now, (horizon_min,)."""
    n = int(horizon_min / config.PROJECTION_STEP_MIN)
    t, clear = clear_sky_curve(t_now, lat, lon, n, site_scale)
    k_fc = forecast_cmf(t, clear, forecast_t, forecast_uvi, lat, lon, site_scale)
    return blend(clear, k_fc, uvi_now, config.PERSIST_TAU_MIN[environment].value)


def blend(clear, k_fc, uvi_now, tau_min, step_min=config.PROJECTION_STEP_MIN):
    """Core of the projection on precomputed arrays (clear[0] is now)."""
    if uvi_now is None or clear[0] < MIN_CLEAR_UVI or tau_min <= 0:
        return clear * k_fc
    k_now = uvi_now / clear[0]
    w = np.exp(-np.arange(len(clear)) * step_min / tau_min)
    return clear * (w * k_now + (1.0 - w) * k_fc)


def minutes_to_dose(uvi_curve_per_min, dose_j_m2):
    """Minutes until the integral of a 1-min UVI curve reaches dose_j_m2 (inf if never)."""
    from .uv import W_PER_UVI
    cum = np.cumsum(np.asarray(uvi_curve_per_min) * W_PER_UVI * 60.0)
    k = np.searchsorted(cum, dose_j_m2)
    if k >= cum.size:
        return np.inf
    prev = cum[k - 1] if k > 0 else 0.0
    step = cum[k] - prev
    return float(k + (dose_j_m2 - prev) / step) if step > 0 else float(k)
