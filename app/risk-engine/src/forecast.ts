/**
 * Forecast-aware expected UV curve for the time-left projection (model/sunrisk/forecast.py).
 *
 * expected(t) = clear_sky(t) x k(t),
 * k(t) = w(t) x k_now + (1 - w(t)) x k_forecast(t),   w(t) = exp(-(t - now) / tau_env)
 *
 * - k_now: what the sensor sees now relative to clear sky (captures current cloud/shade).
 * - k_forecast: forecast all-sky / clear-sky for that hour (1.0 if no forecast, i.e. assume
 *   clear sky: the conservative default).
 * - tau_env: how long the current condition is expected to last (config.PERSIST_TAU_MIN).
 */

import { clearSkyUvi } from "./clearsky";
import * as config from "./config";
import type { ForecastPoint } from "./contracts";
import { interp } from "./numeric";
import { solarPosition } from "./solar";
import { W_PER_UVI } from "./uv";

export const MIN_CLEAR_UVI = 0.3;

export function clearSkyCurve(tNow: number, lat: number, lon: number, nMin: number,
                              siteScale = 1.0, stepMin = 1.0): { t: Float64Array; clear: Float64Array } {
  const t = new Float64Array(nMin);
  const clear = new Float64Array(nMin);
  for (let i = 0; i < nMin; i++) {
    t[i] = tNow + i * stepMin * 60.0;
    clear[i] = clearSkyUvi(solarPosition(t[i], lat, lon).elevation) * siteScale;
  }
  return { t, clear };
}

/** Forecast/clear-sky ratio at times t. Forecast values are hour averages starting at each point's t_unix_s. */
export function forecastCmf(t: Float64Array, forecast: ForecastPoint[] | null | undefined,
                            lat: number, lon: number, siteScale = 1.0): Float64Array {
  if (!forecast || forecast.length === 0) return new Float64Array(t.length).fill(1.0);
  const clip = (x: number) => Math.min(Math.max(x, 0.0), 1.2);
  const ft = forecast.map((f) => f.t_unix_s + 1800.0); // hour centres
  const cmf = forecast.map((f, i) => {
    const fcClear = clearSkyUvi(solarPosition(ft[i], lat, lon).elevation) * siteScale;
    return clip(fcClear > MIN_CLEAR_UVI ? f.uvi / Math.max(fcClear, 1e-9) : 1.0);
  });
  return t.map((x) => clip(interp(x, ft, cmf)));
}

export interface ExpectedCurveOptions {
  uviNow?: number | null;
  environment?: string;
  forecast?: ForecastPoint[] | null;
  horizonMin?: number;
  siteScale?: number;
}

/** Expected local-horizontal UVI at 1-minute steps from now; pass it to ExposureTracker.output(). */
export function expectedUviCurve(tNow: number, lat: number, lon: number,
                                 opts: ExpectedCurveOptions = {}): Float64Array {
  const { uviNow = null, environment = "sun", forecast = null,
          horizonMin = config.PROJECTION_HORIZON_MIN, siteScale = 1.0 } = opts;
  const tau = config.PERSIST_TAU_MIN[environment];
  if (tau === undefined) throw new Error(`unknown environment: ${environment}`);
  const n = Math.trunc(horizonMin / config.PROJECTION_STEP_MIN);
  const { t, clear } = clearSkyCurve(tNow, lat, lon, n, siteScale);
  const kFc = forecastCmf(t, forecast, lat, lon, siteScale);
  return blend(clear, kFc, uviNow, tau);
}

/** Core of the projection on precomputed arrays (clear[0] is now). */
export function blend(clear: Float64Array, kFc: Float64Array, uviNow: number | null, tauMin: number,
                      stepMin = config.PROJECTION_STEP_MIN): Float64Array {
  if (uviNow === null || clear[0] < MIN_CLEAR_UVI || tauMin <= 0) return clear.map((c, i) => c * kFc[i]);
  const kNow = uviNow / clear[0];
  return clear.map((c, i) => {
    const w = Math.exp((-i * stepMin) / tauMin);
    return c * (w * kNow + (1.0 - w) * kFc[i]);
  });
}

/** Minutes until the integral of a 1-min UVI curve reaches doseJm2 (Infinity if never). */
export function minutesToDose(uviCurvePerMin: ArrayLike<number>, doseJm2: number): number {
  let prev = 0.0;
  for (let k = 0; k < uviCurvePerMin.length; k++) {
    const cum = prev + uviCurvePerMin[k] * W_PER_UVI * 60.0;
    if (cum >= doseJm2) {
      const step = cum - prev;
      return step > 0 ? k + (doseJm2 - prev) / step : k;
    }
    prev = cum;
  }
  return Infinity;
}
