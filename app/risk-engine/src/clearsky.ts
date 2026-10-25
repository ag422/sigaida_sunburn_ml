/** Clear-sky UV Index from solar elevation, Madronich 2007 (model/sunrisk/clearsky.py). */

import * as config from "./config";

/**
 * Clear-sky UVI on a horizontal surface; zero when the sun is below the horizon.
 * `scale` is a site correction (e.g. fitted against NASA POWER clear days for aerosols).
 */
export function clearSkyUvi(elevDeg: number, ozoneDu = config.DEFAULT_OZONE_DU,
                            altitudeKm = 0.0, scale = 1.0): number {
  const mu0 = Math.sin((Math.min(Math.max(elevDeg, 0.0), 90.0) * Math.PI) / 180.0);
  const uvi = config.CLEARSKY_A * mu0 ** config.CLEARSKY_B * (ozoneDu / 300.0) ** config.CLEARSKY_C;
  return uvi * (1.0 + config.UV_INCREASE_PER_KM * altitudeKm) * scale;
}
