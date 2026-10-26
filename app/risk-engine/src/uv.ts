/** UV Index <-> erythemal irradiance and dose integration (model/sunrisk/uv.py). */

import * as config from "./config";

export const W_PER_UVI = config.W_PER_M2_PER_UVI;
export const J_PER_SED = 100.0; // Standard Erythema Dose, by definition

/** UV Index -> erythemally weighted irradiance (W/m^2). */
export function uviToIrradiance(uvi: number): number {
  return uvi * W_PER_UVI;
}

export function irradianceToUvi(eEry: number): number {
  return eEry / W_PER_UVI;
}

/** Raw LTR390 UVS counts (at the reference gain/resolution) -> UVI. */
export function countsToUvi(counts: number, countsPerUvi = config.LTR390_COUNTS_PER_UVI): number {
  return counts / countsPerUvi;
}

/** Trapezoidal dose (J/m^2) from irradiance samples (W/m^2) at times tS (seconds). */
export function integrateDose(tS: ArrayLike<number>, eEry: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 1; i < tS.length; i++) sum += 0.5 * (eEry[i] + eEry[i - 1]) * (tS[i] - tS[i - 1]);
  return sum;
}

// LTR390 integration time per resolution setting (datasheet; verify against the part we have).
const INTEGRATION_MS: Record<number, number> = {
  20: 400.0, 19: 200.0, 18: 100.0, 17: 50.0, 16: 25.0, 13: 3.125,
};

/** Sensitivity scales with gain and integration time; reference is gain 18, 20-bit. */
export function countsPerUvi(gain: number, resBits: number,
                             refCountsPerUvi = config.LTR390_COUNTS_PER_UVI): number {
  const ms = INTEGRATION_MS[resBits];
  if (ms === undefined) throw new Error(`unsupported LTR390 resolution: ${resBits} bits`);
  return refCountsPerUvi * (gain / 18.0) * (ms / 400.0);
}
