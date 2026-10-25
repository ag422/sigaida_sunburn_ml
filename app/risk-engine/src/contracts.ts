/**
 * Risk engine interface types, mirroring model/sunrisk/contracts.py (prose: docs/contracts.md).
 * Field names stay snake_case so JSON from the Python side maps across unchanged.
 */

export const CONTRACT_VERSION = 1;

// "forearm" is the sensor site (device on the wrist, facing out from the back of the forearm).
export const BODY_PARTS = [
  "forearm", "face", "neck", "shoulders", "upper_back", "chest", "thighs", "shins",
] as const;
export const ENVIRONMENTS = ["sun", "shade", "cloud", "indoor"] as const;
export const ACTIVITIES = [
  "lying", "sitting", "standing", "walking", "light_activity", "vigorous", "swimming",
] as const;

export type BodyPart = (typeof BODY_PARTS)[number];
export type Environment = (typeof ENVIRONMENTS)[number];
export type Activity = (typeof ACTIVITIES)[number];

export interface MedObservation {
  effective_dose_j_m2: number; // dose to the observed skin site, after sunscreen
  burned: number; // 0/1, or a probability from the redness classifier
}

export interface UserProfile {
  fitzpatrick: number; // 1-6
  med_observations?: MedObservation[];
}

export interface SunscreenApplication {
  t_unix_s: number;
  spf_label: number;
  amount_mg_cm2?: number | null; // null -> typical (assumption)
  body_parts?: string[] | null; // null -> all exposed parts
}

export interface ForecastPoint {
  t_unix_s: number;
  uvi: number; // forecast all-sky UVI (e.g. Open-Meteo hourly)
}

/** One engine update, typically every few seconds. */
export interface RiskStep {
  t_unix_s: number;
  uvi_ambient: number; // sensor-derived, orientation-corrected ambient UVI
  environment?: string; // one of ENVIRONMENTS, default "sun"
  activity?: string; // one of ACTIVITIES, default "standing"
  exposure_ratios?: Record<string, number> | null; // per body part, relative to ambient; null -> 1.0
}

export interface Quantiles {
  p10: number;
  p50: number;
  p90: number;
}

export interface BodyPartRisk {
  body_part: string;
  minutes_left: Quantiles; // alert on p10 (pessimistic); Infinity beyond the horizon
  fraction_med_used: Quantiles;
  effective_spf: number; // median across particles
  beyond_horizon: boolean; // true if even p10 did not reach MED within the horizon
}

export interface RiskOutput {
  t_unix_s: number;
  parts: BodyPartRisk[];
  first_to_burn: string; // body part with the lowest p10 minutes_left
  alert: boolean; // p10 minutes_left below the alert threshold
  projection: "constant" | "forecast";
  contract_version: number;
}
