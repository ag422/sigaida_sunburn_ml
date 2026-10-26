/**
 * Numbers the engine depends on, copied from model/sunrisk/config.py.
 *
 * config.py is the source of truth (it also holds sources and confidence levels). Change values
 * there first, then here; test/parity.test.ts fails if the two disagree.
 */

export const W_PER_M2_PER_UVI = 0.025;

export const MED_MEDIAN_BY_FITZPATRICK: Record<number, number> = {
  1: 200.0,
  2: 250.0,
  3: 350.0,
  4: 450.0,
  5: 600.0,
  6: 1000.0,
};
export const MED_PRIOR_LOG_SD = 0.35;
export const BURN_LOGISTIC_SCALE = 0.15;

export const LAB_SUNSCREEN_AMOUNT = 2.0; // mg/cm^2
export const TYPICAL_APPLIED_AMOUNT = 0.8; // mg/cm^2
export const APPLIED_AMOUNT_LOG_SD = 0.4;
export const SUNSCREEN_WEAR_TAU_H = 3.0;
export const SUNSCREEN_WEAR_TAU_ACTIVE_H = 1.5;
export const SWIM_RETENTION = 0.5;

export const CLEARSKY_A = 12.5;
export const CLEARSKY_B = 2.42;
export const CLEARSKY_C = -1.23;
export const DEFAULT_OZONE_DU = 300.0;
export const UV_INCREASE_PER_KM = 0.1;

export const LTR390_COUNTS_PER_UVI = 2300.0;
export const SENSOR_CAL_LOG_SD = 0.2;

export const PERSIST_TAU_MIN: Record<string, number> = {
  sun: 60.0,
  cloud: 10.0,
  shade: 30.0,
  indoor: 0.0,
};

export const N_PARTICLES = 500;
export const ALERT_QUANTILE = 0.1;
export const ALERT_LEAD_MIN = 15.0;
export const PROJECTION_HORIZON_MIN = 12 * 60;
export const PROJECTION_STEP_MIN = 1.0;
