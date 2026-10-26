/**
 * A scripted beach day that stands in for the wearable until the hardware exists.
 *
 * Ambient UV comes from the engine's own sun position + clear-sky model, so it rises toward
 * solar noon and falls after it. The schedule (what the person is doing) and the per-body-part
 * exposure ratios are hand-written placeholders: the real ratios will come from geometry.py /
 * pipeline.py once they're ported.
 */

import { clearSkyUvi, solarPosition, type RiskStep, type SunscreenApplication } from "@sunrisk/risk-engine";

// Santa Monica, rounded to 0.1 degrees like the contract asks
export const LAT = 34.0;
export const LON = -118.5;
export const UTC_OFFSET_H = -7; // PDT
export const FITZPATRICK = 2;

// 2024-07-15, 10:00 to 18:00 local
export const T_START = Date.UTC(2024, 6, 15, 10 - UTC_OFFSET_H) / 1000;
export const DURATION_MIN = 8 * 60;
export const T_END = T_START + DURATION_MIN * 60;

export interface Segment {
  fromMin: number; // minutes after T_START
  environment: "sun" | "shade" | "cloud";
  activity: string;
  label: string;
}

export const SCHEDULE: Segment[] = [
  { fromMin: 0, environment: "sun", activity: "walking", label: "Walking to the beach" },
  { fromMin: 20, environment: "sun", activity: "lying", label: "Lying on a towel" },
  { fromMin: 80, environment: "sun", activity: "swimming", label: "Swimming" },
  { fromMin: 110, environment: "sun", activity: "sitting", label: "Sitting on the sand" },
  { fromMin: 140, environment: "cloud", activity: "sitting", label: "Sitting, cloud over the sun" },
  { fromMin: 152, environment: "sun", activity: "sitting", label: "Sitting on the sand" },
  { fromMin: 170, environment: "shade", activity: "sitting", label: "Under an umbrella" },
  { fromMin: 230, environment: "sun", activity: "walking", label: "Walking along the shore" },
  { fromMin: 300, environment: "sun", activity: "lying", label: "Lying on a towel" },
  { fromMin: 390, environment: "sun", activity: "light_activity", label: "Beach volleyball" },
];

// SPF 30 at the typical (thin) amount, everywhere except the upper back, the spot people can't
// reach. Protection wears off and the swim removes half, so alerts arrive during the day.
export const MORNING_SUNSCREEN: SunscreenApplication = {
  t_unix_s: T_START, spf_label: 30, amount_mg_cm2: null,
  body_parts: ["forearm", "face", "neck", "shoulders", "chest", "thighs", "shins"],
};

// Share of horizontal ambient UV reaching each body part, by posture. PLACEHOLDERS.
const UPRIGHT = { forearm: 0.6, face: 0.5, neck: 0.4, shoulders: 0.8, upper_back: 0.4, chest: 0.45, thighs: 0.4, shins: 0.35 };
const RATIOS: Record<string, Record<string, number>> = {
  walking: UPRIGHT,
  light_activity: UPRIGHT,
  standing: UPRIGHT,
  sitting: { ...UPRIGHT, thighs: 0.8, shins: 0.5 },
  lying: { forearm: 0.9, face: 0.9, neck: 0.6, shoulders: 0.9, upper_back: 0.0, chest: 1.0, thighs: 1.0, shins: 1.0 },
  // mostly under water (~0.4 transmission); back and shoulders break the surface
  swimming: { forearm: 0.5, face: 0.3, neck: 0.6, shoulders: 0.9, upper_back: 0.8, chest: 0.2, thighs: 0.3, shins: 0.3 },
};

// Rough transmission relative to full sun (config.py: CLOUD_ATTENUATION 0.35; shade is diffuse only)
const ENV_FACTOR = { sun: 1.0, cloud: 0.35, shade: 0.3 };

export function segmentAt(tUnixS: number): Segment {
  const min = (tUnixS - T_START) / 60;
  let seg = SCHEDULE[0];
  for (const s of SCHEDULE) if (s.fromMin <= min) seg = s;
  return seg;
}

/** Next schedule change after t (or the end of the day). */
export function nextChange(tUnixS: number): number {
  const min = (tUnixS - T_START) / 60;
  const next = SCHEDULE.find((s) => s.fromMin > min);
  return next ? T_START + next.fromMin * 60 : T_END;
}

export function clearSkyAt(tUnixS: number): number {
  return clearSkyUvi(solarPosition(tUnixS, LAT, LON).elevation);
}

/** What the wearable + event detection would report at time t. */
export function stepAt(tUnixS: number): RiskStep {
  const seg = segmentAt(tUnixS);
  return {
    t_unix_s: tUnixS,
    uvi_ambient: clearSkyAt(tUnixS) * ENV_FACTOR[seg.environment],
    environment: seg.environment,
    activity: seg.activity,
    exposure_ratios: RATIOS[seg.activity],
  };
}
