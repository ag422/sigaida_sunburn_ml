// Ports of model/tests/test_tracker.py and the forecast/MED checks. These need no fixtures;
// test/parity.test.ts compares against the Python engine number for number.
import { describe, expect, it } from "vitest";
import {
  config, effectiveSpf, ExposureTracker, expectedUviCurve, minutesToDose, PersonalMED, seededRng,
  type UserProfile,
} from "../src";

const det = (fitzpatrick = 2, bodyParts: string[] = ["forearm"]) =>
  new ExposureTracker({ fitzpatrick }, { uncertain: false, bodyParts });

// 250 J/m^2 / (8 * 0.025 W/m^2) = 1250 s = 20.83 min
const TYPE2_UVI8_MIN = 1250 / 60;

describe("ExposureTracker", () => {
  it("UVI 8, type 2, no sunscreen is 20.8 minutes", () => {
    const t = det();
    t.step({ t_unix_s: 0, uvi_ambient: 8 });
    expect(t.minutesLeft()[0]).toBeCloseTo(TYPE2_UVI8_MIN, 1);
  });

  it("time left drops as dose accumulates", () => {
    const t = det();
    t.step({ t_unix_s: 0, uvi_ambient: 8 });
    t.step({ t_unix_s: 600, uvi_ambient: 8 });
    expect(t.minutesLeft()[0]).toBeCloseTo(TYPE2_UVI8_MIN - 10, 1);
    expect(t.dose[0]).toBeCloseTo(120.0, 9);
  });

  it("already burned is zero", () => {
    const t = det();
    t.step({ t_unix_s: 0, uvi_ambient: 10 });
    t.step({ t_unix_s: 3600, uvi_ambient: 10 });
    expect(t.minutesLeft()[0]).toBe(0);
  });

  it("zero UV is beyond the horizon", () => {
    const t = det();
    t.step({ t_unix_s: 0, uvi_ambient: 0 });
    expect(t.minutesLeft()[0]).toBe(Infinity);
    expect(t.output().parts[0].beyond_horizon).toBe(true);
  });

  it("sunscreen extends time left but wears off", () => {
    const plain = det();
    const prot = det();
    prot.applySunscreen({ t_unix_s: 0, spf_label: 30, amount_mg_cm2: 2.0 });
    for (const t of [plain, prot]) t.step({ t_unix_s: 0, uvi_ambient: 8 });
    const m = prot.minutesLeft()[0];
    expect(m).toBeGreaterThan(plain.minutesLeft()[0] * 2);
    expect(m).toBeLessThan(30 * TYPE2_UVI8_MIN);
  });

  it("sunscreen only on listed parts", () => {
    const t = det(2, ["forearm", "face"]);
    t.applySunscreen({ t_unix_s: 0, spf_label: 30, amount_mg_cm2: 2.0, body_parts: ["face"] });
    t.step({ t_unix_s: 0, uvi_ambient: 8 });
    const m = t.minutesLeft();
    expect(m[1]).toBeGreaterThan(m[0]);
    expect(t.output().first_to_burn).toBe("forearm");
  });

  it("swimming reduces protection", () => {
    const t = det();
    t.applySunscreen({ t_unix_s: 0, spf_label: 30, amount_mg_cm2: 2.0 });
    t.step({ t_unix_s: 0, uvi_ambient: 5, activity: "swimming" });
    const before = t.protect[0];
    t.step({ t_unix_s: 1, uvi_ambient: 5, activity: "standing" });
    expect(t.protect[0] / (before * config.SWIM_RETENTION)).toBeCloseTo(1, 3);
  });

  it("exposure ratio scales dose", () => {
    const t = det(2, ["forearm", "shoulders"]);
    t.step({ t_unix_s: 0, uvi_ambient: 8, exposure_ratios: { shoulders: 0.5 } });
    t.step({ t_unix_s: 600, uvi_ambient: 8 });
    expect(t.dose[1]).toBeCloseTo(0.5 * t.dose[0], 9);
  });

  it("Monte Carlo range brackets the point estimate and alerts on p10", () => {
    const t = new ExposureTracker({ fitzpatrick: 2 }, { rng: seededRng(42), bodyParts: ["forearm"] });
    t.step({ t_unix_s: 0, uvi_ambient: 8 });
    const q = t.output().parts[0].minutes_left;
    expect(q.p10).toBeLessThan(TYPE2_UVI8_MIN);
    expect(q.p90).toBeGreaterThan(TYPE2_UVI8_MIN);
    expect(q.p10).toBeLessThan(q.p50);
    expect(q.p50).toBeLessThan(q.p90);
    t.step({ t_unix_s: 1200, uvi_ambient: 8 });
    expect(t.output().alert).toBe(true);
  });

  it("MED observations feed the posterior", () => {
    const obs = { effective_dose_j_m2: 150, burned: 1 };
    const burnedEarly: UserProfile = { fitzpatrick: 3, med_observations: [obs, obs, obs] };
    const t1 = new ExposureTracker(burnedEarly, { uncertain: false });
    const t2 = new ExposureTracker({ fitzpatrick: 3 }, { uncertain: false });
    expect(t1.med[0]).toBeLessThan(t2.med[0]);
  });

  it("uses a forecast curve when given one", () => {
    const t = det();
    t.step({ t_unix_s: 0, uvi_ambient: 8 });
    const halved = t.minutesLeft(new Float64Array(720).fill(4))[0];
    expect(Math.abs(halved - 2 * TYPE2_UVI8_MIN)).toBeLessThan(0.1);
    expect(t.output(new Float64Array(720).fill(4)).projection).toBe("forecast");
  });

  it("rejects out-of-order steps and unknown body parts", () => {
    const t = det();
    t.step({ t_unix_s: 10, uvi_ambient: 8 });
    expect(() => t.step({ t_unix_s: 5, uvi_ambient: 8 })).toThrow();
    expect(() => t.applySunscreen({ t_unix_s: 0, spf_label: 30, body_parts: ["elbow"] })).toThrow();
  });
});

describe("MED, sunscreen, forecast", () => {
  it("prior median is the Fitzpatrick table value", () => {
    expect(new PersonalMED(3).median()).toBeCloseTo(350, 6);
  });

  it("SPF is linear in amount", () => {
    expect(effectiveSpf(30, 2.0)).toBe(30);
    expect(effectiveSpf(30, 1.0)).toBe(15.5);
  });

  it("minutes to dose interpolates within the step", () => {
    // UVI 10 -> 15 J/m^2 per minute
    expect(minutesToDose(new Float64Array(100).fill(10), 150)).toBeCloseTo(10, 9);
    expect(minutesToDose(new Float64Array(10).fill(10), 1000)).toBe(Infinity);
  });

  it("clear-sky curve rises toward solar noon", () => {
    // 2024-07-01 10:00 PDT in Los Angeles; solar noon is ~13:00 PDT
    const c = expectedUviCurve(1719853200, 34.0, -118.2);
    expect(c.length).toBe(720);
    expect(c[150]).toBeGreaterThan(c[0]);
    expect(c[700]).toBeLessThan(0.01); // after sunset
  });
});
