// TypeScript vs Python on shared cases. Fixtures come from model/scripts/export_ts_fixtures.py;
// rerun it after changing the Python engine or config.py.
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  blend, clearSkyUvi, config, ExposureTracker, expectedUviCurve, minutesToDose, PersonalMED,
  solarPosition, type RiskOutput,
} from "../src";

const FIXTURE = fileURLToPath(new URL("./fixtures/parity.json", import.meta.url));
const haveFixture = existsSync(FIXTURE);

// JSON has no Infinity; the exporter writes "inf"
const revive = (_k: string, v: unknown) => (v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const data: any = haveFixture ? JSON.parse(readFileSync(FIXTURE, "utf8"), revive) : {};

// Same operations in the same order, so differences are rounding-level. The tolerance is loose
// enough for libm differences (exp/log/sin) between numpy and V8.
const REL = 1e-9;

function close(actual: number, expected: number, what: string) {
  if (expected === Infinity || expected === -Infinity || actual === Infinity || actual === -Infinity) {
    expect(actual, what).toBe(expected);
    return;
  }
  const tol = REL * Math.max(1, Math.abs(expected));
  expect(Math.abs(actual - expected), `${what}: ${actual} vs ${expected}`).toBeLessThanOrEqual(tol);
}

function closeArray(actual: ArrayLike<number>, expected: number[], what: string) {
  expect(actual.length, `${what} length`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) close(actual[i], expected[i], `${what}[${i}]`);
}

describe.skipIf(!haveFixture)("parity with the Python engine", () => {
  it("config matches config.py", () => {
    const py: Record<string, number> = data.config;
    const ts: Record<string, unknown> = { ...config };
    for (const [name, value] of Object.entries(py)) {
      const m = name.match(/^(\w+)\[(\w+)\]$/);
      const tsValue = m ? (ts[m[1]] as Record<string, number> | undefined)?.[m[2]] : ts[name];
      if (tsValue === undefined) continue; // not used by the engine (geometry, albedo, ...)
      expect(tsValue, name).toBe(value);
    }
  });

  it("MED posterior", () => {
    for (const c of data.posterior) {
      const post = new PersonalMED(c.fitzpatrick);
      for (const o of c.observations) post.update(o.effective_dose_j_m2, o.burned);
      for (const [q, v] of Object.entries(c.quantiles)) close(post.quantile(Number(q)), v as number, `type ${c.fitzpatrick} q${q}`);
      closeArray(post.fromUniform(c.from_uniform.u), c.from_uniform.med, `type ${c.fitzpatrick} fromUniform`);
    }
  });

  it("sun position and clear-sky UVI", () => {
    for (const c of data.solar) {
      const p = solarPosition(c.t, c.lat, c.lon);
      const where = `(${c.lat}, ${c.lon}) t=${c.t}`;
      close(p.elevation, c.elevation, `elevation ${where}`);
      close(p.azimuth, c.azimuth, `azimuth ${where}`);
      close(clearSkyUvi(c.elevation), c.clear_sky_uvi, `clear sky ${where}`);
    }
  });

  it("forecast-aware UVI curve", () => {
    for (const c of data.forecast) {
      const curve = expectedUviCurve(c.t_now, c.lat, c.lon, {
        uviNow: c.uvi_now, environment: c.environment, forecast: c.forecast,
      });
      closeArray(curve, c.curve, `curve ${c.environment}`);
      close(minutesToDose(curve, 250), c.minutes_to_250, `minutes to 250 J/m^2 ${c.environment}`);
    }
    const b = data.blend;
    closeArray(blend(Float64Array.from(b.clear), Float64Array.from(b.k_fc), b.uvi_now, b.tau), b.expect, "blend");
  });

  for (const c of haveFixture ? data.trackers : []) {
    it(`tracker: ${c.name}`, () => {
      const t = new ExposureTracker(c.profile, {
        uncertain: c.uncertain, bodyParts: c.body_parts, draws: c.draws ?? undefined,
      });
      let nOut = 0;
      for (const op of c.ops) {
        if (op.op === "sunscreen") t.applySunscreen(op.app);
        else if (op.op === "step") t.step(op.step);
        else compareOutput(t.output(op.uvi_curve), op.expect, `output #${nOut++}`);
      }
    });
  }
});

function compareOutput(ts: RiskOutput, py: RiskOutput, where: string) {
  expect(ts.first_to_burn, `${where} first_to_burn`).toBe(py.first_to_burn);
  expect(ts.alert, `${where} alert`).toBe(py.alert);
  expect(ts.projection, `${where} projection`).toBe(py.projection);
  close(ts.t_unix_s, py.t_unix_s, `${where} t_unix_s`);
  expect(ts.parts.map((p) => p.body_part)).toEqual(py.parts.map((p) => p.body_part));
  ts.parts.forEach((p, j) => {
    const q = py.parts[j];
    const w = `${where} ${p.body_part}`;
    for (const k of ["p10", "p50", "p90"] as const) {
      close(p.minutes_left[k], q.minutes_left[k], `${w} minutes_left.${k}`);
      close(p.fraction_med_used[k], q.fraction_med_used[k], `${w} fraction_med_used.${k}`);
    }
    close(p.effective_spf, q.effective_spf, `${w} effective_spf`);
    expect(p.beyond_horizon, `${w} beyond_horizon`).toBe(q.beyond_horizon);
  });
}
