/**
 * Monte Carlo exposure tracker: the risk engine the app calls (model/sunrisk/tracker.py).
 *
 * Each particle is one plausible "world": a MED drawn from the user's posterior, a sensor
 * calibration error, and a multiplier on how much sunscreen was really applied. Particles are
 * drawn once per session (common random numbers), so the output moves smoothly over time.
 *
 * State is (particles x parts) arrays stored row-major in Float64Arrays: index i * nParts + j.
 */

import * as config from "./config";
import {
  BODY_PARTS, CONTRACT_VERSION, type BodyPartRisk, type Quantiles, type RiskOutput, type RiskStep,
  type SunscreenApplication, type UserProfile,
} from "./contracts";
import { PersonalMED } from "./med";
import { median, quantileInvertedCdf, sortedCopy } from "./numeric";
import { seededRng, type Rng } from "./random";
import { W_PER_UVI } from "./uv";

/**
 * Per-particle random draws. Normally generated from `rng`; parity tests pass the draws numpy
 * made, so the TypeScript and Python engines run on identical particles.
 */
export interface TrackerDraws {
  /** uniforms in [0, 1), mapped through the MED posterior */
  medUniform: ArrayLike<number>;
  /** standard normals for the sensor calibration error */
  calNormal: ArrayLike<number>;
  /** standard normals for the applied-amount multiplier */
  amountNormal: ArrayLike<number>;
}

export interface TrackerOptions {
  rng?: Rng;
  draws?: TrackerDraws;
  nParticles?: number;
  uncertain?: boolean;
  bodyParts?: readonly string[];
}

export class ExposureTracker {
  readonly parts: readonly string[];
  readonly posterior: PersonalMED;
  readonly n: number;
  readonly med: Float64Array;
  readonly cal: Float64Array;
  readonly amountMult: Float64Array;
  /** effective (post-sunscreen) dose to skin, J/m^2 */
  readonly dose: Float64Array;
  /** sum (SPF-1) * amount / A_lab, i.e. SPF_eff - 1 */
  readonly protect: Float64Array;
  tUnixS: number | null = null;
  private lastStep: RiskStep | null = null;
  private readonly partIdx: Map<string, number>;

  constructor(profile: UserProfile, opts: TrackerOptions = {}) {
    const { uncertain = true, bodyParts = BODY_PARTS } = opts;
    this.parts = [...bodyParts];
    this.partIdx = new Map(this.parts.map((p, i) => [p, i]));

    const posterior = new PersonalMED(profile.fitzpatrick);
    for (const obs of profile.med_observations ?? []) posterior.update(obs.effective_dose_j_m2, obs.burned);
    this.posterior = posterior;

    if (uncertain) {
      let d = opts.draws;
      if (!d) {
        const rng = opts.rng ?? seededRng(0);
        const n = opts.nParticles ?? config.N_PARTICLES;
        d = {
          medUniform: Array.from({ length: n }, () => rng.random()),
          calNormal: Array.from({ length: n }, () => rng.normal(0, 1)),
          amountNormal: Array.from({ length: n }, () => rng.normal(0, 1)),
        };
      }
      this.n = d.medUniform.length;
      if (d.calNormal.length !== this.n || d.amountNormal.length !== this.n) {
        throw new Error("draws must all have the same length");
      }
      this.med = posterior.fromUniform(d.medUniform);
      this.cal = Float64Array.from(d.calNormal, (z) => Math.exp(z * config.SENSOR_CAL_LOG_SD));
      this.amountMult = Float64Array.from(d.amountNormal, (z) => Math.exp(z * config.APPLIED_AMOUNT_LOG_SD));
    } else {
      this.n = 1;
      this.med = Float64Array.of(posterior.median());
      this.cal = Float64Array.of(1);
      this.amountMult = Float64Array.of(1);
    }

    this.dose = new Float64Array(this.n * this.parts.length);
    this.protect = new Float64Array(this.n * this.parts.length);
  }

  // -- inputs ------------------------------------------------------------------------------

  applySunscreen(app: SunscreenApplication): void {
    const amount = app.amount_mg_cm2 ?? config.TYPICAL_APPLIED_AMOUNT;
    const idx = this.indices(app.body_parts);
    const gain = ((app.spf_label - 1.0) * amount) / config.LAB_SUNSCREEN_AMOUNT;
    const P = this.parts.length;
    for (let i = 0; i < this.n; i++) {
      for (const j of idx) this.protect[i * P + j] += gain * this.amountMult[i];
    }
  }

  /** Advance to s.t_unix_s. The previous step's conditions apply over the interval. */
  step(s: RiskStep): void {
    const prev = this.lastStep;
    if (prev !== null) {
      const dt = s.t_unix_s - prev.t_unix_s;
      if (dt < 0) throw new Error("steps must be in time order");
      this.accumulate(prev, dt);
      if (activityOf(prev) === "swimming" && activityOf(s) !== "swimming") {
        for (let k = 0; k < this.protect.length; k++) this.protect[k] *= config.SWIM_RETENTION;
      }
    }
    this.lastStep = s;
    this.tUnixS = s.t_unix_s;
  }

  private accumulate(s: RiskStep, dtS: number): void {
    const ratios = this.ratios(s.exposure_ratios);
    const P = this.parts.length;
    for (let i = 0; i < this.n; i++) {
      const e = s.uvi_ambient * W_PER_UVI * this.cal[i];
      for (let j = 0; j < P; j++) {
        const k = i * P + j;
        this.dose[k] += ((e * ratios[j]) / (1.0 + this.protect[k])) * dtS;
      }
    }
    const decay = Math.exp(-dtS / 3600.0 / wearTauH(activityOf(s)));
    for (let k = 0; k < this.protect.length; k++) this.protect[k] *= decay;
  }

  // -- outputs -----------------------------------------------------------------------------

  /**
   * Minutes until each particle reaches its MED, as a row-major (particles x parts) array.
   *
   * uviCurve: UVI at 1-minute steps starting now (see expectedUviCurve). Omitted -> hold the
   * current reading constant. Values beyond the horizon are Infinity.
   */
  minutesLeft(uviCurve?: ArrayLike<number> | null, exposureRatios?: Record<string, number> | null,
              activity?: string | null): Float64Array {
    const stepMin = config.PROJECTION_STEP_MIN;
    const horizon = Math.trunc(config.PROJECTION_HORIZON_MIN / stepMin);
    const last = this.lastStep;
    const curve = uviCurve
      ? Float64Array.from(uviCurve).subarray(0, horizon)
      : new Float64Array(horizon).fill(last ? last.uvi_ambient : 0.0);
    const T = curve.length;
    const ratios = this.ratios(exposureRatios ?? (last ? last.exposure_ratios : null));
    const tauH = wearTauH(activity || (last ? activityOf(last) : "standing"));

    const dtS = stepMin * 60.0;
    const decay = new Float64Array(T);
    for (let t = 0; t < T; t++) decay[t] = Math.exp((-t * dtS) / 3600.0 / tauH);

    const P = this.parts.length;
    const out = new Float64Array(this.n * P);
    for (let i = 0; i < this.n; i++) {
      const med = this.med[i];
      for (let j = 0; j < P; j++) {
        const k = i * P + j;
        const dose = this.dose[k];
        if (med - dose <= 0) {
          out[k] = 0.0;
          continue;
        }
        out[k] = Infinity;
        // running sum of increments, added to dose afterwards (same rounding as numpy's cumsum)
        let sum = 0.0;
        let before = dose;
        for (let t = 0; t < T; t++) {
          const e = curve[t] * W_PER_UVI * this.cal[i];
          const incr = ((e * ratios[j]) / (1.0 + this.protect[k] * decay[t])) * dtS;
          sum += incr;
          const cum = dose + sum;
          if (cum >= med) {
            // linear interpolation inside the crossing step
            const frac = incr > 0 ? (med - before) / incr : 0.0;
            out[k] = (t + frac) * stepMin;
            break;
          }
          before = cum;
        }
      }
    }
    return out;
  }

  output(uviCurve?: ArrayLike<number> | null, projection?: "constant" | "forecast"): RiskOutput {
    const mins = this.minutesLeft(uviCurve);
    const q = config.ALERT_QUANTILE;
    const P = this.parts.length;
    const column = (a: Float64Array, j: number, f: (v: number, i: number) => number = (v) => v) =>
      Float64Array.from({ length: this.n }, (_, i) => f(a[i * P + j], i));

    const parts: BodyPartRisk[] = this.parts.map((name, j) => {
      const m = quantiles(column(mins, j), q);
      return {
        body_part: name,
        minutes_left: m,
        fraction_med_used: quantiles(column(this.dose, j, (v, i) => v / this.med[i]), q),
        effective_spf: median(column(this.protect, j, (v) => 1.0 + v)),
        beyond_horizon: m.p10 === Infinity,
      };
    });
    // first minimum wins on ties, like Python's min()
    const first = parts.reduce((a, b) => (b.minutes_left.p10 < a.minutes_left.p10 ? b : a));
    return {
      t_unix_s: this.tUnixS || 0.0,
      parts,
      first_to_burn: first.body_part,
      alert: first.minutes_left.p10 < config.ALERT_LEAD_MIN,
      projection: projection ?? (uviCurve ? "forecast" : "constant"),
      contract_version: CONTRACT_VERSION,
    };
  }

  // -- helpers -----------------------------------------------------------------------------

  private indices(parts: readonly string[] | null | undefined): number[] {
    if (!parts) return this.parts.map((_, i) => i);
    return parts.map((p) => this.partIndex(p));
  }

  private ratios(ratios: Record<string, number> | null | undefined): Float64Array {
    const r = new Float64Array(this.parts.length).fill(1.0);
    if (ratios) for (const [p, v] of Object.entries(ratios)) r[this.partIndex(p)] = v;
    return r;
  }

  private partIndex(p: string): number {
    const j = this.partIdx.get(p);
    if (j === undefined) throw new Error(`unknown body part: ${p}`);
    return j;
  }
}

const activityOf = (s: RiskStep) => s.activity ?? "standing";

const wearTauH = (activity: string) =>
  activity === "vigorous" ? config.SUNSCREEN_WEAR_TAU_ACTIVE_H : config.SUNSCREEN_WEAR_TAU_H;

// inverted_cdf avoids interpolating between finite and Infinity (beyond-horizon) particles
function quantiles(x: Float64Array, qLow: number): Quantiles {
  const s = sortedCopy(x);
  return {
    p10: quantileInvertedCdf(s, qLow),
    p50: quantileInvertedCdf(s, 0.5),
    p90: quantileInvertedCdf(s, 1.0 - qLow),
  };
}
