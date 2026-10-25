/** Minimal erythemal dose (MED): skin-type prior and per-user Bayesian update (model/sunrisk/med.py). */

import * as config from "./config";
import { interp, linspace } from "./numeric";

export function priorMedian(fitzpatrick: number): number {
  const v = config.MED_MEDIAN_BY_FITZPATRICK[fitzpatrick];
  if (v === undefined) throw new Error(`Fitzpatrick type must be 1-6, got ${fitzpatrick}`);
  return v;
}

/** P(visible erythema | dose, MED): logistic in ln(dose/MED). */
export function burnProbability(dose: number, med: number, scale = config.BURN_LOGISTIC_SCALE): number {
  const z = Math.log(Math.max(dose, 1e-9) / med) / scale;
  return 1.0 / (1.0 + Math.exp(-z));
}

/**
 * Posterior over one user's MED on a log-spaced grid.
 *
 * Starts from a log-normal prior centred on the Fitzpatrick median and is updated with
 * observed outcomes: (effective dose received, whether redness was seen).
 */
export class PersonalMED {
  readonly logGrid: Float64Array;
  readonly grid: Float64Array;
  weights!: Float64Array;
  nObservations = 0;

  constructor(readonly fitzpatrick: number, nGrid = 401, logSd = config.MED_PRIOR_LOG_SD) {
    const mu = Math.log(priorMedian(fitzpatrick));
    this.logGrid = linspace(mu - 5 * logSd, mu + 5 * logSd, nGrid);
    this.grid = this.logGrid.map(Math.exp);
    this.setLogPost(this.logGrid.map((x) => -0.5 * ((x - mu) / logSd) ** 2));
  }

  private setLogPost(logp: Float64Array): void {
    let max = -Infinity;
    for (const v of logp) if (v > max) max = v;
    const p = logp.map((v) => Math.exp(v - max));
    let sum = 0;
    for (const v of p) sum += v;
    this.weights = p.map((v) => v / sum);
  }

  /**
   * Bayesian update with one observation. `burned` may be a probability in [0, 1]
   * (e.g. from a photo-redness classifier that is not certain), giving a soft-label likelihood.
   */
  update(dose: number, burned: boolean | number): void {
    const pObs = Number(burned);
    if (!(pObs >= 0 && pObs <= 1)) throw new Error("burned must be bool or a probability in [0, 1]");
    const logp = new Float64Array(this.grid.length);
    for (let i = 0; i < logp.length; i++) {
      const pBurn = burnProbability(dose, this.grid[i]);
      const lik = pObs * pBurn + (1.0 - pObs) * (1.0 - pBurn);
      logp[i] = Math.log(this.weights[i] + 1e-300) + Math.log(lik + 1e-300);
    }
    this.setLogPost(logp);
    this.nObservations += 1;
  }

  // CDF at bin centres (not right edges), so a symmetric posterior has its median exactly on
  // the centre grid point
  private cdf(): Float64Array {
    const out = new Float64Array(this.weights.length);
    let cum = 0;
    for (let i = 0; i < out.length; i++) {
      cum += this.weights[i];
      out[i] = cum - 0.5 * this.weights[i];
    }
    return out;
  }

  quantile(q: number): number {
    return Math.exp(interp(q, this.cdf(), this.logGrid));
  }

  median(): number {
    return this.quantile(0.5);
  }

  /** Inverse-CDF transform of uniform draws in [0, 1) (interpolated in log space). */
  fromUniform(u: ArrayLike<number>): Float64Array {
    const cdf = this.cdf();
    return Float64Array.from(u, (x) => Math.exp(interp(x, cdf, this.logGrid)));
  }
}
