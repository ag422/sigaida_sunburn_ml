/**
 * The few numpy routines the engine relies on, reimplemented to give the same results
 * (same operation order where it affects rounding).
 */

/** np.linspace(start, stop, n) */
export function linspace(start: number, stop: number, n: number): Float64Array {
  const out = new Float64Array(n);
  const step = (stop - start) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = i * step + start;
  if (n > 1) out[n - 1] = stop;
  return out;
}

/** np.interp(x, xp, fp) for increasing xp; clamps to the end values outside the range. */
export function interp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>): number {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  // largest j with xp[j] <= x
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  if (x === xp[lo]) return fp[lo];
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

/** np.quantile(x, q, method="inverted_cdf"): no interpolation, so Infinity values stay intact. */
export function quantileInvertedCdf(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  const index = n * q - 1;
  const prev = Math.floor(index);
  let k = index - prev === 0 ? prev : prev + 1;
  if (k < 0) k = 0;
  if (k > n - 1) k = n - 1;
  return sorted[k];
}

/** np.median */
export function median(x: ArrayLike<number>): number {
  const s = sortedCopy(x);
  const n = s.length;
  const h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

export function sortedCopy(x: ArrayLike<number>): Float64Array {
  return Float64Array.from(x).sort();
}
