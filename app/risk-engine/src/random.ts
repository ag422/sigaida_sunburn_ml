/**
 * Seeded random numbers for the app.
 *
 * This does NOT reproduce numpy's generator. Parity tests with Python pass the particle draws
 * in directly (see TrackerDraws), so the generator only needs to be seeded and decent.
 */

export interface Rng {
  random(): number; // uniform in [0, 1)
  normal(mean: number, sd: number): number;
}

/** mulberry32 uniforms + Box-Muller normals. */
export function seededRng(seed = 0): Rng {
  let a = seed >>> 0;
  let spare: number | null = null;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random,
    normal(mean, sd) {
      if (spare !== null) {
        const z = spare;
        spare = null;
        return mean + sd * z;
      }
      let u = 0;
      while (u === 0) u = random();
      const r = Math.sqrt(-2 * Math.log(u));
      const theta = 2 * Math.PI * random();
      spare = r * Math.sin(theta);
      return mean + sd * r * Math.cos(theta);
    },
  };
}
