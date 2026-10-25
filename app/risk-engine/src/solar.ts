/**
 * Sun position from the NOAA solar calculator equations (model/sunrisk/solar.py).
 *
 * - elevation in degrees above the horizon, with an atmospheric-refraction correction
 * - azimuth in degrees clockwise from true north (90 = east, 180 = south)
 */

const R = Math.PI / 180.0;

const clip = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
// np.mod: result has the sign of the divisor
const mod = (x: number, m: number) => ((x % m) + m) % m;

export interface SolarPosition {
  elevation: number;
  azimuth: number;
}

export function solarPosition(tUnixS: number, latDeg: number, lonDeg: number): SolarPosition {
  const lat = latDeg * R;

  const jd = tUnixS / 86400.0 + 2440587.5;
  const jc = (jd - 2451545.0) / 36525.0;

  const l0 = mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0); // mean longitude
  const m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc); // mean anomaly
  const ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
  const mr = m * R;
  const c = Math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
    + Math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
    + Math.sin(3 * mr) * 0.000289;
  const omega = (125.04 - 1934.136 * jc) * R;
  const appLong = (l0 + c - 0.00569 - 0.00478 * Math.sin(omega)) * R;
  const meanObliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
  const obliq = (meanObliq + 0.00256 * Math.cos(omega)) * R;
  const decl = Math.asin(Math.sin(obliq) * Math.sin(appLong));

  const y = Math.tan(obliq / 2.0) ** 2;
  const l0r = l0 * R;
  const eqTimeMin = (4.0 / R) * (y * Math.sin(2 * l0r) - 2 * ecc * Math.sin(mr)
    + 4 * ecc * y * Math.sin(mr) * Math.cos(2 * l0r)
    - 0.5 * y * y * Math.sin(4 * l0r) - 1.25 * ecc * ecc * Math.sin(2 * mr));

  const minutesUtc = mod(tUnixS, 86400.0) / 60.0;
  const trueSolarMin = mod(minutesUtc + eqTimeMin + 4.0 * lonDeg, 1440.0);
  const hourAngle = (trueSolarMin / 4.0 - 180.0) * R;

  const cosZen = clip(Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.cos(hourAngle), -1.0, 1.0);
  const zen = Math.acos(cosZen);
  const elev = 90.0 - zen / R;

  // azimuth (NOAA form), guarded at the zenith/poles
  const denom = Math.cos(lat) * Math.sin(zen);
  const cosAz = clip((Math.sin(lat) * cosZen - Math.sin(decl)) / (Math.abs(denom) < 1e-12 ? 1e-12 : denom), -1.0, 1.0);
  const a = Math.acos(cosAz) / R;
  const az = hourAngle > 0 ? mod(a + 180.0, 360.0) : mod(540.0 - a, 360.0);

  return { elevation: elev + refractionDeg(elev), azimuth: az };
}

/** NOAA approximation of atmospheric refraction (degrees). */
function refractionDeg(e: number): number {
  const te = Math.tan(clip(e, -89.0, 89.9) * R);
  let arcsec: number;
  if (e > 85.0) arcsec = 0.0;
  else if (e > 5.0) arcsec = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5;
  else if (e > -0.575) arcsec = 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)));
  else arcsec = -20.772 / te;
  return arcsec / 3600.0;
}
