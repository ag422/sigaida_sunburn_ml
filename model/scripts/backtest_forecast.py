"""Backtest the time-left projection on NASA POWER days with minute-scale synthetic clouds.

For each warm-season day of a test year and each "now" (10:00-15:00 local, every 30 min):
  truth    = minute UV curve: clear sky x site scale x minute clouds matched to that day's POWER
  question = minutes until 250 J/m^2 (Fitzpatrick II median MED), starting from zero dose
Methods:
  constant     hold the current reading (the prototype)
  forecast     current reading blended into a perfect hourly forecast (that day's POWER)
  climatology  current reading blended into the mean of the same dates (+-7 d) in other years
  clear_sky    ignore the current reading; clear sky only (upper bound on UV)
Error = predicted - actual minutes. Positive = told the user they had more time than they did.

    python scripts/backtest_forecast.py [--test-year 2024]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from sunrisk import config  # noqa: E402
from sunrisk.clearsky import clear_sky_uvi  # noqa: E402
from sunrisk.datasets import nasa_power as npw  # noqa: E402
from sunrisk.forecast import blend, minutes_to_dose  # noqa: E402
from sunrisk.solar import solar_position  # noqa: E402
from sunrisk.synth.clouds import minute_clouds  # noqa: E402
from sunrisk.synth.generate import cmf_from_power  # noqa: E402

RAW = ROOT / "data" / "raw" / "nasa_power"
DOSE = 250.0
NOW_LOCAL_MIN = np.arange(10 * 60, 15 * 60 + 1, 30)
METHODS = ("constant", "forecast", "climatology", "clear_sky")


def day_of_year(t):
    return ((t % (365.25 * 86400)) // 86400).astype(int)


def run(test_year: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sites = json.loads((RAW / "sites.json").read_text())
    errors = {m: [] for m in METHODS}
    stability = {m: [] for m in METHODS}
    for name, (lat, lon, off) in sites.items():
        s = npw.merge([npw.parse_hourly(json.loads(Path(f).read_text()))
                       for f in sorted(glob.glob(str(RAW / f"{name}_*.json")))])
        scale = npw.fit_site_scale(s)
        starts, rows = npw.daily_uv_curves(s, off)
        years = np.array([int(np.datetime64(int(t + 43200), "s").astype("datetime64[Y]").astype(int) + 1970) for t in starts])
        doy = day_of_year(starts + off * 3600)
        for i in np.where((years == test_year) & (rows.max(axis=1) >= 5))[0]:
            t_min = starts[i] + np.arange(2 * 1440) * 60.0          # today + tomorrow (for long horizons)
            elev, _ = solar_position(t_min, lat, lon)
            clear = clear_sky_uvi(elev) * scale
            cmf_h = cmf_from_power(rows[i], starts[i], lat, lon, scale)
            trans, cloudy = minute_clouds(cmf_h, rng)
            truth = clear * np.concatenate([trans, np.ones(1440)])
            k_fc = np.concatenate([np.repeat(np.clip(cmf_h, 0, 1.2), 60), np.ones(1440)])
            others = (years != test_year) & (np.abs(doy - doy[i]) <= 7)
            clim_h = np.nanmean([cmf_from_power(r, st, lat, lon, scale) for r, st in zip(rows[others], starts[others])], axis=0) \
                if others.any() else np.ones(24)
            k_clim = np.concatenate([np.repeat(np.clip(clim_h, 0, 1.2), 60), np.ones(1440)])

            def predict(m, now):
                c = clear[now: now + 720]
                env = "cloud" if cloudy[now] else "sun"
                tau = config.PERSIST_TAU_MIN[env].value
                if m == "constant":
                    curve = np.full(720, truth[now])
                elif m == "forecast":
                    curve = blend(c, k_fc[now: now + 720], truth[now], tau)
                elif m == "climatology":
                    curve = blend(c, k_clim[now: now + 720], truth[now], tau)
                else:
                    curve = c
                return minutes_to_dose(curve, DOSE)

            for now in NOW_LOCAL_MIN:
                actual = minutes_to_dose(truth[now: now + 720], DOSE)
                if not np.isfinite(actual) or actual > 240:
                    continue
                for m in METHODS:
                    errors[m].append(min(predict(m, now), 720) - actual)
            # stability: minute-to-minute change of the prediction over 11:00-13:00
            for m in METHODS:
                p = np.array([min(predict(m, now), 720) for now in range(11 * 60, 13 * 60)])
                stability[m].append(np.mean(np.abs(np.diff(p))))
    return errors, stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-year", type=int, default=2024)
    args = ap.parse_args()
    errors, stability = run(args.test_year)
    n = len(errors["constant"])
    print(f"{n} cases (7 sites, warm-season days of {args.test_year}, now = 10:00-15:00 local)")
    print(f"{'method':12s} {'median|err|':>11s} {'p90|err|':>9s} {'unsafe>5min':>11s} {'cautious>15min':>14s} {'jitter min/min':>14s}")
    for m in METHODS:
        e = np.array(errors[m])
        print(f"{m:12s} {np.median(np.abs(e)):11.1f} {np.quantile(np.abs(e), .9):9.1f} "
              f"{np.mean(e > 5):11.1%} {np.mean(e < -15):14.1%} {np.mean(stability[m]):14.2f}")


if __name__ == "__main__":
    main()
