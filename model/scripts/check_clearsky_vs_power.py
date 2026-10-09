"""Compare our clear-sky UVI formula with NASA POWER all-sky UVI on clear hours.

Clear hour: CLOUD_AMT < 5% and solar elevation > 20 deg at mid-hour. The ratio
POWER / formula (hour averages) is a per-site scale factor that absorbs aerosols, ozone
differences and POWER's own biases. Prints a table; writes nothing.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from sunrisk.clearsky import clear_sky_uvi  # noqa: E402
from sunrisk.datasets import nasa_power as npw  # noqa: E402
from sunrisk.solar import solar_position  # noqa: E402

RAW = ROOT / "data" / "raw" / "nasa_power"


def site_ratios(name, lat, lon):
    s = npw.merge([npw.parse_hourly(json.loads(Path(f).read_text()))
                   for f in sorted(glob.glob(str(RAW / f"{name}_*.json")))])
    sub = s.t_unix_s[:, None] + (np.arange(6) * 10 + 5) * 60.0      # 6 samples inside each hour
    elev, _ = solar_position(sub, lat, lon)
    model = clear_sky_uvi(elev).mean(axis=1)
    mid_elev = elev[:, 3]
    clear = (s.values["CLOUD_AMT"] < 5) & (mid_elev > 20) & ~np.isnan(s.values[npw.UV_PARAM])
    return s.values[npw.UV_PARAM][clear] / model[clear], mid_elev[clear]


def main():
    sites = json.loads((RAW / "sites.json").read_text())
    print(f"{'site':12s} {'n':>5s} {'median':>7s} {'p10':>6s} {'p90':>6s}  median by elevation 20-40/40-60/60+")
    for name, (lat, lon, _) in sites.items():
        r, e = site_ratios(name, lat, lon)
        bands = [np.median(r[(e >= lo) & (e < hi)]) if np.any((e >= lo) & (e < hi)) else np.nan
                 for lo, hi in ((20, 40), (40, 60), (60, 91))]
        print(f"{name:12s} {r.size:5d} {np.median(r):7.2f} {np.quantile(r, .1):6.2f} {np.quantile(r, .9):6.2f}  "
              + " / ".join(f"{b:.2f}" for b in bands))


if __name__ == "__main__":
    main()
