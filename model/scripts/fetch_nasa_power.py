"""Download NASA POWER hourly UV + cloud data for a set of sites into data/raw/nasa_power/.

    python scripts/fetch_nasa_power.py                      # default sites, 2022-2024
    python scripts/fetch_nasa_power.py --years 2024 --site home 40.0 -88.0 -6

One JSON file per site per year (~0.5 MB). Already-downloaded files are skipped.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from sunrisk.datasets.nasa_power import PARAMETERS  # noqa: E402

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
OUT = ROOT / "data" / "raw" / "nasa_power"

# name: (lat, lon, standard-time UTC offset in hours). Chosen to span latitudes, climates
# and both hemispheres.
SITES = {
    "honolulu": (21.31, -157.86, -10),
    "miami": (25.76, -80.19, -5),
    "phoenix": (33.45, -112.07, -7),
    "los_angeles": (34.05, -118.24, -8),
    "chicago": (41.88, -87.63, -6),
    "seattle": (47.61, -122.33, -8),
    "sydney": (-33.87, 151.21, 10),
}


def fetch(name: str, lat: float, lon: float, year: int) -> Path:
    path = OUT / f"{name}_{year}.json"
    if path.exists():
        return path
    q = urllib.parse.urlencode({
        "parameters": ",".join(PARAMETERS), "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": f"{year}0101", "end": f"{year}1231",
        "format": "JSON", "time-standard": "UTC",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
        payload = json.load(r)
    if "properties" not in payload:
        raise RuntimeError(f"{name} {year}: {payload.get('messages')}")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--site", nargs=4, action="append", metavar=("NAME", "LAT", "LON", "UTC_OFFSET"),
                    help="add a custom site (replaces the defaults)")
    args = ap.parse_args()
    sites = ({n: (float(a), float(o), float(z)) for n, a, o, z in args.site} if args.site else SITES)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sites.json").write_text(json.dumps(sites, indent=1))
    for name, (lat, lon, _) in sites.items():
        for y in args.years:
            p = fetch(name, lat, lon, y)
            print(f"{p.name}: {p.stat().st_size / 1e3:.0f} kB")
            time.sleep(1)  # be polite to the public API


if __name__ == "__main__":
    main()
