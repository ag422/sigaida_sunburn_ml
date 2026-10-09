# sunrisk (model)

The Python risk model. An awareness tool, not a medical device.

```bash
cd model
python -m pytest            # Python 3.11+, numpy, matplotlib, pytest
```

If pytest crashes on import with an unrelated plugin (e.g. some Anaconda installs), run
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.

| Module | What it does |
|---|---|
| `sunrisk/config.py` | every non-definitional number, with source + confidence (renders `docs/assumptions.md`) |
| `sunrisk/contracts.py` | BLE packet, session CSV columns, risk engine input/output (see `docs/contracts.md`) |
| `sunrisk/uv.py` | UVI ↔ erythemal irradiance, dose integration |
| `sunrisk/med.py` | MED priors by Fitzpatrick type, `PersonalMED` Bayesian posterior |
| `sunrisk/sunscreen.py` | linear-in-amount SPF, wear-off, swim loss |
| `sunrisk/tracker.py` | `ExposureTracker`: Monte Carlo risk engine (p10/p50/p90 time-left per body part) |
| `sunrisk/datasets/nasa_power.py` | parse NASA POWER hourly UV |

Data (git-ignored, under `../data/`): `scripts/fetch_pamap2.sh`, `scripts/fetch_nasa_power.py`.
After editing `config.py`, run `python scripts/render_assumptions.py`.

The core (`sunrisk/` except `datasets/`) does no I/O, so it can be ported to TypeScript.
Port notes are in module docstrings.
