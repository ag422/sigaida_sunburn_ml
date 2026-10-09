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
| `sunrisk/solar.py` | NOAA sun position (elevation, azimuth) |
| `sunrisk/clearsky.py` | clear-sky UVI (Madronich 2007) |
| `sunrisk/geometry.py` | UV on tilted surfaces, body-part normals per posture |
| `sunrisk/bodydose.py` | sensor gain from tilt/compass, ambient UV estimate |
| `sunrisk/pipeline.py` | device stream → ambient UVI + per-body-part ratios |
| `sunrisk/events/` | event detection: features, rules detector, learned (PAMAP2) classifier, confusion matrices |
| `sunrisk/datasets/pamap2.py` | PAMAP2 wrist IMU → our units, axes and labels |
| `sunrisk/forecast.py` | forecast-aware expected UV curve for time-left |
| `sunrisk/session.py` | end-to-end: device arrays → detected events → risk over time |
| `sunrisk/synth/` | synthetic sessions: scenarios, wrist motion, clouds, LTR390 counts |
| `sunrisk/datasets/nasa_power.py` | parse NASA POWER hourly UV, fit site scale |

Scripts (need the downloaded data):

| Script | Output |
|---|---|
| `scripts/simulate_day.py` | `docs/figures/beach_day.png`: end-to-end beach day, per body part |
| `scripts/evaluate_events.py` | event-detection confusion matrices (`docs/figures/event_confusion.png`) |
| `scripts/train_activity.py` | PAMAP2 classifier: leave-one-subject-out comparison, exports `data/models/activity_forest.json` (needs `pip install .[ml]`) |
| `scripts/backtest_forecast.py` | forecast vs constant projection errors |
| `scripts/check_clearsky_vs_power.py` | clear-sky formula vs NASA POWER per site |

Data (git-ignored, under `../data/`): `scripts/fetch_pamap2.sh`, `scripts/fetch_nasa_power.py`.
After editing `config.py`, run `python scripts/render_assumptions.py`.

The core (`sunrisk/` except `datasets/` and `synth/`) does no I/O, so it can be ported to TypeScript.
Port notes are in module docstrings.
