# @sunrisk/risk-engine

TypeScript port of the Python risk engine (`model/sunrisk/`), for the app. An awareness tool,
not a medical device.

Part of the npm workspace in `app/` (with `mobile/`), so install from there:

```bash
cd app && npm install
npm test --workspace risk-engine          # vitest
npm run typecheck --workspace risk-engine
```

| Module | Ported from |
|---|---|
| `src/tracker.ts` | `tracker.py`: `ExposureTracker`, p10/p50/p90 minutes left per body part |
| `src/med.ts` | `med.py`: MED prior and Bayesian posterior |
| `src/forecast.ts` | `forecast.py`: forecast-aware expected UVI curve |
| `src/solar.ts`, `src/clearsky.ts` | `solar.py` (position only), `clearsky.py` |
| `src/uv.ts`, `src/sunscreen.ts` | `uv.py`, `sunscreen.py` |
| `src/config.ts` | values from `config.py` (the source of truth) |
| `src/contracts.ts` | risk engine types from `contracts.py` |

Not ported yet: body-part geometry (`geometry.py`, `bodydose.py`, `pipeline.py`), event
detection (`events/`) and the BLE packet decoder.

## Usage

```ts
import { ExposureTracker, expectedUviCurve } from "@sunrisk/risk-engine";

const tracker = new ExposureTracker({ fitzpatrick: 2 });
tracker.applySunscreen({ t_unix_s: now, spf_label: 30, body_parts: ["face", "neck"] });
tracker.step({ t_unix_s: now, uvi_ambient: 7.2, environment: "sun", activity: "walking" });

const curve = expectedUviCurve(now, 34.0, -118.2, { uviNow: 7.2, forecast });
const out = tracker.output(curve);   // headline = out.parts[i].minutes_left.p10
```

## Matching the Python engine

`test/parity.test.ts` replays the cases in `test/fixtures/parity.json` and checks every output
against Python to a relative 1e-9. Regenerate the fixtures whenever the Python engine or
`config.py` changes:

```bash
cd model && python scripts/export_ts_fixtures.py
```

numpy's random generator can't be reproduced in TypeScript, so each case stores numpy's raw
draws (`TrackerDraws`: MED uniforms, calibration and amount normals) and the TypeScript tracker
uses them instead of its own generator. In the app the tracker draws its own particles from a
seeded mulberry32 generator (`src/random.ts`); the results are statistically the same but not
identical to Python's for a given seed.
