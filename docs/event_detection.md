# Event detection

Code: `model/sunrisk/events/` (`features.py`, `rules.py`, `base.py`, `evaluate.py`).
To reproduce: `python model/scripts/evaluate_events.py --days 40` writes
`docs/figures/event_confusion.png`.

## Design
- **Windows:** non-overlapping 5 s windows (100 IMU samples at 20 Hz).
- **IMU features** (accelerometer + gyroscope only, so the same code runs on PAMAP2):
  - mean acceleration vector and its length
  - spread of |acc| (`acc_mag_std`)
  - gyro RMS and gyro max
  - step-rhythm autocorrelation
  - tilt
- **UV feature:** `uv_level` = measured UVI ÷ (expected sensor gain × clear-sky UVI).
- **Forecast feature:** the forecast cloud factor for the hour (forecast all-sky ÷ clear-sky).
- **Interface:** `EventDetector.predict(windows) → (environment, activity)`. The rules
  implement it today; a trained classifier can drop in later. `ActivityClassifier` is the
  activity-only interface for the PAMAP2-trained model.

## Rules (thresholds are fields on `RuleDetector`)

| Output | Rule | Why |
|---|---|---|
| swimming | \|mean acc\| < 0.75 g and gyro > 60 °/s | Gravity averages out over full arm circles |
| vigorous | `acc_mag_std` > 0.25 g and gyro > 150 °/s | Impacts + fast rotation |
| walking | step autocorrelation > 0.5 and `acc_mag_std` > 0.08 g | Steps repeat at 1.4–2.6 Hz |
| standing | gyro < 15 °/s, forearm hanging (acc_x < −0.7 g) | Arm hangs at the side |
| sitting / lying | gyro < 15 °/s, forearm level; **lying** if no hand gesture within ±90 s | Weakest rule |
| light_activity | anything else | |
| sun | `uv_level` ≥ 0.5 | Direct beam visible |
| indoor | `uv_level` < 0.1 **for 5 min** | Before then, keep the previous label and **hold the UV estimate** |
| shade vs cloud | `uv_level` in between; **shade** if the forecast says clear, or the wearer was moving when it started; else **cloud** | One sensor can't see the difference |

**Why indoor needs 5 minutes:** a sleeve or hand over the sensor looks exactly like going
indoors. Holding the last outdoor UV estimate for up to 5 minutes means a covered sensor can't
silently zero the dose. The cost: the first 5 minutes indoors still count as outdoors, which
is the conservative direction.

## Results on 40 synthetic days

Setup: 7 NASA POWER sites, warm-season days, random 6-hour scenarios, ~180k windows.
Windows where clear-sky UVI ≥ 3:

| Environment | Precision | Recall | F1 |
|---|---:|---:|---:|
| sun | 0.88 | 0.96 | 0.92 |
| shade | 0.80 | 0.88 | 0.84 |
| cloud | 0.38 | 0.29 | 0.33 |
| indoor | 1.00 | 0.72 | 0.84 |

- **Indoor recall is low by design:** the first 5 minutes of every indoor stay are held as
  outdoor.
- **Cloud vs. shade is poor, and that's expected.** Both mean "no direct beam", and the dose
  model treats them identically (same geometry). The distinction only matters for the
  forecast projection: cloud passes on its own, shade is the wearer's choice.

| Activity | Precision | Recall | F1 |
|---|---:|---:|---:|
| lying | 0.80 | 0.90 | 0.84 |
| sitting | 0.90 | 0.79 | 0.84 |
| standing / walking / light / vigorous / swimming | ≥ 0.98 | ≥ 0.99 | ≥ 0.99 |

**These activity scores are optimistic. Don't quote them as performance.** The synthetic
motions were written by us and are cleanly separable. Real performance will come from:
- **PAMAP2** (Phase 1b): real people's wrists, leave-one-subject-out;
- **our own labeled recordings** (Phase 4).

Lying vs. sitting is the one real ambiguity, even in synthetic data.

## Also known to be optimistic
- The forecast used in evaluation *is* the NASA POWER day the sky was built from: a perfect
  forecast.
- No real-world nuisances: watch-strap slip, wrist-specific quirks, sweat on the sensor,
  reflective surfaces other than sand or grass.
