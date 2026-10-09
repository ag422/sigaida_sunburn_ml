# Datasets

Datasets are downloaded into `data/` (git-ignored) and never committed.

## 1. PAMAP2 Physical Activity Monitoring (primary public dataset)

- **Source:** UCI Machine Learning Repository, dataset 231.
  Fetch with `model/scripts/fetch_pamap2.sh` (~656 MB zip, ~1.6 GB extracted).
- **Terms:** the readme says the dataset is "freely available for academic research" and asks
  that publications cite:
  1. A. Reiss and D. Stricker. *Introducing a New Benchmarked Dataset for Activity Monitoring.* ISWC 2012.
  2. A. Reiss and D. Stricker. *Creating and Benchmarking a New Dataset for Physical Activity Monitoring.* ABRA 2012.
- **Why this one:** it has an IMU **on the wrist of the dominant arm**, which is where our device
  is worn. It also includes a magnetometer, so it matches if we buy an IMU that has one.

### Format (from the readme)

Each `.dat` file is one subject in one session (`Protocol/` or `Optional/`). The file is
space-separated, has 54 columns and is sampled at 100 Hz.

| Columns (1-based) | Content |
|---|---|
| 1 | timestamp (s) |
| 2 | activityID (0 = transient → **discard**) |
| 3 | heart rate (bpm, ~9 Hz, NaN between samples) |
| 4–20 | **wrist IMU** |
| 21–37 | chest IMU |
| 38–54 | ankle IMU |

Within each 17-column IMU block: temperature; acc ±16 g xyz (m/s², **use this one**);
acc ±6 g xyz (saturates, poorly calibrated, so don't use it); gyro xyz (rad/s); magnetometer
xyz (µT); orientation ×4 (**invalid**, ignore).

### What's in it (wrist IMU, activityID ≠ 0, measured 2026-10-08)

7.6 labeled hours in total. Up to 0.9% of wrist rows per file have NaN from wireless dropouts.

| Activity | Minutes | Subjects | → our class |
|---|---:|---:|---|
| lying | 32.1 | 8 | `lying` |
| sitting | 30.9 | 8 | `sitting` |
| standing | 31.7 | 8 | `standing` |
| walking | 39.8 | 8 | `walking` |
| Nordic walking | 31.4 | 7 | `walking` |
| ascending / descending stairs | 19.5 / 17.5 | 8 | `walking` |
| running | 16.4 | 7 | `vigorous` |
| rope jumping | 8.2 | 6 | `vigorous` |
| soccer | 7.8 | 2 | `vigorous` |
| cycling | 27.4 | 7 | `vigorous` (TBD: arms are fairly still) |
| computer work | 51.7 | 4 | `sitting` |
| watching TV | 13.9 | 1 | `sitting` |
| car driving | 9.1 | 1 | excluded (not a sun-exposure posture we model) |
| vacuum / ironing / laundry / cleaning | 29.2 / 39.8 / 16.6 / 31.2 | 4–8 | `light_activity` |

The class mapping is a proposal. It's grouped by what matters for sunburn: body posture
(which body parts face the sky) and exertion (sweat wears off sunscreen faster).

### Known gaps / caveats

- **No swimming.** We need a second, wrist-worn swimming dataset (see §2).
- **Only 9 subjects (8 male) and about 30 min per class.** Evaluate leave-one-subject-out,
  never with a random split of windows.
- **Different device.** The data is 100 Hz with an unknown axis convention relative to ours.
  We downsample to our device rate and use features that don't depend on sensor rotation
  (total acceleration, tilt relative to gravity). The gap between the two devices gets
  measured in Phase 4 on our own recordings.
- **Telling lying from sitting using only the wrist is harder than with a chest sensor.** We'll
  report it honestly. It matters because it decides which body parts face the sky.
- **IMU only, no UV.** PAMAP2 trains the activity/posture detector. It says nothing about UV.

## 2. Swimming (TBD, not yet downloaded)

Candidate: the smartwatch swimming dataset from Brunner et al., *Swimming style recognition and
lap counting using a smartwatch and deep learning*, ISWC 2019 (wrist accel + gyro).
**Verify availability and license before using it.** If no usable public dataset turns up,
swim detection stays rules-based (sustained periodic arm motion plus UV dropping to near zero
underwater) and gets validated on our own data.

## 3. Measured UV (TBD)

For checking the clear-sky model and the forecast-aware countdown: historical measured UV
from USDA UV-B Monitoring and Research Program sites, or Open-Meteo historical
`uv_index`. Not yet chosen; check availability and terms first.

## 4. Our own recordings (Phase 3)

Session CSVs from the device (format in `docs/contracts.md`, still to be written), with
event labels entered in the app.
