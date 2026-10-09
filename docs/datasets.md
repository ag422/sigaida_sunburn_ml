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

## 3. NASA POWER hourly UV (second public dataset)

- **Source:** NASA POWER Hourly API (`https://power.larc.nasa.gov/api/temporal/hourly/point`).
  Fetch with `python model/scripts/fetch_nasa_power.py`. Uses the standard library only;
  about 0.7 MB per site-year.
- **Parameters:** `ALLSKY_SFC_UV_INDEX` (all-sky UV Index), `CLOUD_AMT` (%),
  `ALLSKY_KT` (clearness index, NaN at night), `T2M` (°C). There is **no clear-sky UV
  parameter** (`CLRSKY_SFC_UV_INDEX` is rejected by the API).
- **Coverage checked:** hourly UV exists from at least 2001 to Oct 2025, at any lat/lon.
- **Downloaded (2026-10-08):** 7 sites × 2022–2024, giving **7,651 complete days** (missing
  hours < 0.05%):

| Site | Summer daily-max UVI (median / p90) | Winter median |
|---|---|---|
| Honolulu | 11.4 / 12.2 | 6.2 |
| Miami | 9.2 / 10.3 | 4.8 |
| Phoenix | 10.3 / 11.3 | 3.1 |
| Los Angeles | 9.8 / 10.8 | 2.6 |
| Sydney | 10.3 / 12.4 | 2.5 |
| Chicago | 7.0 / 8.8 | 1.0 |
| Seattle | 6.4 / 8.1 | 0.6 |

**How we use it:**
1. **Synthetic generator (step 3):** real hourly UV days provide the large-scale shape (time
   of day, season, cloudy days). Our generator adds minute-scale clouds, shade, indoor periods
   and the wearer's IMU motion on top.
2. **Checking our clear-sky model:** compare our formula to POWER on clear days (high
   `ALLSKY_KT`, low `CLOUD_AMT`).
3. **Testing the forecast-aware countdown (step 6) on past data:** predict the rest of a
   past day from what was known in the morning (a clear-sky curve, or the same date in other
   years), then compare with the UV that actually happened. Report the error in time-left
   minutes.

**Caveats:**
- **Satellite-derived** (CERES SYN1deg, ~1° grid ≈ 100 km), not a ground measurement.
- **Hourly averages:** these slightly understate instantaneous peak UVI, and they average
  out single passing clouds.
- **Not usable for calibrating our sensor.** That still needs an official ground UVI or a
  reference meter at the same spot (Phase 3).

## 4. Datasets considered and not used for modeling

- **NOAA ClimateBits UV Index:** a **monthly-average** UV climatology, built for
  visualization. POWER covers the same information at hourly resolution for any point, so
  this adds nothing for the model. It could be used for a background figure in the report.
- **WHO UV radiation / disease burden:** population-level deaths and DALYs attributable to
  UV. There are no individual exposure or sunburn labels, so it can't train or validate
  anything here. Use it in the report's **motivation** section only.
- **Measured ground UV (USDA UVMRP, EPA):** still worth considering in Phase 3 as ground
  truth near the calibration site. Not downloaded.

## 5. Our own recordings (Phase 3)

Session CSVs from the device (format in `docs/contracts.md`), with
event labels entered in the app.
