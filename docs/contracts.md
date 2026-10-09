# Data contracts (version 1)

The source of truth is `model/sunrisk/contracts.py`. This doc explains it for the firmware and
app groups. If you change one, change both and bump `CONTRACT_VERSION`.

## 1. BLE packet (device → phone)

- One **notification** per UV conversion. At 20-bit resolution the LTR390 converts every
  400 ms, so that's **2.5 packets/s**.
- Each packet carries the **8 IMU samples** (20 Hz) taken during that 400 ms window.
- **164 bytes, little-endian.** This fits one notification at the iOS default ATT MTU (185),
  so no MTU negotiation is needed. Throughput is about 410 B/s, far below BLE limits.

| Offset | Type | Field | Notes |
|---:|---|---|---|
| 0 | u8 | `version` | = 1 |
| 1 | u8 | `flags` | bit0 `HAS_MAG`, bit1 `UV_SATURATED`, bit2 `LOW_BATTERY`, bit3 `UV_INVALID` |
| 2 | u16 | `seq` | increments per packet, wraps; gaps = dropped packets |
| 4 | u32 | `t_device_ms` | ms since boot when the UV conversion finished |
| 8 | u32 | `uv_counts` | raw LTR390 UVS counts (20-bit max) |
| 12 | u8 | `uv_gain` | 1, 3, 6, 9 or 18 |
| 13 | u8 | `uv_res_bits` | 13, 16, 17, 18, 19 or 20 |
| 14 | u16 | `battery_mv` | |
| 16 | u16 | `uvi_x100` | firmware's own UVI estimate × 100. **Display only**; the phone recomputes from raw counts with the current calibration |
| 18 | u8 | `imu_n` | = 8 |
| 19 | u8 | `imu_rate_hz` | = 20 |
| 20 + 18·i | 9 × i16 | IMU sample i | `ax ay az` (milli-g), `gx gy gz` (0.1 °/s), `mx my mz` (0.1 µT, zero if no magnetometer) |

Rules:
- **Send raw counts plus gain and resolution.** Calibration happens on the phone, so we can
  recalibrate without reflashing.
- **Sample timing:** the IMU samples are evenly spaced, and the last one coincides with
  `t_device_ms`.
- **Values that don't fit in int16 saturate (clamp to the min/max) rather than wrap around.**
- **IMU axes** are the IMU chip's own axes, as mounted. Firmware must document the mounting
  in `firmware/README.md` (which way +z points relative to the wrist: out of the back of the
  forearm or into it). **TBD once the IMU is chosen.**
- **Auto-gain:** if `UV_SATURATED` is set, firmware should step the gain down. The phone
  handles any gain/resolution combination.

## 2. Recorded session files

A session is a folder with three files:

**`session.csv`** has one row per IMU sample (20 Hz). Columns, in order:

```
t_unix_s, t_device_ms, seq, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps,
mx_ut, my_ut, mz_ut, uv_counts, uv_gain, uv_res_bits, uvi, uv_new, battery_mv
```

- `t_unix_s`: UTC seconds (float). The phone maps the device clock onto phone time, using the
  first packet and fitting for drift.
- The UV columns repeat on all 8 rows of a packet. `uv_new = 1` marks the last row, where the
  conversion ended.
- `mx/my/mz` are empty when there is no magnetometer.
- `uvi` is computed by the phone using the calibration recorded in `meta.json`.

**`events.csv`** has the columns `t_unix_s, kind, value`:

| kind | value |
|---|---|
| `label_environment` | `sun` / `shade` / `cloud` / `indoor`, in effect from this time on |
| `label_activity` | `lying` / `sitting` / `standing` / `walking` / `light_activity` / `vigorous` / `swimming` |
| `sunscreen` | `spf=30;amount=0.8;parts=face\|neck` (amount and parts optional) |
| `note` | free text |

**`meta.json`** holds:
- session and device IDs, firmware version, `CONTRACT_VERSION`
- wrist (`left`/`right`) and IMU mounting
- calibration (`counts_per_uvi` at the reference gain/resolution, plus the date calibrated)
- approximate lat/lon
- the user's Fitzpatrick type

Location is personal data: round it to 0.1° (~10 km), which is plenty for sun position and
forecasts.

## 3. Risk engine interface (what the app calls)

Implemented in Python as `ExposureTracker` (`model/sunrisk/tracker.py`), to be ported to
TypeScript. The engine is pure: no network, files or clock. All randomness comes from a seeded
RNG, so the TypeScript port can be tested against the Python version on the same inputs.

**Setup:** `ExposureTracker(UserProfile)`

- `UserProfile.fitzpatrick`: 1–6.
- `UserProfile.med_observations`: a list of `{effective_dose_j_m2, burned}`.
  - `burned` is 0/1, or a probability coming from the redness-photo classifier.
  - The engine rebuilds the MED posterior from these each session.

**Inputs over time:**

| Call | Input |
|---|---|
| `apply_sunscreen(SunscreenApplication)` | `t_unix_s`, `spf_label`, `amount_mg_cm2` (null → typical 0.8), `body_parts` (null → all) |
| `step(RiskStep)` every few seconds | `t_unix_s`, `uvi_ambient` (orientation-corrected), `environment`, `activity`, `exposure_ratios` per body part (null → 1.0) |
| `output(uvi_curve?)` | optional expected UVI at 1-min steps from now (forecast); default holds the current UVI constant |

**Output, `RiskOutput`:**

```
t_unix_s
first_to_burn          body part with the lowest p10 minutes_left
alert                  true if that p10 < ALERT_LEAD_MIN (15 min, placeholder)
projection             "constant" | "forecast"
parts[]:
  body_part
  minutes_left         {p10, p50, p90}   (Infinity if beyond the 12 h horizon)
  fraction_med_used    {p10, p50, p90}
  effective_spf        median
  beyond_horizon       bool
```

**UI rule:** headline numbers and alerts use **p10** (the pessimistic end). Show the p10–p90
range, never a single precise-looking number.

Body parts: `forearm` (the sensor site), `face`, `neck`, `shoulders`, `upper_back`, `chest`,
`thighs`, `shins`.
