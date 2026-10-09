# Roadmap

An awareness tool, not a medical device. It estimates time until sunburn per body part, with
honest uncertainty.

## Hardware (decided)

| Part | Choice | Notes |
|---|---|---|
| MCU / BLE | nRF52840 "SuperMini" (nice!nano-compatible Pro Micro clone) | BLE 5, LiPo charger, B+/B− pads. **No IMU on board.** Arduino: Adafruit nRF52 core, nice!nano board definition. |
| UV | LTR390 (I²C) | UVS mode; ~2300 counts/UVI at max gain/resolution (to be calibrated outdoors) |
| IMU | **To buy**, I²C, shares the bus with the LTR390 | Prefer one **with a magnetometer** (e.g. ICM-20948, LSM9DS1, LSM6DSOX + LIS3MDL), which tells us heading relative to the sun |
| Battery | 3.7 V LiPo | |
| Placement | **Wrist** (watch strap / band) | Matches the PAMAP2 wrist IMU |

### What wrist placement means for the model

- The sensor directly measures roughly what the **back of the hand and forearm** receive.
  Other body parts are inferred from the sun's position plus estimated posture.
- The UV reading swings as the arm moves. To estimate ambient UV, we correct for sensor tilt
  (using the accelerometer) or take the top of the readings over a short window when the
  sensor faces up, rather than using raw samples.
- Sleeves and the hand can shade the sensor. We need to detect this, not mistake it for shade
  or cloud.
- Posture (lying vs. sitting vs. standing) has to be inferred from the wrist alone, which is
  harder than from a chest sensor.

## Model

Mostly a physics/physiology model with Monte Carlo uncertainty. Two parts are learned:

1. **Activity/posture classifier.** Trained on the public **PAMAP2** wrist IMU data
   (see `docs/datasets.md`), evaluated leave-one-subject-out, then tested on our own device
   to measure how well it carries over.
2. **Personal MED.** A Bayesian update from skin-redness photos (prior from skin type).

Public datasets: **PAMAP2** (wrist IMU → activity/posture classifier) and **NASA POWER**
(hourly UV for 7 sites × 3 years → realistic synthetic days, and testing the forecast-aware
countdown on past data). Details in `docs/datasets.md`.

## Status (2026-10-08)

Phase 0 done. Phase 1 steps 1–7 done on synthetic data:
- core model
- contracts
- synthetic generator
- solar position and body-part dose
- event detection
- forecast-aware projection
- end-to-end beach-day figure

Next: Phase 1b (PAMAP2 classifier) and Phase 2 (hardware). Results and caveats:
`docs/body_dose.md`, `docs/event_detection.md`, `docs/forecast.md`.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | **Setup & contracts:** repo layout, `docs/contracts.md` (BLE packet, session CSV, risk engine API), `docs/assumptions.md`, dataset fetch | Firmware, model and app groups agree on the contracts |
| 1 | **Model on simulated data:** dose/MED/sunscreen/Monte Carlo core, synthetic generator, solar position, body-part dose, event detection, forecast-aware countdown | All tests pass; beach-day figure with per-body-part output; event-detection confusion matrix on synthetic data |
| 1b | **PAMAP2 activity classifier:** windowed features, rules baseline vs. small trained model (e.g. decision tree / gradient boosting), leave-one-subject-out | Confusion matrix + per-class F1 on PAMAP2, compared against the rules baseline |
| 2 | **Hardware bring-up** (*start now, alongside Phase 1*): wire LTR390 + IMU, firmware, BLE streaming in the contract format, phone logging | A 1-hour outdoor session recorded to a valid session CSV; battery life measured |
| 3 | **Calibration & data collection:** counts→UVI against official UVI or a reference meter at many sun angles; zero reading, angular response, temperature; labeled sessions | Calibration fit with a stated error; ≥ N hours of labeled own-device data (team picks N) |
| 4 | **Model on real data:** PAMAP2-trained classifier on our data (the cross-device result); check body-part model against a second sensor placed elsewhere on the body; update assumptions | Metrics on real data, including failures |
| 5 | **App:** React Native + BLE; risk engine ported to TypeScript, checked against Python on a shared set of test cases; countdown, alerts at p10, forecast feed, sunscreen logging | Runs end to end on a phone with the real device |
| 6 | **Personalization:** redness photo + color card → Bayesian MED update. *Use incidental exposure only; never expose anyone on purpose. Check whether IRB review applies.* | Demo on simulated users + any real observations |
| 7 | **Field test & write-up** | Report with limitations + "not a medical device" disclaimer |

**Timing risk:** calibration needs sunny days across a range of UV levels, and midday UVI keeps
falling through Oct–Dec. Phase 2 should run in parallel with Phase 1, not after it.

## Definition of done

**Must-have**
1. Wear the device outdoors; the phone shows a live, forecast-aware time-left range (p10/p50/p90).
2. Sensor calibrated, with its error measured and reported.
3. Per-body-part estimates, checked against at least one sensor placed directly on another body part.
4. Activity classifier trained on PAMAP2 and evaluated both on PAMAP2 (leave-one-subject-out)
   and on our own labeled data.
5. Alerts fire at the cautious (p10) estimate; sunscreen wear-off and swim loss are modeled.
6. Python and TypeScript tests pass, the two match on the shared test cases, and the
   contracts and assumptions are documented.

**Stretch:** trained swim detector on a public swimming dataset; personalization validated on
real redness data; firmware power optimization; multiple users.
