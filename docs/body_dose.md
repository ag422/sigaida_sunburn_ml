# Per-body-part dose: method and assumptions

Code: `model/sunrisk/solar.py`, `geometry.py`, `bodydose.py`, `pipeline.py`.
Numbers: `docs/assumptions.md`.

## 1. Sun position
NOAA Solar Calculator equations (Meeus), numpy only, accurate to ~0.01°. Tests check
solstice elevation, equation-of-time extremes and the southern hemisphere.

## 2. UV on a tilted surface
Horizontal erythemal irradiance `E_h` is split into a direct part and a diffuse part, using a
diffuse fraction `f_d` that grows as the sun gets lower (erythemal UV is ~45% diffuse at high
sun, and more at low sun):

```
E(n) = E_h(1−f_d)·max(0, n·s)/sin(elev)   direct beam, if visible
     + E_h·f_d·(1+n_z)/2                    isotropic sky
     + albedo·E_ground·(1−n_z)/2            ground reflection (sand 0.15, grass 0.02, ...)
```

- **Cloud covering the sun:** treated as all diffuse (`f_d = 1`).
- **Shade:** the direct beam is blocked, and only `SHADE_SKY_VIEW` (0.5) of the sky reaches
  the wearer.
- **Indoors:** `INDOOR_UV_FACTOR` (0.02).

**Flagged simplifications:**
- The real UV sky is brighter near the sun and the horizon (not isotropic).
- Each body part is one flat surface; no curvature, and no shading by other body parts.
- No cloud-edge enhancement.
- The `f_d` curve is hand-fitted (low confidence).

## 3. What the wrist sensor tells us
- **Tilt** (from the accelerometer): how far the sensor face points from straight up.
- **Compass direction of the sensor face:** only with a magnetometer (tilt-compensated
  compass). Without one we average over all directions.
- **Ambient UV estimate:** sum of readings ÷ sum of expected gains over a 60 s window. Samples
  where the sensor faces away from the sky (gain < 0.25) are skipped.
- **Swimming:** the expected gain is reduced because the sensor is under water ~40% of each
  stroke.

## 4. Body parts
For each posture (upright, sitting, supine, prone, swimming) every body part has a normal in
the wearer's frame. The wrist **cannot observe which way the torso faces**, so body-part
ratios are always **averaged over all headings**. That's a permanent limitation of wrist
placement, not something the magnetometer fixes, because the wrist turns independently of the
torso. The forearm (the sensor site) uses the measured/estimated ratio directly.

## 5. Results on synthetic data (beach day, Miami, 15 Jul)

| | Ambient estimate / truth (outdoors) |
|---|---|
| Tilt only, no sleeve occlusion | 1.00 |
| With magnetometer, no occlusion | 0.99 |
| Tilt only, 0.5 sleeve occlusions/h | 0.97 |

Per-body-part daily dose estimate / truth: **0.93–1.00**. It's a little low on shoulders, neck
and upper back, because averaging over heading can't capture the posture-specific facing
toward the sun.

Walking alone: tilt-only overestimates ambient UV by ~17% (the conservative direction). The
magnetometer brings this to ~1%.

**Caveat:** the synthetic truth and the estimator use the *same* geometry. These numbers
measure what is lost to noise, arm motion and unknown heading. They do **not** show the
geometry is right. That needs a second sensor worn on another body part (Phase 4).

## 6. Known risks
- **A sleeve or hand covering the sensor lowers the estimate**, which is the unsafe direction.
  Event detection must flag sudden ~10× drops with no change in orientation or environment.
- **Shade and cloud labels change the geometry** (no direct beam), so event-detection errors
  carry through to dose.
