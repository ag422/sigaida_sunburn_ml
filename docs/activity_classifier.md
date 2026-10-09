# Activity classifier (Phase 1b, public dataset: PAMAP2)

Code: `model/sunrisk/datasets/pamap2.py`, `model/sunrisk/events/features.py` (`imu_features_ml`),
`model/sunrisk/events/learned.py`. Reproduce: `python model/scripts/train_activity.py`
(needs scikit-learn, `pip install ./model[ml]`, plus the PAMAP2 download).

## Data preparation
- **Data used:** PAMAP2 wrist IMU only (the dominant wrist), Protocol + Optional sessions,
  9 subjects. Activity 0 (transient) and car driving are dropped, as are runs shorter than
  10 s. The remaining 17 activities are mapped to our 6 classes (`LABEL_MAP`;
  `docs/datasets.md`). PAMAP2 has **no swimming**.
- **Matching our device:**
  - accelerometer: the ±16 g sensor, converted to g
  - gyroscope: converted to °/s
  - dropouts filled by interpolation
  - downsampled 100 → 20 Hz by block mean
- **Axis alignment, checked against gravity:**
  - PAMAP2 x points along the forearm toward the hand, like ours: about −1 g with the arm
    hanging while walking or standing.
  - z points out of the back of the wrist.
  - The only left-handed subject (108) has the opposite y sign to everyone else, exactly as
    a mirror image predicts. Right-wrist data are mirrored to our left-wrist convention
    (y → −y; gyroscope as a pseudovector).
  - The classifier's features also ignore the sign of y, so wrist roll and left/right
    mounting don't matter.
- **Windows:** 5,280 windows of 5 s. Features: 21 per window (`ML_FEATURES`): gravity
  direction along the forearm, |acc| statistics and percentiles, per-axis acc/gyro spread,
  step-rhythm autocorrelation, and spectral energy in 0.3–1 / 1–3 / 3–8 Hz.

## Evaluation: leave-one-subject-out
Each person is held out in turn and the model is trained on the other 8. The scores
therefore estimate performance on **someone the model has never seen**. Splitting windows at
random would leak the same person into training and test, and inflate the scores.

| Model | Macro-F1 | Accuracy |
|---|---:|---:|
| Rules baseline (tuned on our synthetic data) | 0.44 | 0.52 |
| Logistic regression | 0.72 | 0.75 |
| **Random forest** (60 trees, depth ≤ 12, balanced) | **0.78** | **0.81** |
| Gradient boosting (HistGradientBoosting) | 0.78 | 0.81 |

Random forest, per class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| lying | 0.70 | 0.52 | 0.60 |
| sitting | 0.78 | 0.81 | 0.80 |
| standing | 0.73 | 0.75 | 0.74 |
| walking | 0.94 | 0.88 | 0.91 |
| light_activity | 0.76 | 0.87 | 0.81 |
| vigorous | 0.86 | 0.78 | 0.82 |

- **Accuracy per held-out subject:** 0.73–0.88.
- **Top features:** gravity along the forearm (`acc_mean_x`), |acc| 90th percentile,
  acc spread along the forearm.
- **Confusion matrices:** `docs/figures/pamap2_confusion.png`.

**Choice:** a random forest. It ties gradient boosting on accuracy, but it exports to plain
arrays. `forest_to_dict` writes `model/models/activity_forest.json` (1.4 MB, trained on all
9 subjects). `ForestPredictor` runs it with numpy only, and agrees with scikit-learn on
99.98% of windows (rounding in the export); a unit test checks exact agreement on a small
forest. The same JSON plus a ~30-line tree walk runs in TypeScript.

## What this tells us
1. **The rules generalise badly to real people (0.44).** They were tuned on our synthetic
   motions, where they scored 0.95. Real people fidget while lying, hold their arms in many
   ways when standing, and walk at varied speeds. The learned model is the default for
   activity.
2. **Lying vs. sitting is the hard pair** (lying recall 0.52). From the wrist alone it's
   often truly ambiguous. The dose model already handles "lying" cautiously (higher of
   face-up and face-down exposure), so confusing the two mostly matters for shoulders,
   chest and back.
3. **Transfer to our synthetic device data is poor:** macro-F1 0.62 for the forest vs. 0.95
   for the rules. The forest calls synthetic *lying* sitting and synthetic *vigorous*
   walking. The rules win there only because they were tuned to it. This tells us our
   synthetic **motion** model is unrealistic, **not** that the forest is wrong.
   - In PAMAP2, lying people's hands rest about 30° below the elbow; our synthetic arm is
     level.
   - Real vigorous activity isn't a faster walk.
   - Synthetic IMU data should not be used to judge activity models. Use PAMAP2 now and our
     own recordings later.

## Limitations
- **Small, homogeneous group:** 9 subjects (8 male, aged 23–32).
- **Different device:** the Colibri IMU differs in noise, mounting and strap. We expect a
  further drop on our device; Phase 4 measures it.
- **No swimming:** the swim rule (|mean acc| < 0.75 g while rotating fast) overrides the
  classifier. A wrist swimming dataset or our own pool sessions are needed to check it.
- **Cycling counts as `vigorous`** (sweat wears off sunscreen), but the wrist is still on the
  handlebars, so it's often predicted as light activity or standing.
