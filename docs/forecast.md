# Forecast-aware time-left

Code: `model/sunrisk/forecast.py`. Backtest: `python model/scripts/backtest_forecast.py`.

## Method
The prototype held the current UV reading constant. So a brief cloud made time-left jump up,
and it ignored UV rising toward noon. Now the countdown projects an expected UV curve over
the next 12 h at 1-minute steps:

```
expected(t) = clear_sky(t) × [ w(t)·k_now + (1 − w(t))·k_forecast(t) ],   w(t) = exp(−Δt / τ)
```

- `clear_sky(t)` comes from the sun's position, so the day's shape is built in.
- `k_now` = current estimated UV ÷ clear sky (captures the current cloud or shade).
- `k_forecast` = hourly forecast ÷ clear sky. **With no forecast it is 1 (clear sky)**,
  which is the conservative default.
- `τ` depends on the detected environment: sun 60 min, cloud 10 min, shade 30 min, indoor 0
  (indoors the countdown means "if you went out now"). All four are low-confidence model
  choices.

The tracker then integrates this curve with sunscreen wear-off for every Monte Carlo
particle.

## Backtest
Setup:
- **Cases:** 16,276. Warm-season days of 2024 at 7 NASA POWER sites; "now" = 10:00–15:00
  every 30 min.
- **Truth:** a minute-level UV curve, made from the day's hourly POWER UV with synthetic
  passing clouds.
- **Question:** minutes until 250 J/m² (skin type II MED).
- **Error** = predicted − actual. Positive means the user was told they had more time than
  they did (the unsafe direction).

| Method | Median \|err\| | p90 \|err\| | Unsafe by > 5 min | Over-cautious by > 15 min | Jitter (min per min) |
|---|---:|---:|---:|---:|---:|
| Constant (prototype) | 2.0 | 27.1 | 15.0% | 5.5% | 1.80 |
| Forecast-aware, perfect hourly forecast | 0.6 | 8.0 | 5.9% | 2.4% | 0.61 |
| Forecast-aware, climatology (2022–23 same dates) | 0.7 | 9.0 | 3.6% | 3.8% | 0.44 |
| Clear sky only | 0.0 | 11.9 | 0.0% | 7.2% | 0.03 |

**Conclusions:**
- The forecast-aware projection cuts unsafe errors by 2.5–4× and makes the countdown
  3–4× steadier.
- "Clear sky only" is never unsafe here, but only by construction: synthetic clouds never
  raise UV above clear sky.

**Caveats:**
- Truth and projection share the clear-sky shape and the cloud model, so this doesn't test
  clear-sky shape errors or real minute-scale variability. That needs measured minute data
  (Phase 3/4).
- "Perfect forecast" uses the actual POWER day. Real forecasts will be worse; the
  climatology row is a more realistic lower bound.
- The Monte Carlo sensor-calibration spread is also applied to the forecast part of the
  curve, as a stand-in for forecast uncertainty.
