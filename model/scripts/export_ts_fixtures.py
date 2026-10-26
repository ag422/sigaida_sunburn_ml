"""Export shared test cases for the TypeScript risk engine (app/risk-engine).

Runs the Python engine on fixed scenarios and writes inputs + expected outputs to
app/risk-engine/test/fixtures/parity.json. numpy's random generator can't be reproduced in
TypeScript, so each tracker case records the raw draws (uniforms for the MED, standard normals
for calibration and sunscreen amount) and the TypeScript test feeds them in.

Rerun after changing the engine or config.py:  python scripts/export_ts_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from sunrisk import config  # noqa: E402
from sunrisk.clearsky import clear_sky_uvi
from sunrisk.contracts import (BODY_PARTS, MedObservation, RiskStep, SunscreenApplication,
                               UserProfile)
from sunrisk.forecast import blend, expected_uvi_curve, minutes_to_dose
from sunrisk.med import PersonalMED
from sunrisk.solar import solar_position
from sunrisk.tracker import ExposureTracker

OUT = ROOT / "app" / "risk-engine" / "test" / "fixtures" / "parity.json"

# 2024-07-01 10:00 local in Los Angeles (17:00 UTC), rounded location
T0 = 1719853200.0
LA = (34.0, -118.2)


def num(x):
    """JSON has no Infinity; encode it as a string the TS loader converts back."""
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def nums(a):
    return [num(v) for v in np.asarray(a, dtype=float).ravel()]


def risk_output(out):
    q = lambda v: {"p10": num(v.p10), "p50": num(v.p50), "p90": num(v.p90)}
    return {
        "t_unix_s": out.t_unix_s,
        "first_to_burn": out.first_to_burn,
        "alert": out.alert,
        "projection": out.projection,
        "parts": [{
            "body_part": p.body_part,
            "minutes_left": q(p.minutes_left),
            "fraction_med_used": q(p.fraction_med_used),
            "effective_spf": num(p.effective_spf),
            "beyond_horizon": p.beyond_horizon,
        } for p in out.parts],
    }


class Case:
    """Runs a tracker and records every call so the TS test can replay it."""

    def __init__(self, name, profile, seed=0, n=200, uncertain=True, body_parts=BODY_PARTS):
        self.name, self.ops = name, []
        self.profile = profile
        self.body_parts = list(body_parts)
        self.uncertain = uncertain
        self.draws = None
        if uncertain:
            # same consumption order as ExposureTracker.__init__
            rng = np.random.default_rng(seed)
            self.draws = {
                "medUniform": nums(rng.random(n)),
                "calNormal": nums(rng.standard_normal(n)),
                "amountNormal": nums(rng.standard_normal(n)),
            }
        self.t = ExposureTracker(profile, rng=np.random.default_rng(seed), n_particles=n,
                                 uncertain=uncertain, body_parts=body_parts)
        if uncertain:  # the recorded draws must be exactly what the tracker used
            assert np.array_equal(self.t.cal, np.exp(config.SENSOR_CAL_LOG_SD.value * np.array(self.draws["calNormal"])))

    def sunscreen(self, t, spf, amount=None, parts=None):
        app = SunscreenApplication(t, spf, amount, tuple(parts) if parts else None)
        self.t.apply_sunscreen(app)
        self.ops.append({"op": "sunscreen", "app": {"t_unix_s": t, "spf_label": spf,
                                                     "amount_mg_cm2": amount, "body_parts": parts}})

    def step(self, t, uvi, environment="sun", activity="standing", ratios=None):
        self.t.step(RiskStep(t, uvi, environment, activity, ratios))
        self.ops.append({"op": "step", "step": {"t_unix_s": t, "uvi_ambient": uvi, "environment": environment,
                                                "activity": activity, "exposure_ratios": ratios}})

    def output(self, uvi_curve=None):
        out = self.t.output(None if uvi_curve is None else np.asarray(uvi_curve))
        self.ops.append({"op": "output", "uvi_curve": None if uvi_curve is None else nums(uvi_curve),
                         "expect": risk_output(out)})

    def to_json(self):
        p = self.profile
        return {
            "name": self.name,
            "profile": {"fitzpatrick": p.fitzpatrick, "med_observations": [
                {"effective_dose_j_m2": o.effective_dose_j_m2, "burned": o.burned} for o in p.med_observations]},
            "uncertain": self.uncertain,
            "body_parts": self.body_parts,
            "draws": self.draws,
            "ops": self.ops,
        }


def tracker_cases():
    cases = []

    c = Case("type 2, no sunscreen, steady UVI 8, default particle count", UserProfile(2), seed=1,
             n=config.N_PARTICLES)
    for k in range(0, 31, 5):
        c.step(T0 + k * 60, 8.0)
        c.output()
    cases.append(c)

    c = Case("deterministic (uncertain=False)", UserProfile(2), uncertain=False, body_parts=("forearm", "face"))
    c.sunscreen(T0, 30, 2.0, ["face"])
    c.step(T0, 8.0)
    c.output()
    c.step(T0 + 600, 8.0)
    c.output()
    cases.append(c)

    ratios = {"face": 0.6, "neck": 0.4, "shoulders": 0.9, "upper_back": 0.2, "chest": 0.5,
              "thighs": 0.3, "shins": 0.25}
    c = Case("type 3, sunscreen on some parts, ratios, shade and cloud, vigorous", UserProfile(3), seed=2)
    c.sunscreen(T0, 50, None, ["face", "neck"])
    c.sunscreen(T0 + 60, 30, 1.2, None)
    c.step(T0, 6.5, "sun", "walking", ratios)
    c.output()
    c.step(T0 + 900, 6.8, "shade", "sitting", {**ratios, "forearm": 0.4})
    c.output()
    c.step(T0 + 1800, 2.4, "cloud", "vigorous", ratios)
    c.step(T0 + 3600, 7.1, "sun", "vigorous", ratios)
    c.output()
    c.step(T0 + 5400, 7.4, "sun", "lying")
    c.output()
    cases.append(c)

    c = Case("swim removes protection", UserProfile(2), seed=3)
    c.sunscreen(T0, 50, 2.0)
    c.step(T0, 8.0, "sun", "standing")
    c.step(T0 + 1200, 3.2, "sun", "swimming")
    c.output()
    c.step(T0 + 2400, 8.5, "sun", "standing")
    c.output()
    cases.append(c)

    c = Case("personal MED from observations (incl. soft label)",
             UserProfile(4, [MedObservation(300.0, 0.0), MedObservation(520.0, 1.0), MedObservation(450.0, 0.7)]),
             seed=4)
    c.step(T0, 9.0)
    c.step(T0 + 1800, 9.0)
    c.output()
    cases.append(c)

    c = Case("already burned and zero UV", UserProfile(1), seed=5, body_parts=("forearm", "face"))
    c.step(T0, 0.0)
    c.output()                                  # everything beyond the horizon
    c.step(T0 + 60, 11.0, ratios={"face": 0.0})
    c.step(T0 + 3 * 3600, 11.0)
    c.output()                                  # forearm burned, face just started
    cases.append(c)

    fc_t = [T0 - 3600 + h * 3600 for h in range(10)]
    fc_uvi = [5.5, 7.0, 8.2, 6.0, 3.1, 4.0, 5.2, 3.3, 1.5, 0.4]
    curve = expected_uvi_curve(T0, *LA, uvi_now=6.0, environment="sun", forecast_t=fc_t, forecast_uvi=fc_uvi)
    c = Case("forecast-aware projection", UserProfile(2), seed=6)
    c.sunscreen(T0, 30)
    c.step(T0, 6.0)
    c.step(T0 + 300, 6.0)
    c.output(curve)
    c.output(curve[:90])                        # short curve
    c.output()                                  # same state, constant projection
    cases.append(c)

    return [c.to_json() for c in cases]


def posterior_cases():
    out = []
    for fitz, obs in [(1, []), (3, []), (6, []), (2, [(200.0, 1.0)]), (4, [(300.0, 0.0), (520.0, 1.0), (450.0, 0.7)])]:
        post = PersonalMED(fitz)
        for d, b in obs:
            post.update(d, b)
        u = [0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 0.9999999]
        out.append({
            "fitzpatrick": fitz,
            "observations": [{"effective_dose_j_m2": d, "burned": b} for d, b in obs],
            "quantiles": {str(q): post.quantile(q) for q in (0.1, 0.5, 0.9)},
            "from_uniform": {"u": u, "med": nums(np.exp(np.interp(u, post._cdf(), post.log_grid)))},
        })
    return out


def solar_cases():
    out = []
    for lat, lon in [LA, (40.1, -88.2), (-33.9, 151.2), (64.8, -147.7), (0.0, 0.0)]:
        for h in range(0, 24, 3):
            t = T0 + h * 3600 + 17.0
            elev, az = solar_position(t, lat, lon)
            out.append({"t": t, "lat": lat, "lon": lon, "elevation": float(elev), "azimuth": float(az),
                        "clear_sky_uvi": float(clear_sky_uvi(elev))})
    return out


def forecast_cases():
    fc_t = [T0 - 3600 + h * 3600 for h in range(10)]
    fc_uvi = [5.5, 7.0, 8.2, 6.0, 3.1, 4.0, 5.2, 3.3, 1.5, 0.4]
    out = []
    for env, uvi_now, use_fc in [("sun", 6.0, True), ("cloud", 2.0, True), ("shade", 3.0, False),
                                 ("indoor", 0.1, True), ("sun", None, False)]:
        kw = dict(forecast_t=fc_t, forecast_uvi=fc_uvi) if use_fc else {}
        curve = expected_uvi_curve(T0, *LA, uvi_now=uvi_now, environment=env, **kw)
        out.append({
            "t_now": T0, "lat": LA[0], "lon": LA[1], "uvi_now": uvi_now, "environment": env,
            "forecast": [{"t_unix_s": t, "uvi": u} for t, u in zip(fc_t, fc_uvi)] if use_fc else None,
            "curve": nums(curve),
            "minutes_to_250": num(minutes_to_dose(curve, 250.0)),
        })
    # blend falls back to clear x forecast when the sun is low
    clear = np.array([0.1, 0.2, 0.5])
    out_blend = {"clear": nums(clear), "k_fc": [1.0, 0.8, 0.9], "uvi_now": 0.05, "tau": 60.0,
                 "expect": nums(blend(clear, np.array([1.0, 0.8, 0.9]), 0.05, 60.0))}
    return out, out_blend


def config_values():
    vals = {name: a.value for name, a in config.all_assumptions().items()}
    vals.update(N_PARTICLES=config.N_PARTICLES, ALERT_QUANTILE=config.ALERT_QUANTILE,
                PROJECTION_HORIZON_MIN=config.PROJECTION_HORIZON_MIN,
                PROJECTION_STEP_MIN=config.PROJECTION_STEP_MIN)
    return vals


def main():
    forecast, blend_case = forecast_cases()
    data = {
        "generated_by": "model/scripts/export_ts_fixtures.py",
        "config": config_values(),
        "posterior": posterior_cases(),
        "solar": solar_cases(),
        "forecast": forecast,
        "blend": blend_case,
        "trackers": tracker_cases(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(data['trackers'])} tracker cases)")


if __name__ == "__main__":
    main()
