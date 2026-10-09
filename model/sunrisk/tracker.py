"""Monte Carlo exposure tracker: the risk engine the app calls.

Each particle is one plausible "world": a MED drawn from the user's posterior, a sensor
calibration error, and a multiplier on how much sunscreen was really applied. Particles are
drawn once per session (common random numbers), so the output moves smoothly over time.

PORT NOTE: state is a handful of (n_particles x n_parts) float arrays; the projection builds
an (n_particles x n_parts x n_steps) array. In TypeScript, loop over steps instead and keep a
running sum to avoid the 3-D allocation.
"""

from __future__ import annotations

import numpy as np

from . import config
from .contracts import (BODY_PARTS, BodyPartRisk, Quantiles, RiskOutput, RiskStep,
                        SunscreenApplication, UserProfile)
from .med import PersonalMED
from .uv import W_PER_UVI


class ExposureTracker:
    def __init__(self, profile: UserProfile, rng: np.random.Generator | None = None,
                 n_particles: int = config.N_PARTICLES, uncertain: bool = True,
                 body_parts: tuple[str, ...] = BODY_PARTS):
        self.parts = tuple(body_parts)
        self._part_idx = {p: i for i, p in enumerate(self.parts)}
        rng = rng if rng is not None else np.random.default_rng(0)

        posterior = PersonalMED(profile.fitzpatrick)
        for obs in profile.med_observations:
            posterior.update(obs.effective_dose_j_m2, obs.burned)
        self.posterior = posterior

        if uncertain:
            n = n_particles
            self.med = posterior.sample(rng, n)
            self.cal = np.exp(rng.normal(0.0, config.SENSOR_CAL_LOG_SD.value, n))
            self.amount_mult = np.exp(rng.normal(0.0, config.APPLIED_AMOUNT_LOG_SD.value, n))
        else:
            n = 1
            self.med = np.array([posterior.median()])
            self.cal = np.ones(1)
            self.amount_mult = np.ones(1)

        shape = (n, len(self.parts))
        self.dose = np.zeros(shape)        # effective (post-sunscreen) dose to skin, J/m^2
        self.protect = np.zeros(shape)     # sum (SPF-1) * amount / A_lab, i.e. SPF_eff - 1
        self.t_unix_s: float | None = None
        self._last_step: RiskStep | None = None

    # -- inputs ----------------------------------------------------------------------------

    def apply_sunscreen(self, app: SunscreenApplication) -> None:
        amount = app.amount_mg_cm2 if app.amount_mg_cm2 is not None else config.TYPICAL_APPLIED_AMOUNT.value
        idx = self._indices(app.body_parts)
        gain = (app.spf_label - 1.0) * amount / config.LAB_SUNSCREEN_AMOUNT.value
        self.protect[:, idx] += gain * self.amount_mult[:, None]

    def step(self, s: RiskStep) -> None:
        """Advance to s.t_unix_s. The previous step's conditions apply over the interval."""
        prev = self._last_step
        if prev is not None:
            dt = s.t_unix_s - prev.t_unix_s
            if dt < 0:
                raise ValueError("steps must be in time order")
            self._accumulate(prev, dt)
            if prev.activity == "swimming" and s.activity != "swimming":
                self.protect *= config.SWIM_RETENTION.value
        self._last_step = s
        self.t_unix_s = s.t_unix_s

    def _accumulate(self, s: RiskStep, dt_s: float) -> None:
        ratios = self._ratios(s.exposure_ratios)
        e = s.uvi_ambient * W_PER_UVI * self.cal                     # (n,)
        self.dose += e[:, None] * ratios[None, :] / (1.0 + self.protect) * dt_s
        tau_h = (config.SUNSCREEN_WEAR_TAU_ACTIVE_H if s.activity == "vigorous"
                 else config.SUNSCREEN_WEAR_TAU_H).value
        self.protect *= np.exp(-dt_s / 3600.0 / tau_h)

    # -- outputs ---------------------------------------------------------------------------

    def minutes_left(self, uvi_curve=None, exposure_ratios=None, activity=None) -> np.ndarray:
        """(n_particles, n_parts) minutes until each particle reaches its MED.

        uvi_curve: UVI at 1-minute steps starting now. None -> hold the current reading
        constant (the prototype behaviour). Values beyond the horizon are np.inf.
        """
        step_min = config.PROJECTION_STEP_MIN
        n_steps = int(config.PROJECTION_HORIZON_MIN / step_min)
        last = self._last_step
        if uvi_curve is None:
            uvi_curve = np.full(n_steps, last.uvi_ambient if last else 0.0)
        uvi_curve = np.asarray(uvi_curve, dtype=float)[:n_steps]
        n_steps = uvi_curve.size
        ratios = self._ratios(exposure_ratios if exposure_ratios is not None
                              else (last.exposure_ratios if last else None))
        activity = activity or (last.activity if last else "standing")
        tau_h = (config.SUNSCREEN_WEAR_TAU_ACTIVE_H if activity == "vigorous"
                 else config.SUNSCREEN_WEAR_TAU_H).value

        dt_s = step_min * 60.0
        decay = np.exp(-np.arange(n_steps) * dt_s / 3600.0 / tau_h)           # (T,)
        spf = 1.0 + self.protect[:, :, None] * decay[None, None, :]            # (n,P,T)
        e = uvi_curve[None, None, :] * W_PER_UVI * self.cal[:, None, None]    # (n,1,T)
        incr = e * ratios[None, :, None] / spf * dt_s                          # (n,P,T)
        cum = self.dose[:, :, None] + np.cumsum(incr, axis=2)                  # dose at end of each step

        remaining = self.med[:, None] - self.dose                              # (n,P)
        out = np.full(remaining.shape, np.inf)
        out[remaining <= 0] = 0.0
        crossed = cum >= self.med[:, None, None]
        hit = crossed.any(axis=2) & (remaining > 0)
        k = np.argmax(crossed, axis=2)                                         # first step index
        # linear interpolation inside the crossing step
        before = np.where(k > 0, np.take_along_axis(cum, np.maximum(k - 1, 0)[..., None], 2)[..., 0], self.dose)
        inc_k = np.take_along_axis(incr, k[..., None], 2)[..., 0]
        frac = np.where(inc_k > 0, (self.med[:, None] - before) / np.where(inc_k > 0, inc_k, 1.0), 0.0)
        out[hit] = ((k + frac) * step_min)[hit]
        return out

    def output(self, uvi_curve=None, projection: str | None = None) -> RiskOutput:
        mins = self.minutes_left(uvi_curve)
        frac = self.dose / self.med[:, None]
        spf = 1.0 + self.protect
        q = config.ALERT_QUANTILE
        parts = []
        for j, name in enumerate(self.parts):
            m = mins[:, j]
            parts.append(BodyPartRisk(
                body_part=name,
                minutes_left=_quantiles(m, q),
                fraction_med_used=_quantiles(frac[:, j], q),
                effective_spf=float(np.median(spf[:, j])),
                beyond_horizon=bool(np.isinf(np.quantile(m, q, method="inverted_cdf"))),
            ))
        first = min(parts, key=lambda p: p.minutes_left.p10)
        return RiskOutput(
            t_unix_s=self.t_unix_s or 0.0,
            parts=parts,
            first_to_burn=first.body_part,
            alert=first.minutes_left.p10 < config.ALERT_LEAD_MIN.value,
            projection=projection or ("constant" if uvi_curve is None else "forecast"),
        )

    # -- helpers ---------------------------------------------------------------------------

    def _indices(self, parts):
        if parts is None:
            return np.arange(len(self.parts))
        return np.array([self._part_idx[p] for p in parts])

    def _ratios(self, ratios: dict[str, float] | None) -> np.ndarray:
        r = np.ones(len(self.parts))
        if ratios:
            for p, v in ratios.items():
                r[self._part_idx[p]] = v
        return r


def _quantiles(x: np.ndarray, q_low: float) -> Quantiles:
    # inverted_cdf avoids interpolating between finite and inf (beyond-horizon) particles
    lo, mid, hi = np.quantile(x, [q_low, 0.5, 1.0 - q_low], method="inverted_cdf")
    return Quantiles(float(lo), float(mid), float(hi))
