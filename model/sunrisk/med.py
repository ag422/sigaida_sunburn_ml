"""Minimal erythemal dose (MED): skin-type prior and per-user Bayesian update.

PORT NOTE: the posterior is a fixed log-spaced grid, so it ports to TypeScript as plain
arrays; no special functions are needed beyond exp/log.
"""

from __future__ import annotations

import numpy as np

from . import config


def prior_median(fitzpatrick: int) -> float:
    try:
        return config.MED_MEDIAN_BY_FITZPATRICK[fitzpatrick].value
    except KeyError:
        raise ValueError(f"Fitzpatrick type must be 1-6, got {fitzpatrick!r}") from None


def burn_probability(dose, med, scale=config.BURN_LOGISTIC_SCALE.value):
    """P(visible erythema | dose, MED): logistic in ln(dose/MED)."""
    dose = np.maximum(np.asarray(dose, dtype=float), 1e-9)
    z = np.log(dose / np.asarray(med, dtype=float)) / scale
    return 1.0 / (1.0 + np.exp(-z))


class PersonalMED:
    """Posterior over one user's MED on a log-spaced grid.

    Starts from a log-normal prior centred on the Fitzpatrick median and is updated with
    observed outcomes: (effective dose received, whether redness was seen).
    """

    def __init__(self, fitzpatrick: int, n_grid: int = 401,
                 log_sd: float = config.MED_PRIOR_LOG_SD.value):
        self.fitzpatrick = fitzpatrick
        mu = np.log(prior_median(fitzpatrick))
        self.log_grid = np.linspace(mu - 5 * log_sd, mu + 5 * log_sd, n_grid)
        self.grid = np.exp(self.log_grid)
        logp = -0.5 * ((self.log_grid - mu) / log_sd) ** 2
        self._set_log_post(logp)
        self.n_observations = 0

    def _set_log_post(self, logp):
        logp = logp - logp.max()
        p = np.exp(logp)
        self.weights = p / p.sum()

    def update(self, dose: float, burned: bool | float):
        """Bayesian update with one observation.

        `burned` may be a probability in [0, 1] (e.g. from a photo-redness classifier that
        is not certain), which gives a soft-label likelihood.
        """
        p_obs = float(burned)
        if not 0.0 <= p_obs <= 1.0:
            raise ValueError("burned must be bool or a probability in [0, 1]")
        p_burn = burn_probability(dose, self.grid)
        lik = p_obs * p_burn + (1.0 - p_obs) * (1.0 - p_burn)
        self._set_log_post(np.log(self.weights + 1e-300) + np.log(lik + 1e-300))
        self.n_observations += 1

    def _cdf(self):
        # CDF at bin centres (not right edges), so a symmetric posterior has its median
        # exactly on the centre grid point
        return np.cumsum(self.weights) - 0.5 * self.weights

    def quantile(self, q: float) -> float:
        return float(np.exp(np.interp(q, self._cdf(), self.log_grid)))

    def median(self) -> float:
        return self.quantile(0.5)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draws from the posterior (inverse-CDF on the grid, interpolated in log space)."""
        return np.exp(np.interp(rng.random(n), self._cdf(), self.log_grid))
