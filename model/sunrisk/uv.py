"""UV Index <-> erythemal irradiance and dose integration. Pure functions."""

from __future__ import annotations

import numpy as np

from . import config

W_PER_UVI = config.W_PER_M2_PER_UVI.value
J_PER_SED = 100.0  # Standard Erythema Dose, by definition


def uvi_to_irradiance(uvi):
    """UV Index -> erythemally weighted irradiance (W/m^2)."""
    return np.asarray(uvi, dtype=float) * W_PER_UVI


def irradiance_to_uvi(e_ery):
    return np.asarray(e_ery, dtype=float) / W_PER_UVI


def counts_to_uvi(counts, counts_per_uvi=config.LTR390_COUNTS_PER_UVI.value):
    """Raw LTR390 UVS counts (at the reference gain/resolution) -> UVI."""
    return np.asarray(counts, dtype=float) / counts_per_uvi


def integrate_dose(t_s, e_ery):
    """Trapezoidal dose (J/m^2) from irradiance samples (W/m^2) at times t_s (seconds)."""
    t = np.asarray(t_s, dtype=float)
    e = np.asarray(e_ery, dtype=float)
    if t.size < 2:
        return 0.0
    return float(np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(t)))
