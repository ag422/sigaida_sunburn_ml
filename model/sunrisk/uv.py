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


# LTR390 integration time per resolution setting (datasheet; verify against the part we have).
_INTEGRATION_MS = {20: 400.0, 19: 200.0, 18: 100.0, 17: 50.0, 16: 25.0, 13: 3.125}


def counts_per_uvi(gain: int, res_bits: int,
                   ref_counts_per_uvi=config.LTR390_COUNTS_PER_UVI.value) -> float:
    """Sensitivity scales with gain and integration time; reference is gain 18, 20-bit."""
    return ref_counts_per_uvi * (gain / 18.0) * (_INTEGRATION_MS[res_bits] / 400.0)
