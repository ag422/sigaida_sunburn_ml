"""Minute-scale cloud process matched to an hourly cloud modification factor (CMF).

A two-state Markov chain (sun visible / sun behind cloud). In the cloudy state transmission is
CLOUD_ATTENUATION; the cloudy fraction is set so each hour's mean equals its CMF. If the CMF is
below the cloud transmission the hour is overcast with transmission = CMF.
Cloud-edge enhancement (brief UV > clear sky) is not modelled.
"""

from __future__ import annotations

import numpy as np

from .. import config

CLOUD_DWELL_MIN = 5.0   # mean length of a cloud passage (synthetic only)


def minute_clouds(cmf_hourly, rng: np.random.Generator, dwell_min: float = CLOUD_DWELL_MIN):
    """Returns (transmission[n_hours*60], cloudy[n_hours*60] bool)."""
    a = config.CLOUD_ATTENUATION.value
    cmf = np.clip(np.nan_to_num(np.asarray(cmf_hourly, dtype=float), nan=1.0), 0.0, 1.0)
    n = cmf.size * 60
    trans = np.ones(n)
    cloudy = np.zeros(n, dtype=bool)
    state = False
    leave = 1.0 / dwell_min
    for h, m in enumerate(cmf):
        sl = slice(h * 60, (h + 1) * 60)
        if m >= 0.98:
            state = False
            continue
        if m <= a:
            trans[sl], cloudy[sl], state = m, True, True
            continue
        p = (1.0 - m) / (1.0 - a)                       # stationary cloudy fraction
        enter = min(1.0, p * leave / (1.0 - p))
        for i in range(h * 60, (h + 1) * 60):
            state = (rng.random() >= leave) if state else (rng.random() < enter)
            cloudy[i] = state
        trans[sl] = np.where(cloudy[sl], a, 1.0)
    return trans, cloudy
