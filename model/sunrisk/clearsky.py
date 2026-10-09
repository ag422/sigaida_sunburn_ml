"""Clear-sky UV Index from solar elevation (Madronich 2007 analytic formula).

PORT NOTE: scalar formula, trivial to port.
"""

from __future__ import annotations

import numpy as np

from . import config


def clear_sky_uvi(elev_deg, ozone_du=config.DEFAULT_OZONE_DU.value, altitude_km=0.0, scale=1.0):
    """Clear-sky UVI on a horizontal surface; zero when the sun is below the horizon.

    `scale` is a site correction (e.g. fitted against NASA POWER clear days for aerosols).
    """
    mu0 = np.sin(np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0) * np.pi / 180.0)
    uvi = (config.CLEARSKY_A.value * mu0 ** config.CLEARSKY_B.value
           * (np.asarray(ozone_du, dtype=float) / 300.0) ** config.CLEARSKY_C.value)
    return uvi * (1.0 + config.UV_INCREASE_PER_KM.value * altitude_km) * scale
