"""Every number the model depends on that is not a physical definition.

Each entry records where it came from and how much we trust it. `docs/assumptions.md` is
generated from this file (`python scripts/render_assumptions.py`); edit values here, never
in the doc.

Confidence levels:
    high   - definition or well-established measurement
    medium - published value, but varies between studies or people
    low    - placeholder / educated guess; must be revisited with our own data
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    value: float
    unit: str
    confidence: str  # "high" | "medium" | "low"
    source: str
    note: str = ""


# --- UV physics -------------------------------------------------------------------------

# UV Index is defined as 40 x erythemally weighted irradiance (W/m^2). Not an assumption,
# kept here so all numbers live in one place.
W_PER_M2_PER_UVI = Assumption(
    0.025, "W/m^2 per UVI", "high",
    "WHO/WMO Global Solar UV Index definition (UVI = 40 m^2/W x E_ery)",
)

# --- Minimal erythemal dose (MED) priors ------------------------------------------------

# Median MED per Fitzpatrick type (erythemally weighted J/m^2). Type is a weak predictor:
# measured MEDs overlap heavily between types, hence the wide prior below.
MED_MEDIAN_BY_FITZPATRICK = {
    1: Assumption(200.0, "J/m^2", "medium", "Fitzpatrick-type MED tables (e.g. Fitzpatrick 1988; WHO INTERSUN)"),
    2: Assumption(250.0, "J/m^2", "medium", "as above"),
    3: Assumption(350.0, "J/m^2", "medium", "as above"),
    4: Assumption(450.0, "J/m^2", "medium", "as above"),
    5: Assumption(600.0, "J/m^2", "medium", "as above"),
    6: Assumption(1000.0, "J/m^2", "low", "as above; very few measurements for type VI"),
}

MED_PRIOR_LOG_SD = Assumption(
    0.35, "ln(J/m^2)", "low",
    "Chosen so the 10-90% prior range spans roughly 0.64x-1.57x the median, reflecting the "
    "reported overlap between skin types (Harrison & Young 2002). Needs a proper literature fit.",
)

# Steepness of P(burn | dose, MED) = sigmoid(ln(dose/MED) / s). Smaller = sharper threshold.
BURN_LOGISTIC_SCALE = Assumption(
    0.15, "ln units", "low",
    "Placeholder: MED is defined as the threshold of just-perceptible erythema; "
    "this only sets how fuzzy that threshold is.",
)

# --- Sunscreen --------------------------------------------------------------------------

LAB_SUNSCREEN_AMOUNT = Assumption(
    2.0, "mg/cm^2", "high", "ISO 24444 / FDA SPF test application density",
)

TYPICAL_APPLIED_AMOUNT = Assumption(
    0.8, "mg/cm^2", "medium",
    "People typically apply 0.5-1.0 mg/cm^2, i.e. ~25-50% of the test amount "
    "(Petersen & Wulf 2014 review). 0.8 = 40%.",
)

APPLIED_AMOUNT_LOG_SD = Assumption(
    0.4, "ln(mg/cm^2)", "low", "Placeholder spread for how much a person actually applied.",
)

# Faurschou & Wulf 2007 found SPF roughly linear in applied amount:
#   SPF_eff = 1 + (SPF_label - 1) * amount / 2 mg/cm^2
# (An exponential relation, SPF_label ** (amount/2), is used elsewhere and gives LOWER
# protection; we use linear as the brief specifies and flag it.)
SPF_AMOUNT_MODEL = "linear"

SUNSCREEN_WEAR_TAU_H = Assumption(
    3.0, "h", "low",
    "Exponential wear-off time constant at rest. Placeholder from the prototype; "
    "real loss depends on rubbing, sweat, product.",
)

SUNSCREEN_WEAR_TAU_ACTIVE_H = Assumption(
    1.5, "h", "low", "Wear-off time constant during vigorous activity / sweating. Placeholder.",
)

SWIM_RETENTION = Assumption(
    0.5, "fraction", "low",
    "Fraction of remaining protection kept after one swim. 'Water resistant' labels only "
    "promise SPF after 40/80 min immersion in lab conditions; towel drying removes more.",
)

# --- Clear-sky UV and geometry -----------------------------------------------------------

# Clear-sky UVI = A * mu0**B * (ozone/300 DU)**C   (mu0 = sin(solar elevation))
CLEARSKY_A = Assumption(12.5, "UVI", "medium", "Madronich 2007, analytic clear-sky UV Index formula (Photochem. Photobiol. 83)")
CLEARSKY_B = Assumption(2.42, "-", "medium", "as above")
CLEARSKY_C = Assumption(-1.23, "-", "medium", "as above")

DEFAULT_OZONE_DU = Assumption(
    300.0, "DU", "medium",
    "Typical total ozone column; real values vary ~250-450 DU by season and latitude. "
    "Use a forecast/climatology value where available.",
)

UV_INCREASE_PER_KM = Assumption(
    0.10, "fraction per km", "medium", "WHO INTERSUN: UV rises ~10-12% per 1000 m altitude",
)

# Diffuse share of erythemal UV under clear sky: f_d = 1 - DIFFUSE_K * mu0**DIFFUSE_P.
# Gives ~0.45 at high sun, ~0.65 at 30 deg elevation, ~0.8 near the horizon. Erythemal UV is
# far more diffuse than visible light because Rayleigh scattering grows strongly at short
# wavelengths.
DIFFUSE_K = Assumption(0.55, "-", "low", "Hand-fitted shape to reported erythemal diffuse fractions (~0.4-0.5 at high sun). Needs a literature fit.")
DIFFUSE_P = Assumption(0.6, "-", "low", "as above")

ALBEDO = {
    "grass": Assumption(0.02, "fraction", "medium", "WHO INTERSUN: grass/soil reflect < 10% of UV"),
    "sand": Assumption(0.15, "fraction", "medium", "WHO INTERSUN: dry beach sand ~15%"),
    "water": Assumption(0.08, "fraction", "medium", "Water surface ~5-10% (higher with foam, ~25%)"),
    "snow": Assumption(0.80, "fraction", "medium", "WHO INTERSUN: fresh snow up to 80%"),
}

SHADE_SKY_VIEW = Assumption(
    0.5, "fraction", "low",
    "Share of sky diffuse still reaching skin under an umbrella/tree (direct beam fully "
    "blocked). Measurements under beach umbrellas range ~0.3-0.6.",
)

INDOOR_UV_FACTOR = Assumption(
    0.02, "fraction", "low", "Erythemal UV indoors near a window; glass blocks most UVB.",
)

CLOUD_ATTENUATION = Assumption(
    0.35, "fraction", "low",
    "UV transmitted when a cloud covers the sun (all diffuse). Broken-cloud edge enhancement "
    "(> 1) is NOT modelled.",
)

WATER_UV_FACTOR = Assumption(
    0.4, "fraction", "low", "Erythemal UV reaching skin ~0.3-0.5 m under water; depends strongly on clarity.",
)

SLEEVE_OCCLUSION_FACTOR = Assumption(
    0.1, "fraction", "low", "UV reaching the sensor when covered by a sleeve/hand (synthetic data only).",
)

# --- Sensor -----------------------------------------------------------------------------

LTR390_COUNTS_PER_UVI = Assumption(
    2300.0, "counts/UVI", "low",
    "LTR390 datasheet sensitivity at gain 18x, 20-bit. Must be replaced by outdoor "
    "calibration against an official UVI (Phase 3).",
)

SENSOR_CAL_LOG_SD = Assumption(
    0.20, "ln(UVI)", "low",
    "Uncertainty of counts->UVI before calibration. Should shrink to the measured residual "
    "spread after Phase 3.",
)

# --- Monte Carlo ------------------------------------------------------------------------

N_PARTICLES = 500
ALERT_QUANTILE = 0.10  # alerts use the pessimistic 10th percentile of time-left

ALERT_LEAD_MIN = Assumption(
    15.0, "min", "low",
    "Product choice, not physiology: alert when the pessimistic (p10) time-left drops below this.",
)
PROJECTION_HORIZON_MIN = 12 * 60
PROJECTION_STEP_MIN = 1.0


def all_assumptions() -> dict[str, Assumption]:
    """Flat name -> Assumption map, used to render docs/assumptions.md."""
    out: dict[str, Assumption] = {}
    for name, val in globals().items():
        if isinstance(val, Assumption):
            out[name] = val
        elif isinstance(val, dict) and val and all(isinstance(v, Assumption) for v in val.values()):
            for k, v in val.items():
                out[f"{name}[{k}]"] = v
    return out
