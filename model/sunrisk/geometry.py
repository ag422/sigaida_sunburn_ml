"""UV on a tilted surface, and body-part surface normals by posture.

Model (documented in docs/body_dose.md):
    E_surface = E_dir_normal * max(0, n . s)              direct beam
              + E_diffuse * (1 + n_z) / 2                  isotropic sky
              + albedo * E_ground * (1 - n_z) / 2          ground reflection
with E_h the horizontal erythemal irradiance, f_d its diffuse share,
E_diffuse = f_d E_h and E_dir_normal = (1 - f_d) E_h / mu0.

Simplifications (all flagged):
- isotropic sky (real UV sky is brighter near the sun and the horizon);
- one flat normal per body part, no self-shading by other body parts, no curvature;
- body parts in shade get only the sky-view share of diffuse light.

Frames: ENU world (x east, y north, z up). Posture normals are given in a "heading frame":
x = the direction the person faces (or where the head points when lying), y = their left,
z = up. `heading_deg` rotates that frame (clockwise from north, like an azimuth).

PORT NOTE: small fixed tables and dot products only.
"""

from __future__ import annotations

import numpy as np

from . import config
from .contracts import BODY_PARTS

MIN_MU0 = 0.05  # below ~3 deg elevation, don't divide by mu0 (direct beam is negligible anyway)


def diffuse_fraction(elev_deg, cloud=False):
    """Diffuse share of horizontal erythemal UV. Under cloud it is all diffuse."""
    mu0 = np.sin(np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0) * np.pi / 180.0)
    fd = 1.0 - config.DIFFUSE_K.value * mu0 ** config.DIFFUSE_P.value
    return np.where(cloud, 1.0, fd)


def surface_ratio(normals, sun_enu, f_d, albedo, sky_view=1.0, direct_visible=1.0):
    """Irradiance on surfaces with unit `normals` (..., 3) relative to unobstructed horizontal.

    sun_enu (..., 3); f_d, sky_view, direct_visible broadcast. Returns (...,).
    """
    n = np.asarray(normals, dtype=float)
    s = np.asarray(sun_enu, dtype=float)
    mu0 = np.maximum(s[..., 2], MIN_MU0)
    cos_inc = np.maximum(np.sum(n * s, axis=-1), 0.0)
    above = s[..., 2] > 0
    direct = np.where(above, (1.0 - f_d) * cos_inc / mu0, 0.0) * direct_visible
    sky = f_d * (1.0 + n[..., 2]) / 2.0 * sky_view
    # nearby ground receives the same (possibly shaded) horizontal irradiance as the person
    ground_lit = np.where(above, (1.0 - f_d) * direct_visible, 0.0) + f_d * sky_view
    ground = albedo * ground_lit * (1.0 - n[..., 2]) / 2.0
    return direct + sky + ground


def rotate_heading(v_heading_frame, heading_deg):
    """Heading frame -> ENU. v (..., 3), heading (...,) broadcast."""
    v = np.asarray(v_heading_frame, dtype=float)
    h = np.asarray(heading_deg, dtype=float) * np.pi / 180.0
    fx, fy = np.sin(h), np.cos(h)          # facing direction in ENU
    lx, ly = -np.cos(h), np.sin(h)         # left of facing
    x = v[..., 0] * fx + v[..., 1] * lx
    y = v[..., 0] * fy + v[..., 1] * ly
    z = np.broadcast_to(v[..., 2], x.shape)
    return np.stack([x, y, z], axis=-1)


def _unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


_c, _s = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))

# Body-part normals in the heading frame. "neck" = back of the neck, "thighs"/"shins" = front.
# Values marked (*) are rough guesses for curved surfaces.
POSTURE_NORMALS = {
    "upright": {        # standing, walking, active
        "forearm":    _unit(0, 1, 0),        # back of wrist facing outward, arm hanging (*)
        "face":       _unit(_c, 0, _s),      # facing forward, tilted 20 deg up (*)
        "neck":       _unit(-_c, 0, _s),
        "shoulders":  _unit(0, 0, 1),
        "upper_back": _unit(-1, 0, 0.27),
        "chest":      _unit(1, 0, 0.27),
        "thighs":     _unit(1, 0, 0),
        "shins":      _unit(1, 0, 0),
    },
    "sitting": {
        "forearm":    _unit(0, 0, 1),        # resting on lap, back of wrist up
        "face":       _unit(_c, 0, _s),
        "neck":       _unit(-_c, 0, _s),
        "shoulders":  _unit(0, 0, 1),
        "upper_back": _unit(-1, 0, 0.27),
        "chest":      _unit(1, 0, 0.27),
        "thighs":     _unit(0, 0, 1),
        "shins":      _unit(1, 0, 0),
    },
    "supine": {         # lying on back; heading = where the head points
        "forearm":    _unit(0, 0, 1),
        "face":       _unit(0, 0, 1),
        "neck":       _unit(0, 0, -1),
        "shoulders":  _unit(1, 0, 0.3),
        "upper_back": _unit(0, 0, -1),
        "chest":      _unit(0, 0, 1),
        "thighs":     _unit(0, 0, 1),
        "shins":      _unit(0, 0, 1),
    },
    "prone": {          # lying on front, head turned to the side
        "forearm":    _unit(0, 0, 1),
        "face":       _unit(0, 1, 0.2),
        "neck":       _unit(0.3, 0, 1),
        "shoulders":  _unit(0.3, 0, 1),
        "upper_back": _unit(0, 0, 1),
        "chest":      _unit(0, 0, -1),
        "thighs":     _unit(0, 0, -1),
        "shins":      _unit(0, 0, -1),
    },
    "swimming": {       # prone at the surface; submerged parts get WATER_UV_FACTOR
        "forearm":    _unit(0, 0, 1),
        "face":       _unit(1, 0, -0.5),
        "neck":       _unit(0.3, 0, 1),
        "shoulders":  _unit(0.3, 0, 1),
        "upper_back": _unit(0, 0, 1),
        "chest":      _unit(0, 0, -1),
        "thighs":     _unit(0, 0, -1),
        "shins":      _unit(0, 0, -1),
    },
}

SUBMERGED_WHEN_SWIMMING = {"face": 0.5, "chest": 1.0, "thighs": 1.0, "shins": 1.0}  # fraction of time under water

ACTIVITY_POSTURE = {
    "standing": "upright", "walking": "upright", "light_activity": "upright", "vigorous": "upright",
    "sitting": "sitting", "lying": "supine", "swimming": "swimming",
}


def posture_normals(posture: str, parts=BODY_PARTS) -> np.ndarray:
    table = POSTURE_NORMALS[posture]
    return np.stack([table[p] for p in parts])


def body_part_ratios(posture, sun_enu, f_d, albedo, heading_deg=None, sky_view=1.0,
                     direct_visible=1.0, parts=BODY_PARTS, n_headings=36):
    """Per-body-part irradiance relative to unobstructed horizontal, shape (len(parts),).

    heading_deg=None averages over all headings: what we must do when the wearer's facing
    direction relative to the sun is unknown (no magnetometer, or the wrist moves independently
    of the torso).
    """
    normals = posture_normals(posture, parts)                               # (P, 3)
    if heading_deg is None:
        headings = np.arange(n_headings) * 360.0 / n_headings
    else:
        headings = np.atleast_1d(heading_deg)
    enu = rotate_heading(normals[None, :, :], headings[:, None])            # (H, P, 3)
    r = surface_ratio(enu, np.asarray(sun_enu)[None, None, :], f_d, albedo, sky_view, direct_visible)
    r = r.mean(axis=0)
    if posture == "swimming":
        w = config.WATER_UV_FACTOR.value
        sub = np.array([SUBMERGED_WHEN_SWIMMING.get(p, 0.0) for p in parts])
        r = r * (1.0 - sub + sub * w)
    return r
