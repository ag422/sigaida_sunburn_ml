"""Reproduce the beach-day figure with per-body-part output, end to end on synthetic data.

    python scripts/simulate_day.py [--site miami] [--date 2024-07-15] [--fitzpatrick 2] [--seed 1]

Chain: synthetic wrist session (NASA POWER weather) -> event detection -> ambient UV and
body-part ratios -> Monte Carlo tracker with sunscreen -> forecast-aware time-left.
Writes docs/figures/beach_day.png and prints a summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sunrisk import config  # noqa: E402
from sunrisk.contracts import ACTIVITIES, BODY_PARTS, ENVIRONMENTS, RiskStep, SunscreenApplication, UserProfile  # noqa: E402
from sunrisk.datasets import nasa_power as npw  # noqa: E402
from sunrisk.med import prior_median  # noqa: E402
from sunrisk.session import analyze_session  # noqa: E402
from sunrisk.synth.generate import generate_session  # noqa: E402
from sunrisk.synth.scenario import beach_day  # noqa: E402
from sunrisk.tracker import ExposureTracker  # noqa: E402

RAW = ROOT / "data" / "raw" / "nasa_power"
ENV_COLORS = {"sun": "#f6d55c", "shade": "#9fc5e8", "cloud": "#cccccc", "indoor": "#8e7cc3"}


def load_day(site, date):
    sites = json.loads((RAW / "sites.json").read_text())
    lat, lon, off = sites[site]
    files = sorted(glob.glob(str(RAW / f"{site}_*.json")))
    s = npw.merge([npw.parse_hourly(json.loads(Path(f).read_text())) for f in files])
    starts, rows = npw.daily_uv_curves(s, off)
    day0 = (datetime.fromisoformat(date).replace(tzinfo=timezone.utc) - timedelta(hours=off)).timestamp()
    k = int(np.argmin(np.abs(starts - day0)))
    if abs(starts[k] - day0) > 1:
        raise SystemExit(f"{date} not in downloaded NASA POWER data for {site}")
    return lat, lon, off, starts[k], rows[k], npw.fit_site_scale(s)


def truth_fraction(ss, sunscreen, fitz):
    """Deterministic dose on each body part from synthetic truth, same sunscreen model."""
    tr = ExposureTracker(UserProfile(fitz), uncertain=False)
    pending = list(sunscreen)
    local = np.maximum(ss.uvi_local, 1e-6)
    out = []
    for j in range(0, len(ss.uv_idx), 12):
        t = ss.t_unix_s[ss.uv_idx[j]]
        while pending and pending[0].t_unix_s <= t:
            tr.apply_sunscreen(pending.pop(0))
        act = ACTIVITIES[ss.activity[ss.uv_idx[j]]]
        tr.step(RiskStep(t, float(local[j]), activity=act,
                         exposure_ratios=dict(zip(BODY_PARTS, ss.body_part_uvi[j] / local[j]))))
        out.append((t, tr.dose[0] / tr.med[0]))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def bands(ax, t_h, labels, colors, y0, y1):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(t_h[start], t_h[min(i, len(t_h) - 1)], ymin=y0, ymax=y1,
                       color=colors.get(labels[start], "white"), lw=0)
            start = i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="miami")
    ap.add_argument("--date", default="2024-07-15")
    ap.add_argument("--fitzpatrick", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    lat, lon, off, day0, hourly, scale = load_day(args.site, args.date)
    sc = beach_day()
    rng = np.random.default_rng(args.seed)
    ss = generate_session(sc, lat, lon, day0, rng, hourly, site_scale=scale)
    sunscreen = [SunscreenApplication(t, spf, amt, parts) for t, spf, amt, parts in ss.sunscreen_events]
    profile = UserProfile(args.fitzpatrick)

    res = analyze_session(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, ss.uv_counts, ss.uv_gain,
                          ss.uv_res_bits, lat, lon, day0, profile, sunscreen, hourly, scale,
                          config.ALBEDO[sc.surface].value, rng=np.random.default_rng(args.seed))
    t_truth, frac_truth = truth_fraction(ss, sunscreen, args.fitzpatrick)

    hours = lambda t: (np.asarray(t) - day0) / 3600.0  # noqa: E731
    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 0.35, 1.6, 1.4]})

    # 1. UV
    tu = ss.t_unix_s[ss.uv_idx]
    ax[0].plot(hours(tu), ss.uvi_sensor * 0 + ss.uv_counts / config.LTR390_COUNTS_PER_UVI.value,
               lw=0.3, color="#bbbbbb", label="sensor reading (nominal calibration)")
    ax[0].plot(hours(tu), ss.uvi_local, lw=1.2, color="black", label="true UVI at wearer (horizontal)")
    ax[0].plot(hours(res.t_window), res.uvi_local, lw=1.2, color="#e06c00", label="estimated (detector + pipeline)")
    ax[0].set_ylabel("UV Index")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Synthetic beach day: {args.site}, {args.date}, Fitzpatrick {args.fitzpatrick}, "
                    f"sensor calibration error x{ss.cal_true:.2f}")

    # 2. events: truth (top) vs detected (bottom)
    win = np.arange(len(res.t_window)) * 100
    true_env = np.array(ENVIRONMENTS)[ss.environment[np.minimum(win, len(ss.environment) - 1)]]
    bands(ax[1], hours(res.t_window), true_env, ENV_COLORS, 0.52, 1.0)
    bands(ax[1], hours(res.t_window), res.environment, ENV_COLORS, 0.0, 0.48)
    ax[1].set_yticks([0.25, 0.75], ["detected", "true"])
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ENV_COLORS.values()],
                 labels=list(ENV_COLORS), ncol=4, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.6))
    for t, *_ in ss.sunscreen_events:
        for a in ax:
            a.axvline(hours(t), color="#2a9d8f", ls=":", lw=1)

    # 3. fraction of MED per body part
    colors = plt.cm.tab10(np.arange(len(BODY_PARTS)))
    for j, p in enumerate(BODY_PARTS):
        ax[2].plot(hours(res.t_out), res.fraction_med[:, j, 1], color=colors[j], lw=1.5, label=p)
        ax[2].plot(hours(t_truth), frac_truth[:, j], color=colors[j], lw=0.8, ls="--")
    first = res.first_to_burn[-1]
    jf = BODY_PARTS.index(first)
    ax[2].fill_between(hours(res.t_out), res.fraction_med[:, jf, 0], res.fraction_med[:, jf, 2],
                       color=colors[jf], alpha=0.15, label=f"{first} p10-p90")
    ax[2].axhline(1.0, color="red", lw=1)
    ax[2].set_ylabel("fraction of MED used")
    ax[2].legend(ncol=5, fontsize=8, loc="upper left")
    ax[2].text(0.99, 0.02, "solid = estimated median, dashed = synthetic truth", transform=ax[2].transAxes,
               ha="right", fontsize=8)

    # 4. minutes left (p10), forecast-aware vs constant projection
    for j, p in enumerate(BODY_PARTS):
        ax[3].plot(hours(res.t_out), np.minimum(res.minutes_left[:, j, 0], 240), color=colors[j], lw=1.2)
    ax[3].plot(hours(res.t_out), np.minimum(res.minutes_left_constant[:, jf], 240), color=colors[jf],
               lw=0.8, ls=":", label=f"{first}: constant-UV projection")
    ax[3].axhline(config.ALERT_LEAD_MIN.value, color="red", lw=1, label="alert threshold")
    ax[3].set_ylabel("minutes left (p10, capped 240)")
    ax[3].set_xlabel("local standard time (h)")
    ax[3].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "beach_day.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)

    # summary
    print(f"MED prior median (type {args.fitzpatrick}): {prior_median(args.fitzpatrick):.0f} J/m^2")
    alert_t = res.t_out[res.alert]
    print("first alert:", f"{hours(alert_t[0]):.2f} h" if alert_t.size else "none")
    print(f"{'part':11s} {'est p10/p50/p90 fraction MED':>30s} {'truth':>6s}")
    for j, p in enumerate(BODY_PARTS):
        q = res.fraction_med[-1, j]
        print(f"{p:11s} {q[0]:9.2f} {q[1]:9.2f} {q[2]:9.2f} {frac_truth[-1, j]:9.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
