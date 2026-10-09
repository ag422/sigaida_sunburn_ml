"""Evaluate the rules-based event detector on synthetic days built from NASA POWER weather.

    python scripts/evaluate_events.py [--days 40] [--seed 0]

Prints confusion matrices and writes docs/figures/event_confusion.png.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sunrisk import config  # noqa: E402
from sunrisk.contracts import ACTIVITIES, ENVIRONMENTS  # noqa: E402
from sunrisk.datasets import nasa_power as npw  # noqa: E402
from sunrisk.events.evaluate import confusion_matrix, format_report  # noqa: E402
from sunrisk.events.features import build_windows, window_labels  # noqa: E402
from sunrisk.events.rules import RuleDetector  # noqa: E402
from sunrisk.synth.generate import generate_session  # noqa: E402
from sunrisk.synth.scenario import random_scenario  # noqa: E402

RAW = ROOT / "data" / "raw" / "nasa_power"
UV_MATTERS = 3.0   # clear-sky UVI at which environment errors start to matter for burns


def load_sites():
    sites = json.loads((RAW / "sites.json").read_text())
    out = {}
    for name, (lat, lon, off) in sites.items():
        s = npw.merge([npw.parse_hourly(json.loads(Path(f).read_text()))
                       for f in sorted(glob.glob(str(RAW / f"{name}_*.json")))])
        starts, rows = npw.daily_uv_curves(s, off)
        # warm-season days only (when sunburn matters): daily max UVI >= 5
        keep = rows.max(axis=1) >= 5
        out[name] = (lat, lon, starts[keep], rows[keep], npw.fit_site_scale(s))
    return out


def run(days: int, seed: int):
    rng = np.random.default_rng(seed)
    sites = load_sites()
    names = list(sites)
    det = RuleDetector()
    env_t, env_p, act_t, act_p, clear = [], [], [], [], []
    for _ in range(days):
        name = names[rng.integers(len(names))]
        lat, lon, starts, rows, scale = sites[name]
        k = rng.integers(len(starts))
        sc = random_scenario(rng, hours=6)
        ss = generate_session(sc, lat, lon, starts[k], rng, rows[k], site_scale=scale)
        w = build_windows(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, ss.uv_counts, ss.uv_gain,
                          ss.uv_res_bits, lat, lon, starts[k], rows[k], scale,
                          config.ALBEDO[sc.surface].value)
        e, a = det.predict(w)
        env_t.append(window_labels(ss.environment)); act_t.append(window_labels(ss.activity))
        env_p.append(e); act_p.append(a); clear.append(w.clear_uvi)
    cat = np.concatenate
    et, ep, matters = cat(env_t), cat(env_p), cat(clear) >= UV_MATTERS
    return (confusion_matrix(et, ep, len(ENVIRONMENTS)),
            confusion_matrix(et[matters], ep[matters], len(ENVIRONMENTS)),
            confusion_matrix(cat(act_t), cat(act_p), len(ACTIVITIES)))


def plot(cm_env, cm_act, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, cm, names, title in ((axes[0], cm_env, ENVIRONMENTS, "Environment"),
                                 (axes[1], cm_act, ACTIVITIES, "Activity")):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j]:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title} (row-normalised)" + (f", clear-sky UVI >= {UV_MATTERS:g}" if cm is cm_env else ""))
    fig.suptitle("Rules-based event detection on synthetic days (5 s windows)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cm_env, cm_env_uv, cm_act = run(args.days, args.seed)
    print("ENVIRONMENT (all daytime windows)\n" + format_report(cm_env, ENVIRONMENTS) + "\n")
    print(f"ENVIRONMENT (clear-sky UVI >= {UV_MATTERS:g})\n" + format_report(cm_env_uv, ENVIRONMENTS) + "\n")
    print("ACTIVITY\n" + format_report(cm_act, ACTIVITIES))
    plot(cm_env_uv, cm_act, ROOT / "docs" / "figures" / "event_confusion.png")


if __name__ == "__main__":
    main()
