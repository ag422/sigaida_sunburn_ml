"""Phase 1b: train and evaluate the activity classifier on PAMAP2 (wrist IMU).

    python scripts/train_activity.py [--synthetic-days 20]

1. Leave-one-subject-out (LOSO) comparison on PAMAP2: rules baseline, logistic regression,
   random forest, gradient boosting. Windows from the held-out person are never seen in
   training, so scores estimate performance on a NEW person.
2. Train the random forest on all 9 subjects, export to model/models/activity_forest.json
   (numpy/TypeScript-portable), and check the numpy predictor matches scikit-learn.
3. Transfer check: PAMAP2-trained forest vs rules on synthetic device sessions.

Writes docs/figures/pamap2_confusion.png. Needs scikit-learn (pip install .[ml]).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sunrisk.contracts import ACTIVITIES  # noqa: E402
from sunrisk.datasets import pamap2  # noqa: E402
from sunrisk.events.evaluate import confusion_matrix, format_report, per_class_scores  # noqa: E402
from sunrisk.events.features import (ML_FEATURES, Windows, build_windows, imu_features,  # noqa: E402
                                     imu_features_ml, window_labels)
from sunrisk.events.learned import (ForestPredictor, LearnedDetector, forest_to_dict,  # noqa: E402
                                    make_boosting, make_forest, make_logistic)
from sunrisk.events.rules import RuleDetector  # noqa: E402
from sunrisk.synth.generate import generate_session  # noqa: E402
from sunrisk.synth.scenario import random_scenario  # noqa: E402

PAMAP = ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset"
MODEL_OUT = ROOT / "model" / "models" / "activity_forest.json"
PAMAP_CLASSES = [a for a in ACTIVITIES if a != "swimming"]
IDX = [ACTIVITIES.index(a) for a in PAMAP_CLASSES]


def build_dataset(runs):
    Xml, Xbase, y, groups = [], [], [], []
    for r in runs:
        f = imu_features_ml(r.acc_g, r.gyro_dps)
        if len(f) == 0:
            continue
        Xml.append(f)
        Xbase.append(imu_features(r.acc_g, r.gyro_dps))
        y.append(np.full(len(f), pamap2.activity_code(r.activity)))
        groups.append(np.full(len(f), r.subject))
    return np.vstack(Xml), np.vstack(Xbase), np.concatenate(y), np.concatenate(groups)


def rules_predict(Xbase):
    det = RuleDetector(smooth_windows=1)   # windows here are not one continuous session
    return det.predict_activity(Windows(np.zeros(len(Xbase)), Xbase))


def loso(Xml, Xbase, y, groups):
    preds = {m: np.zeros_like(y) for m in ("rules", "logistic", "forest", "boosting")}
    per_subject = {}
    for s in np.unique(groups):
        test, train = groups == s, groups != s
        preds["rules"][test] = rules_predict(Xbase[test])
        for name, make in (("logistic", make_logistic), ("forest", make_forest), ("boosting", make_boosting)):
            preds[name][test] = make().fit(Xml[train], y[train]).predict(Xml[test])
        per_subject[int(s)] = (int(test.sum()), float(np.mean(preds["forest"][test] == y[test])))
    return preds, per_subject


def sub_cm(y, p):
    return confusion_matrix(y, p, len(ACTIVITIES))[np.ix_(IDX, IDX)]


def macro_f1(cm):
    return float(np.nanmean(per_class_scores(cm)[2]))


def transfer(predictor, days, seed=0):
    rng = np.random.default_rng(seed)
    day = datetime(2024, 7, 15, 5, tzinfo=timezone.utc).timestamp()
    truth, rules_p, learned_p = [], [], []
    for _ in range(days):
        sc = random_scenario(rng, hours=4)
        ss = generate_session(sc, 25.76, -80.19, day, rng)
        w = build_windows(ss.t_unix_s, ss.acc_g, ss.gyro_dps, ss.uv_idx, ss.uv_counts, 18, 20,
                          25.76, -80.19, day)
        truth.append(window_labels(ss.activity))
        rules_p.append(RuleDetector().predict(w)[1])
        learned_p.append(LearnedDetector(predictor).predict_session(ss.acc_g, ss.gyro_dps, w)[1])
    t = np.concatenate(truth)
    return (confusion_matrix(t, np.concatenate(rules_p), len(ACTIVITIES)),
            confusion_matrix(t, np.concatenate(learned_p), len(ACTIVITIES)))


def plot(cms, titles, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 5.4))
    for ax, cm, title in zip(axes, cms, titles):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(PAMAP_CLASSES)):
            for j in range(len(PAMAP_CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(PAMAP_CLASSES)), PAMAP_CLASSES, rotation=40, ha="right")
        ax.set_yticks(range(len(PAMAP_CLASSES)), PAMAP_CLASSES)
        ax.set_title(f"{title}\nmacro-F1 {macro_f1(cm):.2f}, accuracy {np.trace(cm) / cm.sum():.2f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("PAMAP2 wrist IMU, leave-one-subject-out (row-normalised)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-days", type=int, default=20)
    args = ap.parse_args()

    runs = pamap2.load_subjects(PAMAP)
    Xml, Xbase, y, groups = build_dataset(runs)
    print(f"PAMAP2: {len(y)} windows of 5 s from {len(np.unique(groups))} subjects")
    for a in PAMAP_CLASSES:
        print(f"  {a:15s} {np.sum(y == ACTIVITIES.index(a)):5d}")

    preds, per_subject = loso(Xml, Xbase, y, groups)
    print("\n=== Leave-one-subject-out ===")
    for name, p in preds.items():
        cm = sub_cm(y, p)
        print(f"\n[{name}]\n" + format_report(cm, PAMAP_CLASSES))
    print("\nforest accuracy per held-out subject:",
          ", ".join(f"{s}: {acc:.2f} (n={n})" for s, (n, acc) in per_subject.items()))

    final = make_forest().fit(Xml, y)
    imp = sorted(zip(final.feature_importances_, ML_FEATURES), reverse=True)[:8]
    print("\nforest top features:", ", ".join(f"{n} {v:.2f}" for v, n in imp))

    spec = forest_to_dict(final)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(spec, separators=(",", ":")))
    predictor = ForestPredictor(spec)
    agree = np.mean(predictor.predict_activity(Xml) == final.predict(Xml))
    print(f"exported {MODEL_OUT.relative_to(ROOT)} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB); "
          f"numpy predictor agrees with scikit-learn on {agree:.4%} of windows")

    plot([sub_cm(y, preds["rules"]), sub_cm(y, preds["forest"]), sub_cm(y, preds["boosting"])],
         ["Rules baseline", "Random forest", "Gradient boosting"],
         ROOT / "docs" / "figures" / "pamap2_confusion.png")

    if args.synthetic_days:
        cm_r, cm_l = transfer(predictor, args.synthetic_days)
        print(f"\n=== Transfer: PAMAP2-trained forest on {args.synthetic_days} synthetic device sessions ===")
        print("[rules]\n" + format_report(cm_r, ACTIVITIES))
        print("[forest + swim rule]\n" + format_report(cm_l, ACTIVITIES))


if __name__ == "__main__":
    main()
