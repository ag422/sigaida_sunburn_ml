"""Learned activity classifier (scikit-learn for training, numpy for inference).

Training uses scikit-learn (optional dependency, `pip install .[ml]`). The trained random
forest is exported to plain arrays (`forest_to_dict`) and run by `ForestPredictor`, which needs
only numpy. That makes the same model portable to the TypeScript app (a JSON file and a
tree walk) and keeps scikit-learn out of the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import ACTIVITIES
from .features import ML_FEATURES, Windows, imu_features_ml
from .rules import RuleDetector, _mode_filter


def make_forest(n_estimators=60, max_depth=12, min_samples_leaf=5, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)


def make_boosting(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)


def make_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def forest_to_dict(model) -> dict:
    """Export a fitted RandomForestClassifier to JSON-serialisable arrays."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        value = t.value[:, 0, :]
        value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)   # per-node class probabilities
        trees.append({
            "feature": t.feature.tolist(),
            "threshold": np.round(t.threshold, 6).tolist(),
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "proba": np.round(value, 5).tolist(),
        })
    return {"format": "sunrisk-forest-v1", "features": list(ML_FEATURES),
            "classes": [ACTIVITIES[int(c)] for c in model.classes_], "trees": trees}


@dataclass
class ForestPredictor:
    """numpy-only random-forest inference from `forest_to_dict` output."""
    spec: dict
    _trees: list = field(init=False, repr=False)

    def __post_init__(self):
        if self.spec.get("format") != "sunrisk-forest-v1":
            raise ValueError("unknown model format")
        if list(self.spec["features"]) != list(ML_FEATURES):
            raise ValueError("model was trained on a different feature set")
        self._trees = [{k: np.asarray(v) for k, v in t.items()} for t in self.spec["trees"]]
        self.class_codes = np.array([ACTIVITIES.index(c) for c in self.spec["classes"]])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.class_codes)))
        rows = np.arange(len(X))
        for t in self._trees:
            node = np.zeros(len(X), dtype=int)
            while True:
                leaf = t["left"][node] == -1
                if leaf.all():
                    break
                f = t["feature"][node]
                go_left = X[rows, np.maximum(f, 0)] <= t["threshold"][node]
                nxt = np.where(go_left, t["left"][node], t["right"][node])
                node = np.where(leaf, node, nxt)
            out += t["proba"][node]
        return out / len(self._trees)

    def predict_activity(self, X: np.ndarray) -> np.ndarray:
        return self.class_codes[np.argmax(self.predict_proba(X), axis=1)]


class LearnedDetector:
    """EventDetector: environment from the rules, activity from a trained classifier.

    PAMAP2 has no swimming, so the swimming rule (whole-arm rotation) overrides the classifier.
    Needs raw IMU arrays to compute the ML features, so call `predict_session`.
    """

    def __init__(self, classifier, rules: RuleDetector | None = None, smooth_windows: int = 5):
        self.classifier = classifier
        self.rules = rules or RuleDetector()
        self.smooth_windows = smooth_windows

    def predict_activity_from_imu(self, acc_g, gyro_dps, w: Windows) -> np.ndarray:
        X = imu_features_ml(acc_g, gyro_dps)[: len(w.t_start)]
        act = np.asarray(self.classifier.predict_activity(X))
        swim = ((w.col("acc_mean_norm") < self.rules.swim_mean_acc_g)
                & (w.col("gyro_rms") > self.rules.swim_gyro_dps))
        act = np.where(swim, ACTIVITIES.index("swimming"), act)
        return _mode_filter(act, self.smooth_windows, len(ACTIVITIES))

    def predict_session(self, acc_g, gyro_dps, w: Windows):
        act = self.predict_activity_from_imu(acc_g, gyro_dps, w)
        return self.rules.predict_environment(w, act), act

    def hold_mask(self):
        return self.rules.hold_mask()
