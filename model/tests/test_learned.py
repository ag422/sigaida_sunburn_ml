import numpy as np
import pytest

from sunrisk.contracts import ACTIVITIES
from sunrisk.datasets import pamap2
from sunrisk.events.evaluate import confusion_matrix, per_class_scores
from sunrisk.events.features import ML_FEATURES, WINDOW_N, Windows, imu_features, imu_features_ml
from sunrisk.events.learned import ForestPredictor, LearnedDetector, forest_to_dict


def fake_pamap(n_per=1500, labels=(3, 0, 4, 11, 1)):
    """100 Hz rows: activity id + wrist acc (m/s^2) / gyro (rad/s) / mag columns."""
    rows = []
    for a in labels:
        block = np.zeros((n_per, 10))
        block[:, 0] = a
        block[:, 1] = -9.80665          # x along forearm toward hand: arm hanging
        block[:, 3] = 0.5
        block[:, 5] = 0.1               # gyro y (rad/s)
        block[:, 8] = 20.0              # mag y
        rows.append(block)
    arr = np.vstack(rows)
    arr[10, 2] = np.nan                 # a dropout
    return arr


def test_pamap2_runs_units_mirror_and_labels():
    runs = pamap2.runs_from_array(fake_pamap(), subject=101)
    assert [r.activity for r in runs] == ["standing", "walking", "lying"]   # 0 (transient) and 11 (driving) dropped
    r = runs[0]
    assert len(r.acc_g) == 1500 // 5                                       # 100 -> 20 Hz
    assert r.acc_g[:, 0] == pytest.approx(-1.0)
    assert not np.isnan(r.acc_g).any()
    assert r.mag_ut[:, 1] == pytest.approx(-20.0)                           # right wrist mirrored: y flips
    assert r.gyro_dps[:, 1] == pytest.approx(np.rad2deg(0.1))              # pseudovector: gyro y keeps sign
    left = pamap2.runs_from_array(fake_pamap(), subject=108)[0]             # left-handed: unchanged
    assert left.mag_ut[:, 1] == pytest.approx(20.0)


def test_short_runs_dropped():
    assert pamap2.runs_from_array(fake_pamap(n_per=500), subject=101) == []   # 5 s < 10 s minimum


def test_ml_features_shape_and_sign_invariance():
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 0.1, (WINDOW_N * 4, 3)) + [0, 0.5, 0.8]
    gyro = rng.normal(0, 5, (WINDOW_N * 4, 3))
    f = imu_features_ml(acc, gyro)
    assert f.shape == (4, len(ML_FEATURES))
    flipped = imu_features_ml(acc * [1, -1, 1], gyro * [-1, 1, -1])
    assert np.allclose(f, flipped)       # left vs right wrist gives the same features


def test_f1_counts_never_predicted_class_as_zero():
    cm = confusion_matrix([0, 0, 1, 1], [1, 1, 1, 1], 2)
    _, _, f1, _ = per_class_scores(cm)
    assert f1[0] == 0.0


def test_forest_export_matches_sklearn():
    pytest.importorskip("sklearn")
    from sunrisk.events.learned import make_forest
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, len(ML_FEATURES)))
    y = np.where(X[:, 0] + 0.5 * X[:, 3] > 0, ACTIVITIES.index("walking"), ACTIVITIES.index("sitting"))
    y[X[:, 5] > 1.2] = ACTIVITIES.index("vigorous")
    model = make_forest(n_estimators=15, max_depth=6).fit(X, y)
    pred = ForestPredictor(forest_to_dict(model))
    Xt = rng.normal(size=(300, len(ML_FEATURES)))
    assert np.array_equal(pred.predict_activity(Xt), model.predict(Xt))
    assert np.allclose(pred.predict_proba(Xt), model.predict_proba(Xt), atol=1e-4)


def test_forest_predictor_rejects_wrong_features():
    with pytest.raises(ValueError):
        ForestPredictor({"format": "sunrisk-forest-v1", "features": ["x"], "classes": [], "trees": []})


class _AlwaysSitting:
    def predict_activity(self, X):
        return np.full(len(X), ACTIVITIES.index("sitting"))


def test_learned_detector_swim_rule_overrides_classifier():
    n = WINDOW_N * 6
    t = np.arange(n) / 20
    ph = 2 * np.pi * 0.45 * t
    acc = np.column_stack([np.cos(ph), np.zeros(n), np.sin(ph)]) * 0.3     # gravity averages out
    gyro = np.column_stack([np.zeros(n), np.full(n, 160.0), np.zeros(n)])
    w = Windows(np.zeros(6), imu_features(acc, gyro))
    act = LearnedDetector(_AlwaysSitting()).predict_activity_from_imu(acc, gyro, w)
    assert (act == ACTIVITIES.index("swimming")).all()
