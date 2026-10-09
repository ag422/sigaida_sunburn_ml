"""Confusion matrices and per-class scores. Pure numpy."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_scores(cm):
    """Returns (precision, recall, f1, support) arrays; NaN where undefined."""
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / cm.sum(axis=0)
        recall = tp / cm.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, cm.sum(axis=1)


def format_report(cm, names):
    p, r, f, s = per_class_scores(cm)
    w = max(len(n) for n in names)
    lines = [f"{'':{w}s}  " + " ".join(f"{n[:6]:>6s}" for n in names) + "   prec  recall    f1  support"]
    for i, n in enumerate(names):
        lines.append(f"{n:{w}s}  " + " ".join(f"{v:6d}" for v in cm[i])
                     + f"  {p[i]:5.2f}  {r[i]:6.2f}  {f[i]:4.2f}  {s[i]:7d}")
    acc = np.trace(cm) / max(cm.sum(), 1)
    macro = np.nanmean(f[s > 0])
    lines.append(f"accuracy {acc:.3f}   macro-F1 {macro:.3f}   (rows = truth, columns = predicted)")
    return "\n".join(lines)
