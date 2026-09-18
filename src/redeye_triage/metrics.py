"""Evaluation metrics, bootstrap confidence intervals, operating points and calibration.

CLAIM checklist items 30-31. This module has no PyTorch dependency.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config as C


def _binarize(y_true, num_classes):
    yb = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:  # label_binarize returns a single column for 2 classes
        yb = np.hstack([1 - yb, yb])
    return yb


# ------------------------------------------------------------------ #
# Overall metrics
# ------------------------------------------------------------------ #
def compute_all_metrics(y_true, y_prob, num_classes):
    """Full metric suite from a probability matrix; predictions are argmax."""
    y_pred = y_prob.argmax(1)
    y_true_bin = _binarize(y_true, num_classes)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        metrics["auroc_macro_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["auroc_weighted_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        metrics["auroc_macro_ovr"] = np.nan
        metrics["auroc_weighted_ovr"] = np.nan
    try:
        metrics["auprc_macro"] = average_precision_score(y_true_bin, y_prob, average="macro")
    except ValueError:
        metrics["auprc_macro"] = np.nan
    return metrics


def bootstrap_metric_fns(num_classes):
    """Metrics reported with bootstrap CIs in Table 1."""
    return {
        "accuracy": lambda t, p: accuracy_score(t, p.argmax(1)),
        "f1_weighted": lambda t, p: f1_score(t, p.argmax(1), average="weighted", zero_division=0),
        "f1_macro": lambda t, p: f1_score(t, p.argmax(1), average="macro", zero_division=0),
        "balanced_acc": lambda t, p: balanced_accuracy_score(t, p.argmax(1)),
        "auroc_ovr": lambda t, p: roc_auc_score(t, p, multi_class="ovr", average="macro"),
        "auprc_macro": lambda t, p: average_precision_score(_binarize(t, num_classes), p, average="macro"),
        "cohen_kappa": lambda t, p: cohen_kappa_score(t, p.argmax(1)),
    }


def bootstrap_ci(y_true, y_prob, num_classes, metric_fn, n_boot=None, seed=None):
    """Percentile bootstrap 95% CI.

    Resamples (with replacement) in which any class is absent are skipped,
    because one-vs-rest AUROC/AUPRC are undefined without every class.
    Returns (point_estimate, lower, upper, replicate_array).
    """
    n_boot = n_boot or C.N_BOOTSTRAP
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    n = len(y_true)
    try:
        point = metric_fn(y_true, y_prob)
    except ValueError:
        point = np.nan
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < num_classes:
            continue
        try:
            reps.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
    return point, lo, hi, reps


# ------------------------------------------------------------------ #
# Per-class metrics
# ------------------------------------------------------------------ #
def perclass_metrics(y_true, y_prob, classes=None):
    """One-vs-rest precision, recall, F1, AUROC, AUPRC and support per class."""
    classes = classes or C.CLASSES
    y_pred = y_prob.argmax(1)
    y_true_bin = _binarize(y_true, len(classes))
    rows = []
    for i, cname in enumerate(classes):
        p = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f = f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        try:
            auroc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        except ValueError:
            auroc = np.nan
        try:
            auprc = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        except ValueError:
            auprc = np.nan
        rows.append({"class": cname, "precision": p, "recall_sensitivity": r, "f1": f,
                     "auroc": auroc, "auprc": auprc, "support": int((y_true == i).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Operating points
# ------------------------------------------------------------------ #
def _stats_at(y_bin, score, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    se = tp / (tp + fn) if (tp + fn) else np.nan
    sp = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return se, sp, ppv, npv


def threshold_analysis(y_true, y_prob, classes=None, target=0.95):
    """Per-class one-vs-rest operating points under three policies.

    max_f1          threshold maximizing F1 on the precision-recall curve.
    high_sens_0.95  the HIGHEST threshold at which sensitivity >= 0.95, i.e. the
                    most specific operating point that still meets the
                    sensitivity target (screening / rule-out).
    high_spec_0.95  the LOWEST threshold at which specificity >= 0.95, i.e. the
                    most sensitive operating point that still meets the
                    specificity target (confirmatory / rule-in).

    sklearn's roc_curve returns thresholds in decreasing order, so the first
    index meeting the sensitivity target is the highest such threshold and the
    last index meeting the specificity target is the lowest such threshold.
    """
    classes = classes or C.CLASSES
    y_true_bin = _binarize(y_true, len(classes))
    rows = []
    for i, cname in enumerate(classes):
        yb, score = y_true_bin[:, i], y_prob[:, i]

        prec, rec, thr_pr = precision_recall_curve(yb, score)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        thr_f1 = thr_pr[np.nanargmax(f1s[:-1])] if len(thr_pr) else 0.5

        fpr, tpr, thr_roc = roc_curve(yb, score)
        spec = 1 - fpr
        ok_se = np.where(tpr >= target)[0]
        thr_hs = thr_roc[ok_se[0]] if len(ok_se) else thr_roc[np.argmax(tpr)]
        ok_sp = np.where(spec >= target)[0]
        thr_hp = thr_roc[ok_sp[-1]] if len(ok_sp) else thr_roc[np.argmax(spec)]

        for policy, thr in [("max_f1", thr_f1), (f"high_sens_{target}", thr_hs),
                            (f"high_spec_{target}", thr_hp)]:
            se, sp, ppv, npv = _stats_at(yb, score, thr)
            rows.append({"class": cname, "policy": policy, "threshold": float(thr),
                         "sensitivity": se, "specificity": sp, "ppv": ppv, "npv": npv})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Calibration
# ------------------------------------------------------------------ #
def reliability_bins(y_true, y_prob, n_bins=10):
    """Return (mean_confidence_per_bin, accuracy_per_bin, ece) using max-probability confidence."""
    conf = y_prob.max(1)
    acc = (y_prob.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ece = [], [], 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(acc[m].mean())
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return xs, ys, ece


def expected_calibration_error(y_true, y_prob, n_bins=10):
    return reliability_bins(y_true, y_prob, n_bins)[2]


def multiclass_brier(y_true, y_prob):
    """Mean over samples of the summed squared error across classes."""
    yb = np.eye(y_prob.shape[1])[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((y_prob - yb) ** 2, axis=1)))
