"""Figures: ROC/PR curves, confusion matrices, calibration and model comparison.

No PyTorch dependency. All figures are saved at 300 dpi (TVST minimum for production).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

from . import config as C  # noqa: E402
from .metrics import _binarize, multiclass_brier, reliability_bins  # noqa: E402

DPI = 300
# Okabe-Ito palette: distinguishable for common colour-vision deficiencies.
CVD_SAFE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def _out(name, output_dir=None):
    d = output_dir or C.OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def plot_roc_pr(y_true, y_prob, model_name, classes=None, output_dir=None):
    classes = classes or C.CLASSES
    yb = _binarize(y_true, len(classes))

    plt.figure(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, color=CVD_SAFE[i],
                 label=f"{c} (AUC={roc_auc_score(yb[:, i], y_prob[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_out(f"fig_roc_{model_name}.png", output_dir), dpi=DPI)
    plt.close()

    plt.figure(figsize=(6, 5))
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(yb[:, i], y_prob[:, i])
        plt.plot(rec, prec, color=CVD_SAFE[i],
                 label=f"{c} (AP={average_precision_score(yb[:, i], y_prob[:, i]):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_out(f"fig_pr_{model_name}.png", output_dir), dpi=DPI)
    plt.close()


def plot_confusion(y_true, y_prob, model_name, classes=None, output_dir=None):
    classes = classes or C.CLASSES
    y_pred = y_prob.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        _out(f"confusion_matrix_{model_name}.csv", output_dir))

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Row-normalized")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for r in range(len(classes)):
        for c in range(len(classes)):
            plt.text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]:.2f})", ha="center", va="center",
                     color="white" if cm_norm[r, c] > 0.5 else "black", fontsize=8)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(f"Confusion - {model_name}")
    plt.tight_layout()
    plt.savefig(_out(f"fig_confusion_{model_name}.png", output_dir), dpi=DPI)
    plt.close()
    return cm


def plot_calibration(y_true, y_prob, model_name, n_bins=10, output_dir=None):
    xs, ys, ece = reliability_bins(y_true, y_prob, n_bins)
    brier = multiclass_brier(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(xs, ys, "o-", color=CVD_SAFE[0], label=model_name)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability diagram - {model_name}\nECE={ece:.3f}  Brier={brier:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_out(f"fig_calibration_{model_name}.png", output_dir), dpi=DPI)
    plt.close()
    return ece, brier


def plot_metric_comparison(bootstrap_json, metrics=("auroc_ovr", "auprc_macro", "f1_weighted",
                                                    "balanced_acc"), save_path=None):
    """Forest-style plot: bootstrap median with 2.5-97.5 percentile whiskers per model."""
    with open(bootstrap_json) as f:
        boot = json.load(f)
    models = list(boot.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for mi, m in enumerate(models):
            reps = np.array(boot[m].get(metric, []))
            if len(reps) == 0:
                continue
            med = np.median(reps)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            ax.errorbar(med, mi, xerr=[[med - lo], [hi - med]], fmt="o",
                        color=CVD_SAFE[mi % len(CVD_SAFE)], capsize=4, markersize=7)
            ax.text(med, mi + 0.18, f"{med:.3f}", ha="center", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.upper() for m in models])
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Model comparison - bootstrap median with 95% CI", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distributions(bootstrap_json, metric="auroc_ovr", save_path=None):
    with open(bootstrap_json) as f:
        boot = json.load(f)
    models = [m for m in boot if len(boot[m].get(metric, []))]
    data = [np.array(boot[m][metric]) for m in models]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel(metric)
    ax.set_title(f"Bootstrap distribution of {metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_overlay(oof_by_model, save_path=None, n_bins=10):
    """oof_by_model: dict model_name -> (y_true, y_prob)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for i, (m, (yt, yp)) in enumerate(oof_by_model.items()):
        xs, ys, ece = reliability_bins(yt, yp, n_bins)
        ax.plot(xs, ys, "o-", color=CVD_SAFE[i % len(CVD_SAFE)],
                label=f"{m.upper()} (ECE={ece:.3f}, Brier={multiclass_brier(yt, yp):.3f})")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
