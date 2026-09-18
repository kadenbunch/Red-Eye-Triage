"""Cross-validation, held-out evaluation and result assembly.

Protocol (manuscript Methods, "Data Partitioning" and "Statistical Analysis"):
  1. A stratified 85/15 train-pool / held-out-test split is created once (seed 42).
  2. Stratified 5-fold CV is run inside the training pool only. Out-of-fold (OOF)
     predictions are pooled across folds and all primary metrics, bootstrap CIs,
     per-class metrics, operating points and figures are computed from them.
  3. Each architecture is then refit on the full training pool, the checkpoint is
     saved, and the held-out test set is evaluated exactly once.
Results are written per model so an interrupted run can resume (CLAIM item 41).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch

from . import config as C
from .data import EyeDiseaseDataset, get_transforms, load_dataset, make_or_load_split
from .metrics import (bootstrap_ci, bootstrap_metric_fns, compute_all_metrics, perclass_metrics,
                      threshold_analysis)
from .models import compute_class_weights, get_model, predict_probs, train_model
from .plots import plot_calibration, plot_confusion, plot_roc_pr


def model_done(m):
    res = os.path.join(C.PER_MODEL_DIR, f"{m}_result.json")
    ckpt = os.path.join(C.CKPT_DIR, f"{m}_heldout_refit.pth")
    return os.path.exists(res) and os.path.exists(ckpt)


def prepare_data():
    """Load images, save the manifest, and create/reuse the held-out split."""
    C.ensure_output_dirs()
    C.set_seed()
    data, labels, manifest = load_dataset()
    manifest.to_csv(os.path.join(C.OUTPUT_DIR, "data_manifest.csv"), index=False)
    print(f"Loaded {len(data)} images across {len(C.CLASSES)} classes.")
    print(manifest["class"].value_counts().to_string())
    idx_train, idx_test = make_or_load_split(labels, manifest)
    print(f"FAST_MODE={C.FAST_MODE} | N_FOLDS={C.N_FOLDS} | EPOCHS={C.EPOCHS} | "
          f"train pool={len(idx_train)} | held-out test={len(idx_test)}")
    return data, labels, manifest, idx_train, idx_test


def run_cross_validation(data, labels, model_name):
    train_tf, eval_tf = get_transforms()
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    oof_true, oof_prob, oof_idx, fold_metrics, cfg = [], [], [], [], None
    for fold, (tr, va) in enumerate(skf.split(data, labels), 1):
        C.set_seed(C.SEED + fold)
        tr_ld = DataLoader(EyeDiseaseDataset(data[tr], labels[tr], train_tf),
                           batch_size=C.BATCH_SIZE, shuffle=True)
        va_ld = DataLoader(EyeDiseaseDataset(data[va], labels[va], eval_tf),
                           batch_size=C.BATCH_SIZE, shuffle=False)
        cv_model, cfg = get_model(model_name, len(C.CLASSES))
        cw = compute_class_weights(labels[tr], len(C.CLASSES))
        cv_model, _ = train_model(cv_model, tr_ld, cw)
        yt, yp = predict_probs(cv_model, va_ld)
        oof_true.append(yt)
        oof_prob.append(yp)
        oof_idx.append(va)
        fold_metrics.append(compute_all_metrics(yt, yp, len(C.CLASSES)))
        print(f"  [{model_name}] fold {fold} acc={fold_metrics[-1]['accuracy']:.3f} "
              f"f1w={fold_metrics[-1]['f1_weighted']:.3f}")
    return (np.concatenate(oof_true), np.concatenate(oof_prob), np.concatenate(oof_idx),
            fold_metrics, cfg)


def run_one_model(m, data, labels, idx_train, idx_test, force=False):
    if model_done(m) and not force:
        print(f"[{m}] already done - skipping (force=True to redo).")
        return
    print(f"\n===== {m.upper()} =====")
    d_tr, l_tr = data[idx_train], labels[idx_train]
    d_te, l_te = data[idx_test], labels[idx_test]
    model_out = os.path.join(C.OUTPUT_DIR, "figures", m)

    # ---- primary analysis: CV inside the training pool ----
    oof_t, oof_p, oof_i, fold_metrics, cfg = run_cross_validation(d_tr, l_tr, m)
    np.savez(os.path.join(C.PER_MODEL_DIR, f"{m}_oof_predictions.npz"),
             y_true=oof_t, y_prob=oof_p, train_pool_index=oof_i,
             dataset_index=np.asarray(idx_train)[oof_i])
    cfg.update({"epochs": C.EPOCHS, "batch_size": C.BATCH_SIZE, "optimizer": "Adam",
                "learning_rate": C.LR, "loss": "class-weighted cross-entropy",
                "augmentation": "hflip(p=0.5)+rotation(15deg)+colorjitter(0.15,0.15,0.10)",
                "seed": C.SEED, "cv_folds": C.N_FOLDS, "bootstrap_resamples": C.N_BOOTSTRAP})

    overall = {"model": m}
    boot = {}
    for name, fn in bootstrap_metric_fns(len(C.CLASSES)).items():
        pt, lo, hi, reps = bootstrap_ci(oof_t, oof_p, len(C.CLASSES), fn)
        overall[name] = f"{pt:.3f} ({lo:.3f}-{hi:.3f})"
        overall[name + "_point"] = pt
        overall[name + "_ci_low"] = lo
        overall[name + "_ci_high"] = hi
        boot[name] = reps.tolist()

    pc = perclass_metrics(oof_t, oof_p)
    pc.insert(0, "model", m)
    th = threshold_analysis(oof_t, oof_p)
    th.insert(0, "model", m)

    plot_roc_pr(oof_t, oof_p, m, output_dir=model_out)
    plot_confusion(oof_t, oof_p, m, output_dir=model_out)
    ece, brier = plot_calibration(oof_t, oof_p, m, output_dir=model_out)
    overall["ece"] = round(ece, 4)
    overall["brier"] = round(brier, 4)

    # ---- secondary: refit on full training pool, single held-out evaluation ----
    train_tf, eval_tf = get_transforms()
    C.set_seed()
    full_ld = DataLoader(EyeDiseaseDataset(d_tr, l_tr, train_tf), batch_size=C.BATCH_SIZE, shuffle=True)
    test_ld = DataLoader(EyeDiseaseDataset(d_te, l_te, eval_tf), batch_size=C.BATCH_SIZE, shuffle=False)
    refit, _ = get_model(m, len(C.CLASSES))
    refit, history = train_model(refit, full_ld, compute_class_weights(l_tr, len(C.CLASSES)))
    ckpt_path = os.path.join(C.CKPT_DIR, f"{m}_heldout_refit.pth")
    torch.save(refit.state_dict(), ckpt_path)
    print(f"  saved checkpoint: {ckpt_path}")

    yt, yp = predict_probs(refit, test_ld)
    np.savez(os.path.join(C.PER_MODEL_DIR, f"{m}_heldout_predictions.npz"),
             y_true=yt, y_prob=yp, dataset_index=np.asarray(idx_test))
    test_m = compute_all_metrics(yt, yp, len(C.CLASSES))
    overall["heldout_test_f1_weighted"] = round(test_m["f1_weighted"], 3)
    overall["heldout_test_auroc_macro"] = round(test_m["auroc_macro_ovr"], 3)
    overall["heldout_test_accuracy"] = round(test_m["accuracy"], 3)

    result = {"hyperparameters": cfg, "overall_cv": overall,
              "perclass_cv": pc.to_dict(orient="records"),
              "thresholds_cv": th.to_dict(orient="records"),
              "fold_metrics": fold_metrics, "refit_history": history,
              "heldout_test": test_m, "bootstrap": boot}
    with open(os.path.join(C.PER_MODEL_DIR, f"{m}_result.json"), "w") as f:
        json.dump(result, f, default=float)
    print(f"[{m}] result saved.")


def assemble_tables(models=None):
    """Stitch per-model JSON into the CSV tables used for Tables 1-3 and S1-S2."""
    models = models or C.MODELS_TO_RUN
    rows = {"hyper": [], "overall": [], "perclass": [], "thr": []}
    boot = {}
    missing = []
    for m in models:
        path = os.path.join(C.PER_MODEL_DIR, f"{m}_result.json")
        if not os.path.exists(path):
            missing.append(m)
            continue
        with open(path) as f:
            r = json.load(f)
        rows["hyper"].append(r["hyperparameters"])
        rows["overall"].append(r["overall_cv"])
        rows["perclass"].append(pd.DataFrame(r["perclass_cv"]))
        rows["thr"].append(pd.DataFrame(r["thresholds_cv"]))
        boot[m] = r["bootstrap"]
    if missing:
        raise RuntimeError(f"Missing results for {missing}; run training first.")
    out = C.OUTPUT_DIR
    pd.DataFrame(rows["hyper"]).to_csv(os.path.join(out, "model_hyperparameters.csv"), index=False)
    pd.DataFrame(rows["overall"]).to_csv(os.path.join(out, "table1_overall_metrics_cv.csv"), index=False)
    pd.concat(rows["perclass"]).to_csv(os.path.join(out, "table2_tableS1_perclass_metrics_cv.csv"), index=False)
    pd.concat(rows["thr"]).to_csv(os.path.join(out, "table3_tableS2_threshold_analysis_cv.csv"), index=False)
    with open(os.path.join(out, "metrics_bootstrap_raw.json"), "w") as f:
        json.dump(boot, f)
    print(f"Tables written to {out}")
