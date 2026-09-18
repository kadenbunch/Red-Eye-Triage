"""Fast tests that need no GPU, no PyTorch, no images and no API key."""
import numpy as np
import pytest

from redeye_triage import config as C
from redeye_triage import llm_grounding as L
from redeye_triage.metrics import (bootstrap_ci, bootstrap_metric_fns, compute_all_metrics,
                                   expected_calibration_error, multiclass_brier, perclass_metrics,
                                   threshold_analysis)


@pytest.fixture
def synthetic_predictions():
    rng = np.random.default_rng(0)
    n, k = 400, len(C.CLASSES)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 3.0
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, p


def test_metrics_ranges(synthetic_predictions):
    y, p = synthetic_predictions
    m = compute_all_metrics(y, p, len(C.CLASSES))
    for key in ("accuracy", "f1_weighted", "auroc_macro_ovr", "auprc_macro"):
        assert 0.0 <= m[key] <= 1.0
    assert perclass_metrics(y, p)["support"].sum() == len(y)
    assert 0 <= expected_calibration_error(y, p) <= 1
    assert 0 <= multiclass_brier(y, p) <= 2


def test_bootstrap_ci_brackets_point(synthetic_predictions):
    y, p = synthetic_predictions
    fn = bootstrap_metric_fns(len(C.CLASSES))["accuracy"]
    point, lo, hi, reps = bootstrap_ci(y, p, len(C.CLASSES), fn, n_boot=200)
    assert lo <= point <= hi and len(reps) > 0


def test_threshold_policies_meet_targets(synthetic_predictions):
    y, p = synthetic_predictions
    th = threshold_analysis(y, p)
    assert len(th) == 3 * len(C.CLASSES)
    assert (th[th.policy == "high_sens_0.95"].sensitivity >= 0.95).all()
    assert (th[th.policy == "high_spec_0.95"].specificity >= 0.95).all()


def test_knowledge_base_loads_and_covers_every_class():
    kb = L.load_knowledge_base()
    assert {it["class"] for it in kb} >= set(C.CLASSES)
    for cls in C.CLASSES:
        system, user, items = L.build_grounded_prompt(cls, kb, 0.9)
        assert items and all(f"[{it['id']}]" in user for it in items)


def test_audit_detects_fabricated_and_uncited():
    kb = L.load_knowledge_base()
    _, _, items = L.build_grounded_prompt("Inflammatory", kb)
    allowed = [it["id"] for it in items]
    good = f"## Red flags\nVision loss requires same-day review [{allowed[0]}]."
    assert L.audit_grounding(good, allowed)["grounding_pass"]
    bad = "## Red flags\nSevere pain needs referral.\nPhotophobia is concerning [FAKE-99]."
    audit = L.audit_grounding(bad, allowed)
    assert audit["fabricated_refs"] == ["FAKE-99"]
    assert audit["n_uncited_clinical_lines"] == 1
    assert not audit["grounding_pass"]


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        L.build_grounded_prompt("Uveitis", L.load_knowledge_base())
