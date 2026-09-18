"""Step 3: Grad-CAM++, Integrated Gradients and nearest-neighbour panels on held-out test images.

Usage:
    python scripts/03_explainability.py --model efficientnet --n-cases 3
    python scripts/03_explainability.py --model efficientnet --case-indices 0 17 42
"""
import os

import numpy as np

from _common import apply_env, base_parser

p = base_parser(__doc__)
p.add_argument("--model", default="efficientnet")
p.add_argument("--n-cases", type=int, default=3, help="Evenly spaced test cases to explain")
p.add_argument("--case-indices", type=int, nargs="*", help="Explicit test-set positions to explain")
args = p.parse_args()
apply_env(args)

from redeye_triage import config as C  # noqa: E402
from redeye_triage.data import EyeDiseaseDataset, get_transforms, load_dataset, make_or_load_split  # noqa: E402
from redeye_triage.explainability import (build_feature_extractor, build_reference_index,  # noqa: E402
                                          explain_case, gradcam_class_grid)
from redeye_triage.models import load_checkpoint  # noqa: E402

split_path = os.path.join(C.CKPT_DIR, "split_indices.npz")
if not os.path.exists(split_path):
    raise SystemExit("No saved split found - run scripts/01_train_evaluate.py first.")

data, labels, manifest = load_dataset()
idx_train, idx_test = make_or_load_split(labels, manifest)
_, eval_tf = get_transforms()
test_ds = EyeDiseaseDataset(data[idx_test], labels[idx_test], eval_tf)

model = load_checkpoint(args.model, os.path.join(C.CKPT_DIR, f"{args.model}_heldout_refit.pth"))
feat = build_feature_extractor(model, args.model)
index, _, kept_imgs, kept_labels = build_reference_index(feat, data[idx_train], labels[idx_train], eval_tf)

os.makedirs(C.EXPLAIN_DIR, exist_ok=True)
gradcam_class_grid(model, args.model, test_ds,
                   save_path=os.path.join(C.EXPLAIN_DIR, f"fig_gradcam_grid_{args.model}.png"))

cases = args.case_indices if args.case_indices else sorted(
    {int(i) for i in np.linspace(0, len(test_ds) - 1, min(args.n_cases, len(test_ds)))})
for ci in cases:
    img_t, lab = test_ds[ci]
    info = explain_case(model, args.model, feat, index, kept_imgs, kept_labels, img_t, true_label=lab,
                        save_path=os.path.join(C.EXPLAIN_DIR, f"fig_explanation_{args.model}_case{ci}.png"))
    print(f"case {ci} ({manifest.iloc[int(idx_test[ci])]['path']}): pred={C.CLASSES[info['pred']]} "
          f"true={C.CLASSES[lab]} IG delta={info['ig_convergence_delta']:.4f}")
print(f"Figures written to {C.EXPLAIN_DIR}")
