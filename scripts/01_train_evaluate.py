"""Step 1: stratified 5-fold CV + held-out refit/evaluation for every architecture.

Usage:
    python scripts/01_train_evaluate.py --data-dir data/images --output-dir outputs
    python scripts/01_train_evaluate.py --models efficientnet --force
"""
from _common import apply_env, base_parser

p = base_parser(__doc__)
p.add_argument("--models", nargs="+", default=None, help="Subset of: efficientnet mobilenet resnet18")
p.add_argument("--force", action="store_true", help="Re-run models that already have results")
args = p.parse_args()
apply_env(args)

from redeye_triage import config as C  # noqa: E402
from redeye_triage.pipeline import assemble_tables, prepare_data, run_one_model  # noqa: E402

data, labels, manifest, idx_train, idx_test = prepare_data()
models = args.models or C.MODELS_TO_RUN
for m in models:
    run_one_model(m, data, labels, idx_train, idx_test, force=args.force)
assemble_tables(models)
