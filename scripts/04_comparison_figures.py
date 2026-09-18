"""Step 4: bootstrap model-comparison figures from outputs/metrics_bootstrap_raw.json.

Usage:
    python scripts/04_comparison_figures.py --output-dir outputs
"""
import os

from _common import apply_env, base_parser

args = base_parser(__doc__).parse_args()
apply_env(args)

from redeye_triage import config as C  # noqa: E402
from redeye_triage.plots import plot_bootstrap_distributions, plot_metric_comparison  # noqa: E402

boot = os.path.join(C.OUTPUT_DIR, "metrics_bootstrap_raw.json")
os.makedirs(C.COMPARISON_DIR, exist_ok=True)
plot_metric_comparison(boot, save_path=os.path.join(C.COMPARISON_DIR, "fig_model_comparison_ci.png"))
for metric in ("auroc_ovr", "auprc_macro"):
    plot_bootstrap_distributions(boot, metric, save_path=os.path.join(C.COMPARISON_DIR, f"fig_{metric}_bootstrap.png"))
print(f"Comparison figures written to {C.COMPARISON_DIR}")
