"""Step 2: exact/near-duplicate audit and train/test leakage check (run after step 1).

Usage:
    python scripts/02_duplicate_audit.py --data-dir data/images --output-dir outputs
"""
from _common import apply_env, base_parser

args = base_parser(__doc__).parse_args()
apply_env(args)

from redeye_triage.duplicates import run_duplicate_audit  # noqa: E402

run_duplicate_audit()
