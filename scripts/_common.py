"""Shared CLI helpers: set path environment variables BEFORE importing the package."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", help="Folder with one sub-folder per class (default: data/images)")
    p.add_argument("--output-dir", help="Where outputs are written (default: outputs)")
    p.add_argument("--fast", action="store_true", help="Quick debug run with tiny settings (not for results)")
    return p


def apply_env(args):
    if args.data_dir:
        os.environ["REDEYE_DATA_DIR"] = os.path.abspath(args.data_dir)
    if args.output_dir:
        os.environ["REDEYE_OUTPUT_DIR"] = os.path.abspath(args.output_dir)
    if args.fast:
        os.environ["REDEYE_FAST_MODE"] = "1"
