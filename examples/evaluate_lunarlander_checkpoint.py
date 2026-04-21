#!/usr/bin/env python3
"""Evaluate a saved LunarLander checkpoint from a run directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.lunarlander_checkpoint_eval import evaluate_run_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="LunarLander run directory containing metrics/history and actor checkpoints")
    parser.add_argument(
        "--mode",
        choices=("best_saved", "latest_saved"),
        default="best_saved",
        help="Which saved checkpoint to evaluate when --step is not provided.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Explicit checkpoint step to evaluate, for example 149 for actor/global_step_149.",
    )
    parser.add_argument(
        "--metric-key",
        default="val/test_reward/success_rate",
        help="Logged metric used to pick the best saved checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. The JSON is always printed to stdout.",
    )
    args = parser.parse_args()

    result = evaluate_run_checkpoint(
        run_dir=Path(args.run_dir),
        mode=args.mode,
        metric_key=args.metric_key,
        explicit_step=args.step,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
