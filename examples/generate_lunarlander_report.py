#!/usr/bin/env python3
"""Aggregate LunarLander ablation runs into a single markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def summarize_run(run_dir: Path):
    config = load_json(run_dir / "run_config_snapshot.json")
    summary = load_json(run_dir / "metrics_summary.json")
    reward = config.get("reward", {})
    weights = reward.get("weights", {})
    trainer = config.get("trainer", {})
    eval_cfg = config.get("eval", {})
    rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
    return {
        "name": trainer.get("experiment_name", run_dir.name),
        "dir": run_dir,
        "weights": weights,
        "summary": summary,
        "eval_seeds": eval_cfg.get("seed_list", []),
        "reward": reward,
        "env_name": rollout_cfg.get("env_name", "LunarLander-v3"),
    }


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(runs: list[dict], output_path: Path):
    if not runs:
        raise ValueError("No run directories provided")

    representative = runs[0]
    reward_cfg = representative["reward"]
    env_name = representative["summary"].get("environment") or representative.get("env_name") or "LunarLander-v3"

    lines = [
        "# LunarLander Reward Sanity Check Report",
        "",
        "## Setup",
        "",
        f"- Environment: `{env_name}`",
        f"- Seed list: `{representative['eval_seeds']}`",
        f"- Phase thresholds: `{reward_cfg.get('phase_thresholds', {})}`",
        f"- Success thresholds: `{reward_cfg.get('success', {})}`",
        "",
        "## Reward Definition",
        "",
        "- Total reward: `w_sub * r_sub + w_prog * r_prog + w_smooth * r_smooth + w_final * r_final_terminal_only`",
        "- Subgoal reward: phase-dependent proximity shaping",
        "- Progress reward: one-time phase transition bonus",
        "- Smoothness reward: vector-L2 action-switch penalty by default",
        "- Final reward: binary landing success at episode end only",
        "",
        "## Ablations",
        "",
        "| Run | w_sub | w_prog | w_smooth | w_final | Final Val Success | Best Val Success | Final Val Reward | Final Fuel Proxy | Warning |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for run in runs:
        weights = run["weights"]
        summary = run["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    format_value(weights.get("sub")),
                    format_value(weights.get("prog")),
                    format_value(weights.get("smooth")),
                    format_value(weights.get("final")),
                    format_value(summary.get("final_val_success_rate")),
                    format_value(summary.get("best_val_success_rate")),
                    format_value(summary.get("final_val_total_reward")),
                    format_value(summary.get("final_val_fuel_proxy")),
                    "misalignment warning" if (summary.get("final_reward_hacking_warning") or 0) > 0 else "ok",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "- Success-rate curves: see each run directory's `plots/task_metrics_train.png` and `plots/reward_overview.png`.",
            "- Shaped-reward curves: see each run directory's `plots/reward_overview.png`.",
            "- Switching behavior: see each run directory's `plots/task_metrics_train.png`.",
            "- Phase progress statistics: inspect `trajectory_dumps/*.json` and `trajectory_dumps/*.png` in each run.",
            "",
            "## Interpretation",
            "",
            "- Compare `final_val_success_rate` and `best_val_success_rate` across the ablation table above.",
            "- If shaped reward rises but success does not, treat that as likely reward misalignment rather than a positive result.",
            "- Smoothness is expected to reduce `mean_num_action_switches`, potentially with a tradeoff in exploration.",
            "- LunarLander remains a classical-control sanity check and does not establish VLA transferability.",
            "",
            "## Run Artifacts",
            "",
        ]
    )

    for run in runs:
        lines.append(f"- `{run['name']}`: `{run['dir']}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="One or more LunarLander run directories")
    parser.add_argument("--output", default="lunarlander_experiment_report.md", help="Output markdown path")
    args = parser.parse_args()

    runs = [summarize_run(Path(path)) for path in args.run_dirs]
    build_report(runs=runs, output_path=Path(args.output))


if __name__ == "__main__":
    main()
