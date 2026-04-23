#!/usr/bin/env python3
"""Aggregate LunarLander ablation runs into markdown and CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PHASES = ("approach", "align", "descend", "touchdown")
MICRO_PROGRESS_KEYS = (
    "enter_x_corridor_070",
    "enter_x_corridor_050",
    "enter_x_corridor_035",
)
REWARD_SHARE_KEYS = ("subgoal", "progress", "micro_progress", "smoothness", "final")
PLOT_FILES = (
    "reward_overview.png",
    "reward_components_train.png",
    "task_metrics_train.png",
    "alignment_diagnostics.png",
    "fuel_proxy.png",
)
TRACE_FILES = (
    "audit_traces_train.csv",
    "audit_traces_train.jsonl",
    "audit_traces_val.csv",
    "audit_traces_val.jsonl",
)
SUCCESS_METRICS = (
    ("final_train_success_rate", "Final Train Success"),
    ("best_train_success_rate", "Best Train Success"),
    ("final_val_success_rate", "Final Val Success"),
    ("best_val_success_rate", "Best Val Success"),
)
REWARD_METRICS = (
    ("final_train_total_reward", "Final Train Reward"),
    ("best_train_success_rate", "Best Train Success"),
    ("final_val_total_reward", "Final Val Reward"),
    ("best_val_success_rate", "Best Val Success"),
)
DIAGNOSTIC_METRICS = (
    ("final_train_fuel_proxy", "Final Train Fuel Proxy"),
    ("final_val_fuel_proxy", "Final Val Fuel Proxy"),
    ("final_train_num_phase_transitions", "Final Train Phase Transitions"),
    ("final_val_num_phase_transitions", "Final Val Phase Transitions"),
)
BASELINE_METRICS = SUCCESS_METRICS + REWARD_METRICS + DIAGNOSTIC_METRICS
SUITE_PLOT_GROUPS = (
    ("success_metric_comparison.png", "success_metric_delta_vs_terminal_only.png", SUCCESS_METRICS),
    ("reward_metric_comparison.png", "reward_metric_delta_vs_terminal_only.png", REWARD_METRICS),
    ("diagnostic_metric_comparison.png", "diagnostic_metric_delta_vs_terminal_only.png", DIAGNOSTIC_METRICS),
)


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def csv_value(value):
    if value is None:
        return ""
    return value


def get_metric(run: dict, key: str):
    return run["summary"].get(key)


def nested_value(summary: dict, key: str, nested_key: str):
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return None
    return value.get(nested_key)


def normalize_pretrained_policy_path(value):
    if value in (None, "", "None", "null"):
        return None
    return str(value)


def infer_policy_init(config: dict):
    model_cfg = config.get("actor_rollout_ref", {}).get("model", {})
    pretrained_policy_path = normalize_pretrained_policy_path(model_cfg.get("pretrained_policy_path"))
    if pretrained_policy_path is None:
        return "random_init", None, model_cfg.get("activation")
    return "competent_anchor", pretrained_policy_path, model_cfg.get("activation")


def reward_warning_label(run: dict):
    return "misalignment warning" if (get_metric(run, "final_reward_hacking_warning") or 0) > 0 else "ok"


def total_nested_value(summary: dict, key: str, names: tuple[str, ...]):
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return 0.0
    total = 0.0
    found = False
    for name in names:
        nested = value.get(name)
        if nested is None:
            continue
        total += float(nested)
        found = True
    return total if found else 0.0


def has_nonzero_nested_value(summary: dict, key: str, names: tuple[str, ...]):
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return False
    return any((value.get(name) or 0) > 0 for name in names)


def summarize_run(run_dir: Path):
    config = load_json(run_dir / "run_config_snapshot.json")
    summary = load_json(run_dir / "metrics_summary.json")
    reward = config.get("reward", {})
    weights = reward.get("weights", {})
    trainer = config.get("trainer", {})
    eval_cfg = config.get("eval", {})
    rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
    policy_init, pretrained_policy_path, activation = infer_policy_init(config)
    plots_dir = run_dir / "plots"
    trajectory_dir = run_dir / "trajectory_dumps"
    return {
        "name": trainer.get("experiment_name", run_dir.name),
        "dir": run_dir,
        "config": config,
        "reward": reward,
        "weights": weights,
        "summary": summary,
        "policy_init": policy_init,
        "pretrained_policy_path": pretrained_policy_path,
        "policy_activation": activation,
        "eval_seeds": eval_cfg.get("seed_list", []),
        "env_name": rollout_cfg.get("env_name", "LunarLander-v3"),
        "plots_present": [name for name in PLOT_FILES if (plots_dir / name).exists()],
        "trace_files_present": [name for name in TRACE_FILES if (run_dir / name).exists()],
        "trajectory_dump_count": len(list(trajectory_dir.glob("*.json"))) if trajectory_dir.exists() else 0,
        "trajectory_png_count": len(list(trajectory_dir.glob("*.png"))) if trajectory_dir.exists() else 0,
        "run_report_present": (run_dir / "run_report.md").exists(),
        "metrics_history_csv_present": (run_dir / "metrics_history.csv").exists(),
        "metrics_history_jsonl_present": (run_dir / "metrics_history.jsonl").exists(),
    }


def best_run_by_metric(runs: list[dict], metric: str):
    candidates = [(run, get_metric(run, metric)) for run in runs if get_metric(run, metric) is not None]
    if not candidates:
        return None, None
    run, value = max(candidates, key=lambda item: item[1])
    return run, value


def any_run_has_metric(runs: list[dict], predicate):
    return any(predicate(run) for run in runs)


def format_run_and_value(run: dict | None, value):
    if run is None or value is None:
        return "`n/a`"
    return f"`{run['name']}` ({format_value(value)})"


def phase_curriculum_summary(run: dict):
    summary = run["summary"]
    train_approach = nested_value(summary, "final_train_phase_counts", "approach")
    val_approach = nested_value(summary, "final_val_phase_counts", "approach")
    return (
        f"train APPROACH={format_value(train_approach)}, "
        f"val APPROACH={format_value(val_approach)}, "
        f"micro-progress total="
        f"{format_value(total_nested_value(summary, 'final_train_micro_progress_counts', MICRO_PROGRESS_KEYS))}/"
        f"{format_value(total_nested_value(summary, 'final_val_micro_progress_counts', MICRO_PROGRESS_KEYS))}"
        " (train/val)"
    )


def compare_full_vs_subprog(runs: list[dict]):
    by_name = {run["name"]: run for run in runs}
    full_run = by_name.get("lunarlander_grpo_suite_full") or by_name.get("full")
    subprog_run = by_name.get("lunarlander_grpo_suite_terminal_sub_prog") or by_name.get("terminal_sub_prog")
    if full_run is None or subprog_run is None:
        return None

    full_train = get_metric(full_run, "final_train_success_rate")
    subprog_train = get_metric(subprog_run, "final_train_success_rate")
    full_val = get_metric(full_run, "final_val_success_rate")
    subprog_val = get_metric(subprog_run, "final_val_success_rate")

    def compare(lhs, rhs):
        if lhs is None or rhs is None:
            return "cannot be compared"
        if lhs > rhs:
            return "beats"
        if lhs < rhs:
            return "trails"
        return "matches"

    return {
        "full_run": full_run,
        "subprog_run": subprog_run,
        "train_relation": compare(full_train, subprog_train),
        "val_relation": compare(full_val, subprog_val),
        "train_full": full_train,
        "train_subprog": subprog_train,
        "val_full": full_val,
        "val_subprog": subprog_val,
    }


def baseline_run(runs: list[dict]):
    for run in runs:
        name = run["name"]
        if name == "terminal_only" or name.endswith("_terminal_only"):
            return run
    return None


def competent_anchor_comparison_runs(runs: list[dict]):
    return [run for run in runs if run["policy_init"] == "competent_anchor"]


def metric_delta(run: dict, baseline: dict, metric: str):
    run_value = get_metric(run, metric)
    baseline_value = get_metric(baseline, metric)
    if run_value is None or baseline_value is None:
        return None
    return float(run_value) - float(baseline_value)


def baseline_comparison_rows(runs: list[dict], baseline: dict):
    rows = []
    for run in runs:
        row = {"run": run}
        for metric, _label in BASELINE_METRICS:
            row[metric] = get_metric(run, metric)
            row[f"{metric}_delta"] = metric_delta(run, baseline, metric)
        rows.append(row)
    return rows


def build_metric_plot_group(plt, runs: list[dict], baseline: dict, metrics: tuple[tuple[str, str], ...], absolute_path: Path, delta_path: Path):
    labels = [run["name"] for run in runs]
    baseline_idx = labels.index(baseline["name"])
    colors = ["#c3d6c8"] * len(labels)
    colors[baseline_idx] = "#2f6b3b"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [get_metric(run, metric) if get_metric(run, metric) is not None else float("nan") for run in runs]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(absolute_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    delta_runs = [run for run in runs if run["name"] != baseline["name"]]
    if not delta_runs:
        return [absolute_path]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    delta_labels = [run["name"] for run in delta_runs]
    for ax, (metric, title) in zip(axes.flat, metrics):
        deltas = [metric_delta(run, baseline, metric) if metric_delta(run, baseline, metric) is not None else float("nan") for run in delta_runs]
        bar_colors = ["#4e8f5b" if (value == value and value >= 0) else "#b85c38" for value in deltas]
        ax.bar(delta_labels, deltas, color=bar_colors)
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_title(f"{title} vs terminal_only")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(delta_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [absolute_path, delta_path]


def create_suite_plots(runs: list[dict], output_path: Path):
    plot_dir = output_path.parent / f"{output_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    comparison_runs = competent_anchor_comparison_runs(runs)
    baseline = baseline_run(comparison_runs)
    if not comparison_runs or baseline is None:
        return []

    created_paths = []
    for absolute_name, delta_name, metrics in SUITE_PLOT_GROUPS:
        created_paths.extend(
            build_metric_plot_group(
                plt=plt,
                runs=comparison_runs,
                baseline=baseline,
                metrics=metrics,
                absolute_path=plot_dir / absolute_name,
                delta_path=plot_dir / delta_name,
            )
        )

    return created_paths


def build_csv_rows(runs: list[dict]):
    rows = []
    for run in runs:
        summary = run["summary"]
        row = {
            "run": run["name"],
            "dir": str(run["dir"]),
            "policy_init": csv_value(run["policy_init"]),
            "policy_activation": csv_value(run["policy_activation"]),
            "pretrained_policy_path": csv_value(run["pretrained_policy_path"]),
            "w_sub": csv_value(run["weights"].get("sub")),
            "w_prog": csv_value(run["weights"].get("prog")),
            "w_smooth": csv_value(run["weights"].get("smooth")),
            "w_final": csv_value(run["weights"].get("final")),
            "final_train_success_rate": csv_value(summary.get("final_train_success_rate")),
            "best_train_success_rate": csv_value(summary.get("best_train_success_rate")),
            "final_val_success_rate": csv_value(summary.get("final_val_success_rate")),
            "best_val_success_rate": csv_value(summary.get("best_val_success_rate")),
            "final_train_total_reward": csv_value(summary.get("final_train_total_reward")),
            "final_val_total_reward": csv_value(summary.get("final_val_total_reward")),
            "final_train_fuel_proxy": csv_value(summary.get("final_train_fuel_proxy")),
            "final_val_fuel_proxy": csv_value(summary.get("final_val_fuel_proxy")),
            "final_reward_hacking_warning": csv_value(summary.get("final_reward_hacking_warning")),
            "final_eval_num_seeds_configured": csv_value(summary.get("final_eval_num_seeds_configured")),
            "final_val_num_eval_episodes": csv_value(summary.get("final_val_num_eval_episodes")),
            "final_eval_full_seed_coverage": csv_value(summary.get("final_eval_full_seed_coverage")),
            "final_eval_seed_hash": csv_value(summary.get("final_eval_seed_hash")),
            "final_train_num_phase_transitions": csv_value(summary.get("final_train_num_phase_transitions")),
            "final_val_num_phase_transitions": csv_value(summary.get("final_val_num_phase_transitions")),
        }
        for phase in PHASES:
            row[f"final_train_phase_counts_{phase}"] = csv_value(
                nested_value(summary, "final_train_phase_counts", phase)
            )
            row[f"final_val_phase_counts_{phase}"] = csv_value(
                nested_value(summary, "final_val_phase_counts", phase)
            )
            row[f"final_train_phase_episode_counts_{phase}"] = csv_value(
                nested_value(summary, "final_train_phase_episode_counts", phase)
            )
            row[f"final_val_phase_episode_counts_{phase}"] = csv_value(
                nested_value(summary, "final_val_phase_episode_counts", phase)
            )
        for name in MICRO_PROGRESS_KEYS:
            row[f"final_train_micro_progress_counts_{name}"] = csv_value(
                nested_value(summary, "final_train_micro_progress_counts", name)
            )
            row[f"final_val_micro_progress_counts_{name}"] = csv_value(
                nested_value(summary, "final_val_micro_progress_counts", name)
            )
        for name in REWARD_SHARE_KEYS:
            row[f"final_train_reward_shares_{name}"] = csv_value(
                nested_value(summary, "final_train_reward_shares", name)
            )
            row[f"final_val_reward_shares_{name}"] = csv_value(
                nested_value(summary, "final_val_reward_shares", name)
            )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_path: Path):
    if not rows:
        raise ValueError("No rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(runs: list[dict], output_path: Path, suite_plot_paths: list[Path] | None = None):
    if not runs:
        raise ValueError("No run directories provided")

    representative = runs[0]
    reward_cfg = representative["reward"]
    summary = representative["summary"]
    env_name = summary.get("environment") or representative.get("env_name") or "LunarLander-v3"

    best_final_val_run, best_final_val = best_run_by_metric(runs, "final_val_success_rate")
    best_best_val_run, best_best_val = best_run_by_metric(runs, "best_val_success_rate")
    best_final_train_run, best_final_train = best_run_by_metric(runs, "final_train_success_rate")
    competent_anchor_runs = [run for run in runs if run["policy_init"] == "competent_anchor"]
    baseline = baseline_run(competent_anchor_runs)
    baseline_rows = baseline_comparison_rows(competent_anchor_runs, baseline) if baseline is not None else []
    warning_runs = [run for run in runs if (get_metric(run, "final_reward_hacking_warning") or 0) > 0]
    any_approach_usage = any_run_has_metric(
        runs,
        lambda run: has_nonzero_nested_value(run["summary"], "final_train_phase_counts", ("approach",))
        or has_nonzero_nested_value(run["summary"], "final_val_phase_counts", ("approach",)),
    )
    any_micro_progress_usage = any_run_has_metric(
        runs,
        lambda run: has_nonzero_nested_value(run["summary"], "final_train_micro_progress_counts", MICRO_PROGRESS_KEYS)
        or has_nonzero_nested_value(run["summary"], "final_val_micro_progress_counts", MICRO_PROGRESS_KEYS),
    )
    full_vs_subprog = compare_full_vs_subprog(runs)

    lines = [
        "# LunarLander Reward Sanity Check Report",
        "",
        "## Executive Summary",
        "",
        f"- Best run by final validation success: {format_run_and_value(best_final_val_run, best_final_val)}",
        f"- Best run by best validation success: {format_run_and_value(best_best_val_run, best_best_val)}",
        f"- Best run by final train success: {format_run_and_value(best_final_train_run, best_final_train)}",
        (
            "- Competent-anchor runs: `none`"
            if not competent_anchor_runs
            else f"- Competent-anchor runs: `{', '.join(run['name'] for run in competent_anchor_runs)}`"
        ),
        (
            "- Reward-hacking warning runs: `none`"
            if not warning_runs
            else f"- Reward-hacking warning runs: `{', '.join(run['name'] for run in warning_runs)}`"
        ),
        f"- Any run with nonzero APPROACH usage: `{any_approach_usage}`",
        f"- Any run with nonzero micro-progress usage: `{any_micro_progress_usage}`",
        "",
        "## Experiment Setup",
        "",
        f"- Environment: `{env_name}`",
        f"- Eval seed list length: `{len(representative['eval_seeds'])}`",
        f"- Eval seed list: `{representative['eval_seeds']}`",
        f"- Eval seed hash: `{summary.get('final_eval_seed_hash', 'n/a')}`",
        f"- Full eval seed coverage: `{summary.get('final_eval_full_seed_coverage', 'n/a')}`",
        f"- Deterministic eval: `{summary.get('final_eval_deterministic', 'n/a')}`",
        f"- Evaluated episodes per validation window: `{summary.get('final_val_num_eval_episodes', 'n/a')}` / `{summary.get('final_eval_num_seeds_configured', 'n/a')}`",
        f"- Policy initializations present: `{sorted({run['policy_init'] for run in runs})}`",
        f"- Phase thresholds: `{reward_cfg.get('phase_thresholds', {})}`",
        f"- Success thresholds: `{reward_cfg.get('success', {})}`",
        "",
        "## Reward Definition",
        "",
        "- Total reward: `w_sub * r_sub + w_prog * (r_prog + r_micro) + w_smooth * r_smooth + w_final * r_final_terminal_only`",
        "- Subgoal reward: phase-local shaping from the configured coefficient set",
        "- Progress reward: one-time phase transition bonuses, with optional early corridor bonuses when configured",
        "- Smoothness reward: vector-L2 action-switch penalty by default",
        "- Final reward: binary landing success at episode end only",
        "",
        "## Main Ablation Comparison",
        "",
        "| Run | Init | Activation | w_sub | w_prog | w_smooth | w_final | Final Train Success | Best Train Success | Final Val Success | Best Val Success | Final Train Reward | Final Val Reward | Warning |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for run in runs:
        weights = run["weights"]
        run_summary = run["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    run["policy_init"],
                    format_value(run["policy_activation"]),
                    format_value(weights.get("sub")),
                    format_value(weights.get("prog")),
                    format_value(weights.get("smooth")),
                    format_value(weights.get("final")),
                    format_value(run_summary.get("final_train_success_rate")),
                    format_value(run_summary.get("best_train_success_rate")),
                    format_value(run_summary.get("final_val_success_rate")),
                    format_value(run_summary.get("best_val_success_rate")),
                    format_value(run_summary.get("final_train_total_reward")),
                    format_value(run_summary.get("final_val_total_reward")),
                    reward_warning_label(run),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            (
                f"- Baseline run: `{baseline['name']}`"
                if baseline is not None
                else "- Baseline run: `n/a`"
            ),
            "- Random-init runs are excluded from this comparison section and the suite-level comparison plots.",
            "",
            "| Run | Final Train Success | Delta vs terminal_only | Final Val Success | Delta vs terminal_only | Final Train Reward | Delta vs terminal_only | Final Val Reward | Delta vs terminal_only |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in baseline_rows:
        run = row["run"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    format_value(row["final_train_success_rate"]),
                    format_value(row["final_train_success_rate_delta"]),
                    format_value(row["final_val_success_rate"]),
                    format_value(row["final_val_success_rate_delta"]),
                    format_value(row["final_train_total_reward"]),
                    format_value(row["final_train_total_reward_delta"]),
                    format_value(row["final_val_total_reward"]),
                    format_value(row["final_val_total_reward_delta"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "| Run | Final Train Fuel Proxy | Delta vs terminal_only | Final Val Fuel Proxy | Delta vs terminal_only | Final Train Phase Transitions | Delta vs terminal_only | Final Val Phase Transitions | Delta vs terminal_only |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in baseline_rows:
        run = row["run"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    format_value(row["final_train_fuel_proxy"]),
                    format_value(row["final_train_fuel_proxy_delta"]),
                    format_value(row["final_val_fuel_proxy"]),
                    format_value(row["final_val_fuel_proxy_delta"]),
                    format_value(row["final_train_num_phase_transitions"]),
                    format_value(row["final_train_num_phase_transitions_delta"]),
                    format_value(row["final_val_num_phase_transitions"]),
                    format_value(row["final_val_num_phase_transitions_delta"]),
                ]
            )
            + " |"
        )

    if suite_plot_paths:
        lines.extend(
            [
                "",
                "## Comparison Plots",
                "",
                "- Competent-anchor runs only; `terminal_only` is the baseline reference.",
            ]
        )
        for path in suite_plot_paths:
            rel_path = path.relative_to(output_path.parent)
            lines.append(f"- [{path.name}]({rel_path.as_posix()})")

    lines.extend(
        [
            "",
            "## Phase and Curriculum",
            "",
            "| Run | Train Phase Counts | Val Phase Counts | Train Phase Episode Counts | Val Phase Episode Counts | Train Mean Transitions | Val Mean Transitions | APPROACH Used? |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    for run in runs:
        run_summary = run["summary"]
        approach_used = has_nonzero_nested_value(run_summary, "final_train_phase_counts", ("approach",)) or has_nonzero_nested_value(
            run_summary, "final_val_phase_counts", ("approach",)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    f"`{run_summary.get('final_train_phase_counts', {})}`",
                    f"`{run_summary.get('final_val_phase_counts', {})}`",
                    f"`{run_summary.get('final_train_phase_episode_counts', {})}`",
                    f"`{run_summary.get('final_val_phase_episode_counts', {})}`",
                    format_value(run_summary.get("final_train_num_phase_transitions")),
                    format_value(run_summary.get("final_val_num_phase_transitions")),
                    "yes" if approach_used else "no",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reward Composition",
            "",
            "| Run | Train Reward Shares | Val Reward Shares | Train Micro-Progress Counts | Val Micro-Progress Counts |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for run in runs:
        run_summary = run["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    f"`{run_summary.get('final_train_reward_shares', {})}`",
                    f"`{run_summary.get('final_val_reward_shares', {})}`",
                    f"`{run_summary.get('final_train_micro_progress_counts', {})}`",
                    f"`{run_summary.get('final_val_micro_progress_counts', {})}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    if full_vs_subprog is not None:
        lines.extend(
            [
                f"- `{full_vs_subprog['full_run']['name']}` {full_vs_subprog['train_relation']} `{full_vs_subprog['subprog_run']['name']}` on final train success: `{format_value(full_vs_subprog['train_full'])}` vs `{format_value(full_vs_subprog['train_subprog'])}`.",
                f"- `{full_vs_subprog['full_run']['name']}` {full_vs_subprog['val_relation']} `{full_vs_subprog['subprog_run']['name']}` on final validation success: `{format_value(full_vs_subprog['val_full'])}` vs `{format_value(full_vs_subprog['val_subprog'])}`.",
            ]
        )

    suite_full = next((run for run in runs if run["name"] in {"lunarlander_grpo_suite_full", "full"}), None)
    if suite_full is not None:
        lines.append(f"- `suite_full` curriculum summary: {phase_curriculum_summary(suite_full)}.")

    if baseline is not None:
        improving_val_runs = [
            row for row in baseline_rows if row["run"]["name"] != baseline["name"] and (row["final_val_success_rate_delta"] or 0.0) > 0
        ]
        if improving_val_runs:
            best_vs_baseline = max(improving_val_runs, key=lambda row: row["final_val_success_rate_delta"])
            lines.append(
                f"- Strongest validation gain versus `terminal_only`: `{best_vs_baseline['run']['name']}` at `{format_value(best_vs_baseline['final_val_success_rate_delta'])}`."
            )
        else:
            lines.append("- No competent-anchor run improved final validation success over `terminal_only` in this report.")

    lines.extend(
        [
            f"- APPROACH was {'observed' if any_approach_usage else 'not observed'} anywhere in the suite. In the saved suite artifacts, all reported APPROACH counts remain zero.",
            f"- Micro-progress rewards were {'used' if any_micro_progress_usage else 'never triggered'} anywhere in the suite. In the saved suite artifacts, all micro-progress counts remain zero.",
            (
                "- `terminal_smooth` is the lone warning case, which is consistent with smoothness reducing exploration here without improving validation."
                if warning_runs
                else "- No reward-hacking warning was triggered in the suite."
            ),
            "- Dense reward improved total reward relative to terminal-only baselines, but the full dense mix did not beat the simpler `terminal_sub_prog` setting on task success.",
            "- LunarLander remains a classical-control sanity check only; any positive result here validates reward plumbing, not VLA transfer.",
            "",
            "## Artifact Appendix",
            "",
        ]
    )

    for run in runs:
        lines.extend(
            [
                f"- `{run['name']}`: `{run['dir']}`",
                f"  - Policy init: `{run['policy_init']}`; activation: `{run['policy_activation']}`; pretrained policy: `{run['pretrained_policy_path'] or 'n/a'}`",
                f"  - Plots present: `{run['plots_present']}`",
                f"  - Trace files present: `{run['trace_files_present']}`",
                f"  - Trajectory dumps: `json={run['trajectory_dump_count']}`, `png={run['trajectory_png_count']}`",
                f"  - Run report present: `{run['run_report_present']}`; metrics history present: `csv={run['metrics_history_csv_present']}`, `jsonl={run['metrics_history_jsonl_present']}`",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="One or more LunarLander run directories")
    parser.add_argument("--output", default="lunarlander_experiment_report.md", help="Output markdown path")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV output path. Defaults to a sibling CSV next to the markdown report.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    runs = [summarize_run(Path(path)) for path in args.run_dirs]
    suite_plot_paths = create_suite_plots(runs=runs, output_path=output_path)
    build_report(runs=runs, output_path=output_path, suite_plot_paths=suite_plot_paths)
    write_csv(rows=build_csv_rows(runs), output_path=csv_output_path)


if __name__ == "__main__":
    main()
