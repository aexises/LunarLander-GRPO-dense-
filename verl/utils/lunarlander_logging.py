"""Local logging and plotting helpers for the LunarLander sanity-check path."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def _nested_get(data: dict[str, Any] | None, keys: list[str], default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_float(value: Any):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_matplotlib():
    try:
        temp_cache_dir = tempfile.gettempdir()
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(temp_cache_dir, "codex-matplotlib-cache"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(temp_cache_dir, "codex-xdg-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


class LunarLanderArtifactLogger:
    """Persist structured metrics and render a few high-signal training plots."""

    def __init__(self, output_dir: str | Path, config: dict[str, Any] | None = None, reset_existing: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = self.output_dir / "metrics_history.jsonl"
        self.metrics_csv_path = self.output_dir / "metrics_history.csv"
        self.summary_json_path = self.output_dir / "metrics_summary.json"
        self.report_path = self.output_dir / "run_report.md"
        self.config = config or {}
        self.rows: list[dict[str, Any]] = []
        self._csv_fieldnames: set[str] = {"timestamp", "step"}
        if reset_existing:
            self._reset_outputs()
        self._write_config_snapshot()

    def _reset_outputs(self):
        for path in [
            self.metrics_jsonl_path,
            self.metrics_csv_path,
            self.summary_json_path,
            self.report_path,
            self.output_dir / "run_config_snapshot.json",
            self.output_dir / "audit_traces_train.csv",
            self.output_dir / "audit_traces_train.jsonl",
            self.output_dir / "audit_traces_val.csv",
            self.output_dir / "audit_traces_val.jsonl",
        ]:
            if path.exists():
                path.unlink()

        for pattern in ("*.png",):
            for file_path in self.plots_dir.glob(pattern):
                file_path.unlink()

        dump_dir = Path(_nested_get(self.config, ["eval", "dump_dir"], self.output_dir / "trajectory_dumps"))
        if dump_dir.exists():
            for pattern in ("*.json", "*.png", "*.csv", "*.jsonl"):
                for file_path in dump_dir.glob(pattern):
                    file_path.unlink()

    def log_metrics(self, step: int, data: dict[str, Any]):
        row = {"timestamp": datetime.now().isoformat(), "step": int(step)}
        for key, value in data.items():
            numeric_value = _safe_float(value)
            row[key] = numeric_value if numeric_value is not None else value

        row.update(self._audit_consistency_metrics(step=int(step), current_row=row))

        self.rows.append(row)
        self._csv_fieldnames.update(row.keys())

        with self.metrics_jsonl_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(row) + "\n")

        self._rewrite_csv()
        self.render_plots()
        self.write_run_report()

    def _write_config_snapshot(self):
        if not self.config:
            return
        with (self.output_dir / "run_config_snapshot.json").open("w", encoding="utf-8") as file_obj:
            json.dump(self.config, file_obj, indent=2)

    def _expected_trace_episode_count(self, split: str) -> int:
        if split == "train":
            return int(_nested_get(self.config, ["data", "train_batch_size"], 0)) * int(_nested_get(self.config, ["data", "n_samples"], 1))
        if split == "val":
            return len(_nested_get(self.config, ["eval", "seed_list"], []))
        return 0

    def _audit_trace_window_stats(self, split: str, step: int) -> dict[str, float] | None:
        trace_path = self.output_dir / f"audit_traces_{split}.jsonl"
        if not trace_path.exists():
            return None

        episodes: dict[str, dict[str, Any]] = {}
        with trace_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if int(row.get("global_steps", -1)) != step:
                    continue
                episode_id = row["episode_id"]
                episodes[episode_id] = {
                    "success": bool(row.get("success", False)),
                    "terminated": bool(row.get("terminated", False)),
                    "truncated": bool(row.get("truncated", False)),
                    "crash": bool(row.get("crash", False)),
                }

        if not episodes:
            return None

        values = list(episodes.values())
        num_episodes = len(values)
        expected = self._expected_trace_episode_count(split)
        return {
            f"audit_trace/{split}_num_episodes": float(num_episodes),
            f"audit_trace/{split}_success_rate": sum(1.0 if item["success"] else 0.0 for item in values) / num_episodes,
            f"audit_trace/{split}_terminated_rate": sum(1.0 if item["terminated"] else 0.0 for item in values) / num_episodes,
            f"audit_trace/{split}_truncated_rate": sum(1.0 if item["truncated"] else 0.0 for item in values) / num_episodes,
            f"audit_trace/{split}_crash_rate": sum(1.0 if item["crash"] else 0.0 for item in values) / num_episodes,
            f"audit_trace/{split}_is_full_window": 1.0 if expected > 0 and num_episodes == expected else 0.0,
        }

    def _audit_consistency_metrics(self, step: int, current_row: dict[str, Any]) -> dict[str, Any]:
        metrics = {}
        for split, success_key, crash_key, terminated_key, truncated_key in [
            ("train", "train_reward/success_rate", "train_reward/crash_rate", "train_reward/terminated_rate", "train_reward/truncated_rate"),
            ("val", "val/test_reward/success_rate", "val/test_reward/crash_rate", "val/test_reward/terminated_rate", "val/test_reward/truncated_rate"),
        ]:
            trace_stats = self._audit_trace_window_stats(split=split, step=step)
            if trace_stats is None:
                continue
            metrics.update(trace_stats)
            for source_key, trace_key, suffix in [
                (success_key, f"audit_trace/{split}_success_rate", "success"),
                (crash_key, f"audit_trace/{split}_crash_rate", "crash"),
                (terminated_key, f"audit_trace/{split}_terminated_rate", "terminated"),
                (truncated_key, f"audit_trace/{split}_truncated_rate", "truncated"),
            ]:
                source_value = _safe_float(current_row.get(source_key))
                trace_value = _safe_float(trace_stats.get(trace_key))
                if source_value is None or trace_value is None:
                    continue
                metrics[f"audit_consistency/{split}_{suffix}_abs_diff"] = abs(source_value - trace_value)
                metrics[f"audit_consistency/{split}_{suffix}_match"] = 1.0 if abs(source_value - trace_value) <= 1e-9 else 0.0
        return metrics

    def _rewrite_csv(self):
        fieldnames = sorted(self._csv_fieldnames, key=lambda item: (item not in {"timestamp", "step"}, item))
        with self.metrics_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def _series(self, metric_name: str) -> tuple[list[int], list[float]]:
        xs = []
        ys = []
        for row in self.rows:
            value = _safe_float(row.get(metric_name))
            if value is None:
                continue
            xs.append(int(row["step"]))
            ys.append(value)
        return xs, ys

    def _plot_lines(self, specs: list[tuple[str, str]], title: str, ylabel: str, filename: str):
        plt = _load_matplotlib()
        if plt is None:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for metric_name, label in specs:
            xs, ys = self._series(metric_name)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
            plotted = True

        if not plotted:
            plt.close(fig)
            return

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plots_dir / filename, dpi=160)
        plt.close(fig)

    def _latest_value(self, metric_name: str):
        for row in reversed(self.rows):
            value = _safe_float(row.get(metric_name))
            if value is not None:
                return value
        return None

    def _peak_value(self, metric_name: str):
        values = [value for _, value in [self._series(metric_name)] if False]
        xs, ys = self._series(metric_name)
        if not ys:
            return None
        return max(ys)

    def _final_summary(self) -> dict[str, Any]:
        summary = {
            "environment": _nested_get(self.config, ["actor_rollout_ref", "rollout", "env_name"], "LunarLander-v3"),
            "final_train_success_rate": self._latest_value("train_reward/success_rate"),
            "best_train_success_rate": self._peak("train_reward/success_rate"),
            "final_val_success_rate": self._latest_value("val/test_reward/success_rate"),
            "best_val_success_rate": self._peak("val/test_reward/success_rate"),
            "final_train_total_reward": self._latest_value("train_reward/mean_total_reward"),
            "final_val_total_reward": self._latest_value("val/test_reward/mean_total_reward"),
            "final_train_fuel_proxy": self._latest_value("train_reward/mean_fuel_proxy"),
            "final_val_fuel_proxy": self._latest_value("val/test_reward/mean_fuel_proxy"),
            "final_reward_hacking_warning": self._latest_value("diagnostics/reward_hacking_warning"),
        }
        with self.summary_json_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)
        return summary

    def _peak(self, metric_name: str):
        _xs, ys = self._series(metric_name)
        if not ys:
            return None
        return max(ys)

    def write_run_report(self):
        summary = self._final_summary()
        reward_cfg = self.config.get("reward", {})
        weights = reward_cfg.get("weights", {})
        eval_cfg = self.config.get("eval", {})
        trainer_cfg = self.config.get("trainer", {})
        data_cfg = self.config.get("data", {})
        rollout_cfg = _nested_get(self.config, ["actor_rollout_ref", "rollout"], {})
        actor_cfg = _nested_get(self.config, ["actor_rollout_ref", "actor"], {})
        current_ablation = str(trainer_cfg.get("experiment_name", "unknown"))
        interpretation_lines = []

        final_val_success = summary.get("final_val_success_rate")
        final_train_success = summary.get("final_train_success_rate")
        warning_flag = summary.get("final_reward_hacking_warning")
        if warning_flag and warning_flag > 0:
            interpretation_lines.append(
                "- Reward-hacking warning triggered: shaped reward rose while success stayed flat or fell. Treat this as likely misalignment until ablations confirm otherwise."
            )
        elif final_val_success is not None or final_train_success is not None:
            interpretation_lines.append(
                "- No reward-hacking warning was triggered in the latest logged window, but alignment still needs to be confirmed across the full ablation suite."
            )
        else:
            interpretation_lines.append("- No training metrics have been logged yet.")

        interpretation_lines.append(
            "- LunarLander remains a classical control sanity check only; any positive result here validates reward plumbing, not VLA transfer."
        )

        report = f"""# LunarLander Run Report

## Setup

- Environment: `{rollout_cfg.get("env_name", "LunarLander-v3")}`
- Experiment name: `{current_ablation}`
- Train batch size: `{data_cfg.get("train_batch_size")}`
- Validation batch size: `{data_cfg.get("val_batch_size")}`
- Samples per prompt: `{data_cfg.get("n_samples")}`
- PPO epochs: `{actor_cfg.get("ppo_epochs")}`
- Learning rate: `{_nested_get(self.config, ["actor_rollout_ref", "actor", "optim", "lr"])}`
- Evaluation seeds: `{eval_cfg.get("seed_list", [])}`
- Eval seed hash: `{hashlib.md5(json.dumps(eval_cfg.get("seed_list", []), sort_keys=True).encode("utf-8")).hexdigest()[:12]}`

## Reward Definition

- Total reward: `w_sub * r_sub + w_prog * r_prog + w_smooth * r_smooth + w_final * r_final_terminal_only`
- Weights: `sub={weights.get("sub")}`, `prog={weights.get("prog")}`, `smooth={weights.get("smooth")}`, `final={weights.get("final")}`
- Phase thresholds: `{reward_cfg.get("phase_thresholds", {})}`
- Success thresholds: `{reward_cfg.get("success", {})}`
- Reward phase thresholds: `{reward_cfg.get("phase_thresholds", {})}`

## Ablations

| Ablation | w_sub | w_prog | w_smooth | w_final | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| terminal_only | 0.0 | 0.0 | 0.0 | 1.0 | compare manually / suite |
| terminal_smooth | 0.0 | 0.0 | >0 | 1.0 | compare manually / suite |
| terminal_sub_prog | >0 | >0 | 0.0 | 1.0 | compare manually / suite |
| full | >0 | >0 | >0 | 1.0 | compare manually / suite |
| dense_only | >0 | >0 | >0 | 0.0 | diagnostic only |

## Results

- Final train success rate: `{summary.get("final_train_success_rate")}`
- Best train success rate: `{summary.get("best_train_success_rate")}`
- Final validation success rate: `{summary.get("final_val_success_rate")}`
- Best validation success rate: `{summary.get("best_val_success_rate")}`
- Final train total reward: `{summary.get("final_train_total_reward")}`
- Final validation total reward: `{summary.get("final_val_total_reward")}`
- Final train fuel proxy: `{summary.get("final_train_fuel_proxy")}`
- Final validation fuel proxy: `{summary.get("final_val_fuel_proxy")}`
- Reward overview plot: `plots/reward_overview.png`
- Reward component plot: `plots/reward_components_train.png`
- Task metrics plot: `plots/task_metrics_train.png`
- Alignment diagnostics plot: `plots/alignment_diagnostics.png`
- Trajectory dumps: `trajectory_dumps/`
- Audit traces: `audit_traces_train.csv`, `audit_traces_val.csv`

## Interpretation

{chr(10).join(interpretation_lines)}
"""
        with self.report_path.open("w", encoding="utf-8") as file_obj:
            file_obj.write(report)

    def render_plots(self):
        self._plot_lines(
            specs=[
                ("train_reward/mean_total_reward", "Train Total Reward"),
                ("val/test_reward/mean_total_reward", "Val Total Reward"),
                ("train_verify_score/all", "Train Success Proxy"),
                ("val/test_score/all", "Val Success Proxy"),
            ],
            title="Reward Overview",
            ylabel="Value",
            filename="reward_overview.png",
        )
        self._plot_lines(
            specs=[
                ("train_reward/mean_r_sub", "Subgoal"),
                ("train_reward/mean_r_prog", "Progress"),
                ("train_reward/mean_r_smooth", "Smoothness"),
                ("train_reward/mean_r_final", "Final"),
            ],
            title="Train Reward Components",
            ylabel="Mean Component Value",
            filename="reward_components_train.png",
        )
        self._plot_lines(
            specs=[
                ("train_reward/success_rate", "Success Rate"),
                ("train_reward/crash_rate", "Crash Rate"),
                ("train_reward/mean_num_action_switches", "Action Switches"),
                ("train_reward/mean_episode_length", "Episode Length"),
            ],
            title="Train Task Metrics",
            ylabel="Value",
            filename="task_metrics_train.png",
        )
        self._plot_lines(
            specs=[
                ("train_reward/corr_total_reward_success", "Corr Total Reward / Success"),
                ("train_reward/corr_r_sub_success", "Corr Subgoal / Success"),
                ("train_reward/corr_r_prog_success", "Corr Progress / Success"),
                ("train_reward/corr_r_smooth_success", "Corr Smoothness / Success"),
            ],
            title="Reward Alignment Diagnostics",
            ylabel="Correlation",
            filename="alignment_diagnostics.png",
        )
        self._plot_lines(
            specs=[
                ("train_reward/mean_fuel_proxy", "Train Fuel Proxy"),
                ("val/test_reward/mean_fuel_proxy", "Val Fuel Proxy"),
            ],
            title="Fuel Proxy",
            ylabel="Value",
            filename="fuel_proxy.png",
        )


def write_trajectory_plot(payload: dict[str, Any], output_path: str | Path):
    plt = _load_matplotlib()
    if plt is None:
        return

    steps = payload.get("steps", [])
    if not steps:
        return

    step_ids = [item["step"] for item in steps]
    xs = [item["next_obs"][0] for item in steps]
    ys = [item["next_obs"][1] for item in steps]
    vx = [item["next_obs"][2] for item in steps]
    vy = [item["next_obs"][3] for item in steps]
    theta = [item["next_obs"][4] for item in steps]
    phases = [item["phase"] for item in steps]
    r_total = [item["reward"]["r_total"] for item in steps]
    r_sub = [item["reward"]["r_sub"] for item in steps]
    r_prog = [item["reward"]["r_prog"] for item in steps]
    r_smooth = [item["reward"]["r_smooth"] for item in steps]
    unique_phases = list(dict.fromkeys(phases))
    phase_to_idx = {label: idx for idx, label in enumerate(unique_phases)}
    phase_idx = [phase_to_idx[label] for label in phases]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(xs, ys, marker="o", linewidth=1.2, markersize=2)
    axes[0, 0].set_title("Trajectory in State Space")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(step_ids, vx, label="vx")
    axes[0, 1].plot(step_ids, vy, label="vy")
    axes[0, 1].plot(step_ids, theta, label="theta")
    axes[0, 1].set_title("Velocity and Angle")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(step_ids, r_total, label="r_total")
    axes[1, 0].plot(step_ids, r_sub, label="r_sub")
    axes[1, 0].plot(step_ids, r_prog, label="r_prog")
    axes[1, 0].plot(step_ids, r_smooth, label="r_smooth")
    axes[1, 0].set_title("Reward Decomposition")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].step(step_ids, phase_idx, where="mid", linewidth=1.2)
    axes[1, 1].scatter(step_ids, phase_idx, c=phase_idx, cmap="viridis", s=12)
    axes[1, 1].set_title("Phase Transition Timeline")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_yticks(list(phase_to_idx.values()))
    axes[1, 1].set_yticklabels(unique_phases)
    axes[1, 1].grid(alpha=0.3)

    outcome = payload.get("final_outcome", {})
    fig.suptitle(
        f"seed={payload.get('seed')} success={outcome.get('success')} crash={outcome.get('crash')} "
        f"length={outcome.get('episode_length')}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
