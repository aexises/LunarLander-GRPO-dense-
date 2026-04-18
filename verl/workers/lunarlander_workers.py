"""Lightweight LunarLander actor-rollout workers.

These workers exist solely to validate step-distributed reward plumbing in the
existing SimpleVLA-RL trainer. LunarLander is a classical control benchmark and
is not intended to be treated as evidence of VLA transfer.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributions import Categorical

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.lunarlander_logging import write_trajectory_plot
from verl.utils.lunarlander_shaped_reward import (
    PHASE_NAMES,
    LunarLanderRewardConfig,
    classify_phase,
    compute_step_reward,
)

FUEL_PROXY_COST = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 1.0,
}


class LunarLanderPolicy(nn.Module):
    def __init__(self, obs_dim: int = 8, hidden_size: int = 128, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class LunarLanderActorRolloutRefWorker(Worker):
    def __init__(self, config, role: str):
        super().__init__()
        self.config = config
        self.role = role
        self._is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = role in ["ref", "actor_rollout_ref"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.distributed.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            torch.distributed.init_process_group(backend=backend)

        if self.world_size != 1:
            raise NotImplementedError("The LunarLander sanity-check worker currently supports a single process only.")

        seed = int(self.config.model.get("seed", 0))
        random.seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        torch.manual_seed(seed + self.rank)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed + self.rank)

    def _make_env(self, seed: int | None):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required for the LunarLander sanity check. "
                "Please install gymnasium[box2d] before running the LunarLander script."
            ) from exc

        env_name = self.config.rollout.get("env_name", "LunarLander-v3")
        env = gym.make(env_name)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def _reward_config(self) -> LunarLanderRewardConfig:
        return LunarLanderRewardConfig.from_config(self.config)

    def _build_scheduler(self):
        total_steps = max(int(self.config.actor.optim.get("total_training_steps", 0)), 1)
        warmup_steps_ratio = float(self.config.actor.optim.get("lr_warmup_steps_ratio", 0.0))
        warmup_steps = int(total_steps * warmup_steps_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(float(step + 1) / float(warmup_steps), 1.0)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _dist_from_observations(self, observations: torch.Tensor, temperature: float = 1.0) -> Categorical:
        logits = self.policy(observations)
        if temperature <= 0:
            temperature = 1.0
        return Categorical(logits=logits / temperature)

    def _mask_from_finish_step(self, finish_step: torch.Tensor, max_steps: int) -> torch.Tensor:
        steps = torch.arange(max_steps, device=finish_step.device)
        return steps.unsqueeze(0) < finish_step.unsqueeze(1)

    def _flatten_valid(self, batch: DataProto):
        observations = batch.batch["observations"].to(self.device)
        actions = batch.batch["responses"].squeeze(-1).to(self.device)
        advantages = batch.batch["advantages"].to(self.device)
        old_log_probs = batch.batch["old_log_probs"].to(self.device)
        finish_step = batch.batch["finish_step"].to(self.device)
        mask = self._mask_from_finish_step(finish_step, observations.size(1))
        return (
            observations[mask],
            actions[mask],
            advantages[mask],
            old_log_probs[mask],
            mask,
        )

    def _compute_log_probs(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist_from_observations(observations, temperature=1.0)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def _write_trajectory_dumps(self, trajectory_payloads: list[dict[str, Any]], global_steps: int):
        dump_limit = int(self.config.get("eval", {}).get("dump_num_trajectories", 0))
        if dump_limit <= 0 or not trajectory_payloads:
            return

        dump_dir = Path(self.config.get("eval", {}).get("dump_dir", "trajectory_dumps"))
        dump_dir.mkdir(parents=True, exist_ok=True)

        success_written = 0
        failure_written = 0
        for payload in trajectory_payloads:
            is_success = bool(payload["final_outcome"]["success"])
            if is_success and success_written >= dump_limit:
                continue
            if (not is_success) and failure_written >= dump_limit:
                continue

            label = "success" if is_success else "failure"
            file_name = f"{label}_seed_{payload['seed']}_step_{global_steps}.json"
            dump_path = dump_dir / file_name
            with dump_path.open("w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, indent=2)
            write_trajectory_plot(payload, dump_path.with_suffix(".png"))

            if is_success:
                success_written += 1
            else:
                failure_written += 1

            if success_written >= dump_limit and failure_written >= dump_limit:
                break

    def _rollout_episode(self, seed: int, do_sample: bool, temperature: float) -> dict[str, Any]:
        env, obs, _info = self._make_env(seed=seed)
        reward_config = self._reward_config()
        max_steps = int(self.config.rollout.get("max_steps", 400))

        observations = []
        actions = []
        old_log_probs = []
        phases = []
        r_sub = []
        r_prog = []
        r_smooth = []
        r_final = []
        r_total = []
        step_dump = []
        fuel_proxy = 0.0

        prev_action = None
        prev_phase = None
        success = False
        crash = False
        terminated = False
        truncated = False

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            dist = self._dist_from_observations(obs_tensor.unsqueeze(0), temperature=temperature)
            if do_sample:
                action_tensor = dist.sample()
            else:
                action_tensor = torch.argmax(dist.logits, dim=-1)
            action = int(action_tensor.item())
            log_prob = float(dist.log_prob(action_tensor).item())

            next_obs, _env_reward, terminated, truncated, info = env.step(action)
            phase = classify_phase(next_obs, reward_config)
            reward_dict = compute_step_reward(
                obs=next_obs,
                prev_obs=obs,
                action=action,
                prev_action=prev_action,
                phase=phase,
                prev_phase=prev_phase,
                terminated=terminated,
                truncated=truncated,
                info=info,
                config=reward_config,
            )

            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            old_log_probs.append(log_prob)
            phases.append(reward_dict["phase"])
            r_sub.append(reward_dict["r_sub"])
            r_prog.append(reward_dict["r_prog"])
            r_smooth.append(reward_dict["r_smooth"])
            r_final.append(reward_dict["r_final"])
            r_total.append(reward_dict["r_total"])
            fuel_proxy += FUEL_PROXY_COST.get(action, 0.0)
            step_dump.append(
                {
                    "step": step,
                    "phase": PHASE_NAMES[reward_dict["phase"]],
                    "obs": np.asarray(obs, dtype=np.float32).tolist(),
                    "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                    "action": action,
                    "reward": {
                        "r_sub": reward_dict["r_sub"],
                        "r_prog": reward_dict["r_prog"],
                        "r_smooth": reward_dict["r_smooth"],
                        "r_final": reward_dict["r_final"],
                        "r_total": reward_dict["r_total"],
                    },
                    "fuel_proxy": fuel_proxy,
                }
            )

            obs = next_obs
            prev_action = action
            prev_phase = phase
            success = reward_dict["success"]
            if terminated or truncated:
                crash = bool(terminated and not success)
                break

        env.close()
        episode_length = len(actions)
        num_action_switches = int(sum(1 for idx in range(1, episode_length) if actions[idx] != actions[idx - 1]))
        final_obs = np.asarray(obs, dtype=np.float32)
        trajectory_payload = {
            "seed": seed,
            "reward_config": asdict(reward_config),
            "note": "LunarLander is used here as a reward-plumbing sanity check, not as a VLA benchmark.",
            "steps": step_dump,
            "final_outcome": {
                "success": bool(success),
                "crash": crash,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "episode_length": episode_length,
                "final_x": float(final_obs[0]),
                "final_vx": float(final_obs[2]),
                "final_vy": float(final_obs[3]),
                "final_theta": float(final_obs[4]),
                "num_action_switches": num_action_switches,
                "fuel_proxy": fuel_proxy,
            },
        }
        return {
            "observations": observations,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "phase": phases,
            "r_sub": r_sub,
            "r_prog": r_prog,
            "r_smooth": r_smooth,
            "r_final": r_final,
            "r_total": r_total,
            "success": bool(success),
            "complete": bool(success),
            "crash": crash,
            "episode_length": episode_length,
            "num_action_switches": num_action_switches,
            "fuel_proxy": fuel_proxy,
            "final_x": float(final_obs[0]),
            "final_vx": float(final_obs[2]),
            "final_vy": float(final_obs[3]),
            "final_theta": float(final_obs[4]),
            "trajectory_dump": json.dumps(trajectory_payload),
            "trajectory_payload": trajectory_payload,
        }

    def _pad_episodes(self, episodes: list[dict[str, Any]]) -> DataProto:
        batch_size = len(episodes)
        max_steps = max(max(ep["episode_length"], 1) for ep in episodes)
        obs_dim = int(self.config.rollout.get("observation_dim", 8))

        responses = torch.zeros((batch_size, max_steps, 1), dtype=torch.long)
        input_ids = torch.zeros((batch_size, max_steps, 1), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_steps, 1), dtype=torch.long)
        pixel_values = torch.zeros((batch_size, max_steps, obs_dim), dtype=torch.float32)
        observations = torch.zeros((batch_size, max_steps, obs_dim), dtype=torch.float32)
        old_log_probs = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        phase = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        r_sub = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        r_prog = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        r_smooth = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        r_final = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        r_total = torch.zeros((batch_size, max_steps), dtype=torch.float32)
        step_token_ends = torch.arange(max_steps, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        finish_step = torch.zeros(batch_size, dtype=torch.long)
        complete = torch.zeros(batch_size, dtype=torch.bool)
        success = torch.zeros(batch_size, dtype=torch.bool)
        crash = torch.zeros(batch_size, dtype=torch.bool)
        episode_length = torch.zeros(batch_size, dtype=torch.long)
        num_action_switches = torch.zeros(batch_size, dtype=torch.long)
        fuel_proxy = torch.zeros(batch_size, dtype=torch.float32)
        final_x = torch.zeros(batch_size, dtype=torch.float32)
        final_vx = torch.zeros(batch_size, dtype=torch.float32)
        final_vy = torch.zeros(batch_size, dtype=torch.float32)
        final_theta = torch.zeros(batch_size, dtype=torch.float32)
        trajectory_dumps = []

        for batch_idx, episode in enumerate(episodes):
            steps = episode["episode_length"]
            finish_step[batch_idx] = steps
            complete[batch_idx] = episode["complete"]
            success[batch_idx] = episode["success"]
            crash[batch_idx] = episode["crash"]
            episode_length[batch_idx] = steps
            num_action_switches[batch_idx] = episode["num_action_switches"]
            fuel_proxy[batch_idx] = episode["fuel_proxy"]
            final_x[batch_idx] = episode["final_x"]
            final_vx[batch_idx] = episode["final_vx"]
            final_vy[batch_idx] = episode["final_vy"]
            final_theta[batch_idx] = episode["final_theta"]
            trajectory_dumps.append(episode["trajectory_dump"])

            if steps == 0:
                continue

            obs_tensor = torch.tensor(np.asarray(episode["observations"], dtype=np.float32))
            action_tensor = torch.tensor(np.asarray(episode["actions"], dtype=np.int64)).unsqueeze(-1)
            log_prob_tensor = torch.tensor(np.asarray(episode["old_log_probs"], dtype=np.float32))
            phase_tensor = torch.tensor(np.asarray(episode["phase"], dtype=np.int64))
            r_sub_tensor = torch.tensor(np.asarray(episode["r_sub"], dtype=np.float32))
            r_prog_tensor = torch.tensor(np.asarray(episode["r_prog"], dtype=np.float32))
            r_smooth_tensor = torch.tensor(np.asarray(episode["r_smooth"], dtype=np.float32))
            r_final_tensor = torch.tensor(np.asarray(episode["r_final"], dtype=np.float32))
            r_total_tensor = torch.tensor(np.asarray(episode["r_total"], dtype=np.float32))

            responses[batch_idx, :steps] = action_tensor
            input_ids[batch_idx, :steps] = action_tensor
            attention_mask[batch_idx, :steps] = 1
            pixel_values[batch_idx, :steps] = obs_tensor
            observations[batch_idx, :steps] = obs_tensor
            old_log_probs[batch_idx, :steps] = log_prob_tensor
            phase[batch_idx, :steps] = phase_tensor
            r_sub[batch_idx, :steps] = r_sub_tensor
            r_prog[batch_idx, :steps] = r_prog_tensor
            r_smooth[batch_idx, :steps] = r_smooth_tensor
            r_final[batch_idx, :steps] = r_final_tensor
            r_total[batch_idx, :steps] = r_total_tensor

        tensors = {
            "responses": responses.to(self.device),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "observations": observations.to(self.device),
            "old_log_probs": old_log_probs.to(self.device),
            "phase": phase.to(self.device),
            "r_sub": r_sub.to(self.device),
            "r_prog": r_prog.to(self.device),
            "r_smooth": r_smooth.to(self.device),
            "r_final": r_final.to(self.device),
            "r_total": r_total.to(self.device),
            "step_token_ends": step_token_ends.to(self.device),
            "finish_step": finish_step.to(self.device),
            "complete": complete.to(self.device),
            "success": success.to(self.device),
            "crash": crash.to(self.device),
            "episode_length": episode_length.to(self.device),
            "num_action_switches": num_action_switches.to(self.device),
            "fuel_proxy": fuel_proxy.to(self.device),
            "final_x": final_x.to(self.device),
            "final_vx": final_vx.to(self.device),
            "final_vy": final_vy.to(self.device),
            "final_theta": final_theta.to(self.device),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors={"trajectory_dump": trajectory_dumps})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        hidden_size = int(self.config.model.get("hidden_size", 128))
        obs_dim = int(self.config.rollout.get("observation_dim", 8))
        action_dim = int(self.config.rollout.get("action_dim", 4))
        self.policy = LunarLanderPolicy(obs_dim=obs_dim, hidden_size=hidden_size, action_dim=action_dim).to(self.device)

        if self._is_actor:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=float(self.config.actor.optim.lr),
                betas=tuple(self.config.actor.optim.get("betas", (0.9, 0.999))),
            )
            self.scheduler = self._build_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        meta_info = prompts.meta_info
        do_sample = bool(meta_info.get("do_sample", self.config.rollout.get("do_sample", True)))
        n_samples = meta_info.get("n_samples", 1)
        temperature = float(meta_info.get("temperature", self.config.rollout.get("temperature", 1.0)))
        if meta_info.get("validate", False):
            do_sample = False

        seeds = prompts.batch["trial_id"].repeat_interleave(n_samples, dim=0).squeeze(-1).tolist()
        episodes = [self._rollout_episode(int(seed), do_sample=do_sample, temperature=temperature) for seed in seeds]

        if meta_info.get("validate", False):
            global_steps = int(meta_info.get("global_steps", 0))
            self._write_trajectory_dumps([episode["trajectory_payload"] for episode in episodes], global_steps)

        return self._pad_episodes(episodes).to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        observations, actions, advantages, old_log_probs, _mask = self._flatten_valid(data)

        if observations.numel() == 0:
            return DataProto(meta_info={"metrics": {"actor/policy_loss": 0.0, "actor/entropy": 0.0, "actor/clipfrac": 0.0, "actor/approx_kl": 0.0, "actor/lr(1e-4)": float(self.optimizer.param_groups[0]["lr"]) * 1e4}})

        clip_low = 1.0 - float(self.config.actor.get("clip_ratio_low", 0.2))
        clip_high = 1.0 + float(self.config.actor.get("clip_ratio_high", 0.28))
        entropy_coeff = float(self.config.actor.get("entropy_coeff", 0.0))
        grad_clip = float(self.config.actor.get("grad_clip", 1.0))
        ppo_epochs = int(self.config.actor.get("ppo_epochs", 1))
        micro_batch_size = min(int(self.config.actor.get("ppo_mini_batch_size", observations.size(0))), observations.size(0))

        loss_values = []
        entropy_values = []
        clipfrac_values = []
        approx_kl_values = []

        for _ in range(ppo_epochs):
            indices = torch.randperm(observations.size(0), device=self.device) if self.config.actor.get("shuffle", True) else torch.arange(observations.size(0), device=self.device)
            for start in range(0, observations.size(0), micro_batch_size):
                mb_idx = indices[start:start + micro_batch_size]
                mb_obs = observations[mb_idx]
                mb_actions = actions[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]

                log_probs, entropy = self._compute_log_probs(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high)
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                loss = policy_loss - entropy_coeff * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
                self.optimizer.step()

                loss_values.append(float(policy_loss.detach().item()))
                entropy_values.append(float(entropy.detach().mean().item()))
                clipfrac_values.append(float((ratio.ne(clipped_ratio)).float().mean().item()))
                approx_kl_values.append(float((mb_old_log_probs - log_probs).mean().abs().item()))

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "actor/policy_loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "actor/entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "actor/clipfrac": float(np.mean(clipfrac_values)) if clipfrac_values else 0.0,
            "actor/approx_kl": float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
            "actor/lr(1e-4)": float(self.optimizer.param_groups[0]["lr"]) * 1e4,
        }
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        assert self._is_actor
        observations, actions, _advantages, _old_log_probs, _mask = self._flatten_valid(data)
        if observations.numel() == 0:
            return DataProto(meta_info={"metrics": {"actor/entropy_eval": 0.0}})
        with torch.no_grad():
            _log_probs, entropy = self._compute_log_probs(observations, actions)
        return DataProto(meta_info={"metrics": {"actor/entropy_eval": float(entropy.mean().item())}})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        observations = data.batch["observations"].to(self.device)
        actions = data.batch["responses"].squeeze(-1).to(self.device)
        finish_step = data.batch["finish_step"].to(self.device)
        mask = self._mask_from_finish_step(finish_step, observations.size(1))
        with torch.no_grad():
            flat_log_probs, _ = self._compute_log_probs(observations[mask], actions[mask])
        ref_log_prob = torch.zeros((observations.size(0), observations.size(1)), dtype=torch.float32, device=self.device)
        ref_log_prob[mask] = flat_log_probs.float()
        return DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob.cpu()})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        if self.rank != 0:
            return
        os.makedirs(local_path, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "config": {
                "model": dict(self.config.model),
                "actor": dict(self.config.actor) if self._is_actor else {},
                "rollout": dict(self.config.rollout),
            },
        }
        torch.save(checkpoint, os.path.join(local_path, "lunarlander_policy.pt"))
