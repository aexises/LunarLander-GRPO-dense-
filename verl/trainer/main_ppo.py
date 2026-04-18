# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import statistics
from functools import partial

import numpy as np
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, countdown, multiply, logic
from verl.trainer.ppo.ray_trainer import RayTrainer
from verl.utils.lunarlander_shaped_reward import LunarLanderRewardConfig
import warnings
warnings.filterwarnings("ignore", message="Batch mode enable graph is only supported with num_graph_seeds==1")


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 0


def _masked_step_mean(values: torch.Tensor, finish_step: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    steps = torch.arange(values.shape[1], device=values.device)
    mask = steps.unsqueeze(0) < finish_step.unsqueeze(1)
    if not mask.any():
        return 0.0
    return float(values[mask].float().mean().item())


def _safe_correlation(values: torch.Tensor, targets: torch.Tensor) -> float:
    if values.numel() == 0 or targets.numel() == 0:
        return 0.0
    x = values.float().detach().cpu().numpy()
    y = targets.float().detach().cpu().numpy()
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _build_step_reward_tensors(data: DataProto, distribution_mode: str) -> dict[str, torch.Tensor]:
    response_shape = data.batch["responses"].shape
    if distribution_mode != "last_token_per_step":
        if distribution_mode == "uniform_within_step":
            raise NotImplementedError("uniform_within_step is reserved for a future extension.")
        raise ValueError(f"Unknown reward distribution mode: {distribution_mode}")

    component_names = ["r_sub", "r_prog", "r_smooth", "r_final", "r_total"]
    reward_tensors = {
        name: torch.zeros(response_shape[0], response_shape[1] * response_shape[2], dtype=torch.float32, device=data.batch["responses"].device)
        for name in component_names
    }
    finish_step = data.batch["finish_step"].long()
    step_token_ends = data.batch["step_token_ends"].long()

    for batch_idx in range(response_shape[0]):
        steps = int(finish_step[batch_idx].item())
        for step_idx in range(steps):
            token_idx = int(step_token_ends[batch_idx, step_idx].item())
            for name in component_names:
                reward_tensors[name][batch_idx, token_idx] += float(data.batch[name][batch_idx, step_idx].item())

    reward_tensors["gt_scores"] = reward_tensors["r_final"].clone()
    reward_tensors["all"] = reward_tensors["r_total"].clone()
    return reward_tensors

class RobRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config

    def verify(self, data):
        success_tensor = data.batch.get('success', data.batch['complete']).float()
        completes = success_tensor.tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        if 'r_total' in data.batch.keys():
            finish_step = data.batch['finish_step'].long()
            success = data.batch.get('success', data.batch['complete']).float()
            episode_total = data.batch['r_total'].sum(dim=1)
            reward_metrics.update({
                'mean_total_reward': float(episode_total.mean().item()),
                'mean_r_sub': _masked_step_mean(data.batch['r_sub'], finish_step),
                'mean_r_prog': _masked_step_mean(data.batch['r_prog'], finish_step),
                'mean_r_smooth': _masked_step_mean(data.batch['r_smooth'], finish_step),
                'mean_r_final': _masked_step_mean(data.batch['r_final'], finish_step),
                'success_rate': float(success.mean().item()),
                'crash_rate': float(data.batch.get('crash', torch.zeros_like(success, dtype=torch.bool)).float().mean().item()),
                'mean_episode_length': float(finish_step.float().mean().item()),
                'mean_num_action_switches': float(data.batch.get('num_action_switches', torch.zeros_like(finish_step)).float().mean().item()),
                'mean_abs_final_x': float(data.batch.get('final_x', torch.zeros_like(success)).abs().float().mean().item()),
                'mean_abs_final_vx': float(data.batch.get('final_vx', torch.zeros_like(success)).abs().float().mean().item()),
                'mean_abs_final_vy': float(data.batch.get('final_vy', torch.zeros_like(success)).abs().float().mean().item()),
                'mean_abs_final_theta': float(data.batch.get('final_theta', torch.zeros_like(success)).abs().float().mean().item()),
                'corr_total_reward_success': _safe_correlation(episode_total, success),
                'corr_r_sub_success': _safe_correlation(data.batch['r_sub'].mean(dim=1), success),
                'corr_r_prog_success': _safe_correlation(data.batch['r_prog'].mean(dim=1), success),
                'corr_r_smooth_success': _safe_correlation(data.batch['r_smooth'].mean(dim=1), success),
            })

        return score, reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict = {}
        reward_metrics = {}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))

        reward_mode = self.config.get('reward', {}).get('mode', 'terminal_only')
        if 'r_total' in data.batch.keys() and reward_mode == 'lunarlander_shaped':
            reward_tensor_dict = _build_step_reward_tensors(
                data=data,
                distribution_mode=self.config.reward.get('distribution_mode', 'last_token_per_step'),
            )
            _, verifier_metrics, _, _ = self.verify(data)
            reward_metrics.update(verifier_metrics)
        else:
            verifier_reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32).reshape((reward_tensor.shape[0], -1))
            valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len
            if 'acc' in data.batch:
                verifier_score = data.batch['acc'].cpu().numpy().tolist()
            else:
                verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
                reward_metrics.update(verifier_metrics)
            for i in range(verifier_reward.shape[0]):
                verifier_reward[i, valid_response_length[i] - 1] += verifier_score[i]
            reward_tensor_dict['gt_scores'] = verifier_reward
            reward_tensor_dict['all'] = verifier_reward.clone()

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     raise  ValueError
        #     reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
        #     reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
        #     if self.config.reward_model.rm_coef!=0:
        #         reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if reward_mode == 'lunarlander_shaped' and 'all' in reward_tensor_dict:
            reward_tensor += reward_tensor_dict['all']
            reward_metrics['verifier'] = float(reward_tensor_dict['gt_scores'].sum(dim=1).mean().item())
        elif self.config.verifier.reward_coef!=0:
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    is_lunarlander = config.data.task_suite_name == 'lunarlander'
    if is_lunarlander:
        from omegaconf import open_dict
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.reward = config.reward
            config.actor_rollout_ref.eval = config.eval

    if is_lunarlander:
        tokenizer = _DummyTokenizer()
    else:
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if is_lunarlander:
        from verl.workers.lunarlander_workers import LunarLanderActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    if is_lunarlander:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(LunarLanderActorRolloutRefWorker),
        }
        if config.algorithm.kl_ctrl.kl_coef > 0:
            role_worker_mapping[Role.RefPolicy] = ray.remote(LunarLanderActorRolloutRefWorker)
    else:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(RobActorRolloutRefWorker)
        }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {role: global_pool_id for role in role_worker_mapping.keys()}

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef!=0.:
        if config.reward_model.rm_type == 'normal':
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == 'prime':
            from verl.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RobRewardManager( num_examine=0, config=config) # note: verifier is called both inside reward_fn and outside.

    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager( num_examine=1,config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
