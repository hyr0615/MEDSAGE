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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .stage_token_mapping import assign_token_stages, decode_with_offsets
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .sa_outcome_credit import (
    adjust_vkc_disagreement,
    bounded_stage_allocation,
    build_stage_similarity_matrix,
    compute_local_outcome_credit,
    disagreement_from_similarity,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    @staticmethod
    def _sagrpo_lcs_len(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        if len(a) < len(b):
            short, long = a, b
        else:
            short, long = b, a
        prev = [0] * (len(short) + 1)
        for token_b in long:
            cur = [0]
            for j, token_a in enumerate(short, start=1):
                if token_a == token_b:
                    cur.append(prev[j - 1] + 1)
                else:
                    cur.append(max(prev[j], cur[-1]))
            prev = cur
        return prev[-1]

    @classmethod
    def _sagrpo_rouge_l_similarity(cls, text_a: str, text_b: str) -> float:
        tokens_a = re.findall(r"[A-Za-z0-9_.+-]+", (text_a or "").lower())
        tokens_b = re.findall(r"[A-Za-z0-9_.+-]+", (text_b or "").lower())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        lcs = cls._sagrpo_lcs_len(tokens_a, tokens_b)
        precision = lcs / max(len(tokens_b), 1)
        recall = lcs / max(len(tokens_a), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _sagrpo_extract_stage_text(text: str, stage: str) -> str:
        tag = "VIS" if stage == "V" else {"L": "LOC", "K": "KNO", "C": "CON"}[stage]
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
        if not match or re.search(r"</?[A-Za-z][^>]*>", match.group(1)):
            return ""
        return match.group(1).strip()

    @staticmethod
    def _sagrpo_stage_char_spans(text: str) -> dict[str, list[tuple[int, int]]]:
        spans: dict[str, list[tuple[int, int]]] = {"L": [], "V": [], "K": [], "C": []}
        tag_to_stage = {"LOC": "L", "VIS": "V", "KNO": "K", "CON": "C"}
        for tag, stage in tag_to_stage.items():
            for match in re.finditer(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.IGNORECASE | re.DOTALL):
                if match.group(1).strip() and not re.search(r"</?[A-Za-z][^>]*>", match.group(1)):
                    spans[stage].append((match.start(1), match.end(1)))
        return spans

    def _sagrpo_token_stage_ids(self, token_ids: torch.Tensor, response_mask: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Decode responses and assign stage IDs: 0=outside, 1=L, 2=V, 3=K, 4=C."""
        texts: list[str] = []
        stage_id_tensor = torch.zeros_like(response_mask, dtype=torch.long)
        for row_idx in range(token_ids.shape[0]):
            positions = response_mask[row_idx].bool().nonzero(as_tuple=True)[0]
            ids = token_ids[row_idx, positions].detach().cpu().tolist()
            text, offsets = decode_with_offsets(self.tokenizer, ids)
            texts.append(text)
            stages = assign_token_stages(offsets, self._sagrpo_stage_char_spans(text))
            stage_id_tensor[row_idx, positions] = torch.as_tensor(stages, dtype=torch.long, device=stage_id_tensor.device)
        return texts, stage_id_tensor

    def _apply_stage_aware_grpo(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Apply bounded stage allocation and stage-specific L/C credit assignment."""
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        device = response_mask.device
        scores = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().float().numpy()
        uids = [str(uid) for uid in batch.non_tensor_batch["uid"]]
        # Outcome calibration uses the trajectory advantage before component mixing.
        global_advantages = batch.batch["advantages"].detach().to(dtype=torch.float32)
        global_advantages = (
            (global_advantages * response_mask.to(dtype=torch.float32)).sum(dim=-1)
            / response_mask.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        ).cpu()

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[uid].append(idx)

        uid_to_avg_score = {uid: float(np.mean([scores[i] for i in indices])) for uid, indices in uid_to_indices.items()}
        alpha = float(self.config.algorithm.sa_alpha)
        min_w = float(self.config.algorithm.sa_min_weight)
        max_w = float(self.config.algorithm.sa_max_weight)
        sample_weights = torch.ones(response_mask.shape[0], dtype=torch.float32, device=device)
        for idx, uid in enumerate(uids):
            difficulty = 1.0 - uid_to_avg_score.get(uid, 0.0)
            sample_weights[idx] = max(min_w, min(max_w, 1.0 + alpha * difficulty))

        decoded_texts, stage_ids = self._sagrpo_token_stage_ids(responses, response_mask)
        stage_ids = stage_ids.to(device=device)
        stages = ["L", "V", "K", "C"]

        def group_standardize(values: torch.Tensor) -> torch.Tensor:
            standardized = torch.zeros_like(values, dtype=torch.float32, device=device)
            values = values.to(device=device, dtype=torch.float32)
            for indices in uid_to_indices.values():
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
                group = values[index_tensor]
                std = group.std(unbiased=False)
                if float(std.item()) > 1e-6:
                    standardized[index_tensor] = (group - group.mean()) / (std + 1e-6)
            return standardized

        component_advantages = bool(self.config.algorithm.sa_component_advantages)
        component_mix = float(self.config.algorithm.sa_component_advantage_mix)
        if not 0.0 <= component_mix <= 1.0:
            raise ValueError(f"sa_component_advantage_mix must be in [0, 1], got {component_mix}")
        component_keys = {
            "iou": "sa_reward_iou",
            "accuracy": "sa_reward_accuracy",
            "format": "sa_reward_format",
            "explicit_conclusion": "sa_reward_explicit_conclusion",
            "bbox_count_match": "sa_reward_bbox_count_match",
        }
        component_advantages = component_advantages and all(
            key in batch.batch for key in component_keys.values()
        )
        if component_advantages:
            fmt = batch.batch[component_keys["format"]].to(device=device, dtype=torch.float32)
            bbox_count_match = batch.batch[component_keys["bbox_count_match"]].to(
                device=device, dtype=torch.float32
            )
            explicit_conclusion = batch.batch[component_keys["explicit_conclusion"]].to(
                device=device, dtype=torch.float32
            )
            l_scores = (
                batch.batch[component_keys["iou"]].to(device=device, dtype=torch.float32)
                * fmt
                * bbox_count_match
            )
            c_scores = (
                batch.batch[component_keys["accuracy"]].to(device=device, dtype=torch.float32)
                * fmt
                * explicit_conclusion
            )
            l_advantages = group_standardize(l_scores)
            c_advantages = group_standardize(c_scores)
            advantages = batch.batch["advantages"].clone()
            for row_index in range(advantages.shape[0]):
                l_residual = (
                    (1.0 - component_mix) * advantages[row_index]
                    + component_mix * l_advantages[row_index].to(dtype=advantages.dtype)
                )
                c_residual = (
                    (1.0 - component_mix) * advantages[row_index]
                    + component_mix * c_advantages[row_index].to(dtype=advantages.dtype)
                )
                advantages[row_index] = torch.where(
                    stage_ids[row_index] == 1,
                    l_residual,
                    advantages[row_index],
                )
                advantages[row_index] = torch.where(
                    stage_ids[row_index] == 4,
                    c_residual,
                    advantages[row_index],
                )
            batch.batch["advantages"] = advantages
            metrics["sagrpo/L_component_score_mean"] = float(l_scores.mean().item())
            metrics["sagrpo/C_component_score_mean"] = float(c_scores.mean().item())

        stage_weight_by_uid: dict[str, dict[str, float]] = {}
        stage_weight_by_row: list[dict[str, float]] = [dict() for _ in range(len(uids))]
        stage_divergence_values: dict[str, list[float]] = {stage: [] for stage in stages}
        outcome_credit_values: dict[str, list[float]] = {"V": [], "K": []}
        outcome_credit_fallback: dict[str, list[bool]] = {"V": [], "K": []}
        adjusted_divergence_values: dict[str, list[float]] = {"V": [], "K": []}
        rollout_stage_weight_values: dict[str, list[float]] = {stage: [] for stage in stages}

        gamma = float(self.config.algorithm.sa_stage_gamma)
        stage_mix = float(self.config.algorithm.sa_stage_mix)
        stage_min = float(self.config.algorithm.sa_stage_min_multiplier)
        stage_max = float(self.config.algorithm.sa_stage_max_multiplier)
        use_l_divergence = bool(self.config.algorithm.sa_use_l_divergence)
        outcome_aware = bool(self.config.algorithm.sa_outcome_aware_credit)
        if not 0.0 <= stage_mix <= 1.0:
            raise ValueError(f"sa_stage_mix must be in [0, 1], got {stage_mix}")
        if not 0.0 < stage_min <= 1.0 <= stage_max:
            raise ValueError(
                "SA stage multipliers must satisfy 0 < min <= 1 <= max, "
                f"got min={stage_min}, max={stage_max}"
            )
        if outcome_aware and use_l_divergence:
            raise ValueError("sa_outcome_aware_credit requires sa_use_l_divergence=false so a_L remains 1")

        accuracy_available = "sa_reward_accuracy" in batch.batch
        if accuracy_available:
            accuracy = batch.batch["sa_reward_accuracy"].detach().to(dtype=torch.float32).cpu()
            # Partial answer credit does not count as a successful outcome.
            binary_correctness = torch.isclose(accuracy, torch.ones_like(accuracy), atol=1e-6, rtol=0.0)
        else:
            binary_correctness = torch.zeros(len(uids), dtype=torch.bool)

        if outcome_aware:
            for uid, indices in uid_to_indices.items():
                stage_similarities: dict[str, torch.Tensor] = {}
                stage_validity: dict[str, torch.Tensor] = {}
                divergences: dict[str, float] = {}
                for stage in stages:
                    stage_texts = [self._sagrpo_extract_stage_text(decoded_texts[i], stage) for i in indices]
                    similarity, valid = build_stage_similarity_matrix(
                        stage_texts, self._sagrpo_rouge_l_similarity
                    )
                    stage_similarities[stage] = similarity
                    stage_validity[stage] = valid
                    divergence = disagreement_from_similarity(similarity, valid)
                    divergences[stage] = divergence
                    stage_divergence_values[stage].append(divergence)

                group_correctness = binary_correctness[indices]
                vk_credits = []
                for stage in ("V", "K"):
                    if accuracy_available:
                        credits, fallback = compute_local_outcome_credit(
                            stage_similarities[stage], stage_validity[stage], group_correctness
                        )
                    else:
                        credits = torch.zeros(len(indices), dtype=torch.float32)
                        fallback = torch.ones(len(indices), dtype=torch.bool)
                    vk_credits.append(credits)
                    outcome_credit_values[stage].extend(credits.tolist())
                    outcome_credit_fallback[stage].extend(fallback.tolist())

                local_vk_credit = torch.stack(vk_credits, dim=1)
                base_vkc = torch.tensor(
                    [divergences["V"], divergences["K"], divergences["C"]], dtype=torch.float32
                )
                adjusted_vkc = adjust_vkc_disagreement(
                    base_vkc, local_vk_credit, global_advantages[indices]
                )
                vkc_multipliers = bounded_stage_allocation(
                    adjusted_vkc, gamma, stage_mix, stage_min, stage_max
                )
                adjusted_divergence_values["V"].extend(adjusted_vkc[:, 0].tolist())
                adjusted_divergence_values["K"].extend(adjusted_vkc[:, 1].tolist())
                for local_row, batch_row in enumerate(indices):
                    weights = {
                        "L": 1.0,
                        "V": float(vkc_multipliers[local_row, 0].item()),
                        "K": float(vkc_multipliers[local_row, 1].item()),
                        "C": float(vkc_multipliers[local_row, 2].item()),
                    }
                    stage_weight_by_row[batch_row] = weights
                    for stage in stages:
                        rollout_stage_weight_values[stage].append(weights[stage])
        else:
            # Without outcome credit, all rollouts share the group's stage allocation.
            for uid, indices in uid_to_indices.items():
                divergences = []
                for stage in stages:
                    stage_texts = [self._sagrpo_extract_stage_text(decoded_texts[i], stage) for i in indices]
                    non_empty = [text for text in stage_texts if text]
                    if len(non_empty) < 2:
                        divergence = 0.0
                    else:
                        sims = []
                        for a_idx in range(len(non_empty)):
                            for b_idx in range(a_idx + 1, len(non_empty)):
                                sims.append(self._sagrpo_rouge_l_similarity(non_empty[a_idx], non_empty[b_idx]))
                        divergence = 1.0 - float(np.mean(sims)) if sims else 0.0
                    divergence = max(0.0, min(1.0, divergence))
                    divergences.append(divergence)
                    stage_divergence_values[stage].append(divergence)

                # L is neutral when localization disagreement is disabled.
                active_indices = list(range(len(stages))) if use_l_divergence else [1, 2, 3]
                logits = torch.tensor([divergences[i] for i in active_indices], dtype=torch.float32)
                dynamic = torch.softmax(logits * gamma, dim=0) * len(active_indices)
                blended = 1.0 + stage_mix * (dynamic - 1.0)
                blended = torch.clamp(blended, min=stage_min, max=stage_max)
                # Redistribute residual mass without exceeding the multiplier bounds.
                target_sum = float(len(active_indices))
                for _ in range(len(active_indices) + 1):
                    correction = target_sum - float(blended.sum().item())
                    if abs(correction) < 1e-6:
                        break
                    eligible = blended < stage_max if correction > 0 else blended > stage_min
                    eligible_count = int(eligible.sum().item())
                    if eligible_count == 0:
                        break
                    blended[eligible] += correction / eligible_count
                    blended = torch.clamp(blended, min=stage_min, max=stage_max)

                multipliers = torch.ones(len(stages), dtype=torch.float32)
                multipliers[active_indices] = blended
                stage_weight_by_uid[uid] = {
                    stage: float(weight) for stage, weight in zip(stages, multipliers.tolist())
                }

        token_weights = sample_weights.unsqueeze(-1).expand_as(response_mask).clone()
        for idx, uid in enumerate(uids):
            for stage_idx, stage in enumerate(stages, start=1):
                stage_weight = (
                    stage_weight_by_row[idx][stage] if outcome_aware else stage_weight_by_uid[uid][stage]
                )
                token_weights[idx] = torch.where(
                    stage_ids[idx] == stage_idx,
                    token_weights[idx] * stage_weight,
                    token_weights[idx],
                )
        token_weights = token_weights * response_mask.float()
        token_weights = torch.where(response_mask.bool(), token_weights, torch.ones_like(token_weights))
        batch.batch["sa_token_weights"] = token_weights

        valid_mask = response_mask.bool()
        metrics["sagrpo/enabled"] = 1.0
        metrics["sagrpo/component_advantages"] = float(component_advantages)
        metrics["sagrpo/component_advantage_mix"] = component_mix
        metrics["sagrpo/L_divergence_used"] = float(use_l_divergence)
        metrics["sagrpo/stage_mix"] = stage_mix
        metrics["sagrpo/outcome_aware_credit"] = float(outcome_aware)
        metrics["sagrpo/difficulty_weight_mean"] = float(sample_weights.mean().detach().cpu().item())
        metrics["sagrpo/difficulty_weight_min"] = float(sample_weights.min().detach().cpu().item())
        metrics["sagrpo/difficulty_weight_max"] = float(sample_weights.max().detach().cpu().item())
        metrics["sagrpo/token_weight_mean"] = float(token_weights[valid_mask].mean().detach().cpu().item()) if valid_mask.any() else 0.0
        for stage in stages:
            values = stage_divergence_values[stage]
            metrics[f"sagrpo/{stage}_divergence"] = float(np.mean(values)) if values else 0.0
            stage_weights = (
                rollout_stage_weight_values[stage]
                if outcome_aware
                else [stage_weight_by_uid[uid][stage] for uid in uid_to_indices]
            )
            metrics[f"sagrpo/{stage}_weight"] = float(np.mean(stage_weights)) if stage_weights else 1.0
        if outcome_aware:
            metrics["sagrpo/outcome_accuracy_available"] = float(accuracy_available)
            for stage in ("V", "K"):
                credits = np.asarray(outcome_credit_values[stage], dtype=np.float32)
                fallback = np.asarray(outcome_credit_fallback[stage], dtype=np.float32)
                adjusted = np.asarray(adjusted_divergence_values[stage], dtype=np.float32)
                metrics[f"sagrpo/{stage}_credit_mean"] = float(credits.mean()) if credits.size else 0.0
                metrics[f"sagrpo/{stage}_credit_std"] = float(credits.std()) if credits.size else 0.0
                metrics[f"sagrpo/{stage}_credit_min"] = float(credits.min()) if credits.size else 0.0
                metrics[f"sagrpo/{stage}_credit_max"] = float(credits.max()) if credits.size else 0.0
                metrics[f"sagrpo/{stage}_credit_fallback_rate"] = float(fallback.mean()) if fallback.size else 1.0
                metrics[f"sagrpo/{stage}_adjusted_divergence"] = float(adjusted.mean()) if adjusted.size else 0.0
            for stage in ("V", "K", "C"):
                weights = np.asarray(rollout_stage_weight_values[stage], dtype=np.float32)
                metrics[f"sagrpo/{stage}_weight_std"] = float(weights.std()) if weights.size else 0.0
                metrics[f"sagrpo/{stage}_weight_min"] = float(weights.min()) if weights.size else 1.0
                metrics[f"sagrpo/{stage}_weight_max"] = float(weights.max()) if weights.size else 1.0
                if weights.size:
                    bound_hits = np.isclose(weights, stage_min, atol=1e-6) | np.isclose(
                        weights, stage_max, atol=1e-6
                    )
                    metrics[f"sagrpo/{stage}_bound_hit_rate"] = float(bound_hits.mean())
                else:
                    metrics[f"sagrpo/{stage}_bound_hit_rate"] = 0.0
            vkc_sums = np.asarray(
                [
                    stage_weight_by_row[row]["V"]
                    + stage_weight_by_row[row]["K"]
                    + stage_weight_by_row[row]["C"]
                    for row in range(len(uids))
                ],
                dtype=np.float32,
            )
            sum_errors = np.abs(vkc_sums - 3.0)
            metrics["sagrpo/VKC_weight_sum_max_error"] = float(sum_errors.max()) if sum_errors.size else 0.0
            if sum_errors.size and float(sum_errors.max()) >= 1e-5:
                raise RuntimeError("outcome-aware V/K/C multipliers do not preserve a per-rollout sum of 3")
        if any(not self._sagrpo_extract_stage_text(text, "L") and not self._sagrpo_extract_stage_text(text, "V") and not self._sagrpo_extract_stage_text(text, "K") and not self._sagrpo_extract_stage_text(text, "C") for text in decoded_texts):
            metrics["sagrpo/no_stage_tag_response_count"] = float(sum(
                1 for text in decoded_texts
                if not any(self._sagrpo_extract_stage_text(text, stage) for stage in stages)
            ))
        else:
            metrics["sagrpo/no_stage_tag_response_count"] = 0.0
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        if self.config.algorithm.stage_aware_grpo:
                            for component in (
                                "iou",
                                "accuracy",
                                "format",
                                "explicit_conclusion",
                                "bbox_count_match",
                            ):
                                values = reward_metrics.get(component)
                                if values is not None and len(values) == len(batch):
                                    batch.batch[f"sa_reward_{component}"] = torch.as_tensor(
                                        values,
                                        dtype=torch.float32,
                                        device=batch.batch["responses"].device,
                                    )
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    if self.config.algorithm.stage_aware_grpo:
                        batch = self._apply_stage_aware_grpo(batch, metrics)

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
