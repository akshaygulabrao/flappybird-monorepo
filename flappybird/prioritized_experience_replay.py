from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (DictReplayBufferSamples,
                                                   DictRolloutBufferSamples,
                                                   ReplayBufferSamples,
                                                   RolloutBufferSamples)
from stable_baselines3.common.vec_env import VecNormalize


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_space, act_space,**kwargs):
        super().__init__(buffer_size, obs_space,act_space,**kwargs)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32) + 1e-6
        self.priority_alpha = 0.6
    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Calculate the priority of each experience
        indices = np.arange(len(self.priorities))

        # Normalize the priorities to follow a uniform distribution
        normalized_priorities = self.priorities ** self.priority_alpha
        normalized_priorities = normalized_priorities / np.sum(normalized_priorities)

        # Sample from the buffer based on the normalized priorities
        sampled_indices = np.random.choice(indices, size=batch_size, replace=False, p=normalized_priorities)

        return self._get_samples(sampled_indices, env)

    def _get_samples(sampled_indices,env):

    def update_priorities(self, indices, priorities):
        pass
