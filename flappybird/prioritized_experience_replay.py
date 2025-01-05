from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_space, act_space,**kwargs):
        super().__init__(buffer_size, obs_space,act_space)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
    def sample(self, batch_size,env):
         pass
    def update_priorities(self, indices, priorities):
        pass
