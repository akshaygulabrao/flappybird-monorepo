"""
Q-Learning agent for flappy bird. Uses simplified observation space of 12 features.

Uses stable-baseline3 DQN implementation. Uses wandb for logging. Used for benchmarking when model
improvement stops. Running 10 million episodes, and storing evaluation metrics every million
episodes.
"""

import argparse
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import flappy_bird_env
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import yaml
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EveryNTimesteps)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import (DictReplayBufferSamples,
                                                   DictRolloutBufferSamples,
                                                   ReplayBufferSamples,
                                                   RolloutBufferSamples)
from stable_baselines3.common.vec_env import VecNormalize

assert flappy_bird_env is not None, "flappy_bird_env is not installed"


class ScoreCallback:
    def __init__(self):
        self.scores = []

    def __call__(self, locals_dict, globals_dict):
        if locals_dict["dones"][0]:
            self.scores.append(locals_dict["infos"][0]["score"])
        return True


class EvalPolicyCallback(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
    def _on_step(self) -> bool:
        self.score_callback = ScoreCallback()
        evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=100,
            deterministic=True,
            callback=self.score_callback,
        )
        scores = self.score_callback.scores
        self.logger.record("eval/mean_score", np.mean(scores))
        self.logger.record("eval/std_score", np.std(scores))
        self.logger.dump(step=self.num_timesteps)
        return True


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, alpha=0.6, beta=0.4,observation_space=spaces.Box(0,1,(12,)),action_space=spaces.Discrete(2),**kwargs):
        super().__init__(buffer_size,observation_space=observation_space,action_space=action_space,**kwargs)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0  # Initial priority for new experiences

    def add(        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        ):
        super().add(obs,next_obs,action,reward, done,infos)
        self.priorities[self.pos] = self.max_priority  # Assign max priority to new experiences
        # Update position and check if buffer is full
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    def sample(self, batch_size,env):
        # Compute sampling probabilities
        return super().sample(batch_size,env)

    def update_priorities(self, indices, priorities):
        # Update priorities for sampled experiences
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

def main():
    parser = argparse.ArgumentParser(description="DQN configuration")
    parser.add_argument("config", help="path to config file (yaml)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    env = gym.make(**config["env"])
    env = Monitor(env, "logs/")

    if config["checkpoint_path"] is None:
        model = sb3.DQN(**config["model"], replay_buffer_class=PrioritizedReplayBuffer, env=env)
    else:
        checkpoint_path = f"data/{config['name']}.zip"
        if os.path.exists(checkpoint_path):
            model = sb3.DQN.load(checkpoint_path, env=env)
        else:
            print("Checkpoint file not found")
            exit(1)

    event_callback = EveryNTimesteps(n_steps=100_000, callback=EvalPolicyCallback(env))
    checkpoint_callback = CheckpointCallback(**config["checkpoint"])

    model.learn(
        **config["training"],
        callback=[event_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(f"data/{config['name']}.zip")


if __name__ == "__main__":
    main()
