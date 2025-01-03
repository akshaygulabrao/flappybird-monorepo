"""
Q-Learning agent for flappy bird. Uses simplified observation space of 12 features.

Uses stable-baseline3 DQN implementation. Uses wandb for logging. Used for benchmarking when model
improvement stops. Running 10 million episodes, and storing evaluation metrics every million
episodes.
"""

import os

import flappy_bird_env
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import yaml
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EveryNTimesteps)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

assert flappy_bird_env is not None, "flappy_bird_env is not installed"

import argparse

parser = argparse.ArgumentParser(description="DQN configuration")


parser.add_argument("config", help="path to config file (yaml)")
args = parser.parse_args()


with open(args.config, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


env = gym.make(**config["env"])
env = Monitor(env, "logs/")


class ScoreCallback:
    def __init__(self):
        self.scores = []

    def __call__(self, locals_dict, globals_dict):
        if locals_dict["dones"][0]:
            self.scores.append(locals_dict["infos"][0]["score"])
        return True


class EvalPolicyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        self.score_callback = ScoreCallback()
        evaluate_policy(
            self.model,
            env,
            n_eval_episodes=100,
            deterministic=True,
            callback=self.score_callback,
        )
        scores = self.score_callback.scores
        self.logger.record("eval/mean_score", np.mean(scores))
        self.logger.record("eval/std_score", np.std(scores))
        self.logger.dump(step=self.num_timesteps)
        return True


if config["checkpoint_path"] is None:
    model = sb3.DQN("MlpPolicy", env,**config["model"])

else:
    checkpoint_path = f"data/{config['name']}.zip"
    if os.path.exists(checkpoint_path):
        model = sb3.DQN.load(checkpoint_path, env=env)
    else:
        print("Checkpoint file not found")
        exit(1)


event_callback = EveryNTimesteps(n_steps=100_000, callback=EvalPolicyCallback())

# Add a checkpoint callback
checkpoint_callback = CheckpointCallback(**config["checkpoint"])

model.learn(
    **config["training"],
    callback=[event_callback, checkpoint_callback],
    progress_bar=True,
)

model.save(f"data/{config['name']}.zip")
