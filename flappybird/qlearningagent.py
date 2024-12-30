"""
Q-Learning agent for flappy bird. Uses simplified observation space of 12 features.

Uses stable-baseline3 DQN implementation. Uses wandb for logging. Used for benchmarking when model
improvement stops. Running 10 million episodes, and storing evaluation metrics every million
episodes.
"""
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import yaml
import flappy_bird_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import wandb

assert flappy_bird_env is not None, "flappy_bird_env is not installed"

with open("config/config.yaml", "r",encoding="utf-8") as file:
    config = yaml.safe_load(file)


env = gym.make("FlappyBird-v0", use_lidar=False)
env = Monitor(env, "logs/")

class ScoreCallback:
    def __init__(self):
        self.scores = []    
    def __call__(self, locals_dict, globals_dict):
        if locals_dict['dones'][0]:
            self.scores.append(locals_dict['infos'][0]['score'])
        return True

model = sb3.DQN(policy="MlpPolicy",
    env=env,
    **config["model"])

score_callback = ScoreCallback()    
print(evaluate_policy(model, env, n_eval_episodes=3, deterministic=True, callback=score_callback)) 