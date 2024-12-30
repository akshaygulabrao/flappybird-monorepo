import gymnasium as gym
import flappy_bird_gymnasium 
import numpy as np
import stable_baselines3 as sb3
import yaml
from stable_baselines3.common.monitor import Monitor

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


if config["train"]:
    env = gym.make("FlappyBird-v0", use_lidar=False)

    if config["train-from-scratch"]:
        model = sb3.DQN(policy="MlpPolicy",
            env=env,
            **config["model"])
    else:
        model = sb3.DQN.load("dqn_flappy_bird",env=env)

    model.learn(total_timesteps=config["total_timesteps"], 
                tb_log_name=config["tb_log_name"],
                reset_num_timesteps=config["reset_num_timesteps"],
                progress_bar=True)

    model.save("dqn_flappy_bird")
else:
    scores = []
    env = gym.make("FlappyBird-v0", use_lidar=False)
    model = sb3.DQN.load("dqn_flappy_bird",env=env)

    for i in range(1000):
        obs,_ = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                scores.append(info['score'])
                break
    env.close()
    print(np.mean(scores))