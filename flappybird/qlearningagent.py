import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import yaml
import flappy_bird_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


assert flappy_bird_env
with open("config/config.yaml", "r",encoding="utf-8") as file:
    config = yaml.safe_load(file)


if config["train"]:
    env = gym.make("FlappyBird-v0", use_lidar=False)
    env = Monitor(env, "logs/")

    if config["train-from-scratch"]:
        model = sb3.DQN(policy="MlpPolicy",
            env=env,
            **config["model"])
    else:
        model = sb3.DQN.load("dqn_flappy_bird",env=env)

    model.learn(**config["training-run"],
                progress_bar=True)

    model.save("dqn_flappy_bird")

else:
    episode_rewards = []
    final_scores = []
    def callback(locals_, globals_):
        if locals_['done']:
            final_scores.append(locals_['info']['score'])
        return True
    
    env = gym.make("FlappyBird-v0", use_lidar=False)
    env = Monitor(env, "logs/")
    model = sb3.DQN.load("dqn_flappy_bird",env=env)
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=100,
        callback=callback,
        deterministic=True,
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Final scores: {np.mean(final_scores)}")

