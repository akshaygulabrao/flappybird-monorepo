import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN



env = gym.make(
    "FlappyBird-v0",
    audio_on=True,
    use_lidar=False,
    normalize_obs=True,
    score_limit=20,
    pipe_gap=100,
)
model = DQN.load("flappybird_dqn", env=env)

model.learn(total_timesteps=1e6, log_interval=1e3)




model.save("flappybird_dqn")



env = gym.make(
        "FlappyBird-v0",
        audio_on=True,
        use_lidar=False,
        render_mode="human",
        normalize_obs=True,
        score_limit=20,
        pipe_gap=100,
    )
model = DQN.load("flappybird_dqn", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        break