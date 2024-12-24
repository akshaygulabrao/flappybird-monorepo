from collections import defaultdict
import os
import tqdm
import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
from src.handcrafted_agent import agent


class Agent:
    def __init__(self, path=None):
        self.path = path
        if not os.path.exists(self.path):
            self.q_table = defaultdict(lambda: [0, 0])
        else:
            self.q_table = defaultdict(lambda: [0, 0])
            df = pd.read_csv(self.path)
            for row in df.itertuples(index=False, name=None):
                self.q_table[tuple(row[:12])] = list(row[12:])

        self.alpha = 0.1
        self.gamma = 0.9

    def decide(self, state):
        if np.random.rand() < 0.1:
            return np.random.randint(2)
        else:
            return max([0, 1], key=lambda x: self.q_table[state][x])

    def update(self, s, a, r, s_next):
        q = self.q_table
        q[s][a] = q[s][a] + self.alpha * (
            r + self.gamma * max([0, 1], key=lambda x: q[s_next][x]) - q[s][a]
        )


columns = [
    "pipe_0_x",
    "pipe_0_top",
    "pipe_0_bot",
    "pipe_1_x",
    "pipe_1_top",
    "pipe_1_bot",
    "pipe_2_x",
    "pipe_2_top",
    "pipe_2_bot",
    "bird_y",
    "bird_vel",
    "bird_rot",
]

if __name__ == "__main__":
    agent = Agent(path="data/qtable.csv")
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        render_mode=None,
        use_lidar=False,
        normalize_obs=True,
        score_limit=1000,
    )
    for i in tqdm.tqdm(range(int(1e3))):
        obs, _ = env.reset()
        while True:
            action = agent.decide(tuple(obs.tolist()))
            next_obs, reward, done, term, info = env.step(action)
            reward = float(reward)
            agent.update(tuple(obs.tolist()), action, reward, tuple(next_obs.tolist()))
            correct = 0
            obs = next_obs
            if done or term:
                break
        if i % 100 == 0:
            df = pd.DataFrame(
                [list(k) + v for k, v in agent.q_table.items()],
                columns=columns + ["flap", "no_flap"],
            )
            df.to_csv(agent.path,index=False)
    df.to_csv(agent.path, index=False)
