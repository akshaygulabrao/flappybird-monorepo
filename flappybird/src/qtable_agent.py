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
                self.q_table[tuple(row[:11])] = list(row[11:])

        self.alpha = 0.1
        self.gamma = 0.9
        self.lambda_ = 0.9
        self.epsilon = 0.5

    def decide(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        else:
            return max([0, 1], key=lambda x: self.q_table[state][x])

    def update(self, s, a, r, s_next):
        self.eligibility_traces[tuple(list(s) + [a])] += 1

        q = self.q_table
        q[s][a] = q[s][a] + self.alpha * (
            r + self.gamma * max([0, 1], key=lambda x: q[s_next][x]) - q[s][a]
        )
        for k in self.eligibility_traces:
            s,a= k[:-1],k[-1]
            td_error = r + self.gamma * max([0, 1], key=lambda x: q[s_next][x]) - q[s][a]
            q[s][a] += self.alpha * self.eligibility_traces[k] * td_error
            self.eligibility_traces[k] *= self.gamma * self.lambda_



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
    score = 0
    for i in tqdm.tqdm(range(int(1e6))):
        agent.eligibility_traces = defaultdict(int)
        obs, _ = env.reset()
        while True:
            action = agent.decide(tuple(obs.tolist()))
            next_obs, reward, done, term, info = env.step(action)
            reward = float(reward)
            agent.update(tuple(obs.tolist()), action, reward, tuple(next_obs.tolist()))
            obs = next_obs
            if done or term:
                score += info["score"]
                break
        if i % 1000 == 0:
            score= 0 
            df = pd.DataFrame(
                [list(k) + v for k, v in agent.q_table.items()],
                columns=columns + ["flap", "no_flap"],
            )
            df.to_csv(agent.path, index=False)
    print(f"Average score: {score / 1000}")
    df.to_csv(agent.path, index=False)
