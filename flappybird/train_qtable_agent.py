from collections import defaultdict
import os
import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
import pickle
from handcrafted_agent import handcrafted_agent


class QTable_Agent:
    def __init__(self, path=None, sanity_check=True):
        """
        Initializes the QTable_Agent.

        Args:
            path (str): The path to the pickle file containing the Q-table.
            sanity_check (bool): Whether to use the handcrafted agent for the epsilon-greedy policy.
        """
        self.sanity_check = sanity_check
        self.eligibility_traces = defaultdict(int)
        self.q_table = defaultdict(lambda: [0, -1])
        self.path = path
        if os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    items = pickle.load(f)
                    self.q_table = defaultdict(lambda: [0, -1], items)
                print(f"Loaded qtable from {self.path}")
                print(f"Qtable shape: {len(self.q_table)}")
            except Exception as e:
                print(f"Error loading qtable: {e}")
                print("Starting from scratch")
        else:
            print(f"No qtable found at {self.path}, starting from scratch")

        self.alpha = 0.01
        self.gamma = 0.9
        self.lambda_ = 0.9
        self.epsilon = 0.01

    def to_tuple(self, state):
        if not isinstance(state, tuple):
            state = tuple(state)
        return state

    def decide(self, state):
        """
        Decides the action to take based on the current state. If the agent is in sanity check mode, it uses the handcrafted agent.
        Otherwise, it uses the epsilon-greedy policy.

        Args:
            state (tuple): The current state.

        Returns:
            int: The action to take (0 for no flap, 1 for flap).
        """
        state = self.to_tuple(state)
        if np.random.rand() < self.epsilon:
            if self.sanity_check:
                return handcrafted_agent(state, normalize=True)
            else:
                return np.random.choice([0, 1], p=[0.95, 0.05])
        else:
            return max([0, 1], key=lambda x: self.q_table[state][x])

    def update(self, s, a, r, s_next):
        """
        Updates the Q-table using the Q-learning algorithm with eligibility traces.

        Args:
            s (tuple): The current state.
            a (int): The action taken.
            r (float): The reward received.
            s_next (tuple): The next state.
        """
        s = self.to_tuple(s)
        s_next = self.to_tuple(s_next)
        self.eligibility_traces[tuple(list(s) + [a])] += 1

        q = self.q_table  # shorter name

        # calculate the td error
        next_max = max([0, 1], key=lambda x: q[s_next][x])
        td_error = r + self.gamma * next_max - q[s][a]

        # update the q table
        q[s][a] = q[s][a] + self.alpha * td_error
        self.eligibility_traces[tuple(list(s) + [a])] += 1

        for k in self.eligibility_traces:
            s, a = k[:-1], k[-1]
            q[s][a] += self.alpha * self.eligibility_traces[k] * td_error
            self.eligibility_traces[k] *= self.gamma * self.lambda_


if __name__ == "__main__":
    assert flappy_bird_gymnasium
    agent = QTable_Agent(path="data/qtable.pkl", sanity_check=True)
    log_frequency = int(1e3)
    max_episodes = int(1e6)
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        use_lidar=False,
        # render_mode='human',
        normalize_obs=True,
        score_limit=1,
    )
    score = 0
    reward_sum = 0
    try:
        for i in range(max_episodes):
            agent.eligibility_traces = defaultdict(int)
            # agent.epsilon = 1 - (i+1)/max_episodes
            obs, _ = env.reset()
            while True:
                action = agent.decide(tuple(obs.tolist()))
                next_obs, reward, done, term, info = env.step(action)
                reward = float(reward)
                reward_sum += reward
                agent.update(tuple(obs.tolist()), action, reward, tuple(next_obs.tolist()))
                obs = next_obs
                if done or term:
                    score += info["score"]
                    reward_sum += reward
                    break
            if (i + 1) % log_frequency == 0:
                print(
                    f"Run {i+1:5d} complete, "
                    f"average score: {score/log_frequency:.2f}, "
                    f"qtable size: {len(agent.q_table)}, "
                    f"epsilon: {agent.epsilon:.2f},"
                    f"reward: {reward_sum/log_frequency:.2f}"
                )
                score = 0
                reward_sum = 0
                with open(agent.path, 'wb') as f:
                    pickle.dump(dict(agent.q_table), f)
    except KeyboardInterrupt:
        print("Catching KeyboardInterrupt, saving qtable")
    finally:
        with open(agent.path, 'wb') as f:
            pickle.dump(dict(agent.q_table), f)
        print("Saved qtable")
