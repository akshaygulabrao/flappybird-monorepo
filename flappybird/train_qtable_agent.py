from collections import defaultdict
import os
import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
from handcrafted_agent import handcrafted_agent

class QTable_Agent:
    def __init__(self, path=None,sanity_check=False):
        """
        Initializes the QTable_Agent.

        Args:
            path (str): The path to the CSV file containing the Q-table.
            sanity_check (bool): Whether to use the handcrafted agent for the epsilon-greedy policy.
        """
        self.sanity_check = sanity_check
        self.path = path
        self.eligibility_traces = defaultdict(int)
        self.q_table = defaultdict(lambda: [0, 0])
        
        if os.path.exists(self.path):
            self.q_table = defaultdict(lambda: [0, 0])
            df = pd.read_csv(self.path)
            for row in df.itertuples(index=False, name=None):
                self.q_table[tuple(row[:11])] = list(row[11:])
            print(f"Loaded qtable from {self.path}")
            print(f"Qtable shape: {len(self.q_table)}")
        else:
            print(f"No qtable found at {self.path}, starting from scratch")

        self.alpha = 0.1
        self.gamma = 0.9
        self.lambda_ = 0.9
        self.epsilon = 0.1
    
    def to_tuple(self,state):
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
                return handcrafted_agent(state,normalize=True)
            else:
                return np.random.choice([0,1],p=[0.9,0.1])
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

        q = self.q_table # shorter name

        # calculate the td error
        next_max = max([0, 1], key=lambda x: q[s_next][x])
        td_error = r + self.gamma * next_max - q[s][a]

        # update the q table
        q[s][a] = q[s][a] + self.alpha * td_error
        self.eligibility_traces[tuple(list(s) + [a])] += 1
        
        for k in self.eligibility_traces:
            s,a= k[:-1],k[-1]
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
    assert flappy_bird_gymnasium
    agent = QTable_Agent(path="data/qtable.csv",sanity_check=False)
    log_frequency= int(1e3)
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        # render_mode='human',
        use_lidar=False,
        normalize_obs=True,
        score_limit=1,
    )
    score = 0
    try:
        for i in range(int(1e6)):
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
            if (i+1) % log_frequency == 0:
                print(f"Run {i+1:5d} complete, average score: {score/log_frequency:.2f}, qtable size: {len(agent.q_table)}")
                score = 0
                df = pd.DataFrame(
                    [list(k) + v for (k, v) in agent.q_table.items()],
                    columns=columns + ["flap", "no_flap"],
                )
                df.to_csv(agent.path, index=False)
    except KeyboardInterrupt:
        print("Catching KeyboardInterrupt, saving qtable`")
        pass
    finally:
        df.to_csv(agent.path, index=False)
        print("Saved qtable")
