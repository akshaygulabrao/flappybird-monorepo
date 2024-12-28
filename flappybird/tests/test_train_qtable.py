from collections import defaultdict
import os
import tqdm
import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
import src.handcrafted_agent


ACTION_FLAP = 1
ACTION_NO_FLAP = 0

class QTable_Agent:
    def __init__(self, path=None):
        self.path = path
        self.q_table = defaultdict(lambda: [0, 0])
        if os.path.exists(self.path):
            self.q_table = defaultdict(lambda: [0, 0])
            df = pd.read_csv(self.path)
            for row in df.itertuples(index=False, name=None):
                self.q_table[tuple(row[:11])] = list(row[11:])

        self.alpha = 0.1
        self.gamma = 0.9
        self.lambda_ = 0.9
        self.epsilon = 0.1

        self.eligibility_traces = defaultdict(int)

    def decide(self, state):
        if not isinstance(state, tuple):
            state = tuple(state)
        if np.random.rand() < self.epsilon:
            # for testing, we want the feedback to be perfect
            return src.handcrafted_agent(state,normalize=True)
        else:
            return max([ACTION_NO_FLAP, ACTION_FLAP], key=lambda x: self.q_table[state][x])

    def update(self, s, a, r, s_next):
        if not isinstance(s, tuple):
            s = tuple(s)
        if not isinstance(s_next, tuple):
            s_next = tuple(s_next)
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
def create_environment(audio_on=True, render_mode=None):
    """Create and return a configured Flappy Bird environment."""
    return gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=False,
        normalize_obs=True,
        score_limit=1,
    )

def run_episode(agent, env):
    """Run a single episode and return the score."""
    obs, _ = env.reset()
    
    while True:
        action = agent.decide(tuple(obs.tolist()))
        next_obs, reward, done, term, info = env.step(action)
        reward = float(reward)
        agent.update(tuple(obs.tolist()), action, reward, tuple(next_obs.tolist()))
        obs = next_obs
        
        if done or term:
            return info["score"]

def save_qtable(agent, path):
    """Save the Q-table to a CSV file."""
    df = pd.DataFrame(
        [list(k) + v for (k, v) in agent.q_table.items()],
        columns=columns + ["flap", "no_flap"],
    )
    df.to_csv(path, index=False)

def test_train_qtable():
    assert flappy_bird_gymnasium
    qtable_path = "data/qtable_tests.csv"
    
    # Training phase
    if not os.path.exists(qtable_path):
        agent = QTable_Agent(path=qtable_path)
        env = create_environment()
        score = 0
        
        try:
            for i in tqdm.tqdm(range(int(1e4))):
                score += run_episode(agent, env)
                
                if i % 100000 == 0:
                    score = 0
                    save_qtable(agent, agent.path)
                    
        except KeyboardInterrupt:
            print("Catching KeyboardInterrupt")
        except Exception as e:
            print("Error: ", e)
        finally:
            save_qtable(agent, agent.path)
            print("Saved qtable")

    # Testing phase
    agent = QTable_Agent(path=qtable_path)
    env = create_environment()
    agent.epsilon = 0.0  # No exploration during testing
    
    total_score = sum(run_episode(agent, env) for _ in tqdm.tqdm(range(5)))
    print('Score: ', total_score)
    assert total_score > 3, f"Expected score greater than 3, but got {total_score}"
