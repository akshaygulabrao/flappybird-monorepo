import torch
import gymnasium
import flappy_bird_gymnasium
import yaml
from pathlib import Path
import numpy as np
import wandb

ACTION_NO_FLAP = 0
ACTION_FLAP = 1

class QLearningAgent:
    def __init__(self, config_path="/Users/ox/Documents/flappybird-monorepo/config/config.yaml"):
        assert flappy_bird_gymnasium
        assert torch
        self.config = self._load_config(config_path)
        self.learning_rate = self.config["learning_rate"]
        self.discount_factor = self.config["discount_factor"]
        self.epsilon = self.config["epsilon"]
        
        wandb.init(project="flappybird", name="qlearningagent-1",
                   notes="learning how wandb works, just testing the notes section")

    def _load_config(self, config_path):
        with open(Path(__file__).parent / config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_env(self, training=True):
        self.env = gymnasium.make(
            self.config["env"]["name"],
            audio_on=self.config["env"]["audio_on"],
            use_lidar=self.config["env"]["use_lidar"],
            normalize_obs=self.config["env"]["normalize_obs"],
            score_limit=self.config["env"]["score_limit"],
        )

    def train(self, episodes=None, max_steps=None):
        episodes = episodes or self.config["training"]["episodes"]
        max_steps = max_steps or self.config["training"]["max_steps"]
        self.setup_env(training=True)
        for i in range(episodes):
            state = self.env.reset()
            for j in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                if done or truncated:
                    break
            wandb.log({"episode": i, "score":1})
            if (i+1) % 100 == 0:
                print(f"Episode {i+1} completed")
            

    def test(self, episodes=100, max_steps=100):
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                if done or truncated:
                    break

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([ACTION_NO_FLAP,ACTION_FLAP],prob=[0.95,.05])
        else:
            return np.argmax(self.q_table[state])

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train()
    agent.test()