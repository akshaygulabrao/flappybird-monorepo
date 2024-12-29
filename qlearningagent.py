"""
This is a simple Q-learning agent for the Flappy Bird game. Uses a simplified observation space.
Which already hardcodes most of the relevant features to learn in the game.
"""
import torch
import gymnasium
import flappy_bird_gymnasium
import yaml
from pathlib import Path
import numpy as np
import wandb
import torch.nn as nn
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
        self.q_network = self.build_network()
        wandb.init(project="flappybird", name="qlearningagent-1",
                   notes="learning how wandb works, just testing the notes section")

    def _load_config(self, config_path):
        with open(Path(__file__).parent / config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build_network(self):
        input_dim = self.config["network"]["input_size"]
        output_dim = self.config["network"]["output_size"]
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def setup_env(self, training=True):
        self.env = gymnasium.make(
            self.config["env"]["name"],
            audio_on=self.config["env"]["audio_on"],
            use_lidar=self.config["env"]["use_lidar"],
            normalize_obs=self.config["env"]["normalize_obs"],
            score_limit=self.config["env"]["score_limit"],
        )

    def train(self, episodes=None, max_steps=None):
        
        episodes = self.config["training"]["episodes"]
        max_steps = self.config["training"]["max_steps"]
        self.setup_env(training=True)
        for i in range(episodes):
            state,_ = self.env.reset()
            for j in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                self.learn(state,action,reward,next_state,done,truncated)
                if done or truncated:
                    break
            wandb.log({"episode": i, "score":1})
            if (i+1) % 100 == 0:
                print(f"Episode {i+1} completed")
    
    def learn(self,state,action,reward,next_state,done,truncated):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        truncated = torch.tensor(truncated, dtype=torch.float32)

        q_sa = self.q_network(state)[action]
        if not done:
            q_sa_prime = torch.max(self.q_network(next_state))
            td_error = reward + self.discount_factor * q_sa_prime - q_sa
        else:
            td_error = reward - q_sa
        loss = td_error ** 2
        self.q_network.train()
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

    def test(self, episodes=100, max_steps=100):
        for episode in range(episodes):
            state,_ = self.env.reset()
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                if done or truncated:
                    break

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([ACTION_NO_FLAP,ACTION_FLAP],p=[0.95,.05])
        else:
            state = torch.tensor(state, dtype=torch.float32)
            logits = self.q_network(state)
            return torch.argmax(logits).item()

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train()