import torch
import gymnasium
import flappy_bird_gymnasium

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        assert flappy_bird_gymnasium
        assert torch
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
    def setup_train_env(self):
        self.env = gymnasium.make(
            "FlappyBird-v0",
            audio_on=True,
            use_lidar=False,
            normalize_obs=True,
            score_limit=10,
        )

    def setup_test_env(self):
        self.env = gymnasium.make(
            "FlappyBird-v0",
            audio_on=True,
            use_lidar=False,
            normalize_obs=True,
            score_limit=10,
        )
    def train(self, episodes=1000, max_steps=100):
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break
    def test(self, episodes=100, max_steps=100):
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                if done:
                    break

agent = QLearningAgent()
agent.train()
agent.test()