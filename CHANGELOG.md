## 0.4.1 (2024-12-29)

### Fix

- use config.yaml file for qlearningagent.py

## 0.4.0 (2024-12-29)

### Feat

- use stable_baselines3 for DQN implementation

### Fix

- integrated env w torchrl

## 0.3.1 (2024-12-28)

### Fix

- Introduce qnetwork_testing.py and transition from torch to torchrl

## 0.3.0 (2024-12-28)

### Feat

- **qlearning-agent**: enhance QLearningAgent with Weights & Biases integration and action selection improvements

## 0.2.0 (2024-12-28)

### Feat

- **qlearning-agent**: add QLearningAgent class for training and testing in FlappyBird environment

### Refactor

- add configuration and new agents for Flappy Bird

## 0.1.1 (2024-12-28)

### Fix

- removed qtable work
- **qtable**: fixed bug with pandas serialization

### Refactor

- **qtable_agent**: removes packaging code

## 0.1.0 (2024-12-28)

### Feat

- Q-Table implemented and can pass one pipe
- used binomial dist. for epsilon greedy
- Eligibility traces for credit assignment
- Off-policy Q-Learning

### Fix

- added research in one repo
