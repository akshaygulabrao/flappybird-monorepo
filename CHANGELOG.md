## v0.12.0 (2025-01-08)

### Feat

- moved back to title page

## v0.11.0 (2025-01-07)

### Feat

- finished README and PPT

### Refactor

- revise README for clarity and depth on reinforcement learning concepts

## v0.10.0 (2025-01-07)

### Feat

- using marp to export markdown to powerpoint
- accidentally ended up with a good idea
- added versions of extra experiments

### Fix

- Finished experiment 2
- updating README

### Refactor

- consolidate career summary and enhance content structure

## v0.7.1 (2025-01-01)

### Fix

- qol changes for outline - weights2mp4 loads DQN Agent by default instead of handcrafted agent - dqn_versions maps my numeric versions to quality attributes of each version - outline.md corrects misinformation about the replay buffer
- renamed config to dqn_v1
- resolve merge

### Refactor

- agents.py contains all agents

## v0.7.0 (2025-01-01)

### Feat

- made weights2py more useful
- benchmark runs benchmarking tests for you
- visualize policy with video

### Fix

- added auto-naming timestamps to weights2mp4
- plan for DQN
- move outline paragraphs to paper

## v0.6.2 (2025-01-01)

### Fix

- multiple sections added to outline
- **paper**: refine DQN section
- **paper**: enhance outline on DQN and postmortem reflections
- **qlearningagent**: update training configuration and checkpointing

## v0.6.1 (2024-12-31)

### Fix

- **config**: update training parameters and model settings
- **qlearningagent**: enhance training with checkpointing and score limit
- expand paper outline on Flappy Bird reinforcement learning

## v0.6.0 (2024-12-30)

### Feat

- added logging to tensorboard
- incorporated eval_policy into checkpoint
- new callback for checking scores during eval

## v0.5.0 (2024-12-29)

### Feat

- add Flappy Bird environment and constants
- **qlearningagent**: enhance training flow and evaluation process
- use stable_baselines3 for DQN implementation
- **qlearning-agent**: enhance QLearningAgent with Weights & Biases integration and action selection improvements
- **qlearning-agent**: add QLearningAgent class for training and testing in FlappyBird environment
- Q-Table implemented and can pass one pipe
- used binomial dist. for epsilon greedy
- Eligibility traces for credit assignment
- Off-policy Q-Learning

### Fix

- use config.yaml file for qlearningagent.py
- integrated env w torchrl
- Introduce qnetwork_testing.py and transition from torch to torchrl
- removed qtable work
- **qtable**: fixed bug with pandas serialization
- added research in one repo

### Refactor

- moved rl files back into flappy bird dir
- add configuration and new agents for Flappy Bird
- **qtable_agent**: removes packaging code
