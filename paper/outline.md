# Catastrophic Forgetting Cover-Up

## Introduction
Reinforcement learning
- heuristic search of sequential decision making
- discover solutions counterintuitive to humans
- foundation of breakthroughs in go, chess, and protein folding

### Reseach Focus
- **Catastrophic Forgetting**
  - Issue in value-based reinforcement learning methods from the adaptive nature of neural networks

This paper covers the following:
- Introduction to Reinforcement Learning
- Introduction to Value-based Methods
- Introduction to Deep Q-Learning
- Flappy Bird Example
- Stability Issues with DQN

Most RL papers
- don't mention catastrophic forgetting
  - makes their research look worse
- catastrophic forgetting
  - fundamental issue in function approximation based
- downplay the difficulty of resolving the problem
- use/not-use data visualization methods to hide the issue

## Background

### Reinforcement Learning
1. Framework for sequential decision making
2. Agent learns by observing environment's response to actions
3. Goal is to maximize the reward signal given by the environment
  - It's a numerical value
  - Good reward signals are consistent and achievable
  - Accurately represent the difference between optimal and suboptimal actions

### Markov Decision Processes
1. Formal name for sequential decision making
2. 4 components of an MDP
  - State space $\mathcal{S}$
  - Action space $\mathcal{A}$
  - Transition function $P(s'|s,a) \rightarrow [0,1]$
  - Reward function $R(s,a) \rightarrow \mathbb{R}$
3. Each interaction with the environment is known as a trajectory

### Types of Reinforcement Learning
1. There are two dimensions along which RL algorithms can be classified:
  - Known/Unknown environment dynamics
  - Learn expected return of each state-action
  - Most optimal action distribution for each state
2. Model-Based RL vs Model-Free RL
  - Sometimes the next state is deterministic wrt action
    - Map the optimal action to each state
  - The next state is uncertain
    - Learn the state transition function
3. Value-based RL vs Policy-based RL
  - Value-based: Learn the value function for each state-action
    - The optimal policy is then trivial
    - Exploration strategy is a hyperparameter
  - Policy-based: Learn the optimal action distribution for each state
    - Exploration strategy is baked into the policy
    - More stable

### Deep Q Learning
Before Q-Learning Networks:
- Q-Tables: learn the value of each state-action pair
  - Suffer from the curse of dimensionality
  - Manual feature engineering to compress the state space
- Q-Learning Networks
  - Learn the value of each state-action pair
  - Automates feature engineering
  - Networks were deep for their time


## Related Work

1. [Chen. Deep Reinforcement Learning for Flappy Bird. Stanford Final Project, 2015](https://cs229.stanford.edu/proj2015/362_report.pdf)
 - Learns from pixels to play flappy bird
 - Uses DQN from Mnih et al. w/ target network
2. [Yenchenlin. Deep Reinforcement Learning for Flappy Bird. Github, 2016](https://github.com/yenchenlin/DeepLearningFlappyBird)
 - Implementation of the approach. Inspired by the work of Chen. Modified the velocity so that the bird doesn't jump as high
3. [johnnycode8. Train the DQN Algorithm on Flappy Bird, Youtube, 2024](https://github.com/johnnycode8/dqn_pytorch)
 - Implements the dueling-dqn approach on the simplified observation space.
4. [xviniette. Flappy Bird with PPO, Github, 2024](https://github.com/xviniette/FlappyLearning)
 - Implements the PPO approach on the simplified observation space.
5. [markub3327. flappy-bird-gymnasium, Github, 2023](https://github.com/markub3327/flappy-bird-gymnasium)
 - Implements the Dueling DQN approach on two state representations: (1) the simplified observation space (2) LIDAR measurements of the environment.
6. [SarvagaVaish. FlappyBirdRL, Github, 2014](https://github.com/SarvagyaVaish/FlappyBirdRL)
 - Simplified observation space and used a QTable to implement a perfect agent.
7. [kyokin78. rl-flappybird, Github, 2019](https://github.com/kyokin78/rl-flappybird)
 - Further simplified the simplified observation space and used a QTable to implement a perfect agent. Inspired by the work of SarvagaVaish.
8. [foodsung. DRL-FlappyBird, Github, 2016](https://github.com/foodsung/DRL-FlappyBird)
 - Inspired by yenchenlin. Provides an even more mature implementation.

## Flappy Bird
- Mobile game from 2013 that went viral
- Simple game mechanics make it an ideal candidate for reinforcement learning tasks
- RL environment exists at [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- Simplified observation space 12 features, [0,1]



## Methods
- Stable-baselines3 open-source implementations of RL Algorithms
- Utilized flappy-bird-gymnasium environment by gymnasium

| Name | Mean Score (1000 runs) | Std Score (1000 runs) |
|------|------------------------|-----------------------|
| dqn_flappybird_v1_1.3M_steps | 20.82| 15.83 |
| dqn_flappybird_v2_2M_steps | 121.43 | 142.98 |

Note: Change the v1 to be more descriptive. Only supposed to be identified by me right now.

### Results
1. Original training run was 30M learning steps.
2. Catastrophic forgetting around 1.2M learning steps, weird jumps in score after that.
3. Random chance tweaked parameters to get 900 average score. `<Include tensorboard chart here>`

### Ablation DQN
Cartesian product of all possible combinations of the following:
1. Double DQN
2. Dueling DQN

## Criticism of Related Work
- No mention of catastrophic forgetting.
- Nobody showed performance over time graph, implying lots of catastrophic forgetting.
- Mnih et al. 2013 didn't encounter it as much because no local maxima to get stuck in. Mnih mentions it and dodges the graphs that would show very poor performance.

### Further Work
- Learn directly from pixels
- Experiment with transformer models
- Experiment with Partially Observable Environments
- Try Hindsight Experience Replay
- Try Policy-based methods
- Duplicated neurons with different learning rates

## Conclusion
- Summary of findings
- Comparative analysis
- future work
- research implications

## Postmortem
One of my biggest mistakes was that I started by trying to reimplment my own QTable approach. This wasted lots of time and effort by trying to resolve problems that had already been solved in prior work. I thought that looking at their code would be "cheating." I did not see that I was not adding anything new to the field by doing this. It would have been better to review all prior work and try to replicate it before I built my own QTable approach.
