# How many neurons do you need to play flappy bird perfectly?

## Introduction

- **Research Focus:**
  - Investigate the neural network complexity required for optimal Flappy Bird performance.
  - Define "perfect performance" as surpassing a handcrafted evaluation agent scoring 900/1000 over 10 runs.

- **Contributions:**
  - Development of a high-performing handcrafted agent.
  - Analysis of network complexity versus performance.

- **Methodology Rationale:**
  - Opt for non-pixel-based learning to explore diverse model types beyond CNNs.

There are lots of interesting ways to extend this project, but I have to start focusing on communicating my findings instead of more discovery.

## Related Work

1. [Chen. Deep Reinforcement Learning for Flappy Bird. Stanford Final Project, 2015](https://cs229.stanford.edu/proj2015/362_report.pdf)
 - Learns from pixels to play flappy bird
 - Uses DQN from Mnih et al.
2. [Yenchenlin. Deep Reinforcement Learning for Flappy Bird. Github, 2016](https://github.com/yenchenlin/DeepLearningFlappyBird)
 - Implementation of the approach. Inspired by the work of Chen.
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

A winning state for this project is finding a perfect scoring bot for significantly less memory than a q-table for every state.

## Flappy Bird
- Mobile game from 2013 that went viral
- Game mechanics
- Simple game mechanics make it an ideal candidate for reinforcement learning tasks
- RL environment exists at [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- Simplified observation space 12 features, [0,1]

## Reinforcement Learning Techniques
Reinforcement learning techniques are roughly divided into two categories:
1. Value-based methods

### Value-based methods
1. Implicitly assume optimal action is not a mixed strategy.
2. More sample efficient, because the value function is computed per action.

### Policy-based methods
1. Assume that there is a mixed strategy for each state that is optimal.
2. More stable, because they naturally explore more in each state instead of using hard-coded exploration strategies.


## Benchmarking Reinforcement Learning
- Best achievable average score with a score limit of 1000.
- Sample inefficiency does not count
- How close does it get to handcrafted agent.

## Handcrafted evaluation function
- I made a handcrafted evaluation function.
- It gets approximately 900 on a training run of 10 runs with a score limit of 1000.
- I can measure the success of the reinforcement learning agents utilizing how

## Simplified State Space
- How well do the other benchmarks perform?
- How well does my agent perform?
  * Parameters that impacted the performance of my agent

## DQN
1. Rough idea from Mnih et al. 2013, but incorporates improvements
2. Uses network architecture of [12, 64, 64, 2]
3. Double DQN
4. Dueling DQN

| Name | Mean Score (1000 runs) | Std Score (1000 runs) |
|------|------------------------|-----------------------|
| Handcrafted Agent | `<mean>` | `<std>` |
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
3. {Prioritized Experience Replay, Hindsight Experience Replay}

### Further Work
- Learn directly from pixels
- Experiment with transformer models
- Experiment with Partially Observable Environments
- Try Hindsight Experience Replay
- Try Policy-based methods

## Conclusion
- Summary of findings
- Comparative analysis
- future work
- research implications

## Postmortem
One of my biggest mistakes was that I started by trying to reimplment my own QTable approach. This wasted lots of time and effort by trying to resolve problems that had already been solved in prior work. I thought that looking at their code would be "cheating." I did not see that I was not adding anything new to the field by doing this. It would have been better to review all prior work and try to replicate it before I built my own QTable approach.
