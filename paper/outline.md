# How many neurons do you need to play flappy bird perfectly?

## Introduction
I am investigating the amount of neurons required to play flappy bird perfectly. Perfectly performance is quantified by beating a handcrafted evaluation agent that I made. It is capable of scoring 900 on a training run of 10 runs with a score limit of 1000.

I contribute:
- A meta-survey of all work that has been done on Flappy Bird
- A hand-crafted agent capable of "perfect" performance
- A review of SOTA reinforcement learning algorithms
- Benchmarking of all the algorithms

## Related Work

1. [Chen. Deep Reinforcement Learning for Flappy Bird. Stanford Final Project, 2015](https://cs229.stanford.edu/proj2015/362_report.pdf) Original approach describing how to use the DQN approach mentioned in Atari applied to flappy bird.
2. [Yenchenlin. Deep Reinforcement Learning for Flappy Bird. Github, 2016](https://cs229.stanford.edu/proj2015/362_report.pdf) Implementation of the approach. Inspired by the work of Chen.
3. [johnnycode8. Train the DQN Algorithm on Flappy Bird, Youtube, 2024](https://github.com/johnnycode8/dqn_pytorch) Implements the dueling-dqn approach on the simplified observation space.
4. [xviniette. Flappy Bird with PPO, Github, 2024](https://github.com/xviniette/FlappyLearning) Implements the PPO approach on the simplified observation space.
5. [markub3327. flappy-bird-gymnasium, Github, 2023](https://github.com/markub3327/flappy-bird-gymnasium) Implements the Dueling DQN approach on two state representations: (1) the simplified observation space (2) LIDAR measurements of the environment.
6. [SarvagaVaish. FlappyBirdRL, Github, 2014](https://github.com/SarvagyaVaish/FlappyBirdRL) Simplified observation space and used a QTable to implement a perfect agent.
7. [kyokin78. rl-flappybird, Github, 2019](https://github.com/kyokin78/rl-flappybird) Further simplified the simplified observation space and used a QTable to implement a perfect agent. Inspired by the work of SarvagaVaish.
8. [foodsung. DRL-FlappyBird, Github, 2016](https://github.com/foodsung/DRL-FlappyBird) Inspired by yenchenlin. Provides an even more mature implementation.



## Flappy Bird
- Mobile game from 2013 that went viral
- Game mechanics
- Simple game mechanics make it an ideal candidate for reinforcement learning tasks
- RL environment exists at [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)

### Flappy Bird Observation Space
Due to computational constraints, I am using a simplified observation space. Current SOTA implementations learn directly from pixels. The observation space consists of 12 features: 
1. The distance to pipe 0
2. the top locattion of the bottom of pillar 1
3. the top locattion of the top of pillar 1
4. the distance to pipe 1
5. the top locattion of the bottom of pillar 2
6. the top locattion of the top of pillar 2
7. the distance to pipe 2
8. the top locattion of the bottom of pillar 3
9. the top locattion of the top of pillar 3
10. The current y position of the bird
11. The current y velocity of the bird
12. The current rotation of the bird

Note that the rotation does not impact any of the collision checking and is purely cosmetic.

These features are normalized to be between 0 and 1.


## Reinforcement Learning Techniques
### DQN
I use the DQN algorithm from stable-baselines3. The current implementation by stable-baseline3 implements many bells and whistles above the original DQN algorithm.

The original DQN algorithm comes from [Mnih et al. Playing Atari with Deep Reinforcement Learning, arxiv 2013](https://arxiv.org/abs/1312.5602). They use a CNN consisting of convolution-relu blocks connected to a final fully connected layer. The CNN then approximates
$$
Q(s,a) \leftarrow 
\begin{cases} 
r & \text{ if } s' \text{ is terminal} \\
r + \gamma \max_{a'} Q(s',a') & \text{otherwise}
\end{cases}
$$
or 
$$
Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s',a')
$$
where Q(s',a') is 0 if the state is terminal.

At each step, the CNN processes data from a replay buffer. They use MSE loss to train the network with stochastic gradient descent. 

[van Hasselt et al. Deep Reinforcement Learning with Double Q-learning, arxiv 2015](https://arxiv.org/abs/1509.06461) points out some problem with the original Q-Learning algorithm. The original Q-Learning algorithm suffers from overestimation bias. Hasselt et al. introduced the double Q-learning algorithm. The double Q-learning algorithm uses two networks to estimate the Q-value. One network is used to select the action, and the other network is used to evaluate the action. This reduces the overestimation bias. Effectively, they compute:

$$
Q_1(s,a) \leftarrow r + \gamma Q_1(s', \argmax_{a'} Q_2(s',a'))
$$
where:
- $Q_1$ is the network used to select the action
- $Q_2$ is the network used to evaluate the action

Note that the paper uses a slightly more complex notation, generalizing Q and specifying that there are two sets of parameters instead.

[van Hasselt et al. Dueling Network Architectures for Deep Reinforcement Learning, arxiv 2015](https://arxiv.org/abs/1511.06581) introduces the dueling network architecture. It splits the Q-value into two components:
$$
Q(s,a) = V(s) + A(s,a)
$$
where:
- $V(s)$ is the value function
- $A(s,a)$ is the advantage function

This allows more efficient state representation and reduces the variance of the Q-value estimate.


I originally used a simplified [64,64] multi-layer perceptron. I expected this to be insufficient and the performance to plateau. I set an over ambitious training step of 30 million. What surprised me was that the performance didn't stop plateauing long after I suspected it would. Setting the epsilon to $1 \times 10^{-4}$ statically instead of using epsilon-greedy exploration helped the agent more because the agent would play things safe by accounting for situations where flap or not flapping keeps it alive. The static $\epsilon$ helped the agent learn significantly faster. I suspect this to be an environment specific feature, where in other environments, the epsilon-greedy exploration would be more beneficial.
### PPO

### SAC

### TD3

### A2C

### A3C

### TRPO

### ACKTR

### Rainbow DQN

### C51/Distributional DQN

### ACER

### Soft Q-Learning

### IMPALA

### APE-X

### R2D2

### NGU

### MuZero

### DrQ

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

## Pixels
- Learning straight from pixels

## Conclusion
- Summary of findings
- Comparative analysis
- future work
- research implications

## Postmortem
One of my biggest mistakes was that I started by trying to reimplment my own QTable approach. This wasted lots of time and effort by trying to resolve problems that had already been solved in prior work. I thought that looking at their code would be "cheating." I did not see that I was not adding anything new to the field by doing this. It would have been better to review all prior work and try to replicate it before I built my own QTable approach.
