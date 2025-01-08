# Flappy Bird with Q-Learning

Reinforcement learning is the systematized approach of learning from sequential decisions. Agents iteratively learn from the environment's response to actions, which consists of the state update and a numerical reward signal. It allows the discovery of solutions that are counterintuitive to humans. It is the foundation of breakthroughs in go, chess, and protein folding.

In this repository, I replicate prior work on the flappy bird game. I technically was successful, but had significant issues with the stability of learning. Upon further reflection, I noticed that all prior work also had significant issues with the stability of learning, but avoided showing this problem through clever display tricks, or chart omission.

## Flappy Bird
Flappy bird is a mobile game from 2013 that went viral. The concept is simple: fly through gaps in a stream of pillars. The pillars are of varying heights. The difficulty lies is precisely managing the velocity and position of the bird against gravity. [Play Flappy Bird](https://flappybird.io).

The observation space in this environment is 12 features: the height of the last pipe, the height of the next 2 pipes, the horizontal distances to each of the 3 pipes, the velocity of the bird, and the y-position of the bird. The action space is 2 actions: jump and do nothing. Jumping makes you move upward on the screen, and you must weight for gravity to naturally bring you back down.

The reward system in this game is:
- 0.1 points for staying alive
- -0.5 points for hitting the top of the screen
- -1 point for dying
- 1 point for passing through a pipe


## Background

### Markov Decision Processes
Reinforcement learning is a framework for learning from sequential decisions. Markov Decision Processes (MDPs) formalize this process.
MDPs consist of two sets and two functions:
- State space $\mathcal{S}$
- Action space $\mathcal{A}$
- Transition function $P(s'|s,a) \rightarrow [0,1]$
- Reward function $R(s,a) \rightarrow \mathbb{R}$

An agent's interaction with their environment is defined by a trajectory. A trajectory is a sequence of states, actions, and rewards.

$S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T$

The agent's goal is to learn a policy $\pi(s) \rightarrow a$ that maximizes the reward signal. The reward signal isn't given by the reward received during a single timestep, but rather the sum of rewards received over the entire trajectory. When making a decision, the agent must consider the future rewards it can expect to receive, but discount them by a factor of $\gamma \in [0,1]$ because they are less certain. $\gamma$ close to 1 makes the agent more forward-looking into the future. $\gamma$ close to 0 makes the agent look for immediate rewards. In most reinforcement learning problems, $\gamma$ is set to 0.99. With calculus, even if t approaches infinity, the sum of rewards is finite.

$$ \sum_{t=0}^{\infty} \gamma r_{t+1} = \frac{1}{1-\gamma} $$

When the environment dynamics, $P(s'|s,a)$, are known, the agent can use dynamic programming to solve the MDP. It can walk backwards from the terminal state to the initial state, and calculate the value of each state-action pair.If environment dynamics is unknown, the agent can use reinforcement learning to learn the best action for each state.

### Two Learning Paradigms of Reinforcement Learning
The agent has two approaches for learning the best action for each state: (1) Learning the expected return of each state-action pair or (2) learning the optimal action distribution for each state.

#### Value-based Reinforcement Learning
Learning the expected return of each state-action pair is faster, but needs a hardcoded exploration strategy, which needs to be tuned for each environment. For example, using a learning rate of .1 causes the agent to learn absolutely nothing in the flappy bird environment because the optimal action distribution is to jump about 1/20 of the time. The probability that this distribution is selected naturally is extremely low.

#### Policy-based Reinforcement Learning
Learning the optimal action distribution for each state is more stable, because it naturally systematically explores the environment. Learning action distributions also naturally expand to deal with continuous action spaces.

### Deep Q Learning
Q-Learning is the value-based policy-free approach to reinforcement learning. The agent needs to learn a function $S \times A \rightarrow \mathbb{R}$ that approximates the expected return of each state-action pair. Classical approaches for this problem is to use a Q-Table, which stores the value of each state-action pair in a table.

The problem with Q-Tables is that it is mathematically impossible to represent the value of each state-action pair in a table, if the state space is continuous. Even if the state space is discrete, the table is too large to be practical. The only solution is to compress the state space so that Q-tables are feasible.[kyokin78. rl-flappybird, Github, 2019](https://github.com/kyokin78/rl-flappybird) utilizes this approach to beat the game.

[Mnih et al. 2013](https://www.nature.com/articles/nature14236) used convolutional neural networks to compress the state space, and beat Atari games. To contend with the dynamics of the environment, they randomly sampled from a replay buffer to decorrelate the samples. Then they learned the Q-values of each state-action pair, using temporal difference learning.

$$Q(s,a,\theta) \leftarrow r + \gamma \max_{a'} Q(s',a',\theta)$$

## Related Work

After flappy bird was released around 2013, the first python implementation of flappy bird was released by [sourabhv FlappyBird Github. 2014](https://github.com/sourabhv/FlapPyBird). It quickly became popular and was used as a benchmark for reinforcement learning. The first replication of Mnih et al. 2013 was [Chen. Deep Reinforcement Learning for Flappy Bird. Stanford Final Project, 2015](https://cs229.stanford.edu/proj2015/362_report.pdf). This paper used the same convolutional neural network architecture as Mnih et al. 2013, and made the gaps wider to make the game easier to play at first, then narrowed them back to make the game hard when the network learned how to navigate the gaps. Widening the pipes made the reward signal less sparse and more consistent, which made leanring faster. [Yenchenlin. Deep Reinforcement Learning for Flappy Bird. Github, 2016](https://github.com/yenchenlin/DeepLearningFlappyBird) replicated the work of Chen. Neither of these papers show a score vs time chart, indicating that they ran into the same problem of catastrophic forgetting that I encountered, but hid the chart to preserve the quality of their research. [markub3327. flappy-bird-gymnasium, Github, 2023](https://github.com/markub3327/flappy-bird-gymnasium)[johnnycode8. Train the DQN Algorithm on Flappy Bird, Youtube, 2024](https://github.com/johnnycode8/dqn_pytorch) use a dueling DQN approach in the simplified state space, but avoids showing the score vs time chart.

One popular method used to solve the issue of catastrophic forgetting is to move back to using a Q-Table to store the state. [SarvagaVaish. FlappyBirdRL, Github, 2014](https://github.com/SarvagyaVaish/FlappyBirdRL) [kyokin78. rl-flappybird, Github, 2019](https://github.com/kyokin78/rl-flappybird) both compressed the state space by storing the relative difference in heights instead of the absolute positions, allowing for a more compressed state space and completely bypassing the problem of catastrophic forgetting.

## Methods

Stable-baselines3 is a library for reinforcement learning, providing wrappers and environments for many common reinforcement learning problems. It is a fork of the open-baselines library started by OpenAI in 2015, that is currently stale. I used the DQN algorithm that stable-baselines3 provides. The DQN algorithm in stable-baselines3 makes two improvements over the original DQN algorithm: (1) A double DQN approach to reduce the overestimation of Q-values, (2) A dueling DQN approach to more efficiently represent the Q-value for each state-action.

### Double DQN
Double DQN attempts to solve the overestimation of Q-values that happen implicitly when applying the TD-update. The original update is:

$$Q(s,a,\theta) \leftarrow r + \gamma \max_{a'} Q(s',a',\theta)$$

`<finish reasoning here>`
### Dueling DQN
The Dueling DQN improves the efficiency of the network approximation by separating the Q-value into two components: the state value $V(s)$ and the advantage function $A(s,a)$. The state value is the mean Q-value of all actions, and the advantage function is the deviation of the Q-value of each action from the state value. This was emprically shown to improve the expressiveness of the Q network.

### Results

| Name | Mean Score (1000 runs) | Std Score (1000 runs) |
|------|------------------------|-----------------------|
| 2 layers 1.3M steps | 172| 192 |
| 3 layers 2M steps | 567| 388 |
| 4 layers 2M steps | 416 | 360 |
| 5 layers 2M steps | 689 | 366 |
| 6 layers 2M steps | 877 | 342 |
| handcrafted agent | **954** | **170** |

### Stability of Learning
![dqn_scores](./paper/dqn_scores.png)
[Video of version 2](videos/dqn_flappybird_v2_20250103-030423.mp4)
- Seems that more layers helped generalize faster
  - Could be due to the fact that the agent is able to learn more complex patterns
- Catastrophic forgetting
  - evident in all prior work
- Mnih et al. 2013,
  - obscures this by choosing optimistic charts
    - probably took multiple attempts
  - arbitrary q-value chart to show "stability"
- No other prior work shows catastrophic forgetting
  - Including this chart makes the research look less valuable


### Further Work
- Learn directly from pixels
- Experiment with transformer models
- Experiment with Partially Observable Environments
- Try Hindsight Experience Replay
- Try Policy-based methods
- Duplicated neurons with different learning rates

## Conclusion
- Stable-baselines3 is a powerful tool for reinforcement learning
- Prior work heavily overstates the stability of learning
- Catastrophic forgetting is a common problem
  - function approximation with dynamic environments

## Postmortem
- should have started with replicating prior work
- had trouble deciding when to stop
  - interesting follow-up questions
  - deadlines exist
- update results differently
  - time-based instead of milestone-based
