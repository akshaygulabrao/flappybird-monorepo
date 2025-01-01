# Reinforcement Learning Techniques in Flappy Bird
Reinforcement learning allows the systematic search of solutions in a problem space. They allow the discover of solutions to problems that are counterintuitive to humans. Reinforcement learning has been the foundation of breakthroughs in go, chess, and protein folding [^1][^2][^3]. This paper provides a bottom-up approach to reinforcement learning by implementing an agent that plays Flappy Bird[^4]. Flappy bird is a simple mobile game from 2015. It is an excellent environment for reinforcement learning because of its simple game mechanics and discrete action space. Here the state space iteratively becomes more sophisticated, and I use different function approximation algorithms to address this

I progress through three stages:
1. A minimal implementation using a handcrafted binary state space
2. An intermediate approach using a simplified observation space
3. A full implementation learning directly from pixel data

## Introduction to Reinforcement Learning
In reinforcement learning, an agent lives in an environment specified by an event loop. The agent must observe the relationship between its actions and the environment's response. The environment's response consists of two things: (1) a reward signal, and (2) a new state.

``` python
observation = env.get_initial_state()
while True:
    action = agent.act(observation)
    observation, reward = env.step(action)
    agent.learn(observation, reward)
```
### Markov Decision Processes
Formally, the environment is defined by four things: (1) the state space $S$, (2) the action space $A$, (3) the transition function $P$, and (4) the reward function $R$.

- The state space $S$ is the set of all possible states the agent can observe.
- The action space $A$ is the set of all possible actions the agent can take.
- The transition function $P(s'|s,a) \rightarrow [0,1]$ is the probability of transitioning from state $s \in S$ to state $s' \in S$ given action $a \in A$.
- The reward function $R(s,a) \rightarrow \mathbb{R}$ is the reward the agent receives for $a$ in $s$

The goal for all agents is to maximize the return. The return is the sum of all cumulative rewards over time. The return is defined as
$$\sum_{k=0}^\infty\gamma^kR_{t+k+1}$$
where $\gamma$ is the discount rate, which places more emphasis on immediate rewards.

When the agent interacts with the environment, its interaction can be described as a sequence of states, actions, and rewards known as a trajectory:
$$
(s_0, a_0, r_0, s_1, a_1, r_1, \cdots)
$$
### Policy
The agent's behavior is characterized by a policy $\pi$. The policy is a mapping from the state space to the action space.
$$\pi : \mathcal{S} \rightarrow \mathcal{A}$$
The policy can also be probability based
$$\pi(a|s) \rightarrow [0,1] \quad s\in\mathcal{S}, a \in \mathcal{A}$$

A popular policy for reinforcement learning tasks is $\epsilon-\text{greedy}$, where the agent selects a random actions with probability $\epsilon$ and chooses the best action with probability $1-\epsilon$, allowing a consistent balance between exploration and exploitation. The epsilon parameter decreses during training to select the best actions more often later in training, and use random actions near the beginning of training.


### Q-Learning
If the transition probabilities $p$ was known, it would be possible to directly compute the Q-Function with the Bellman Equation.

In Q-Learning, the agent's goal is to create a mapping that predicts the value of taking action $a$ in state $s$ by sampling trajectories from the environment.  The agent assigns the value based on three components:
1. the reward received in the current state
2. the value of the best next state-action $\max_aQ(s',a)$, where $s'$ is the next state, and $a$ is the action which results in the highest value
3. value of the current state-action $Q(s,a)$

The Q-Learning update at time $t$ is
$$
Q(s,a) = Q(s,a) + \alpha[R_t + \gamma\max_aQ(s',a) - Q(s,a)]
$$
where $\alpha = 1 \times 10^{-5}$, and $\gamma = 0.9$.

While sampling trajectories, the agent must systematically choose between exploring different state-actions and exploiting the best state-action to achieve variance in every trajectory. The most popular policy in Q-Learning scenarios is $\epsilon\text{-decay}$, where the agent performs a random action with probability $\epsilon$, and performs its best action, $Q(s,a)$, with probability $1 - \epsilon$. The $\epsilon-\text{decay}$ allows significant exploration in the beginning and significant exploitation in the end. Episode $i$ has an $\epsilon$ of
$$
\epsilon = \max(0.01, 0.5 \times 0.99^i)
$$



This completes a theoretical framework to reinforcement learning, I now apply it to iteratively more complex scenarios, starting with a minimal binary state space.

## Step 1: Handcrafted State Space
To properly compare the performance of different function approximation algorithms, I first implement the agent using a handcrafted binary state space and bezier curves. By computing the optimal control point for


One of the issues I encountered throughout this step was parameter hypertuning. The learning rate $\alpha$ and $\epsilon-\text{decay}$ functions needed to be carefully chosen after lots of failure. In the first step, the state space is handcrafted $s \in \{0,1\}$. There are two states: (0) where not flapping is the optimal choice and (1) where flapping is optimal. The action space consists of 2 discrete actions, $a \in \{\text{no\_flap}, \text{flap}\}$. Here the agent must simply learn that $0 \rightarrow \text{no\_flap}$ and $1 \rightarrow \text{flap}$.

For this step, the learning rate $\alpha = 1 \times 10^{-2}$, $\gamma = 0.999$. I utilized $\epsilon-\text{decay}$ with 10000 timesteps. The $\epsilon$ for episode $i$ was
$$
\epsilon(i) = \max(0.01, .999^{i})
$$

### Comparison with Atari Paper
The popular Atari paper introducing DQN to the field specifically mentions that they did not run into stability issues or hyperparameter sensitivity [^10]. I could be doing something wrong, or the function approximation encourages far more exploration than the $\epsilon-\text{greedy}$ policy would normally allow. It is possible that their deep network was capable of adapting to their hyperparameter choices. The function approximation state space may have emphasized a large amount of exploration during the early stages when the network was untrained, that was not fully represented during my approach.

### Observation State Space


## References
[^1]: [Mastering the Game of Go without Human Knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)\
[^2]: [Mastering Chess and Shogi by Planning with a Tree Search](https://arxiv.org/pdf/1712.01815.pdf)\
[^3]: [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)\
[^4]: [Flappy Bird](https://flappy-bird.co)\
[^5]: This is a continuation of a project that I started two years ago, a final project for a deep learning course. I didn't end up getting it working then, but have been working on it since.
[^6]: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)
[^7]: [Flappy Bird Gymnasium](https://github.com/Kautenja/flappy-bird-gymnasium)\
[^8]: [Flappy Bird State Vector](https://github.com/markub3327/flappy-bird-gymnasium)\
[^9]: Reinforcement learning: an introduction Section
[^10]: [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
