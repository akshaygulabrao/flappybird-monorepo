# DQN versions

## DQN v1
1. Stable Baselines 3 implementation:
    - buffer_size 50K
    - gamma 0.99
    - learning_rate $5 \times 10^{-4}$
    - batch_size 32
    - epsilon_start $1 \times 10^{-4}$
    - prioritized_replay False
    - target_network_update_freq 10K
    - Double DQN & Dueling DQN
2. Network architecture:
    - [12, 64, 64, 2]
    - ReLU
    - Adam

- Very unstable training
- Reached high score of 886 randomly during training
- 1.3M steps
- 20.82 mean score
- 15.83 std score

## DQN v2
1. Increased neural network layers from 2 layers to 3 layers.
2. The new network architecture is [12,64,64,2]
- 5M steps

- 566.6 mean score
- 388.1 std score

## DQN v3

## DQN v4
