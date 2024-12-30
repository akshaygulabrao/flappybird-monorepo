# Experiments
The following details experiments run on the flappy bird environment. They should correspond under the experiment header inside the logs subdirectory.

## dqn-0
This is a baseline run using the native stable-baselines3 implementation. The metrics for 100 runs
after training for 1 million episodes. I use a low learning rate of 1e-4 because random actions impact 
the training run a lot.

* Average Reward: 30.75 
* Std Reward: 26.26
* Final Score: 5.8