import gymnasium as gym
import flappy_bird_gymnasium 
import stable_baselines3 as sb3
import yaml

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)



env = gym.make("FlappyBird-v0", use_lidar=False)

if config["train-from-scratch"]:
    model = sb3.DQN(policy="MlpPolicy",
        env=env,
        learning_rate=config["model"]["learning_rate"],
        buffer_size=config["model"]["buffer_size"],  # 1e6
        learning_starts=config["model"]["learning_starts"],
        batch_size=config["model"]["batch_size"],
        tau=config["model"]["tau"],
        gamma= config["model"]["gamma"],
        train_freq=config["model"]["train_freq"],
        gradient_steps=config["model"]["gradient_steps"],
        replay_buffer_class=config["model"]["replay_buffer_class"],
        replay_buffer_kwargs=config["model"]["replay_buffer_kwargs"],
        optimize_memory_usage=config["model"]["optimize_memory_usage"],
        target_update_interval=config["model"]["target_update_interval"],
        exploration_fraction=config["model"]["exploration_fraction"],
        exploration_initial_eps=config["model"]["exploration_initial_eps"],
        exploration_final_eps=config["model"]["exploration_final_eps"],
        max_grad_norm=config["model"]["max_grad_norm"],
        stats_window_size=config["model"]["stats_window_size"],
        tensorboard_log=config["model"]["tensorboard_log"],
        policy_kwargs=config["model"]["policy_kwargs"],
        verbose=config["model"]["verbose"],
        seed=config["model"]["seed"],
        device=config["model"]["device"],
        _init_setup_model=config["model"]["_init_setup_model"])

    model.learn(total_timesteps=config["total_timesteps"], 
                tb_log_name=config["tb_log_name"],
                reset_num_timesteps=config["reset_num_timesteps"])

    model.save("dqn_flappy_bird")
else:
    model = sb3.DQN.load("dqn_flappy_bird")
