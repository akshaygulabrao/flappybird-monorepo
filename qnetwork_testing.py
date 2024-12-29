from matplotlib.dviread import Box
from torchrl.modules import EGreedyModule, MLP, QValueModule
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
import flappy_bird_gymnasium
import gymnasium
import torchrl.data
import torch
assert flappy_bird_gymnasium
from torchrl.envs import GymWrapper
from torchrl.objectives import DQNLoss, SoftUpdate

env = GymWrapper(
    gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        use_lidar=False,
        normalize_obs=True,
        score_limit=10,
    )
)

action_spec = torchrl.data.tensor_specs.Categorical(n=2)

value_mlp = MLP(in_features=12, out_features=2, num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=action_spec))
exploration_module = EGreedyModule(
    action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)