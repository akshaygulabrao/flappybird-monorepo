from matplotlib.dviread import Box
from torchrl.modules import EGreedyModule, MLP, QValueModule
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
import flappy_bird_gymnasium
import gymnasium
import torchrl.data
import torch
assert flappy_bird_gymnasium

env = gymnasium.make(
    "FlappyBird-v0",
    audio_on=True,
    use_lidar=False,
    normalize_obs=True,
    score_limit=10,
)



observation_spec = torchrl.data.tensor_specs.Bounded(shape=(12,),dtype=torch.float32,low=0,high=1)
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
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))