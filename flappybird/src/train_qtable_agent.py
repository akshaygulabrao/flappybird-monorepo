import gymnasium
import flappy_bird_gymnasium
from src.qtable_agent import Agent

agent = Agent()


env = gymnasium.make(
    "FlappyBird-v0",
    audio_on=True,
    render_mode=None,
    use_lidar=False,
    normalize_obs=False,
    score_limit=1000,
)

for i in range(1000):
    obs, _ = env.reset()
    while True:
        action = agent.act(obs)
        obs, _, done, term, info = env.step(action)
        if done or term:
            break
agent.save_q_table()
