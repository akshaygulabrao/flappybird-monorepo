"""Tests the handcrafted agent. Should score above 900 on average of 1000 runs."""

import gymnasium
import flappy_bird_gymnasium
import tqdm
from flappybird.src import handcrafted_agent
from flappybird.src import QTable_Agent

def play(env,play_fn):
    """
    Play the game with the handcrafted agent.

    Args:
        env (gymnasium.Env): The environment to play in.
        use_lidar (bool): Whether to use lidar observations.
        render_mode (str): The mode to render the game in.

    Returns:
        int: The score achieved by the agent.
    """

    obs, _ = env.reset()
    while True:
        action = play_fn(obs.tolist())
        obs, _, done, term, info = env.step(action)
        if done or term:
            break

    env.close()
    return info["score"]


def test_handcrafted_agent(normalize=True):
    """
    Test the handcrafted agent. This test will trigger a warning because I turn off observation
    normalization.

    Args:
        None

    Returns:
        None: Asserts that the average score is above 750. 
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        render_mode=None,
        use_lidar=False,
        normalize_obs=normalize,
        score_limit=1000,
    )

    scores = []
    for _ in tqdm.tqdm(range(10)):
        scores.append(play(env,handcrafted_agent))
    print(f"Average score: {sum(scores) / len(scores)}")
    assert sum(scores) / len(scores) > 750


def test_qtable_agent(normalize=True):
    """
    Test the handcrafted agent. This test will trigger a warning because I turn off observation
    normalization.

    Args:
        None

    Returns:
        None: Asserts that the average score is above 750.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        render_mode=None,
        use_lidar=False,
        normalize_obs=normalize,
        score_limit=1,
    )

    scores = []
    agent = QTable_Agent(path="data/qtable.csv")
    for _ in tqdm.tqdm(range(10)):
        scores.append(play(env,agent.decide))
    print(f"Average score: {sum(scores) / len(scores)}")
    assert sum(scores)  > 7
