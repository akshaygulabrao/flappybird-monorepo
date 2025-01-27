"""
Benchmarking the performance of all models in the paper.
Input:
A list of defined models
Output:
Markdown pastable table with the performance of each model.
"""
from pathlib import Path

import numpy as np
import stable_baselines3
from agents import DQNAgent, HandcraftedAgent
from tqdm import tqdm
from weights2mp4 import create_environment, record_gameplay


def evaluate_agent(agent):
    env = create_environment(render_mode=None)
    scores = []
    for i in tqdm(range(1000)):
        score = record_gameplay(env, agent.decide)
        scores.append(score)
    return {'name': agent.name, 'mean score (1000 runs)': np.mean(scores), 'std score (1000 runs)': np.std(scores), 'training_time(hours)': 200}

# Function to print markdown table
def print_markdown_table(evals):
    # Print the header
    print("| Name | Mean Score (1000 runs) | Std Score (1000 runs) |")
    print("|------|------------------------|-----------------------|")

    # Print each row
    for eval in evals:
        print(f"| {eval['name']} | {eval['mean score (1000 runs)']:.2f} | {eval['std score (1000 runs)']:.2f} |")

# Evaluate agents
evals = []
model0 = HandcraftedAgent()
model1 = DQNAgent(Path("data/dqn_flappybird_v1_21700000_steps.zip"))

evals.append(evaluate_agent(model0))

# Print the markdown table
print_markdown_table(evals)
