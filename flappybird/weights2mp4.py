"""
This script takes in a policy and outputs a video of the agent playing the game.
The policy must be defined in the policy method and should return an action
provided the state.
"""

import glob
import os
import time
from pathlib import Path

import flappy_bird_env
import gymnasium as gym
import stable_baselines3
from agents import BaseAgent, DQNAgent, HandcraftedAgent
from moviepy import VideoFileClip, concatenate_videoclips


def create_environment(render_mode=None):
    return gym.make("FlappyBird-v0", use_lidar=False, render_mode=render_mode, audio_on=True, score_limit=1000)

def record_gameplay(env, decision_function):
    obs, info = env.reset()
    while True:
        action = decision_function(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()
    return info['score']

def process_video(input_path, output_path):
    try:
        video = VideoFileClip(input_path)
        total_duration = video.duration

        first_segment = video.subclipped(0, 10)
        middle_segment = video.subclipped(10, max(10, total_duration - 3)).with_fps(600, change_duration=True)
        last_segment = video.subclipped(max(10, total_duration - 3), total_duration)

        final_video = concatenate_videoclips([first_segment, middle_segment, last_segment])
        final_video.write_videofile(output_path, codec="libx264")
    except Exception as e:
        print(f"Error processing video: {e}")

def get_latest_video_file(folder, pattern):
    files = glob.glob(f"{folder}/{pattern}")
    if not files:
        raise FileNotFoundError("No video files found.")
    return max(files, key=os.path.getctime)

def main(model_path="data/dqn_flappybird_v1_1300000_steps.zip", video_folder="videos"):
    weights_file_name = Path(model_path).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_video = f"{video_folder}/{weights_file_name}_{timestamp}.mp4"

    agent = DQNAgent(model_path)
    env = create_environment(render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix=weights_file_name
    )

    try:
        input_video = get_latest_video_file(video_folder, f"{weights_file_name}*")
        record_gameplay(env, agent.decide)
        process_video(input_video, output_video)
    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
