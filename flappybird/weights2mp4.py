"""
This script takes in a policy and outputs a video of the agent playing the game.
The policy must be defined in the policy method and should return an action
provided the state.
"""

import gymnasium as gym
import flappy_bird_env
import stable_baselines3
from moviepy import VideoFileClip, concatenate_videoclips

def load_model(model_path):
    return stable_baselines3.DQN.load(model_path)

def decide_action(model, obs):
    return model.predict(obs, deterministic=True)

def create_environment():
    return gym.make("FlappyBird-v0", use_lidar=False, render_mode="rgb_array", audio_on=True, score_limit=1000)

def record_gameplay(env, model):
    obs, info = env.reset()
    while True:
        action = int(decide_action(model, obs)[0])
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()

def process_video(input_path, output_path):
    try:
        video = VideoFileClip(input_path)
        total_duration = video.duration

        first_segment = video.subclip(0, 10)
        middle_segment = video.subclip(10, max(10, total_duration - 3)).with_fps(600, change_duration=True)
        last_segment = video.subclip(max(10, total_duration - 3), total_duration)

        final_video = concatenate_videoclips([first_segment, middle_segment, last_segment])
        final_video.write_videofile(output_path, codec="libx264")
    except Exception as e:
        print(f"Error processing video: {e}")

def main():
    model_path = "/Users/ox/Documents/flappybird-monorepo/data/dqn_flappybird_v1_21700000_steps.zip"
    video_folder = "videos"
    input_video = f"{video_folder}/rl-video-episode-0.mp4"
    output_video = f"{video_folder}/output_video.mp4"

    model = load_model(model_path)
    env = create_environment()
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: True)

    record_gameplay(env, model)
    process_video(input_video, output_video)

if __name__ == "__main__":
    main()
