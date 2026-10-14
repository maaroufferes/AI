import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


name = sys.argv[1] if len(sys.argv) > 1 else "ppo_timer"
folder = os.path.dirname(os.path.abspath(__file__))
model = PPO.load(os.path.join(folder, name))

env=gym.make("LunarLander-v3",render_mode="rgb_array")
env=RecordVideo(env,video_folder=os.path.join(folder,"videos"),name_prefix=name,episode_trigger=lambda ep:True)

episodes=0
obs,info=env.reset()
while episodes<100:
    action,_= model.predict(obs, deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)
    if terminated or truncated:
        episodes+=1
        obs,info=env.reset()


env.close()