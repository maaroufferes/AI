import gymnasium as gym
from stable_baselines3 import PPO
from Wrapper import TimePenalty
import os

folder = os.path.dirname(os.path.abspath(__file__))

env = TimePenalty(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

path = os.path.join(folder, "ppo_timer")
model.save(path)
print("SAVED TO:", path + ".zip")