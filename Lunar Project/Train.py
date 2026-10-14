import gymnasium as gym
from stable_baselines3 import PPO
import os

folder = os.path.dirname(os.path.abspath(__file__))


env=gym.make("LunarLander-v3")
model=PPO("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=1_000_000)


path = os.path.join(folder, "ppo_lander")
model.save(path)
print("SAVED TO:", path + ".zip")
