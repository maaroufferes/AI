import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else "ppo_lander"
folder = os.path.dirname(os.path.abspath(__file__))
model = PPO.load(os.path.join(folder, name))


env=gym.make("LunarLander-v3")
scores=[]
total=0
obs,info=env.reset()

while len(scores)<10:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)
    total+=reward
    if terminated or truncated:
        scores.append(total)
        print("episode score:", round(total, 1))
        total=0
        obs,info=env.reset()

env.close()
print("Average over 10 episodes:", np.mean(scores))