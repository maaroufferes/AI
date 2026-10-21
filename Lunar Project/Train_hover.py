from collections import deque
import json
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from Wrapper import HoverBonus

folder = os.path.dirname(os.path.abspath(__file__))


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, window_size=100):
        super().__init__()
        self.rew_moy = []
        self.len_moy = []
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_lengths = deque(maxlen=window_size)

    def _on_step(self):
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.recent_rewards.append(episode["r"])
                self.recent_lengths.append(episode["l"])
                self.rew_moy.append(
                    sum(self.recent_rewards) / len(self.recent_rewards)
                )
                self.len_moy.append(
                    sum(self.recent_lengths) / len(self.recent_lengths)
                )
        return True


env = Monitor(HoverBonus(gym.make("LunarLander-v3")))
model = PPO("MlpPolicy", env, verbose=1)
stats_callback = EpisodeStatsCallback()
model.learn(total_timesteps=1_000_000, callback=stats_callback)
env.close()

stats_path = os.path.join(folder, "training_stats_hover.json")
with open(stats_path, "w", encoding="utf-8") as stats_file:
    json.dump(
        {
            "average_episode_lengths": stats_callback.len_moy,
            "average_episode_rewards": stats_callback.rew_moy,
        },
        stats_file,
    )
print("Training statistics saved to:", stats_path)

path = os.path.join(folder, "ppo_hover")
model.save(path)
print("SAVED TO:", path + ".zip")