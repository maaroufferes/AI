import gymnasium as gym


class TimePenalty(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= 0.05
        return obs, reward, terminated, truncated, info

class HoverBonus(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if abs(obs[4]) < 0.2 and abs(obs[2]) < 0.5 and abs(obs[3]) < 0.5:
            reward += 1
        return obs, reward, terminated, truncated, info