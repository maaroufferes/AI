import gymnasium as gym

env=gym.make("LunarLander-v3",render_mode="human")
obs,info=env.reset()

for i in range (0,1000):
    action=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action)
    print(obs,reward)
    if terminated or truncated:
        obs,info=env.reset()
env.close()