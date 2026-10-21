import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO


FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FOLDER, "ppo_timer.zip")
VIDEO_FOLDER = os.path.join(FOLDER, "landing_timer_videos")
NUMBER_OF_VIDEOS = 5
NUMBER_OF_CANDIDATE_EPISODES = 1000
SEED = 42


def evaluate_episodes(model):
    env = gym.make("LunarLander-v3")
    episodes = []

    try:
        for episode_index in range(NUMBER_OF_CANDIDATE_EPISODES):
            seed = SEED + episode_index
            observation, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

            episodes.append((total_reward, seed))
    finally:
        env.close()

    return episodes


def select_episodes(episodes):
    if len(episodes) < NUMBER_OF_VIDEOS:
        raise RuntimeError(
            f"Need at least {NUMBER_OF_VIDEOS} evaluated episodes, "
            f"but only got {len(episodes)}."
        )

    lowest = min(episodes, key=lambda item: item[0])
    highest = max(episodes, key=lambda item: item[0])
    minimum_reward = lowest[0]
    maximum_reward = highest[0]

    targets = [
        ("lowest", minimum_reward, lowest),
        ("halfway_to_zero", minimum_reward / 2, None),
        ("closest_to_zero", 0.0, None),
        ("halfway_to_max", maximum_reward / 2, None),
        ("highest", maximum_reward, highest),
    ]

    selected = []
    used_seeds = {lowest[1], highest[1]}
    for label, target, fixed_episode in targets:
        if fixed_episode is not None:
            episode = fixed_episode
        else:
            remaining = [
                episode for episode in episodes if episode[1] not in used_seeds
            ]
            episode = min(remaining, key=lambda item: abs(item[0] - target))
            used_seeds.add(episode[1])

        reward, seed = episode
        selected.append((label, target, reward, seed))

    return selected


def record_episode(model, video_folder, index, label, target_reward, seed):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda _: True,
        name_prefix=f"{index:02d}_{label}",
    )

    try:
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
    finally:
        env.close()

    print(
        f"Video {index}/{NUMBER_OF_VIDEOS}: {label}, "
        f"reward {total_reward:.1f} (target {target_reward:.1f})"
    )


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run Train.py first, or change MODEL_PATH to the model you trained."
        )

    model = PPO.load(MODEL_PATH)
    print(f"Evaluating {NUMBER_OF_CANDIDATE_EPISODES} attempts...")
    episodes = evaluate_episodes(model)
    selected_episodes = select_episodes(episodes)

    run_folder = os.path.join(
        VIDEO_FOLDER,
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_folder, exist_ok=False)

    for index, (label, target, reward, seed) in enumerate(
        selected_episodes, start=1
    ):
        print(f"Selected {label}: target {target:.1f}, attempt reward {reward:.1f}")
        record_episode(model, run_folder, index, label, target, seed)

    print(f"Videos saved in: {run_folder}")


if __name__ == "__main__":
    main()
