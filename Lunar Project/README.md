# Lunar Lander: PPO Training and Analysis

This project trains and evaluates reinforcement-learning agents in Gymnasium's
`LunarLander-v3` environment. It compares a standard Proximal Policy
Optimization (PPO) agent with two reward-shaping variations, records selected
landings as videos, and provides interactive charts for exploring training
progress.

## Features

- **Three PPO training experiments:** a baseline agent, an agent with a bonus
  for low-motion/upright states, and an agent with a per-step time penalty.
- **Training metrics:** rolling averages of episode reward and length, computed
  over up to the most recent 100 completed episodes.
- **Saved models and metrics:** each experiment writes a Stable-Baselines3 model
  archive and a JSON file containing its training metrics.
- **Interactive charts:** view reward and episode length over training, inspect
  values by hovering over a curve, zoom with the mouse wheel, and press `R` to
  reset the view.
- **Landing evaluation videos:** evaluate a trained policy on 1,000 seeded
  episodes, select five representative results, and record those episodes.

## How it works

Each training script creates the Lunar Lander environment and trains an
`MlpPolicy` using Stable-Baselines3 PPO for **1,000,000 environment steps**.
Gymnasium's `Monitor` wrapper reports completed-episode results to a callback.
The callback maintains rolling episode reward and length averages, which are
saved to JSON after training. The trained policy is saved as a `.zip` archive.

The experiments differ in how rewards are calculated:

| Experiment | Environment | Reward behavior | Model | Statistics |
| --- | --- | --- | --- | --- |
| Baseline | `LunarLander-v3` | Uses the environment's standard rewards. | `ppo_lander.zip` | `training_stats.json` |
| Hover bonus | `LunarLander-v3` with `HoverBonus` | Adds a reward bonus when the lander is upright and its horizontal and vertical velocities are low. | `ppo_hover.zip` | `training_stats_hover.json` |
| Time penalty | `LunarLander-v3` with `TimePenalty` | Subtracts `0.1` reward on each step to encourage shorter episodes. | `ppo_timer.zip` | `training_stats_time.json` |

`Record_Landing_Progress.py` loads a model, runs deterministic predictions
across 1,000 episodes with different seeds, and records five selected episodes:
the lowest- and highest-reward episodes, plus episodes whose rewards are closest
to halfway between the minimum and zero, zero, and halfway between zero and the
maximum. Videos are saved in a timestamped run folder. The script currently
loads `ppo_timer.zip`; edit `MODEL_PATH` and `VIDEO_FOLDER` in that script to
evaluate another experiment.

## Getting started

Use Python with the project dependencies installed. From this directory, for
example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "gymnasium[box2d]" stable-baselines3 matplotlib
```

Train one experiment:

```powershell
python Train.py
# Or: python Train_hover.py
# Or: python Train_timer.py
```

Each training run can take a while. After training, show its chart:

```powershell
python Plot_Training_Graph.py
# Or: python Plot_Training_Graph_Hover.py
# Or: python Plot_Training_Graph_Time.py
```

The chart opens in a Matplotlib window. Hover over a line to inspect a point,
scroll to zoom, and press `R` to restore the original chart limits.

To evaluate the default time-penalty model and record videos:

```powershell
python Record_Landing_Progress.py
```

The recorder requires the configured model archive to exist. Change `MODEL_PATH`
in `Record_Landing_Progress.py` if you want to use a different saved policy.

## Files and folders

### Source files

| File | Purpose |
| --- | --- |
| `Train.py` | Trains the baseline PPO agent; writes `training_stats.json` and `ppo_lander.zip`. |
| `Train_hover.py` | Trains PPO with the `HoverBonus` reward wrapper; writes `training_stats_hover.json` and `ppo_hover.zip`. |
| `Train_timer.py` | Trains PPO with the `TimePenalty` reward wrapper; writes `training_stats_time.json` and `ppo_timer.zip`. |
| `Wrapper.py` | Defines the Gymnasium reward wrappers shared by the modified-reward experiments. |
| `Training_Graph_Viewer.py` | Loads a training-statistics JSON file and displays interactive reward and episode-length plots. |
| `Plot_Training_Graph.py` | Opens the baseline training chart using `training_stats.json`. |
| `Plot_Training_Graph_Hover.py` | Opens the hover-bonus chart using `training_stats_hover.json`. |
| `Plot_Training_Graph_Time.py` | Opens the time-penalty chart using `training_stats_time.json`. |
| `Record_Landing_Progress.py` | Evaluates the configured saved model, selects five representative episodes, and records them as videos. |
| `README.md` | Documents the experiments, setup, usage, and files in this project. |

### Saved models and training data

| File | Purpose |
| --- | --- |
| `ppo_lander.zip` | Saved baseline PPO policy; produced by `Train.py`. |
| `ppo_hover.zip` | Saved hover-bonus PPO policy; produced by `Train_hover.py`. |
| `ppo_timer.zip` | Saved time-penalty PPO policy; produced by `Train_timer.py` and loaded by the video recorder by default. |
| `training_stats.json` | Baseline rolling-average episode rewards and lengths. |
| `training_stats_hover.json` | Hover-bonus rolling-average episode rewards and lengths. |
| `training_stats_time.json` | Time-penalty rolling-average episode rewards and lengths. |

The statistics JSON files contain `average_episode_rewards` and
`average_episode_lengths` arrays. The values are rolling averages for completed
episodes, not raw per-step observations.

### Recorded videos

| Folder | Purpose |
| --- | --- |
| `landing_videos/` | Baseline-model evaluation videos. |
| `landing_hover_videos/` | Hover-bonus evaluation videos. |
| `landing_timer_videos/` | Time-penalty evaluation videos; this is the recorder's default output folder. |

Each folder can contain timestamped `run_YYYYMMDD_HHMMSS/` subfolders. A run
contains up to five `.mp4` recordings named for their selected reward category:
`lowest`, `halfway_to_zero`, `closest_to_zero`, `halfway_to_max`, and `highest`.
These videos are outputs of evaluation, not inputs needed to train the agents.

### Local environment and generated files

- `venv/` is a machine-specific Python virtual environment, not application
  source. Create a new environment and install the dependencies above rather
  than relying on the checked-in environment folder.
- `__pycache__/` contains Python bytecode generated when scripts are run; it is
  not required to run the project.

