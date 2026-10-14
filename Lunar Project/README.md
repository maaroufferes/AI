# Lunar Lander: Reinforcement Learning and Reward Shaping

Training an AI agent to land a rocket in the Lunar Lander game using reinforcement learning, then experimenting with **reward shaping** to see how changing the reward changes the agent's behavior, including how it can exploit a badly designed reward (**reward hacking**).

## What this project does

The agent plays Lunar Lander thousands of times. It gets points for good behavior (moving toward the pad, landing softly) and loses points for bad behavior (crashing, wasting fuel). Over time it learns to land on its own.

The project has three parts:

1. **Baseline:** train an agent with the default Lunar Lander reward.
2. **Break it:** add a flawed reward (+1 per frame while upright and slow) and observe the agent learning to hover instead of landing.
3. **Fix it:** design a reward that pays for *improvement between frames* plus a small time fee, so hovering can't be farmed.

## How it works

* **Environment:** `LunarLander-v3` from Gymnasium (Box2D physics). It runs at 50 frames per second.
* **Observation:** 8 numbers per frame: x position, height, x speed, y speed, tilt, spin, and two leg-contact flags.
* **Actions:** 4 choices: do nothing, fire left engine, fire main engine, fire right engine.
* **AI algorithm:** **PPO (Proximal Policy Optimization)** from Stable-Baselines3, using the `MlpPolicy` (a small neural network with two hidden layers of 64 units).
* **Reward shaping:** custom Gymnasium wrappers sit between the game and the agent and modify the reward each step.
* **Solved:** an average score of 200 or more over many episodes.

## Libraries used

|Library|Purpose|
|-|-|
|[Gymnasium](https://gymnasium.farama.org/) (with `box2d`)|The Lunar Lander environment, wrappers, video recording|
|[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)|PPO implementation|
|[PyTorch](https://pytorch.org/)|Neural network backend (installed with Stable-Baselines3)|
|[NumPy](https://numpy.org/)|Averaging scores|
|[MoviePy](https://zulko.github.io/moviepy/)|Writing video files for `RecordVideo`|

## Project structure

```
Lunar Project/
├── random\\\_test.py      # Watch a random agent crash (explore the environment)
├── train.py            # Train PPO on the default reward, saves ppo\\\_lander.zip
├── Test.py             # Evaluate a saved model over 10 episodes (plain env, no wrapper)
├── Record.py           # Record videos of a saved model
├── Wrapper.py      # Custom reward wrappers (TimePenalty, HoverBonus...)
├── train\\\_Time.py    # Train with the time-penalty wrapper, saves ppo\\\_time.zip
├── train\\\_hover.py      # Train with the broken HoverBonus reward, saves ppo\\\_hover.zip
└── videos/             # Recorded clips
```

## Custom reward wrappers (`Wrapper.py`)

|Wrapper|Idea|
|-|-|
|`TimePenalty`|Subtracts a small amount every frame so wasting time costs points.|
|`HoverBonus`|The **flawed** reward: +1 per frame while tilt and speeds are small. Hovering for 1,000 frames earns far more than landing, so the agent learns not to land.|

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv venv
venv\\\\Scripts\\\\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install "gymnasium\\\[box2d]" stable-baselines3 moviepy
```

If the Box2D install fails with an error mentioning `swig`, install swig first and retry.

## Usage

```bash
# 1. Watch a random agent
python random\\\_test.py

# 2. Train the baseline agent (about 20-30 minutes for 1,000,000 steps)
python train.py

# 3. Evaluate it (average over 10 episodes, plain environment)
python Test.py
python Test.py ppo\\\_lander

# 4. Record a video
python Record.py ppo\\\_lander

# 5. Train with custom rewards, then evaluate and record them
python train\\\_time.py
python train\\\_hover.py
python Test.py ppo\\\_hover
python Record.py ppo\\\_time
```

Models are always evaluated on the **plain environment without wrappers**, so a custom bonus cannot inflate the score.

## Results

Replace the placeholders with your own numbers after running each experiment. Training is random, so run each experiment at least twice.

|Model|Reward|Avg. score (10 episodes)|Behavior|
|-|-|-|-|
|`ppo\\\_lander`|Default|*your result*|*e.g. lands between the flags*|
|`ppo\\\_time`|Default + time penalty|*your result*|*your observation*|
|`ppo\\\_hover`|Flawed hover bonus|*your result*|*e.g. hovers, never lands*|
|`ppo\\\_fixed`|Improvement-based|*your result*|*your observation*|

Videos: *add links or GIFs here*

## Key takeaways

* **What you reward is what the agent learns.** The agent optimizes the reward you write, not the behavior you intended.
* **A high training score does not mean good behavior.** With the hover bonus, the training reward can rise while the rocket never lands. Always watch the video.
* **Reward improvement, not position.** Paying for the change between frames removes the incentive to sit still and farm points.
* **Evaluate honestly.** Test with wrappers off, over many episodes, and over more than one training run.

## Credits

* Lunar Lander environment: [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
* PPO: Schulman et al., 2017, "Proximal Policy Optimization Algorithms"
* Project idea and guide: Hari (@harixganesh), "Training an AI to land a rocket with reward shaping"

