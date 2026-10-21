import json
import os

import matplotlib.pyplot as plt


def show_training_graph(stats_filename, graph_filename=None):
    folder = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(folder, stats_filename)
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"Training statistics not found: {stats_path}\n"
            "Run the matching training script first."
        )

    with open(stats_path, encoding="utf-8") as stats_file:
        stats = json.load(stats_file)

    lengths = stats["average_episode_lengths"]
    rewards = stats["average_episode_rewards"]
    if not lengths or not rewards:
        raise ValueError(
            "The training statistics contain no completed episodes to plot."
        )
    if len(lengths) != len(rewards):
        raise ValueError(
            "Episode length and reward statistics have different lengths."
        )

    episodes = range(1, len(rewards) + 1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(episodes, lengths)
    axes[0].set_title("Average Episode Length (100-episode moving average)")
    axes[0].set_ylabel("Steps")
    axes[0].grid(True)

    axes[1].plot(episodes, rewards)
    axes[1].set_title("Average Episode Reward (100-episode moving average)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True)

    fig.tight_layout()
    original_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    annotations = []
    for ax in axes:
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->"},
        )
        annotation.set_visible(False)
        annotations.append(annotation)
        for line in ax.lines:
            line.set_picker(5)

    def show_point(event):
        if event.inaxes not in axes or event.x is None or event.y is None:
            return

        ax_index = list(axes).index(event.inaxes)
        ax = event.inaxes
        annotation = annotations[ax_index]
        nearest = None

        for line in ax.lines:
            contains, details = line.contains(event)
            if not contains:
                continue
            for index in details["ind"]:
                x, y = line.get_data()
                pixel_x, pixel_y = ax.transData.transform((x[index], y[index]))
                distance = (pixel_x - event.x) ** 2 + (pixel_y - event.y) ** 2
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, x[index], y[index])

        if nearest is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        _, x, y = nearest
        annotation.xy = (x, y)
        annotation.set_text(f"Episode: {x:.0f}\nAverage: {y:.2f}")
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    def zoom(event):
        ax = event.inaxes
        if ax not in axes or event.xdata is None or event.ydata is None:
            return

        factor = 1 / 1.2 if event.button == "up" else 1.2
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        ax.set_xlim(
            event.xdata + (left - event.xdata) * factor,
            event.xdata + (right - event.xdata) * factor,
        )
        ax.set_ylim(
            event.ydata + (bottom - event.ydata) * factor,
            event.ydata + (top - event.ydata) * factor,
        )
        fig.canvas.draw_idle()

    def reset_view(event):
        if event.key and event.key.lower() == "r":
            for ax, (x_limits, y_limits) in zip(axes, original_limits):
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", show_point)
    fig.canvas.mpl_connect("scroll_event", zoom)
    fig.canvas.mpl_connect("key_press_event", reset_view)

    print("Scroll to zoom, hover over the lines for values, and press R to reset.")
    plt.show()
