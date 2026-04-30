import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BASE_COLOR_PULLING = "#7fcdbb"
BASE_COLOR_EXTRA = "#edf8b1"
LIGHTD_COLOR_PULLING = "#d9ebd4"
LIGHTD_COLOR_EXTRA = "#f8ac8c"

HATCH_1 = "||"
HATCH_2 = "--"
HATCH_3 = "\\"
HATCH_4 = "/"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 0.3

ALGO_ORDER = ["BTS", "P2P", "LinUCB", "EpsGreedy"]
ALGO_LABELS = {
    "BTS": "BTS",
    "P2P": "P2P",
    "LinUCB": "LinUCB",
    "EpsGreedy": "Cache-aware $\\varepsilon$-Bandit",
}
ALGO_COLORS = {
    "BTS": LIGHTD_COLOR_EXTRA,
    "P2P": LIGHTD_COLOR_PULLING,
    "LinUCB": BASE_COLOR_EXTRA,
    "EpsGreedy": BASE_COLOR_PULLING,
}
ALGO_HATCHES = {
    "BTS": HATCH_4,
    "P2P": HATCH_3,
    "LinUCB": HATCH_2,
    "EpsGreedy": HATCH_1,
}


def _setup_fonts():
    return {"weight": "normal", "size": 24}


def _load_metrics():
    x_labels = [100, 200, 500, 1000]
    metrics = {
        "BTS": [2, 4, 30, 30],
        "P2P": [2, 6, 40, 42],
        "LinUCB": [2, 6, 40, 42],
        "EpsGreedy": [2, 4, 32, 35],
    }
    return x_labels, metrics


def _plot_grouped_bars(x_labels, metrics, xlabel, ylabel, filename):
    font = _setup_fonts()

    ind = np.arange(len(x_labels))
    bar_width = 0.18

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i, algo in enumerate(ALGO_ORDER):
        offset = (i - 1.5) * bar_width
        ax.bar(
            ind + offset,
            height=metrics[algo],
            width=bar_width,
            color=ALGO_COLORS[algo],
            linewidth=0,
            edgecolor="black",
            label=ALGO_LABELS[algo],
            hatch=ALGO_HATCHES[algo],
        )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#A8BAC4", lw=1.2)
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font, loc="center")
    plt.tick_params(labelsize=18)

    ax.set_xticks(ind)
    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 51, 10))

    out_path = os.path.join(os.path.dirname(__file__), filename)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    xs, metrics = _load_metrics()
    _plot_grouped_bars(
        xs,
        metrics,
        xlabel="Number of user requests",
        ylabel="Deployment overhead (GB)",
        filename="online_default_deployment_overhead.pdf",
    )
