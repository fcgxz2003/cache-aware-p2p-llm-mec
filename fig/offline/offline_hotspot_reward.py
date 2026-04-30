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


ALGO_ORDER = ["BTS", "P2P", "Knapsack", "RAA"]
ALGO_LABELS = {
    "RAA": "RAA-Greedy",
    "Knapsack": "Knapsack-Greedy",
    "P2P": "P2P",
    "BTS": "BTS",
}
ALGO_COLORS = {
    "RAA": BASE_COLOR_PULLING,
    "Knapsack": BASE_COLOR_EXTRA,
    "P2P": LIGHTD_COLOR_PULLING,
    "BTS": LIGHTD_COLOR_EXTRA,
}
ALGO_HATCHES = {
    "RAA": HATCH_1,
    "Knapsack": HATCH_2,
    "P2P": HATCH_3,
    "BTS": HATCH_4,
}


def _setup_fonts():
    return {"weight": "normal", "size": 28}


def _load_hotspot_results_reward():
    hotspot_list = [10, 30, 50, 100]
    metrics = {
        "RAA": [
            84878,
            87537,
            87324,
            87751,
        ],
        "Knapsack": [
            76118,
            85423,
            81119,
            83938,
        ],
        "P2P": [
            76320,
            85750,
            81119,
            86223,
        ],
        "BTS": [
            31883,
            66132,
            68175,
            67712,
        ],
    }
    return hotspot_list, metrics


def _plot_grouped_bars(x_labels, metrics, xlabel, ylabel, filename):
    font = _setup_fonts()

    N = len(x_labels)
    ind = np.arange(N)
    bar_width = 0.18

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i, algo in enumerate(ALGO_ORDER):
        vals = metrics[algo]
        offset = (i - 1.5) * bar_width
        ax.bar(
            ind + offset,
            height=vals,
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

    plt.tick_params(labelsize=20)

    ax.set_xticks(ind)
    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_ylim(0, 100000)
    ax.set_yticks(np.arange(0, 100001, 20000))

    out_path = os.path.join(os.path.dirname(__file__), filename)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    xs, metrics = _load_hotspot_results_reward()
    _plot_grouped_bars(
        xs,
        metrics,
        xlabel="Number of hotspot cloudlets",
        ylabel="Total reward",
        filename="offline_hotspot_total_reward.pdf",
    )
