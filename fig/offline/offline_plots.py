import copy
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from ex1_multi_cloudlet_hotspot import init_environment, init_users, run_hotspot_once
from offline.raa_greedy import raa_greedy
from offline.knapsack_greedy import knapsack_greedy
from offline.BTS import bts
from offline.P2P import p2p


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


ALGO_ORDER = ["RAA", "Knapsack", "P2P", "BTS"]
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
    font = {
        "weight": "normal",
        "size": 24,
    }
    return font


def _save_fig(fig, filename: str):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, filename)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    print(f"Saved figure to {path}")


def run_avg_experiments():
    """在 avg 场景下，跑 4 个算法在不同请求数下的接纳率和 reward。
    请求数: 100, 200, 500, 1000。
    """

    num_users_list = [100, 200, 500, 1000]

    random.seed(4)
    np.random.seed(4)

    G, edges, foundation_models, adapters, fm_dict = init_environment()

    results_accept = {algo: [] for algo in ALGO_ORDER}
    results_reward = {algo: [] for algo in ALGO_ORDER}

    for n_users in num_users_list:
        users = init_users(n_users)

        # RAA-Greedy
        admitted = raa_greedy(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            fm_dict,
            lambda_delay=1e-3,
        )
        results_accept["RAA"].append(len(admitted) / len(users))
        results_reward["RAA"].append(sum(u.request.reward for u in admitted))

        # Knapsack-Greedy
        admitted = knapsack_greedy(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            lambda_delay=1e-3,
        )
        results_accept["Knapsack"].append(len(admitted) / len(users))
        results_reward["Knapsack"].append(sum(u.request.reward for u in admitted))

        # BTS
        admitted = bts(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            lambda_delay=1e-3,
        )
        results_accept["BTS"].append(len(admitted) / len(users))
        results_reward["BTS"].append(sum(u.request.reward for u in admitted))

        # P2P
        admitted = p2p(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            lambda_delay=1e-3,
        )
        results_accept["P2P"].append(len(admitted) / len(users))
        results_reward["P2P"].append(sum(u.request.reward for u in admitted))

    return num_users_list, results_accept, results_reward


def run_hotspot_experiments(seed_best: int = 4):
    """在 hotspot 场景下，跑 4 个算法在不同热点 cloudlet 数下的接纳率和 reward。
    热点数: 10, 30, 50, 100。
    """

    hotspot_list = [10, 30, 50, 100]
    num_users = 3000

    results_accept = {algo: [] for algo in ALGO_ORDER}
    results_reward = {algo: [] for algo in ALGO_ORDER}

    for h in hotspot_list:
        res = run_hotspot_once(h, num_users=num_users, seed=seed_best)
        for algo in ALGO_ORDER:
            admitted = res[algo]["admitted"]
            reward = res[algo]["reward"]
            results_accept[algo].append(admitted / num_users)
            results_reward[algo].append(reward)

    return hotspot_list, results_accept, results_reward


def plot_grouped_bars(x_labels, metrics, ylabel, filename):
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

    plt.xlabel("")
    plt.ylabel(ylabel, font, loc="top")

    plt.tick_params(labelsize=20)

    ax.set_xticks(ind)
    ax.set_xticklabels([str(x) for x in x_labels])

    ax.legend(fontsize=16, frameon=False)

    _save_fig(fig, filename)
