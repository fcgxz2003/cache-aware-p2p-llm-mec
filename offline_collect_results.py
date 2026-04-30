import copy
import csv
import os
import random

import numpy as np

from ex1_multi_cloudlet_hotspot import (
    init_environment,
    init_users,
    init_users_hotspot,
)
from offline.BTS import bts
from offline.P2P import p2p
from offline.knapsack_greedy import knapsack_greedy
from offline.raa_greedy import raa_greedy


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGO_ORDER = ["RAA", "Knapsack", "P2P", "BTS"]


def _run_offline_algorithms(
    G,
    users,
    edges,
    foundation_models,
    adapters,
    fm_dict,
    lambda_delay: float = 1e-3,
):
    rows = []

    admitted, total_reward = raa_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        fm_dict,
        lambda_delay=lambda_delay,
        return_total_reward=True,
    )
    rows.append(("RAA", admitted, total_reward))

    admitted, total_reward = knapsack_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
        return_total_reward=True,
    )
    rows.append(("Knapsack", admitted, total_reward))

    admitted, total_reward = p2p(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
        return_total_reward=True,
    )
    rows.append(("P2P", admitted, total_reward))

    admitted, total_reward = bts(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
        return_total_reward=True,
    )
    rows.append(("BTS", admitted, total_reward))

    return rows


def collect_offline_avg(seed: int = 4, lambda_delay: float = 1e-3):
    num_users_list = [100, 200, 500, 1000]

    random.seed(seed)
    np.random.seed(seed)

    G, edges, foundation_models, adapters, fm_dict = init_environment()

    rows = []
    for n in num_users_list:
        users = init_users(n)
        algo_rows = _run_offline_algorithms(
            G,
            users,
            edges,
            foundation_models,
            adapters,
            fm_dict,
            lambda_delay=lambda_delay,
        )
        for algo, admitted, total_reward in algo_rows:
            rows.append(
                {
                    "scenario": "avg",
                    "num_users": n,
                    "algo": algo,
                    "admitted": len(admitted),
                    "total_users": len(users),
                    "accept_rate": len(admitted) / len(users),
                    "total_reward": total_reward,
                }
            )

    out_path = os.path.join(RESULTS_DIR, "offline_avg.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "num_users",
                "algo",
                "admitted",
                "total_users",
                "accept_rate",
                "total_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Offline-Avg] Results written to {out_path}")


def collect_offline_hotspot(seed: int = 4, lambda_delay: float = 1e-3):
    hotspot_list = [10, 30, 50, 100]
    num_users = 3000

    rows = []
    for h in hotspot_list:
        random.seed(seed)
        np.random.seed(seed)

        G, edges, foundation_models, adapters, fm_dict = init_environment()
        users = init_users_hotspot(num_users, num_hot_edges=h)

        algo_rows = _run_offline_algorithms(
            G,
            users,
            edges,
            foundation_models,
            adapters,
            fm_dict,
            lambda_delay=lambda_delay,
        )
        for algo, admitted, total_reward in algo_rows:
            rows.append(
                {
                    "scenario": "hotspot",
                    "hotspots": h,
                    "num_users": num_users,
                    "algo": algo,
                    "admitted": len(admitted),
                    "accept_rate": len(admitted) / num_users,
                    "total_reward": total_reward,
                }
            )

    out_path = os.path.join(RESULTS_DIR, "offline_hotspot.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "hotspots",
                "num_users",
                "algo",
                "admitted",
                "accept_rate",
                "total_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Offline-Hotspot] Results written to {out_path}")


if __name__ == "__main__":
    collect_offline_avg(seed=4)
    collect_offline_hotspot(seed=4)
