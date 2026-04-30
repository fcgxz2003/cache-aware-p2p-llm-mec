import csv
import os
import random

import numpy as np

from ex1_multi_cloudlet_hotspot import (
    init_environment,
    init_users,
    init_users_hotspot,
)
from online.common import benchmark_online_algorithms, prepare_online_environment


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGO_ORDER = ["LinUCB", "P2P", "BTS", "EpsGreedy"]


def _build_base_environment():
    G, edges, foundation_models, adapters, fm_dict = init_environment()
    prepare_online_environment(edges, foundation_models)
    return G, edges, foundation_models, adapters


def collect_online_default(seed: int = 4):
    num_users_list = [100, 200, 500, 1000]

    random.seed(seed)
    np.random.seed(seed)

    G, edges, foundation_models, adapters = _build_base_environment()
    full_users = init_users(max(num_users_list))

    rows = []
    for n in num_users_list:
        # 所有点共享同一条 online 到达序列，只比较前缀长度。
        users = full_users[:n]

        for result in benchmark_online_algorithms(
            G,
            users,
            edges,
            foundation_models,
            adapters,
            alpha=1.0,
            epsilon=0.12,
        ):
            rows.append(
                {
                    "scenario": "avg",
                    "num_users": n,
                    "algo": result["algo"],
                    "admitted": result["admitted"],
                    "total_users": result["total_users"],
                    "accept_rate": result["accept_rate"],
                    "total_reward": result["total_reward"],
                }
            )

    out_path = os.path.join(RESULTS_DIR, "online_avg.csv")
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

    print(f"[Online-Default] Results written to {out_path}")


def collect_online_hotspot(seed: int = 4):
    hotspot_list = [10, 20, 50, 100]
    num_users = 3000

    rows = []
    for h in hotspot_list:
        random.seed(seed)
        np.random.seed(seed)

        G, edges, foundation_models, adapters = _build_base_environment()

        users = init_users_hotspot(num_users, num_hot_edges=h)
        for result in benchmark_online_algorithms(
            G,
            users,
            edges,
            foundation_models,
            adapters,
            alpha=1.0,
            epsilon=0.12,
        ):
            rows.append(
                {
                    "scenario": "hotspot",
                    "hotspots": h,
                    "num_users": num_users,
                    "algo": result["algo"],
                    "admitted": result["admitted"],
                    "accept_rate": result["accept_rate"],
                    "total_reward": result["total_reward"],
                }
            )

    out_path = os.path.join(RESULTS_DIR, "online_hotspot.csv")
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

    print(f"[Online-Hotspot] Results written to {out_path}")


if __name__ == "__main__":
    collect_online_default(seed=4)
    collect_online_hotspot(seed=4)
