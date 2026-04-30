import time
import copy

from ex1_multi_cloudlet_hotspot import init_environment, init_users
from offline.raa_greedy import raa_greedy
from offline.knapsack_greedy import knapsack_greedy
from offline.BTS import bts
from offline.P2P import p2p


if __name__ == "__main__":
    # 多点平均分布场景：仍然使用多 cloudlet，但用户按距离平均分布
    G, edges, foundation_models, adapters, fm_dict = init_environment()
    users = init_users(3000)

    # RAA-Greedy
    start_time = time.time()
    admitted_requests = raa_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        fm_dict,
        lambda_delay=1e-3,
    )
    print(f"[Multi-Cloudlet-Avg] RAA-Greedy 耗时: {time.time() - start_time:.2f} s")
    total_reward = sum(u.request.reward for u in admitted_requests)
    print(
        f"[Multi-Cloudlet-Avg] RAA 接纳率: {len(admitted_requests)} / {len(users)} "
        f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
    )

    # Knapsack-Greedy
    start_time = time.time()
    admitted_requests = knapsack_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=1e-3,
    )
    print(
        f"[Multi-Cloudlet-Avg] Knapsack-Greedy 耗时: {time.time() - start_time:.2f} s"
    )
    total_reward = sum(u.request.reward for u in admitted_requests)
    print(
        f"[Multi-Cloudlet-Avg] Knapsack-Greedy 接纳率: {len(admitted_requests)} / {len(users)} "
        f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
    )

    # BTS
    start_time = time.time()
    admitted_requests = bts(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
    )
    print(f"[Multi-Cloudlet-Avg] BTS 耗时: {time.time() - start_time:.2f} s")
    total_reward = sum(u.request.reward for u in admitted_requests)
    print(
        f"[Multi-Cloudlet-Avg] BTS 接纳率: {len(admitted_requests)} / {len(users)} "
        f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
    )

    # P2P
    start_time = time.time()
    admitted_requests = p2p(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
    )
    print(f"[Multi-Cloudlet-Avg] P2P 耗时: {time.time() - start_time:.2f} s")
    total_reward = sum(u.request.reward for u in admitted_requests)
    print(
        f"[Multi-Cloudlet-Avg] P2P 接纳率: {len(admitted_requests)} / {len(users)} "
        f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
    )
