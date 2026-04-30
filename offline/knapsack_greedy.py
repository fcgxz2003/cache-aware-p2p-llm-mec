from .raa_greedy import calculate_delay


def knapsack_greedy(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    lambda_delay=1e-3,
    return_total_reward=False,
):
    """Knapsack-Greedy baseline (offline, per-cloudlet ratio greedy).
    - 对每个用户，枚举所有满足精度与时延约束的 (foundation model, adapter),
      计算 eff_reward = R_k - lambda_delay * D_{k,j}。
    - 对每个用户只保留“性价比”最高的模型:ratio = eff_reward / combined_cost。
    - 在每个 cloudlet 内，将用户按 ratio 从大到小排序，依次尝试接纳；
    """
    fm_map = {fm.id: fm for fm in foundation_models}
    user_best_plan = {}

    for user in users:
        req = user.request
        home_id = req.homeCloudlet
        edge = edges[home_id]

        best_ratio = None
        best_mid = None
        best_eff_reward = None

        for fm in foundation_models:
            adapter = adapters_dict.get((fm.id, req.type))
            if adapter is None:
                continue

            # 精度达标
            if user.accuracy > adapter.accuracy:
                continue

            # 延迟达标
            delay = calculate_delay(home_id, fm, adapter, req.instruction, G, edges)
            if delay > user.delay:
                continue

            eff_reward = req.reward - lambda_delay * delay
            if eff_reward <= 0:
                continue

            # 相对当前 edge 初始缓存状态估算额外资源成本
            extra_storage = 0.0
            extra_gpu = 0.0
            if fm.id not in edge.cached_models:
                extra_storage += fm.size
            if (adapter.model_id, adapter.service_type) not in edge.cached_adapters:
                extra_storage += adapter.size
            if fm.id not in edge.loaded_models:
                extra_gpu += edge.delta * fm.size
            if (adapter.model_id, adapter.service_type) not in edge.loaded_adapters:
                extra_gpu += edge.delta * adapter.size

            combined_cost = extra_storage + extra_gpu
            ratio = eff_reward / (combined_cost + 1e-6)

            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_mid = fm.id
                best_eff_reward = eff_reward

        if best_mid is not None:
            user_best_plan[user] = {
                "mid": best_mid,
                "best_ratio": best_ratio,
                "best_eff_reward": best_eff_reward,
            }

    admitted_requests = []
    total_reward = 0.0

    # 按 cloudlet 分组做背包贪心
    # edge之间互不影响，因此可以分别处理
    edge_to_users = {}
    for user, info in user_best_plan.items():
        eid = user.request.homeCloudlet
        edge_to_users.setdefault(eid, []).append(user)

    for edge_id, user_list in edge_to_users.items():
        edge = edges[edge_id]

        # 按 best_ratio 从大到小排序，近似 0/1 knapsack 比值贪心
        sorted_users = sorted(
            user_list,
            key=lambda u: user_best_plan[u]["best_ratio"],
            reverse=True,
        )

        for user in sorted_users:
            plan = user_best_plan[user]
            mid = plan["mid"]
            req = user.request

            fm = fm_map[mid]
            adapter = adapters_dict[(mid, req.type)]

            if edge.check_capacity(fm, adapter):
                edge.allocate(fm, adapter)
                admitted_requests.append(user)
                total_reward += plan["best_eff_reward"]

    if return_total_reward:
        return admitted_requests, total_reward
    return admitted_requests
