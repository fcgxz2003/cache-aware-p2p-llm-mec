from online.common import DEPLOYMENT_COST_WEIGHT, cold_start_edges


def calculate_delay(homeCloudlet_id, fm, adp, instruction, G, edges):
    """
    在线版本自己的延迟计算函数，格式和 offline/P2P.py 中的 calculate_delay 对齐。
    总延迟 = Model 拉取延迟 + Adapter 拉取延迟 + 推理延迟
    """
    homeCloudlet = edges[homeCloudlet_id]

    # 基座模型拉取延迟
    model_pull_delay = 0
    if fm.id not in homeCloudlet.cached_models:
        model_providers = []
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and fm.id in edges[neighbor].cached_models:
                model_providers.append(neighbor)

        if not model_providers:
            model_pull_delay = G[homeCloudlet.id]["DC"]["weight"] * fm.size
        else:
            model_providers.sort(key=lambda p: G[homeCloudlet.id][p]["weight"])
            selected_model_peers = model_providers[:3]
            max_weight = max(
                G[homeCloudlet.id][p]["weight"] for p in selected_model_peers
            )
            model_pull_delay = max_weight * (fm.size / len(selected_model_peers))

    # Adapter 拉取延迟
    adp_pull_delay = 0
    adp_key = (adp.model_id, adp.service_type)
    if adp_key not in homeCloudlet.cached_adapters:
        adp_providers = []
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and adp_key in edges[neighbor].cached_adapters:
                adp_providers.append(neighbor)

        if not adp_providers:
            bts_weight = G[homeCloudlet.id]["DC"]["weight"]
            adp_pull_delay = bts_weight * adp.size
        else:
            adp_providers.sort(key=lambda p: G[homeCloudlet.id][p]["weight"])
            selected_adp_peers = adp_providers[:3]
            max_weight = max(
                G[homeCloudlet.id][p]["weight"] for p in selected_adp_peers
            )
            adp_pull_delay = max_weight * (adp.size / len(selected_adp_peers))

    # 推理延迟
    gamma = 0.001
    inference_delay = gamma * instruction * (fm.size + adp.size) / homeCloudlet.eta
    return model_pull_delay + adp_pull_delay + inference_delay


def run_p2p_online(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    delay_weight: float = 1e-3,
):
    """在线 P2P 基线（顺序到达，冷启动）。
    - 返回 (admitted, total_reward)，只累计被接纳请求的 reward - delay_weight * delay；
    - 候选排序采用净收益密度启发式；
    - 资源不足时通过 Cloudlet 的 LRU 驱逐接口回收缓存/显存后再尝试分配。
    """

    # 冷启动
    cold_start_edges(edges)
    deployment_cost_weight = DEPLOYMENT_COST_WEIGHT  # 每新增写入 1GB 产生的部署开销

    fm_map = {fm.id: fm for fm in foundation_models}
    admitted = []
    total_reward = 0.0

    for user in users:
        req = user.request
        home_id = req.homeCloudlet
        edge = edges[home_id]

        candidates = []
        for fm in foundation_models:
            adp = adapters_dict.get((fm.id, req.type))
            if adp is None:
                continue
            if user.accuracy <= adp.accuracy:
                delay = calculate_delay(home_id, fm, adp, req.instruction, G, edges)
                if delay <= user.delay:
                    candidates.append((fm.id, delay))

        if not candidates:
            continue

        # (reward - delay_weight * delay) / 资源增量。
        def score(item):
            """计算候选模型的资源收益密度，用于在线 P2P 排序。"""
            mid, delay = item
            fm = fm_map[mid]
            adp = adapters_dict[(mid, req.type)]
            extra_storage = 0.0
            extra_gpu = 0.0
            if mid not in edge.cached_models:
                extra_storage += fm.size
            if (adp.model_id, adp.service_type) not in edge.cached_adapters:
                extra_storage += adp.size
            if mid not in edge.loaded_models:
                extra_gpu += edge.delta * fm.size
            if (adp.model_id, adp.service_type) not in edge.loaded_adapters:
                extra_gpu += edge.delta * adp.size

            combined = extra_storage + extra_gpu
            net_utility = req.reward - delay_weight * delay
            return (-(net_utility / (combined + 1e-6)), delay)

        candidates.sort(key=score)

        for mid, delay in candidates:
            fm = fm_map[mid]
            adp = adapters_dict[(mid, req.type)]

            new_storage_mb = 0.0
            if fm.id not in edge.cached_models:
                new_storage_mb += fm.size
            if (adp.model_id, adp.service_type) not in edge.cached_adapters:
                new_storage_mb += adp.size

            # 在线 P2P：若资源不足，则按 LRU 驱逐旧缓存/加载项，再尝试分配
            if edge.try_allocate_with_lru(fm, adp):
                deployment_cost = deployment_cost_weight * (new_storage_mb / 1024.0)
                total_reward += req.reward - delay_weight * delay - deployment_cost
                admitted.append(user)
                break
    return admitted, total_reward
