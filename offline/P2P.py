import networkx as nx


def calculate_delay(homeCloudlet_id, fm, adp, instruction, G, edges):
    """
    计算延迟
    总延迟 = Model拉取延迟 + Adapter拉取延迟 + 推理延迟
    严格对应论文公式: D_{k,j} = D^{model}_{k,j} + D^{weight}_{k,j} + D^{inf}_{k,j}
    """
    homeCloudlet = edges[homeCloudlet_id]

    # 基座模型拉取延迟
    model_pull_delay = 0
    if fm.id not in homeCloudlet.cached_models:
        model_providers = []
        # 寻找拥有该模型的cloudlet
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and fm.id in edges[neighbor].cached_models:
                model_providers.append(neighbor)

        if not model_providers:
            # 如果全网都没缓存触发回源
            model_pull_delay = G[homeCloudlet.id]["DC"]["weight"] * fm.size
        else:
            # 如果有缓存，则P2P 并行拉取, 按网络延迟排序，取最多前 3 个最佳节点
            model_providers.sort(key=lambda p: G[homeCloudlet.id][p]["weight"])
            selected_model_peers = model_providers[:3]
            max_weight = max(
                G[homeCloudlet.id][p]["weight"] for p in selected_model_peers
            )
            model_pull_delay = max_weight * (fm.size / len(selected_model_peers))

    # 计算 Adapter 拉取延迟
    adp_pull_delay = 0
    adp_key = (adp.model_id, adp.service_type)
    if adp_key not in homeCloudlet.cached_adapters:
        adp_providers = []
        # 寻找拥有该Adapter的cloudlet
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and adp_key in edges[neighbor].cached_adapters:
                adp_providers.append(neighbor)

        if not adp_providers:
            # 如果全网都没缓存触发回源
            bts_weight = G[homeCloudlet.id]["DC"]["weight"]
            adp_pull_delay = bts_weight * adp.size
        else:
            # P2P 并行拉取, 按网络延迟排序，取最多前 3 个最佳节点
            adp_providers.sort(key=lambda p: G[homeCloudlet.id][p]["weight"])
            selected_adp_peers = adp_providers[:3]
            max_weight = max(
                G[homeCloudlet.id][p]["weight"] for p in selected_adp_peers
            )
            adp_pull_delay = max_weight * (adp.size / len(selected_adp_peers))

    # 计算推理延迟
    # D^{inf} = (gamma * Ins_k / eta) * (S^{model} + S^{weight})
    gamma = 0.001
    inference_delay = gamma * instruction * (fm.size + adp.size) / homeCloudlet.eta
    return model_pull_delay + adp_pull_delay + inference_delay


def p2p(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    lambda_delay=1e-3,
    return_total_reward=False,
):
    """P2P Baseline 算法（加入 delay 惩罚）
    - 延迟计算允许 P2P 相邻边缘节点拉取
    - 对每个候选 (user, model) 使用 eff_reward = reward - lambda_delay * delay
    - 用户按 eff_reward 从大到小排序；eff_reward <= 0 的用户不接纳
    - 候选模型按共享后的额外成本做排序；接纳阶段仍允许真实复用。
    """
    user_candidates_map = {}
    fm_map = {fm.id: fm for fm in foundation_models}

    for user in users:
        req = user.request
        home_id = req.homeCloudlet
        valid_models = []

        for fm in foundation_models:
            adapter = adapters_dict.get((fm.id, req.type))
            if adapter is None:
                continue

            if user.accuracy <= adapter.accuracy:
                delay = calculate_delay(home_id, fm, adapter, req.instruction, G, edges)
                if delay <= user.delay:
                    eff_reward = req.reward - lambda_delay * delay
                    valid_models.append((fm.id, delay, eff_reward))

        if valid_models:

            def score_for(item):
                mid, delay, eff_reward = item
                fm = fm_map[mid]
                adp = adapters_dict[(mid, req.type)]
                home_edge = edges[home_id]

                extra_storage = 0.0
                extra_gpu = 0.0
                if mid not in home_edge.cached_models:
                    extra_storage += fm.size
                if (adp.model_id, adp.service_type) not in home_edge.cached_adapters:
                    extra_storage += adp.size
                if mid not in home_edge.loaded_models:
                    extra_gpu += home_edge.delta * fm.size
                if (adp.model_id, adp.service_type) not in home_edge.loaded_adapters:
                    extra_gpu += home_edge.delta * adp.size

                combined_cost = extra_storage + extra_gpu
                return -(eff_reward / (combined_cost + 1e-6)), delay

            valid_models.sort(key=score_for)

            best_eff_reward = max(v[2] for v in valid_models)
            if best_eff_reward > 0:
                user_candidates_map[user] = {
                    "models": [m[0] for m in valid_models],
                    "best_eff_reward": best_eff_reward,
                    "eff_reward_by_model": {m[0]: m[2] for m in valid_models},
                }

    admitted_requests = []
    total_reward = 0.0
    sorted_users = sorted(
        list(user_candidates_map.keys()),
        key=lambda u: user_candidates_map[u]["best_eff_reward"],
        reverse=True,
    )

    for user in sorted_users:
        req = user.request
        home_id = req.homeCloudlet
        edge = edges[home_id]

        if user_candidates_map[user]["best_eff_reward"] <= 0:
            continue

        for m_id in user_candidates_map[user]["models"]:
            fm = next(m for m in foundation_models if m.id == m_id)
            adapter = adapters_dict[(m_id, req.type)]

            if edge.check_capacity(fm, adapter):
                edge.allocate(fm, adapter)
                admitted_requests.append(user)
                total_reward += user_candidates_map[user]["eff_reward_by_model"][m_id]
                break

    if return_total_reward:
        return admitted_requests, total_reward
    return admitted_requests
