import networkx as nx


def calculate_delay(homeCloudlet_id, fm, adp, instruction, G, edges):
    """
    纯回源拉取延迟计算，只要本地没缓存强制从DC拉取。
    """
    homeCloudlet = edges[homeCloudlet_id]
    bts_weight = G[homeCloudlet_id]["DC"]["weight"]

    # 基座模型拉取延迟
    model_pull_delay = 0
    if fm.id not in homeCloudlet.cached_models:
        model_pull_delay = bts_weight * fm.size

    # Adapter 拉取延迟
    adp_pull_delay = 0
    if (adp.model_id, adp.service_type) not in homeCloudlet.cached_adapters:
        adp_pull_delay = bts_weight * adp.size

    # 推理延迟
    gamma = 0.001
    inference_delay = gamma * instruction * (fm.size + adp.size) / homeCloudlet.eta

    return model_pull_delay + adp_pull_delay + inference_delay


def bts(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    lambda_delay=1e-3,
    return_total_reward=False,
):
    """BTS 算法
    - 延迟计算强制回源
    - 对每个候选 (user, model) 使用 eff_reward = reward - lambda_delay * delay
    - 用户按 eff_reward 从大到小排序, eff_reward <= 0 的用户不接纳
    - 候选模型按共享后的额外成本做排序, 接纳阶段仍允许真实复用。
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
            # 基于共享后的额外成本排序：优先选择单位共享成本更高的模型。
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

            best_mid, best_delay, best_eff_reward = valid_models[0]
            if best_eff_reward > 0:
                user_candidates_map[user] = {
                    "models": [m[0] for m in valid_models],
                    "best_eff_reward": best_eff_reward,
                    "eff_reward_by_model": {
                        mid: eff_reward for mid, delay, eff_reward in valid_models
                    },
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

        # 若该用户的最优有效收益已不大于 0，则不再尝试接纳
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
