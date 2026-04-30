from offline.BTS import calculate_delay as bts_delay
from online.common import DEPLOYMENT_COST_WEIGHT, cold_start_edges


def run_bts_online(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    delay_weight: float = 1e-3,
):
    """
    - 顺序到达的 BTS
    - 冷启动
    - 每个请求：强制回源延迟计算，按性价比选择可行模型并分配
    - 返回 (admitted, total_reward)其中 total_reward 只累计被接纳请求的 reward - delay_weight * delay
    - 当资源不足时，采用 LRU 驱逐释放缓存与显存，再尝试接纳请求
    """
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
                delay = bts_delay(home_id, fm, adp, req.instruction, G, edges)
                if delay <= user.delay:
                    candidates.append((fm.id, delay))

        if not candidates:
            continue

        # 恢复资源感知的启发式：综合考虑 reward 与新增资源占用。
        def score(item):
            """计算候选模型的单位资源净收益排序分数。"""
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
            deployment_cost = deployment_cost_weight * (extra_storage / 1024.0)
            net_utility = req.reward - delay_weight * delay - deployment_cost
            combined = extra_storage + extra_gpu
            return -(net_utility / (combined + 1e-6)), delay

        candidates.sort(key=score)

        for mid, delay in candidates:
            fm = fm_map[mid]
            adp = adapters_dict[(mid, req.type)]

            new_storage_mb = 0.0
            if fm.id not in edge.cached_models:
                new_storage_mb += fm.size
            if (adp.model_id, adp.service_type) not in edge.cached_adapters:
                new_storage_mb += adp.size

            net_reward = (
                req.reward
                - delay_weight * delay
                - deployment_cost_weight * (new_storage_mb / 1024.0)
            )
            if net_reward <= 0:
                continue

            # 尝试按 LRU 驱逐策略为新模型+适配器腾出资源
            if edge.try_allocate_with_lru(fm, adp):
                total_reward += net_reward
                admitted.append(user)
                break
    return admitted, total_reward
