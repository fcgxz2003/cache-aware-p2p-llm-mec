import random

from online.common import DEPLOYMENT_COST_WEIGHT, cold_start_edges
from online.P2P import calculate_delay as p2p_delay


CACHE_ACTIONS = {
    "none": (0, 0),
    "adapter": (0, 1),
    "model_only": (1, 0),
    "full": (1, 1),
}


def run_eps_greedy_online(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    epsilon: float = 0.05,
    rng_seed: int = 0,
    delay_weight: float = 1e-3,
    penalty: float = 50.0,
    min_epsilon: float = 0.0,
    cache_bonus_weight: float = 1.2,
    resource_penalty_weight: float = 0.2,
    future_value_weight: float = 1.6,
    eviction_penalty_weight: float = 0.6,
    stats_decay_interval: int = 80,
    stats_decay_factor: float = 0.8,
):
    """在线 ε-greedy：在 QoS 可行动作集合上直接学习模型与缓存动作。
    action (foundation model, cache_model, cache_adapter)，并为每个 home cloudlet
    维护局部经验均值。其余参数保留仅为兼容已有调用。
    """

    del cache_bonus_weight
    del resource_penalty_weight
    del future_value_weight
    del eviction_penalty_weight
    del stats_decay_interval
    del stats_decay_factor

    cold_start_edges(edges)
    deployment_cost_weight = DEPLOYMENT_COST_WEIGHT
    rng = random.Random(rng_seed)
    q_values = {}
    visit_counts = {}
    admitted = []
    total_reward = 0.0
    total_steps = max(len(users) - 1, 1)

    def current_epsilon(step_idx):
        """按请求进度线性衰减探索率。"""
        progress = min(step_idx / total_steps, 1.0)
        return max(min_epsilon, epsilon * (1.0 - progress))

    def action_key(home_id, model_id, req_type, action_name):
        """构造局部 bandit 动作键。"""
        return (home_id, model_id, req_type, action_name)

    def storage_write_mb(edge, model, adapter, action_name):
        """计算执行缓存动作时的新增磁盘写入量。"""
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        new_storage_mb = 0.0
        if cache_model and model.id not in edge.cached_models:
            new_storage_mb += model.size
        adapter_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adapter_key not in edge.cached_adapters:
            new_storage_mb += adapter.size
        return new_storage_mb

    def is_feasible_action(edge, model, adapter, action_name):
        """检查动作在当前 edge 上是否能通过 LRU 驱逐成功执行。"""
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        snapshot = edge._snapshot_online_state()
        feasible = edge.try_apply_action_with_lru(
            model,
            adapter,
            cache_model,
            cache_adapter,
        )
        edge._restore_online_state(snapshot)
        return feasible

    def enumerate_feasible_actions(user, edge):
        """枚举满足精度、时延和容量约束的 bandit 动作。"""
        req = user.request
        candidates = []
        for model in foundation_models:
            adapter = adapters_dict.get((model.id, req.type))
            if adapter is None or adapter.accuracy < user.accuracy:
                continue

            delay = p2p_delay(
                req.homeCloudlet, model, adapter, req.instruction, G, edges
            )
            if delay > user.delay:
                continue

            for action_name in CACHE_ACTIONS:
                if not is_feasible_action(edge, model, adapter, action_name):
                    continue
                candidates.append((model, adapter, delay, action_name))
        return candidates

    for step_idx, user in enumerate(users):
        req = user.request
        home_id = req.homeCloudlet
        edge = edges[home_id]
        feasible_actions = enumerate_feasible_actions(user, edge)
        if not feasible_actions:
            continue

        epsilon_t = current_epsilon(step_idx)
        if rng.random() < epsilon_t:
            model, adapter, delay, action_name = rng.choice(feasible_actions)
        else:
            model, adapter, delay, action_name = max(
                feasible_actions,
                key=lambda item: q_values.get(
                    action_key(home_id, item[0].id, req.type, item[3]),
                    0.0,
                ),
            )

        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        act_key = action_key(home_id, model.id, req.type, action_name)
        new_storage_mb = storage_write_mb(edge, model, adapter, action_name)
        deployment_cost = deployment_cost_weight * (new_storage_mb / 1024.0)

        if edge.try_apply_action_with_lru(model, adapter, cache_model, cache_adapter):
            reward = req.reward - delay_weight * delay - deployment_cost
            total_reward += reward
            admitted.append(user)
        else:
            reward = -penalty

        visit_counts[act_key] = visit_counts.get(act_key, 0) + 1
        q_values[act_key] = (
            q_values.get(act_key, 0.0)
            + (reward - q_values.get(act_key, 0.0)) / visit_counts[act_key]
        )

    return admitted, total_reward
