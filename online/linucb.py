import numpy as np

from online.common import DEPLOYMENT_COST_WEIGHT, cold_start_edges
from online.P2P import calculate_delay as p2p_delay


class ContextAwareLinUCB:
    def __init__(self, alpha, foundation_models, adapters_dict, edge_ids):
        """LinUCB：仅学习模型选择，不显式学习缓存动作。"""
        self.alpha = alpha
        self.models = foundation_models
        self.model_ids = [
            getattr(fm, "id", None) or getattr(fm, "model_id", None)
            for fm in self.models
        ]
        self.model_lookup = {
            getattr(fm, "id", None) or getattr(fm, "model_id", None): fm
            for fm in self.models
        }
        self.service_types = sorted(
            {adapter.service_type for adapter in adapters_dict.values()}
        )

        # 请求特征 + 服务类型 + 本地资源利用率。
        self.d = 4 + len(self.service_types) + 2

        # 动作空间只保留 foundation model 选择。
        self.actions = list(self.model_ids)

        # 初始化 Ridge 参数
        self.A = {a: np.eye(self.d) for a in self.actions}
        self.b = {a: np.zeros(self.d) for a in self.actions}

    def _build_context_vector(self, user, edges_dict):
        """为当前请求构造轻量上下文向量。"""
        req = user.request
        phi_r = [
            req.instruction / 10000.0,
            user.accuracy,
            user.delay / 30000.0,
            req.reward / 100.0,
        ]

        service_one_hot = [
            1.0 if req.type == svc else 0.0 for svc in self.service_types
        ]

        local_edge = edges_dict[req.homeCloudlet]
        local_state = [
            local_edge.used_storage / max(local_edge.storage_capacity, 1.0),
            local_edge.used_memory / max(local_edge.memory_capacity, 1.0),
        ]

        c_t = np.array(
            phi_r + service_one_hot + local_state,
            dtype=float,
        )
        return c_t

    def get_valid_actions(self, user, G, edges_dict, adapters_dict):
        """筛选满足 QoS 约束的可行动作集合。"""
        valid_actions = []
        req = user.request
        for fm in self.models:
            mid = getattr(fm, "id", None) or getattr(fm, "model_id", None)
            adapter = adapters_dict.get((mid, req.type))
            if adapter is None:
                continue
            if adapter.accuracy < user.accuracy:
                continue

            delay = p2p_delay(
                req.homeCloudlet,
                fm,
                adapter,
                req.instruction,
                G,
                edges_dict,
            )
            if delay > user.delay:
                continue

            valid_actions.append(mid)
        return valid_actions

    def decide(self, user, G, edges_dict, adapters_dict):
        """基于 LinUCB 上置信界选择当前请求的 foundation model。"""
        req = user.request
        edge = edges_dict[req.homeCloudlet]
        c_t = self._build_context_vector(user, edges_dict)

        valid_actions = self.get_valid_actions(user, G, edges_dict, adapters_dict)

        def heuristic_score(action):
            """在 UCB 分数相近时提供资源感知的启发式打分。"""
            mid = action
            fm = self.model_lookup.get(mid)
            if fm is None:
                return -float("inf")

            adapter = adapters_dict.get((mid, req.type))
            if adapter is None:
                return -float("inf")

            extra_storage = 0.0
            extra_gpu = 0.0
            if mid not in edge.cached_models:
                extra_storage += fm.size
            if (adapter.model_id, adapter.service_type) not in edge.cached_adapters:
                extra_storage += adapter.size
            if mid not in edge.loaded_models:
                extra_gpu += edge.delta * fm.size
            if (adapter.model_id, adapter.service_type) not in edge.loaded_adapters:
                extra_gpu += edge.delta * adapter.size

            delay = p2p_delay(
                req.homeCloudlet,
                fm,
                adapter,
                req.instruction,
                G,
                edges_dict,
            )
            net_utility = req.reward - 1e-3 * delay
            combined = extra_storage + extra_gpu

            return net_utility / (combined + 1e-6)

        best_a = None
        max_p = -float("inf")
        best_h = -float("inf")

        if not valid_actions:
            return None, c_t

        for a in valid_actions:
            try:
                theta_a = np.linalg.solve(self.A[a], self.b[a])
                Ainv_ct = np.linalg.solve(self.A[a], c_t)
            except np.linalg.LinAlgError:
                continue

            expected_reward = theta_a.T @ c_t
            confidence_bound = self.alpha * np.sqrt(float(c_t.T @ Ainv_ct))
            p_ta = expected_reward + confidence_bound
            h_ta = heuristic_score(a)

            if p_ta > max_p + 1e-9 or (abs(p_ta - max_p) <= 1e-9 and h_ta > best_h):
                max_p = p_ta
                best_a = a
                best_h = h_ta

        return best_a, c_t

    def update(self, action, c_t, reward):
        self.A[action] += np.outer(c_t, c_t)
        self.b[action] += reward * c_t


def run_linucb_online(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    alpha=1.0,
    delay_weight=1e-3,
    penalty=5.0,
):
    """
    - 顺序到达环境下运行轻量版 LinUCB
    - 冷启动
    - 只学习 foundation model 选择，缓存/加载由固定 LRU 逻辑处理
    """
    cold_start_edges(edges)
    deployment_cost_weight = DEPLOYMENT_COST_WEIGHT  # 每新增写入 1GB 产生的部署开销

    agent = ContextAwareLinUCB(alpha, foundation_models, adapters_dict, edges.keys())

    admitted = []
    total_reward = 0.0
    model_lookup = {
        getattr(model, "id", None) or getattr(model, "model_id", None): model
        for model in foundation_models
    }

    for user in users:
        action, c_t = agent.decide(user, G, edges, adapters_dict)
        # 若当前请求在 QoS 约束下没有任何可行动作，直接跳过，
        # 不对 bandit 做 update，避免把无法服务的样本错误归因到某个 arm。
        if action is None:
            continue

        # action 格式: model_id
        model_id = action
        fm = model_lookup.get(model_id)
        if fm is None:
            agent.update(action, c_t, 0.0)
            continue
        adapter = adapters_dict.get((model_id, user.request.type))
        if adapter is None:
            agent.update(action, c_t, 0.0)
            continue

        edge = edges[user.request.homeCloudlet]
        # 检查延迟约束（按 P2P 拉取模型与 adapter）以及容量约束
        delay = p2p_delay(
            user.request.homeCloudlet,
            fm,
            adapter,
            user.request.instruction,
            G,
            edges,
        )
        meets_qos = adapter.accuracy >= user.accuracy and delay <= user.delay

        new_storage_mb = 0.0
        if fm.id not in edge.cached_models:
            new_storage_mb += fm.size
        if (adapter.model_id, adapter.service_type) not in edge.cached_adapters:
            new_storage_mb += adapter.size

        # 使用固定的 LRU 分配逻辑，不再显式学习缓存动作。
        if meets_qos and edge.try_allocate_with_lru(fm, adapter):
            deployment_cost = deployment_cost_weight * (new_storage_mb / 1024.0)
            reward = user.request.reward - delay_weight * delay - deployment_cost
            total_reward += reward
            admitted.append(user)
        else:
            reward = -penalty

        agent.update(action, c_t, reward)
    return admitted, total_reward
