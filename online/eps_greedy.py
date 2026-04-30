import math
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
    """在线 ε-greedy：联合学习服务模型、缓存动作和缓存驱逐方案。

    - 直接在满足 QoS 的联合动作上进行 ε-greedy，而不是先用 P2P 风格预排序 foundation model；
    - 联合动作包含 foundation model、缓存动作以及显式 eviction plan；
    - 利用“全局先验 + 本地偏置”的统计来适应不同 home cloudlet 的热点差异；
    - 在高压且低收益场景下允许显式 reject，避免低价值请求持续挤占后续资源；
    - 若存储不足，则使用 retention score 生成显式 eviction plan，并将其作为学习动作的一部分；
    - 返回的 total_reward 保持为真实收益，便于与 P2P / LinUCB / BTS 对比。
    """

    cold_start_edges(edges)
    deployment_cost_weight = DEPLOYMENT_COST_WEIGHT
    rng = random.Random(rng_seed)

    global_Q = {}
    global_N = {}
    local_Q = {}
    local_N = {}

    global_service_seen = {}
    local_service_seen = {}
    global_model_success = {}
    local_model_success = {}
    item_hits = {}

    admitted = []
    total_reward = 0.0
    total_steps = max(len(users) - 1, 1)
    fm_map = {fm.id: fm for fm in foundation_models}
    service_types = sorted({service_type for _model_id, service_type in adapters_dict})
    model_service_types = {}
    for model_id, service_type in adapters_dict:
        model_service_types.setdefault(model_id, set()).add(service_type)

    def current_epsilon(step_idx):
        progress = min(step_idx / total_steps, 1.0)
        # 长 horizon 下更快收敛到 exploitation，减少后期持续探索带来的 reward 损失。
        return max(
            min_epsilon, min_epsilon + (epsilon - min_epsilon) * (1.0 - progress) ** 2
        )

    def decay_stats():
        for stats in (
            global_service_seen,
            local_service_seen,
            global_model_success,
            local_model_success,
            item_hits,
        ):
            stale_keys = []
            for stat_key, stat_value in stats.items():
                decayed = stat_value * stats_decay_factor
                if decayed < 1e-3:
                    stale_keys.append(stat_key)
                else:
                    stats[stat_key] = decayed
            for stat_key in stale_keys:
                del stats[stat_key]

    def remove_fifo_entry(fifo_list, kind, key):
        try:
            fifo_list.remove((kind, key))
        except ValueError:
            pass

    def ensure_action_stats(home_id, action_key):
        if action_key not in global_Q:
            global_Q[action_key] = 0.0
            global_N[action_key] = 0

        local_stat_key = (home_id, action_key)
        if local_stat_key not in local_Q:
            local_Q[local_stat_key] = 0.0
            local_N[local_stat_key] = 0

    def coarse_action_key(action_key):
        if (
            isinstance(action_key, tuple)
            and len(action_key) == 4
            and action_key[0] != "__reject__"
        ):
            return action_key[:3]
        return action_key

    def posterior_mean_for_key(home_id, action_key):
        ensure_action_stats(home_id, action_key)

        global_mean = global_Q[action_key]
        local_stat_key = (home_id, action_key)
        local_mean = local_Q[local_stat_key]
        local_count = local_N[local_stat_key]
        global_count = global_N[action_key]

        if local_count <= 0:
            return global_mean

        prior_strength = min(8.0, 2.0 + 0.2 * global_count)
        return (prior_strength * global_mean + local_count * local_mean) / (
            prior_strength + local_count
        )

    def posterior_action_value(home_id, action_key):
        exact_value = posterior_mean_for_key(home_id, action_key)
        shared_key = coarse_action_key(action_key)
        if shared_key == action_key:
            return exact_value

        shared_value = posterior_mean_for_key(home_id, shared_key)
        exact_local_count = local_N.get((home_id, action_key), 0)
        exact_global_count = global_N.get(action_key, 0)
        exact_support = exact_local_count + 0.25 * exact_global_count
        exact_weight = exact_support / (exact_support + 6.0)
        return exact_weight * exact_value + (1.0 - exact_weight) * shared_value

    def update_action_stats(home_id, action_key, reward):
        def _update_single(key):
            ensure_action_stats(home_id, key)

            global_N[key] += 1
            global_Q[key] += (reward - global_Q[key]) / global_N[key]

            local_stat_key = (home_id, key)
            local_N[local_stat_key] += 1
            local_Q[local_stat_key] += (reward - local_Q[local_stat_key]) / local_N[
                local_stat_key
            ]

        _update_single(action_key)
        shared_key = coarse_action_key(action_key)
        if shared_key != action_key:
            _update_single(shared_key)

    def blended_service_seen(home_id, req_type):
        return local_service_seen.get(
            (home_id, req_type), 0.0
        ) + 0.35 * global_service_seen.get(req_type, 0.0)

    def blended_model_success(home_id, model_id, req_type):
        return local_model_success.get(
            (home_id, model_id, req_type), 0.0
        ) + 0.35 * global_model_success.get((model_id, req_type), 0.0)

    def blended_home_activity(home_id):
        local_total = sum(
            local_service_seen.get((home_id, service_type), 0.0)
            for service_type in service_types
        )
        global_total = sum(
            global_service_seen.get(service_type, 0.0) for service_type in service_types
        )
        return local_total + 0.2 * global_total

    def neighbor_availability_factor(home_id, item_kind, item_key):
        neighbor_count = 0
        nearest_weight = float("inf")
        for neighbor in G.neighbors(home_id):
            if neighbor == "DC":
                continue
            neighbor_edge = edges[neighbor]
            if item_kind == "model":
                present = item_key in neighbor_edge.cached_models
            else:
                present = item_key in neighbor_edge.cached_adapters
            if not present:
                continue
            neighbor_count += 1
            nearest_weight = min(nearest_weight, G[home_id][neighbor]["weight"])

        if neighbor_count == 0:
            return 1.0

        proximity = 1.0 / (1.0 + nearest_weight)
        return 1.0 / (1.0 + 0.45 * neighbor_count + 1.8 * proximity)

    def pull_components(home_id, model, adapter):
        home_edge = edges[home_id]

        model_pull_delay = 0.0
        if model.id not in home_edge.cached_models:
            providers = [
                neighbor
                for neighbor in G.neighbors(home_edge.id)
                if neighbor != "DC" and model.id in edges[neighbor].cached_models
            ]
            if not providers:
                model_pull_delay = G[home_edge.id]["DC"]["weight"] * model.size
            else:
                providers.sort(key=lambda node: G[home_edge.id][node]["weight"])
                selected = providers[:3]
                max_weight = max(G[home_edge.id][node]["weight"] for node in selected)
                model_pull_delay = max_weight * (model.size / len(selected))

        adp_key = (adapter.model_id, adapter.service_type)
        adapter_pull_delay = 0.0
        if adp_key not in home_edge.cached_adapters:
            providers = [
                neighbor
                for neighbor in G.neighbors(home_edge.id)
                if neighbor != "DC" and adp_key in edges[neighbor].cached_adapters
            ]
            if not providers:
                adapter_pull_delay = G[home_edge.id]["DC"]["weight"] * adapter.size
            else:
                providers.sort(key=lambda node: G[home_edge.id][node]["weight"])
                selected = providers[:3]
                max_weight = max(G[home_edge.id][node]["weight"] for node in selected)
                adapter_pull_delay = max_weight * (adapter.size / len(selected))

        return model_pull_delay, adapter_pull_delay

    def future_cache_value(home_id, req_type, model, adapter, edge, action_name):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        model_pull_delay, adapter_pull_delay = pull_components(home_id, model, adapter)
        adp_key = (adapter.model_id, adapter.service_type)

        future_value = 0.0
        if cache_model and model.id not in edge.cached_models:
            service_scope = model_service_types.get(model.id, {req_type})
            model_reuse = 0.0
            for service_type in service_scope:
                model_reuse += 0.7 * math.log1p(
                    blended_service_seen(home_id, service_type)
                )
                model_reuse += 1.0 * math.log1p(
                    blended_model_success(home_id, model.id, service_type)
                )
            model_reuse /= max(len(service_scope), 1)
            future_value += (
                neighbor_availability_factor(home_id, "model", model.id)
                * model_reuse
                * (model_pull_delay / 1000.0)
            )
        if cache_adapter and adp_key not in edge.cached_adapters:
            adapter_reuse = 0.8 * math.log1p(blended_service_seen(home_id, req_type))
            adapter_reuse += 1.2 * math.log1p(
                blended_model_success(home_id, model.id, req_type)
            )
            future_value += (
                1.2
                * neighbor_availability_factor(home_id, "adapter", adp_key)
                * adapter_reuse
                * (adapter_pull_delay / 1000.0)
            )

        return future_value

    def cache_action_bonus(home_id, req_type, model, adapter, edge, action_name):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        if action_name == "none":
            return 0.12 * math.log1p(blended_model_success(home_id, model.id, req_type))

        reuse_signal = math.log1p(blended_service_seen(home_id, req_type))
        reuse_signal += math.log1p(
            blended_model_success(home_id, model.id, req_type) + 1.0
        )
        bonus = 0.0
        adp_key = (adapter.model_id, adapter.service_type)
        if cache_model and model.id not in edge.cached_models:
            service_scope = model_service_types.get(model.id, {req_type})
            cross_service_reuse = sum(
                math.log1p(blended_service_seen(home_id, service_type))
                for service_type in service_scope
            ) / max(len(service_scope), 1)
            bonus += (
                0.65
                * (reuse_signal + 0.6 * cross_service_reuse)
                / (1.0 + model.size / 1024.0)
            )
        if cache_adapter and adp_key not in edge.cached_adapters:
            bonus += 0.9 * reuse_signal / (1.0 + adapter.size)
        return bonus

    def storage_write_mb(model, adapter, edge, action_name):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        new_storage_mb = 0.0
        if cache_model and model.id not in edge.cached_models:
            new_storage_mb += model.size
        adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adp_key not in edge.cached_adapters:
            new_storage_mb += adapter.size
        return new_storage_mb

    def immediate_action_score(req, model, adapter, delay, edge, action_name):
        new_storage_mb = storage_write_mb(model, adapter, edge, action_name)
        deployment_cost = deployment_cost_weight * (new_storage_mb / 1024.0)
        pressure = resource_penalty(model, adapter, edge, action_name)
        base_reward = req.reward - delay_weight * delay - deployment_cost
        return (base_reward / 10.0) - 0.45 * pressure, base_reward

    def transient_pull_penalty(home_id, user, model, adapter, edge, action_name, delay):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        model_pull_delay, adapter_pull_delay = pull_components(home_id, model, adapter)
        adp_key = (adapter.model_id, adapter.service_type)
        delay_slack = max(user.delay - delay, 0.0) / max(user.delay, 1.0)

        penalty = 0.0
        if not cache_model and model.id not in edge.cached_models:
            model_scope = len(model_service_types.get(model.id, {user.request.type}))
            penalty += (
                0.15
                * model_scope
                * neighbor_availability_factor(home_id, "model", model.id)
                * (model_pull_delay / max(user.delay, 1.0))
                * (1.2 - delay_slack)
            )

        if not cache_adapter and adp_key not in edge.cached_adapters:
            service_hotness = math.log1p(
                blended_service_seen(home_id, user.request.type)
            )
            penalty += 0.2 + 0.45 * service_hotness
            penalty += (
                (0.55 + service_hotness)
                * neighbor_availability_factor(home_id, "adapter", adp_key)
                * (adapter_pull_delay / max(user.delay, 1.0))
                * (1.3 - delay_slack)
            )

        return penalty

    def should_skip_transient_action(home_id, req_type, adapter, edge, action_name):
        if action_name not in {"none", "model_only"}:
            return False

        adp_key = (adapter.model_id, adapter.service_type)
        if adp_key in edge.cached_adapters:
            return False

        local_repeat_count = local_service_seen.get((home_id, req_type), 0.0)
        return local_repeat_count >= 2.0

    def reject_action_value(home_id, req_type, edge, best_service_score):
        reject_key = ("__reject__", req_type, "reject")
        posterior = posterior_action_value(home_id, reject_key)
        storage_pressure = edge.used_storage / max(edge.storage_capacity, 1.0)
        memory_pressure = edge.used_memory / max(edge.memory_capacity, 1.0)
        combined_pressure = storage_pressure + 1.2 * memory_pressure
        scarcity = math.log1p(blended_service_seen(home_id, req_type))
        relief_gain = max(0.0, combined_pressure - 1.05) * max(
            0.0, 1.1 - best_service_score
        )
        reject_score = posterior + 0.15 * scarcity + 1.2 * relief_gain - 0.5
        return reject_key, reject_score

    def resource_penalty(model, adapter, edge, action_name):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        extra_storage = 0.0
        extra_gpu = 0.0
        adp_key = (adapter.model_id, adapter.service_type)

        if cache_model and model.id not in edge.cached_models:
            extra_storage += model.size
        if cache_adapter and adp_key not in edge.cached_adapters:
            extra_storage += adapter.size
        if model.id not in edge.loaded_models:
            extra_gpu += edge.delta * model.size
        if adp_key not in edge.loaded_adapters:
            extra_gpu += edge.delta * adapter.size

        storage_ratio = extra_storage / max(edge.storage_capacity, 1.0)
        gpu_ratio = extra_gpu / max(edge.memory_capacity, 1.0)
        return storage_ratio + 1.6 * gpu_ratio

    def enumerate_feasible_joint_actions(user, edge):
        req = user.request
        joint_candidates = []

        for fm in foundation_models:
            adp = adapters_dict.get((fm.id, req.type))
            if adp is None or adp.accuracy < user.accuracy:
                continue

            delay = p2p_delay(req.homeCloudlet, fm, adp, req.instruction, G, edges)
            if delay > user.delay:
                continue

            for action_name in CACHE_ACTIONS:
                if should_skip_transient_action(
                    req.homeCloudlet,
                    req.type,
                    adp,
                    edge,
                    action_name,
                ):
                    continue

                eviction_plan, eviction_cost = build_eviction_plan(
                    req.homeCloudlet,
                    edge,
                    fm,
                    adp,
                    action_name,
                )
                if eviction_plan is None:
                    continue

                action_key = (fm.id, req.type, action_name, eviction_plan)
                ensure_action_stats(req.homeCloudlet, action_key)

                bonus = cache_action_bonus(
                    req.homeCloudlet, req.type, fm, adp, edge, action_name
                )
                future_value = future_cache_value(
                    req.homeCloudlet,
                    req.type,
                    fm,
                    adp,
                    edge,
                    action_name,
                )
                immediate_score, _base_reward = immediate_action_score(
                    req,
                    fm,
                    adp,
                    delay,
                    edge,
                    action_name,
                )
                stability_penalty = transient_pull_penalty(
                    req.homeCloudlet,
                    user,
                    fm,
                    adp,
                    edge,
                    action_name,
                    delay,
                )
                pressure = resource_penalty(fm, adp, edge, action_name)
                repeat_signal = math.log1p(
                    blended_service_seen(req.homeCloudlet, req.type)
                )
                home_signal = math.log1p(blended_home_activity(req.homeCloudlet))
                reuse_lift = 0.0
                if (
                    action_name in {"model_only", "full"}
                    and fm.id not in edge.cached_models
                ):
                    model_scope = len(model_service_types.get(fm.id, {req.type}))
                    reuse_lift += (0.08 + 0.02 * model_scope) * repeat_signal
                    reuse_lift += 0.03 * home_signal
                if action_name == "full":
                    reuse_lift += 0.22 * repeat_signal
                elif action_name == "model_only":
                    reuse_lift -= 0.20 * repeat_signal
                elif action_name == "adapter" and fm.id not in edge.cached_models:
                    reuse_lift -= 0.03 * repeat_signal
                elif action_name == "none":
                    reuse_lift -= 0.10 * repeat_signal

                # 将启发式项降为弱先验，由 bandit 统计主导动作价值。
                action_prior = (
                    0.18 * immediate_score
                    + 0.10 * bonus
                    + 0.12 * future_value
                    + reuse_lift
                    - 0.08 * pressure
                    - 0.10 * stability_penalty
                    - 0.06 * eviction_cost
                )
                action_value = (
                    posterior_action_value(req.homeCloudlet, action_key) + action_prior
                )
                joint_candidates.append(
                    (
                        fm,
                        adp,
                        delay,
                        action_key,
                        action_name,
                        action_value,
                        eviction_plan,
                        eviction_cost,
                        bonus,
                        future_value,
                        immediate_score,
                    )
                )

        return joint_candidates

    def retention_score(home_id, edge, kind, key):
        if kind == "model":
            size = edge.model_sizes.get(key, fm_map[key].size)
        else:
            size = edge.adapter_sizes.get(key, adapters_dict[key].size)

        hits = math.log1p(item_hits.get((home_id, kind, key), 0.0))
        total_entries = max(len(edge.cache_fifo), 1)
        try:
            recency_rank = edge.cache_fifo.index((kind, key)) + 1
        except ValueError:
            recency_rank = 0
        recency = recency_rank / total_entries
        scarcity = neighbor_availability_factor(home_id, kind, key)
        size_penalty = size / 1024.0
        return 1.4 * hits + 0.8 * recency + 1.2 * scarcity - 0.35 * size_penalty

    def build_eviction_plan(home_id, edge, model, adapter, action_name):
        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        extra_storage = 0.0
        if cache_model and model.id not in edge.cached_models:
            extra_storage += model.size
        adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adp_key not in edge.cached_adapters:
            extra_storage += adapter.size

        need_to_free = max(
            0.0, edge.used_storage + extra_storage - edge.storage_capacity
        )
        if need_to_free <= 1e-9:
            return (), 0.0

        protected_entries = {("model", model.id), ("adapter", adp_key)}
        candidates = []
        for kind, key in edge.cache_fifo:
            if (kind, key) in protected_entries:
                continue
            if kind == "model":
                size = edge.model_sizes.get(key, fm_map[key].size)
            else:
                size = edge.adapter_sizes.get(key, adapters_dict[key].size)
            candidates.append(
                (retention_score(home_id, edge, kind, key), size, kind, key)
            )

        candidates.sort(key=lambda item: (item[0], -item[1], item[2], str(item[3])))

        freed = 0.0
        plan = []
        eviction_cost = 0.0
        for score, size, kind, key in candidates:
            plan.append((kind, key))
            freed += size
            eviction_cost += max(score, 0.0)
            if freed + 1e-9 >= need_to_free:
                return tuple(plan), eviction_cost

        return None, float("inf")

    def update_item_hits(home_id, edge, model, adapter, action_name):
        adp_key = (adapter.model_id, adapter.service_type)
        touched_entries = []
        if model.id in edge.cached_models or CACHE_ACTIONS[action_name][0]:
            touched_entries.append(("model", model.id))
        if adp_key in edge.cached_adapters or CACHE_ACTIONS[action_name][1]:
            touched_entries.append(("adapter", adp_key))

        for kind, key in touched_entries:
            stat_key = (home_id, kind, key)
            item_hits[stat_key] = item_hits.get(stat_key, 0.0) + 1.0

    for step_idx, user in enumerate(users):
        if step_idx > 0 and step_idx % stats_decay_interval == 0:
            decay_stats()

        req = user.request
        home_id = req.homeCloudlet
        edge = edges[home_id]

        joint_candidates = enumerate_feasible_joint_actions(user, edge)
        if not joint_candidates:
            global_service_seen[req.type] = global_service_seen.get(req.type, 0.0) + 1.0
            local_service_seen[(home_id, req.type)] = (
                local_service_seen.get((home_id, req.type), 0.0) + 1.0
            )
            continue

        selected = None
        if joint_candidates:
            best_service_score = max(item[10] for item in joint_candidates)
            reject_key, reject_score = reject_action_value(
                home_id,
                req.type,
                edge,
                best_service_score,
            )

            epsilon_t = current_epsilon(step_idx)
            if rng.random() < epsilon_t:
                selected = rng.choice(joint_candidates)
            else:
                best_service_action = max(
                    joint_candidates,
                    key=lambda item: (item[5], item[10], -item[2]),
                )
                storage_pressure = edge.used_storage / max(edge.storage_capacity, 1.0)
                memory_pressure = edge.used_memory / max(edge.memory_capacity, 1.0)
                reject_enabled = (
                    (storage_pressure >= 0.98 or memory_pressure >= 0.98)
                    and best_service_score < 0.65
                    and reject_score > best_service_action[5] + 0.15
                )
                selected = None if reject_enabled else best_service_action

        if selected is None:
            if joint_candidates:
                update_action_stats(home_id, reject_key, 0.0)

            global_service_seen[req.type] = global_service_seen.get(req.type, 0.0) + 1.0
            local_service_seen[(home_id, req.type)] = (
                local_service_seen.get((home_id, req.type), 0.0) + 1.0
            )
            continue

        (
            fm,
            adp,
            delay,
            action_key,
            action_name,
            _action_value,
            eviction_plan,
            eviction_cost,
            bonus,
            future_value,
            _immediate_score,
        ) = selected

        cache_model, cache_adapter = CACHE_ACTIONS[action_name]
        adp_key = (adp.model_id, adp.service_type)
        model_preloaded = fm.id in edge.loaded_models
        adapter_preloaded = adp_key in edge.loaded_adapters
        model_will_write = cache_model and fm.id not in edge.cached_models
        adapter_will_write = cache_adapter and adp_key not in edge.cached_adapters

        if edge.try_apply_action_with_eviction_plan(
            fm,
            adp,
            cache_model,
            cache_adapter,
            eviction_plan,
        ):
            new_storage_mb = 0.0
            if model_will_write:
                new_storage_mb += fm.size
            if adapter_will_write:
                new_storage_mb += adp.size

            deployment_cost = deployment_cost_weight * (new_storage_mb / 1024.0)
            realized_reward = req.reward - delay_weight * delay - deployment_cost
            total_reward += realized_reward
            admitted.append(user)

            if not cache_model and not model_preloaded and fm.id in edge.loaded_models:
                edge.loaded_models.remove(fm.id)
                edge.used_memory -= edge.delta * fm.size
                remove_fifo_entry(edge.load_fifo, "model", fm.id)

            if (
                not cache_adapter
                and not adapter_preloaded
                and adp_key in edge.loaded_adapters
            ):
                edge.loaded_adapters.remove(adp_key)
                edge.used_memory -= edge.delta * adp.size
                remove_fifo_entry(edge.load_fifo, "adapter", adp_key)

            global_model_success[(fm.id, req.type)] = (
                global_model_success.get((fm.id, req.type), 0.0) + 1.0
            )
            local_model_success[(home_id, fm.id, req.type)] = (
                local_model_success.get((home_id, fm.id, req.type), 0.0) + 1.0
            )
            update_item_hits(home_id, edge, fm, adp, action_name)
            reward = realized_reward
        else:
            reward = -penalty

        update_action_stats(home_id, action_key, reward)
        global_service_seen[req.type] = global_service_seen.get(req.type, 0.0) + 1.0
        local_service_seen[(home_id, req.type)] = (
            local_service_seen.get((home_id, req.type), 0.0) + 1.0
        )

    return admitted, total_reward
