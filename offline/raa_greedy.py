import copy


def calculate_delay(homeCloudlet_id, fm, adp, instruction, G, edges):
    """
    计算延迟
    总延迟 = Model拉取延迟 + Adapter拉取延迟 + 推理延迟
    严格对应论文公式: D_{k,j} = D^{model}_{k,j} + D^{weight}_{k,j} + D^{inf}_{k,j}
    """
    homeCloudlet = edges[homeCloudlet_id]

    # 基座模型拉取延迟
    model_pull_delay = 0
    # 如果本地已经缓存了基座模型延迟为 0
    if fm.id not in homeCloudlet.cached_models:
        model_providers = []
        # 寻找拥有该模型的cloudlet
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and fm.id in edges[neighbor].cached_models:
                model_providers.append(neighbor)

        if not model_providers:
            # 如果全网都没缓存 -> 触发回源
            model_pull_delay = G[homeCloudlet.id]["DC"]["weight"] * fm.size
        else:
            # P2P 并行拉取
            model_providers.sort(key=lambda p: G[homeCloudlet.id][p]["weight"])
            selected_model_peers = model_providers[:3]
            max_weight = max(
                G[homeCloudlet.id][p]["weight"] for p in selected_model_peers
            )
            model_pull_delay = max_weight * (fm.size / len(selected_model_peers))

    # 计算 Adapter 拉取延迟
    adp_pull_delay = 0
    adp_key = (adp.model_id, adp.service_type)

    # 如果本地已经缓存了 Adapter延迟为 0
    if adp_key not in homeCloudlet.cached_adapters:
        adp_providers = []
        # 寻找拥有该Adapter的cloudlet
        for neighbor in G.neighbors(homeCloudlet.id):
            if neighbor != "DC" and adp_key in edges[neighbor].cached_adapters:
                adp_providers.append(neighbor)

        if not adp_providers:
            # 如果全网都没缓存，触发回源
            bts_weight = G[homeCloudlet.id]["DC"]["weight"]
            adp_pull_delay = bts_weight * adp.size
        else:
            # P2P 并行拉取
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


from MHS import MHS


def _build_user_plans(
    users,
    foundation_models,
    adapters_dict,
    G,
    edges,
    lambda_delay,
):
    # 对某个 edge 上的每个用户，记录 “用户可接受哪些 foundation model”。
    # 之后 MHS 会从这些候选模型里挑一个尽量小的集合，使每个用户都至少被一个模型覆盖。
    candidate = {}
    # 后面做 reward 比较和剪枝时直接复用，不重复计算 delay / reward。
    user_plan_map = {}

    for user in users:
        req = user.request
        home_id = req.homeCloudlet

        valid_models = []
        plan_by_model = {}
        for fm in foundation_models:
            adapter = adapters_dict.get((fm.id, req.type))
            if adapter is None:
                continue

            # 用户的精度要求高于 adapter 可提供精度时，模型不可用。
            if user.accuracy > adapter.accuracy:
                continue

            delay = calculate_delay(home_id, fm, adapter, req.instruction, G, edges)
            # 单个请求仍然必须满足自己的时延约束。
            if delay > user.delay:
                continue

            eff_reward = req.reward - lambda_delay * delay
            valid_models.append(fm.id)
            plan_by_model[fm.id] = {
                "fm": fm,
                "adapter": adapter,
                "delay": delay,
                "eff_reward": eff_reward,
            }

        if valid_models:
            candidate.setdefault(home_id, []).append(valid_models)
            user_plan_map[user] = plan_by_model

    return candidate, user_plan_map


def _best_plan_for_user(user, user_plan_map, current_models, blocked_adapters=None):
    # 在当前保留下来的模型集合里，为这个用户挑一个最优执行方案。
    best_mid = None
    best_plan = None
    blocked_adapters = blocked_adapters or set()

    for mid, plan in user_plan_map[user].items():
        if mid not in current_models:
            continue
        adapter_key = (plan["adapter"].model_id, plan["adapter"].service_type)
        if adapter_key in blocked_adapters:
            continue
        if best_plan is None:
            best_mid = mid
            best_plan = plan
            continue
        if plan["eff_reward"] > best_plan["eff_reward"] + 1e-9 or (
            abs(plan["eff_reward"] - best_plan["eff_reward"]) <= 1e-9
            and plan["delay"] < best_plan["delay"]
        ):
            best_mid = mid
            best_plan = plan

    return best_mid, best_plan


def _collect_edge_plans(
    edge_id,
    users,
    user_plan_map,
    current_models,
    blocked_adapters=None,
):
    # 先忽略容量，只看“每个用户最想选哪个模型”，形成一个临时分配方案。
    admitted_users = []
    allocation = {}
    total_eff_reward = 0.0

    for user in users:
        if user.request.homeCloudlet != edge_id or user not in user_plan_map:
            continue

        _, best_plan = _best_plan_for_user(
            user,
            user_plan_map,
            current_models,
            blocked_adapters=blocked_adapters,
        )
        # eff_reward <= 0 说明即便能服务，这个请求也不值得接。
        if best_plan is None or best_plan["eff_reward"] <= 0:
            continue

        admitted_users.append(user)
        allocation[user] = (best_plan["fm"], best_plan["adapter"])
        total_eff_reward += best_plan["eff_reward"]

    return admitted_users, allocation, total_eff_reward


def _allocation_is_feasible(edge, allocation):
    # 复用收益。
    unique_models = {}
    unique_adapters = {}

    for fm, adapter in allocation.values():
        unique_models[fm.id] = fm
        unique_adapters[(adapter.model_id, adapter.service_type)] = adapter

    return edge.check_batch_capacity(
        list(unique_models.values()), list(unique_adapters.values())
    )


def _choose_pruned_model(edge_id, users, user_plan_map, current_models, fm_map):
    # 当前模型集合超容量时，尝试删掉一个模型。
    # 枚举，看删完后还能保留多少总有效收益；如果收益差不多了，就删掉更大的模型（因为大模型更占资源，更可能是瓶颈）。
    best_model_to_remove = None
    best_remaining_reward = float("-inf")
    best_removed_size = float("-inf")

    for mid in current_models:
        remaining_models = set(current_models)
        remaining_models.remove(mid)
        _, _, remaining_reward = _collect_edge_plans(
            edge_id, users, user_plan_map, remaining_models
        )
        removed_size = fm_map[mid].size

        if remaining_reward > best_remaining_reward + 1e-9 or (
            abs(remaining_reward - best_remaining_reward) <= 1e-9
            and removed_size > best_removed_size
        ):
            best_model_to_remove = mid
            best_remaining_reward = remaining_reward
            best_removed_size = removed_size

    return best_model_to_remove


def _choose_pruned_adapter(
    edge_id,
    users,
    user_plan_map,
    current_models,
    blocked_adapters,
    allocation,
    adapters_dict,
):
    best_adapter_to_remove = None
    best_remaining_reward = float("-inf")
    best_removed_size = float("-inf")

    current_adapter_keys = {
        (adapter.model_id, adapter.service_type) for _, adapter in allocation.values()
    }
    for adapter_key in current_adapter_keys:
        if adapter_key in blocked_adapters:
            continue

        remaining_blocked = set(blocked_adapters)
        remaining_blocked.add(adapter_key)
        _, _, remaining_reward = _collect_edge_plans(
            edge_id,
            users,
            user_plan_map,
            current_models,
            blocked_adapters=remaining_blocked,
        )
        removed_size = adapters_dict[adapter_key].size

        if remaining_reward > best_remaining_reward + 1e-9 or (
            abs(remaining_reward - best_remaining_reward) <= 1e-9
            and removed_size > best_removed_size
        ):
            best_adapter_to_remove = adapter_key
            best_remaining_reward = remaining_reward
            best_removed_size = removed_size

    return best_adapter_to_remove


def _estimate_incremental_cost(edge, fm, adapter):
    extra_storage = 0.0
    extra_gpu = 0.0

    if fm.id not in edge.cached_models:
        extra_storage += fm.size
    adapter_key = (adapter.model_id, adapter.service_type)
    if adapter_key not in edge.cached_adapters:
        extra_storage += adapter.size

    if fm.id not in edge.loaded_models:
        extra_gpu += edge.delta * fm.size
    if adapter_key not in edge.loaded_adapters:
        extra_gpu += edge.delta * adapter.size

    return extra_storage + extra_gpu


def _greedy_partial_admission(edge, admitted_users, allocation, user_plan_map):
    temp_edge = copy.deepcopy(edge)

    def user_rank(user):
        fm, adapter = allocation[user]
        eff_reward = user_plan_map[user][fm.id]["eff_reward"]
        incremental_cost = _estimate_incremental_cost(temp_edge, fm, adapter)
        ratio = eff_reward / (incremental_cost + 1e-6)
        return (ratio, eff_reward, -user_plan_map[user][fm.id]["delay"])

    ranked_users = sorted(admitted_users, key=user_rank, reverse=True)

    selected_users = []
    selected_allocation = {}
    total_eff_reward = 0.0

    for user in ranked_users:
        fm, adapter = allocation[user]
        if not temp_edge.check_capacity(fm, adapter):
            continue
        temp_edge.allocate(fm, adapter)
        selected_users.append(user)
        selected_allocation[user] = (fm, adapter)
        total_eff_reward += user_plan_map[user][fm.id]["eff_reward"]

    return selected_users, selected_allocation, total_eff_reward


def _best_single_model_fallback(edge_id, users, user_plan_map, edge):
    best_users = []
    best_allocation = {}
    best_reward = float("-inf")

    candidate_models = set()
    for user in users:
        if user.request.homeCloudlet != edge_id or user not in user_plan_map:
            continue
        candidate_models.update(user_plan_map[user].keys())

    for mid in candidate_models:
        admitted_for_edge, allocation, _ = _collect_edge_plans(
            edge_id,
            users,
            user_plan_map,
            {mid},
        )
        if not admitted_for_edge:
            continue

        partial_users, partial_allocation, partial_reward = _greedy_partial_admission(
            edge,
            admitted_for_edge,
            allocation,
            user_plan_map,
        )
        if partial_reward > best_reward + 1e-9 or (
            abs(partial_reward - best_reward) <= 1e-9
            and len(partial_users) > len(best_users)
        ):
            best_users = partial_users
            best_allocation = partial_allocation
            best_reward = partial_reward

    return best_users, best_allocation, best_reward


def raa_greedy(
    G,
    users,
    edges,
    foundation_models,
    adapters_dict,
    fm_dict=None,
    lambda_delay=1e-3,
    return_total_reward=False,
):
    """
    RAA-Greedy 算法
    思路：
    1. 先对每个用户筛出“哪些模型能满足精度和时延”；
    2. 对每个 edge, 用 MHS 找一个尽量小的模型集合去覆盖这些用户；
    3. 如果整批方案不可行，先在当前模型集合内做 adapter 级剪枝；
    4. 若仍不可行，则在保留的模型和 adapter 上做部分接纳；
    5. 若当前 hitting-set 连部分接纳都不可行，则切换到单模型 fallback；
    6. 只有前面都失败时，才继续剪掉一个“删了以后损失最小”的模型。
    """
    fm_map = fm_dict or {fm.id: fm for fm in foundation_models}
    candidate, user_plan_map = _build_user_plans(
        users,
        foundation_models,
        adapters_dict,
        G,
        edges,
        lambda_delay,
    )

    admitted_requests = []
    total_reward = 0.0

    for edge_id, cand_list in candidate.items():
        edge = edges[edge_id]
        # MHS 返回一个最小命中集
        hitman = MHS(cand_list)
        base_hit_set = hitman.get()
        if not base_hit_set:
            continue

        current_models = set(base_hit_set)
        while current_models:
            blocked_adapters = set()
            best_partial_users = []
            best_partial_allocation = {}
            best_partial_reward = float("-inf")

            while True:
                admitted_for_edge, allocation, edge_reward = _collect_edge_plans(
                    edge_id,
                    users,
                    user_plan_map,
                    current_models,
                    blocked_adapters=blocked_adapters,
                )

                if not admitted_for_edge:
                    break

                # 这里优先尝试整批接纳，保留 RAA 的共享模型导向。
                if _allocation_is_feasible(edge, allocation):
                    for user in admitted_for_edge:
                        fm, adapter = allocation[user]
                        edge.allocate(fm, adapter)
                    admitted_requests.extend(admitted_for_edge)
                    total_reward += edge_reward
                    current_models = set()
                    break

                partial_users, partial_allocation, partial_reward = (
                    _greedy_partial_admission(
                        edge,
                        admitted_for_edge,
                        allocation,
                        user_plan_map,
                    )
                )
                if partial_reward > best_partial_reward + 1e-9:
                    best_partial_users = partial_users
                    best_partial_allocation = partial_allocation
                    best_partial_reward = partial_reward

                pruned_adapter = _choose_pruned_adapter(
                    edge_id,
                    users,
                    user_plan_map,
                    current_models,
                    blocked_adapters,
                    allocation,
                    adapters_dict,
                )
                if pruned_adapter is None:
                    break
                blocked_adapters.add(pruned_adapter)

            if not current_models:
                break

            single_model_users, single_model_allocation, single_model_reward = (
                _best_single_model_fallback(edge_id, users, user_plan_map, edge)
            )
            if single_model_reward > best_partial_reward + 1e-9:
                best_partial_users = single_model_users
                best_partial_allocation = single_model_allocation
                best_partial_reward = single_model_reward

            if best_partial_users:
                for user in best_partial_users:
                    fm, adapter = best_partial_allocation[user]
                    edge.allocate(fm, adapter)
                admitted_requests.extend(best_partial_users)
                total_reward += best_partial_reward
                break

            # 兜底时才回退到模型级剪枝，避免一个大热点 edge 被整体放弃。
            pruned_model = _choose_pruned_model(
                edge_id, users, user_plan_map, current_models, fm_map
            )
            if pruned_model is None:
                break
            current_models.remove(pruned_model)

    if return_total_reward:
        return admitted_requests, total_reward
    return admitted_requests
