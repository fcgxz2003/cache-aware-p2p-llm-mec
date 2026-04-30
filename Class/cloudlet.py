class Cloudlet:
    """
    边缘节点的类
    """

    def __init__(self, id, memory_capacity, storage_capacity, eta):
        self.id = id
        self.memory_capacity = memory_capacity
        self.storage_capacity = storage_capacity
        self.eta = eta

        self.cached_models = set()  # x_{i,j}
        self.cached_adapters = set()  # y_{i,j,m} (model_id, service_type)
        self.loaded_models = set()  # u_{i,j}
        self.loaded_adapters = set()  # (model_id, service_type)

        self.used_storage = 0.0
        self.used_memory = 0.0

        self.delta = 2.0  # FP16 占 2 Bytes

        # 记录模型/适配器大小，便于驱逐时更新资源使用量
        self.model_sizes = {}
        self.adapter_sizes = {}

        # LRU 队列：按“最近使用”顺序记录缓存与加载的条目，用于在线算法驱逐
        # 缓存队列元素: ("model"/"adapter", key)
        #   - model: key 为 model.id
        #   - adapter: key 为 (model_id, service_type)
        # 显存队列元素同理
        self.cache_fifo = []
        self.load_fifo = []

    def check_capacity(self, model, adapter):
        """
        检查单个模型和Adapter能否放入当前的Cloudlet。
        考虑到缓存复用：如果已经存在，则额外占用为 0。
        """
        extra_storage = 0.0
        extra_gpu = 0.0

        # 检查磁盘存储的额外需求
        if model.id not in self.cached_models:
            extra_storage += model.size
        if (adapter.model_id, adapter.service_type) not in self.cached_adapters:
            extra_storage += adapter.size

        # 检查GPU显存的额外需求
        if model.id not in self.loaded_models:
            extra_gpu += self.delta * model.size
        if (adapter.model_id, adapter.service_type) not in self.loaded_adapters:
            extra_gpu += self.delta * adapter.size

        # 容量判定
        has_space = (self.used_storage + extra_storage) <= self.storage_capacity and (
            self.used_memory + extra_gpu
        ) <= self.memory_capacity

        return has_space

    def allocate(self, model, adapter):
        """
        正式接纳请求后，分配资源，更新节点状态
        """
        # 扣除存储资源并更新状态
        if model.id not in self.cached_models:
            self.cached_models.add(model.id)
            self.used_storage += model.size
            self.model_sizes[model.id] = model.size
            self.cache_fifo.append(("model", model.id))

        adp_key = (adapter.model_id, adapter.service_type)
        if adp_key not in self.cached_adapters:
            self.cached_adapters.add(adp_key)
            self.used_storage += adapter.size
            self.adapter_sizes[adp_key] = adapter.size
            self.cache_fifo.append(("adapter", adp_key))

        # 扣除显存资源并更新状态
        if model.id not in self.loaded_models:
            self.loaded_models.add(model.id)
            self.used_memory += self.delta * model.size
            self.load_fifo.append(("model", model.id))

        if adp_key not in self.loaded_adapters:
            self.loaded_adapters.add(adp_key)
            self.used_memory += self.delta * adapter.size
            self.load_fifo.append(("adapter", adp_key))

    def check_batch_capacity(self, model_list, adapter_list):
        """
        批量检查一组模型和Adapter作为一个整体能否同时放入当前节点
        """
        extra_storage = 0.0
        extra_gpu = 0.0

        for model in model_list:
            if model.id not in self.cached_models:
                extra_storage += model.size
            if model.id not in self.loaded_models:
                extra_gpu += self.delta * model.size

        for adapter in adapter_list:
            if (adapter.model_id, adapter.service_type) not in self.cached_adapters:
                extra_storage += adapter.size
            if (adapter.model_id, adapter.service_type) not in self.loaded_adapters:
                extra_gpu += self.delta * adapter.size

        has_space = (self.used_storage + extra_storage) <= self.storage_capacity and (
            self.used_memory + extra_gpu
        ) <= self.memory_capacity
        return has_space

    def check_action_capacity(self, model, adapter, cache_model, cache_adapter):
        """针对 LinUCB 的动作 (cache_model, cache_adapter) 检查容量。

        cache_model / cache_adapter 表示**是否写入本地缓存(磁盘)**；
        无论是否缓存，只要当前请求要执行推理，就必须把基座模型和适配器加载到 GPU，
        因此显存占用始终按 "尚未 loaded" 来计算额外开销。
        """
        extra_storage = 0.0
        extra_gpu = 0.0

        # 磁盘缓存：只在 cache_* 为 1 且当前未缓存时增加存储开销
        if cache_model and model.id not in self.cached_models:
            extra_storage += model.size
        edge_adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and edge_adp_key not in self.cached_adapters:
            extra_storage += adapter.size

        # GPU 显存：只要当前还没加载，就需要为本次推理分配显存
        if model.id not in self.loaded_models:
            extra_gpu += self.delta * model.size
        if edge_adp_key not in self.loaded_adapters:
            extra_gpu += self.delta * adapter.size

        return (self.used_storage + extra_storage) <= self.storage_capacity and (
            self.used_memory + extra_gpu
        ) <= self.memory_capacity

    def apply_action(self, model, adapter, cache_model, cache_adapter):
        """根据 LinUCB 决策执行一次动作：更新缓存与显存状态。

        - 若 cache_model == 1，则将基座模型写入磁盘缓存；
        - 若 cache_adapter == 1，则将适配器写入磁盘缓存；
        - 无论缓存与否，只要本次请求被接纳，均需确保模型与适配器已加载到 GPU，
          并相应增加 used_memory（若此前未加载）。
        """
        # 更新磁盘缓存状态
        if cache_model and model.id not in self.cached_models:
            self.cached_models.add(model.id)
            self.used_storage += model.size
            self.model_sizes[model.id] = model.size
            self.cache_fifo.append(("model", model.id))

        adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adp_key not in self.cached_adapters:
            self.cached_adapters.add(adp_key)
            self.used_storage += adapter.size
            self.adapter_sizes[adp_key] = adapter.size
            self.cache_fifo.append(("adapter", adp_key))

        # 更新 GPU 显存占用（只要这次请求执行，就需要加载到 GPU）
        if model.id not in self.loaded_models:
            self.loaded_models.add(model.id)
            self.used_memory += self.delta * model.size
            self.load_fifo.append(("model", model.id))

        if adp_key not in self.loaded_adapters:
            self.loaded_adapters.add(adp_key)
            self.used_memory += self.delta * adapter.size
            self.load_fifo.append(("adapter", adp_key))

    # ==== 以下方法仅供 online P2P / BTS / LinUCB 等在线算法使用（支持 LRU 驱逐）====

    def _touch_entry(self, fifo_list, kind, key):
        """在 LRU 队列中将某个条目标记为最近使用。"""
        try:
            idx = fifo_list.index((kind, key))
        except ValueError:
            return
        fifo_list.pop(idx)
        fifo_list.append((kind, key))

    def _remove_entry_from_fifo(self, fifo_list, kind, key):
        try:
            fifo_list.remove((kind, key))
        except ValueError:
            pass

    def _drop_cache_entry(self, kind, key):
        self._remove_entry_from_fifo(self.cache_fifo, kind, key)
        if kind == "model" and key in self.cached_models:
            self.cached_models.remove(key)
            self.used_storage -= self.model_sizes.get(key, 0.0)
        elif kind == "adapter" and key in self.cached_adapters:
            self.cached_adapters.remove(key)
            self.used_storage -= self.adapter_sizes.get(key, 0.0)

    def _drop_load_entry(self, kind, key):
        self._remove_entry_from_fifo(self.load_fifo, kind, key)
        if kind == "model" and key in self.loaded_models:
            self.loaded_models.remove(key)
            self.used_memory -= self.delta * self.model_sizes.get(key, 0.0)
        elif kind == "adapter" and key in self.loaded_adapters:
            self.loaded_adapters.remove(key)
            self.used_memory -= self.delta * self.adapter_sizes.get(key, 0.0)

    def _snapshot_online_state(self):
        return {
            "cached_models": set(self.cached_models),
            "cached_adapters": set(self.cached_adapters),
            "loaded_models": set(self.loaded_models),
            "loaded_adapters": set(self.loaded_adapters),
            "used_storage": self.used_storage,
            "used_memory": self.used_memory,
            "cache_fifo": list(self.cache_fifo),
            "load_fifo": list(self.load_fifo),
        }

    def _restore_online_state(self, snapshot):
        self.cached_models = snapshot["cached_models"]
        self.cached_adapters = snapshot["cached_adapters"]
        self.loaded_models = snapshot["loaded_models"]
        self.loaded_adapters = snapshot["loaded_adapters"]
        self.used_storage = snapshot["used_storage"]
        self.used_memory = snapshot["used_memory"]
        self.cache_fifo = snapshot["cache_fifo"]
        self.load_fifo = snapshot["load_fifo"]

    def _build_cache_eviction_order(self, policy, protected_entries=None):
        protected_entries = protected_entries or set()
        candidates = [
            entry for entry in self.cache_fifo if entry not in protected_entries
        ]

        if policy == "keep_model":
            adapters = [entry for entry in candidates if entry[0] == "adapter"]
            models = [entry for entry in candidates if entry[0] == "model"]
            return adapters + models

        if policy == "keep_adapter":
            models = [entry for entry in candidates if entry[0] == "model"]
            adapters = [entry for entry in candidates if entry[0] == "adapter"]
            return models + adapters

        return candidates

    def _evict_cache_for_online(
        self,
        extra_storage,
        policy="lru",
        protected_entries=None,
    ):
        if (self.used_storage + extra_storage) <= self.storage_capacity:
            return True

        eviction_order = self._build_cache_eviction_order(policy, protected_entries)
        idx = 0
        while (self.used_storage + extra_storage) > self.storage_capacity and idx < len(
            eviction_order
        ):
            kind, key = eviction_order[idx]
            idx += 1
            self._drop_cache_entry(kind, key)

        return (self.used_storage + extra_storage) <= self.storage_capacity

    def _evict_load_for_online(self, extra_gpu, protected_entries=None):
        if (self.used_memory + extra_gpu) <= self.memory_capacity:
            return True

        protected_entries = protected_entries or set()
        eviction_order = [
            entry for entry in self.load_fifo if entry not in protected_entries
        ]
        idx = 0
        while (self.used_memory + extra_gpu) > self.memory_capacity and idx < len(
            eviction_order
        ):
            kind, key = eviction_order[idx]
            idx += 1
            self._drop_load_entry(kind, key)

        return (self.used_memory + extra_gpu) <= self.memory_capacity

    def _evict_lru_for_online(
        self,
        extra_storage,
        extra_gpu,
        protected_cache=None,
        protected_load=None,
    ):
        """在在线场景下按 LRU 驱逐缓存/加载条目以腾出资源。"""
        cache_ok = self._evict_cache_for_online(
            extra_storage,
            policy="lru",
            protected_entries=protected_cache,
        )
        if not cache_ok:
            return False

        return self._evict_load_for_online(
            extra_gpu,
            protected_entries=protected_load,
        )

    def try_allocate_with_lru(self, model, adapter):
        """在线 P2P / BTS 使用的分配接口：必要时按 LRU 驱逐再分配。

        - 若当前资源足够，直接调用 allocate；
        - 若不足，则按 LRU 驱逐旧的缓存/加载项后再尝试分配；
        - 若驱逐到极限仍不足，则返回 False，不接纳该请求。
        """
        # 计算该模型与适配器的额外资源需求
        extra_storage = 0.0
        extra_gpu = 0.0

        if model.id not in self.cached_models:
            extra_storage += model.size
        adp_key = (adapter.model_id, adapter.service_type)
        if adp_key not in self.cached_adapters:
            extra_storage += adapter.size

        if model.id not in self.loaded_models:
            extra_gpu += self.delta * model.size
        if adp_key not in self.loaded_adapters:
            extra_gpu += self.delta * adapter.size

        # 访问命中的条目，更新为最近使用（LRU）
        if model.id in self.cached_models:
            self._touch_entry(self.cache_fifo, "model", model.id)
        if adp_key in self.cached_adapters:
            self._touch_entry(self.cache_fifo, "adapter", adp_key)
        if model.id in self.loaded_models:
            self._touch_entry(self.load_fifo, "model", model.id)
        if adp_key in self.loaded_adapters:
            self._touch_entry(self.load_fifo, "adapter", adp_key)

        protected_entries = {("model", model.id), ("adapter", adp_key)}

        # 若无需驱逐即可放下，直接分配
        if (self.used_storage + extra_storage) <= self.storage_capacity and (
            self.used_memory + extra_gpu
        ) <= self.memory_capacity:
            self.allocate(model, adapter)
            return True

        snapshot = self._snapshot_online_state()
        if not self._evict_lru_for_online(
            extra_storage,
            extra_gpu,
            protected_cache=protected_entries,
            protected_load=protected_entries,
        ):
            self._restore_online_state(snapshot)
            return False

        self.allocate(model, adapter)
        return True

    def try_apply_action_with_policy(
        self,
        model,
        adapter,
        cache_model,
        cache_adapter,
        eviction_policy="lru",
    ):
        """在线 bandit 使用的分配接口：缓存动作与驱逐偏好共同构成动作。"""

        extra_storage = 0.0
        extra_gpu = 0.0

        if cache_model and model.id not in self.cached_models:
            extra_storage += model.size
        adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adp_key not in self.cached_adapters:
            extra_storage += adapter.size

        if model.id not in self.loaded_models:
            extra_gpu += self.delta * model.size
        if adp_key not in self.loaded_adapters:
            extra_gpu += self.delta * adapter.size

        if model.id in self.cached_models:
            self._touch_entry(self.cache_fifo, "model", model.id)
        if adp_key in self.cached_adapters:
            self._touch_entry(self.cache_fifo, "adapter", adp_key)
        if model.id in self.loaded_models:
            self._touch_entry(self.load_fifo, "model", model.id)
        if adp_key in self.loaded_adapters:
            self._touch_entry(self.load_fifo, "adapter", adp_key)

        protected_entries = {("model", model.id), ("adapter", adp_key)}

        if (self.used_storage + extra_storage) <= self.storage_capacity and (
            self.used_memory + extra_gpu
        ) <= self.memory_capacity:
            self.apply_action(model, adapter, cache_model, cache_adapter)
            return True

        snapshot = self._snapshot_online_state()

        if not self._evict_cache_for_online(
            extra_storage,
            policy=eviction_policy,
            protected_entries=protected_entries,
        ):
            self._restore_online_state(snapshot)
            return False

        if not self._evict_load_for_online(
            extra_gpu,
            protected_entries=protected_entries,
        ):
            self._restore_online_state(snapshot)
            return False

        self.apply_action(model, adapter, cache_model, cache_adapter)
        return True

    def try_apply_action_with_eviction_plan(
        self,
        model,
        adapter,
        cache_model,
        cache_adapter,
        evict_cache_entries=None,
    ):
        """在线 bandit 使用的分配接口：按显式给定的缓存驱逐计划执行动作。"""

        extra_storage = 0.0
        extra_gpu = 0.0

        if cache_model and model.id not in self.cached_models:
            extra_storage += model.size
        adp_key = (adapter.model_id, adapter.service_type)
        if cache_adapter and adp_key not in self.cached_adapters:
            extra_storage += adapter.size

        if model.id not in self.loaded_models:
            extra_gpu += self.delta * model.size
        if adp_key not in self.loaded_adapters:
            extra_gpu += self.delta * adapter.size

        if model.id in self.cached_models:
            self._touch_entry(self.cache_fifo, "model", model.id)
        if adp_key in self.cached_adapters:
            self._touch_entry(self.cache_fifo, "adapter", adp_key)
        if model.id in self.loaded_models:
            self._touch_entry(self.load_fifo, "model", model.id)
        if adp_key in self.loaded_adapters:
            self._touch_entry(self.load_fifo, "adapter", adp_key)

        protected_entries = {("model", model.id), ("adapter", adp_key)}
        evict_cache_entries = evict_cache_entries or ()

        snapshot = self._snapshot_online_state()

        if (self.used_storage + extra_storage) > self.storage_capacity:
            for kind, key in evict_cache_entries:
                if (kind, key) in protected_entries:
                    continue
                self._drop_cache_entry(kind, key)

        if (self.used_storage + extra_storage) > self.storage_capacity:
            self._restore_online_state(snapshot)
            return False

        if not self._evict_load_for_online(
            extra_gpu,
            protected_entries=protected_entries,
        ):
            self._restore_online_state(snapshot)
            return False

        self.apply_action(model, adapter, cache_model, cache_adapter)
        return True

    def try_apply_action_with_lru(self, model, adapter, cache_model, cache_adapter):
        """在线 LinUCB / ε-greedy 使用的分配接口：支持缓存动作并按 LRU 驱逐。

        - 根据 cache_model/cache_adapter 决定是否将模型/适配器写入磁盘缓存；
        - 无论是否缓存，只要本次请求被接纳，都需要将模型和适配器加载到 GPU；
        - 若当前资源不足，则按 LRU 驱逐旧缓存/加载项后再尝试分配；
        - 若驱逐到极限仍不足，则返回 False，不接纳该请求。
        """

        return self.try_apply_action_with_policy(
            model,
            adapter,
            cache_model,
            cache_adapter,
            eviction_policy="lru",
        )

    # 兼容旧调用名；当前实现语义已是 LRU。
    def try_allocate_with_fifo(self, model, adapter):
        return self.try_allocate_with_lru(model, adapter)

    def try_apply_action_with_fifo(self, model, adapter, cache_model, cache_adapter):
        return self.try_apply_action_with_lru(
            model, adapter, cache_model, cache_adapter
        )
