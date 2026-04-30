import random

random.seed(1)  # 方便debug

import numpy as np
import pandas as pd
import time
import copy

from load_dataset import read_edge_servers, build_networkx, nearestNeighbors
from Class.cloudlet import Cloudlet
from Class.model import Model, Adapter
from Class.request import Request
from Class.user import User

# from linucb import ContextAwareLinUCB
from offline.raa_greedy import raa_greedy
from offline.knapsack_greedy import knapsack_greedy
from offline.BTS import bts
from offline.P2P import p2p

foundation_models = [
    Model(id="LlaMA-2-7b", size=3.0 * 1024),
    Model(id="LlaMA-2-13b", size=6.5 * 1024),
    Model(id="Baichuan-7B", size=3.5 * 1024),
    Model(id="Qwen-4b", size=2.0 * 1024),
    Model(id="Qwen-8b", size=4.5 * 1024),
    Model(id="Qwen-14b", size=7.0 * 1024),
    Model(id="CodeLlama-7b", size=3.0 * 1024),
    Model(id="CodeLlama-13b", size=6.5 * 1024),
    Model(id="CodeLlama-34b", size=19 * 1024),
]

fm_dict = {fm.id: fm for fm in foundation_models}

# assign a per-model TFLOPS factor in [100,300] to match paper's influencing factor
for fm in foundation_models:
    fm.tflops = random.uniform(100, 300)

service_types = [
    "Chatbot",
    "Summarization",
    "CodeGeneration",
    "MathReasoning",
    "DomainQA",
]

adapters = {
    # LlaMA-2-7b：偏 Chat/Summ，其他任务一般
    ("LlaMA-2-7b", "Chatbot"): Adapter(
        model_id="LlaMA-2-7b", service_type="Chatbot", size=0.52, accuracy=0.76
    ),
    ("LlaMA-2-7b", "Summarization"): Adapter(
        model_id="LlaMA-2-7b", service_type="Summarization", size=0.56, accuracy=0.74
    ),
    ("LlaMA-2-7b", "CodeGeneration"): Adapter(
        model_id="LlaMA-2-7b", service_type="CodeGeneration", size=0.62, accuracy=0.66
    ),
    ("LlaMA-2-7b", "MathReasoning"): Adapter(
        model_id="LlaMA-2-7b", service_type="MathReasoning", size=0.66, accuracy=0.64
    ),
    ("LlaMA-2-7b", "DomainQA"): Adapter(
        model_id="LlaMA-2-7b", service_type="DomainQA", size=0.70, accuracy=0.70
    ),
    # LlaMA-2-13b：Summ/Domain 偏强，Code/Math 适中
    ("LlaMA-2-13b", "Chatbot"): Adapter(
        model_id="LlaMA-2-13b", service_type="Chatbot", size=0.52, accuracy=0.80
    ),
    ("LlaMA-2-13b", "Summarization"): Adapter(
        model_id="LlaMA-2-13b", service_type="Summarization", size=0.58, accuracy=0.84
    ),
    ("LlaMA-2-13b", "CodeGeneration"): Adapter(
        model_id="LlaMA-2-13b", service_type="CodeGeneration", size=0.64, accuracy=0.74
    ),
    ("LlaMA-2-13b", "MathReasoning"): Adapter(
        model_id="LlaMA-2-13b", service_type="MathReasoning", size=0.68, accuracy=0.74
    ),
    ("LlaMA-2-13b", "DomainQA"): Adapter(
        model_id="LlaMA-2-13b", service_type="DomainQA", size=0.72, accuracy=0.82
    ),
    # Baichuan-7B：偏 Chat/Domain，Code/Math 略弱
    ("Baichuan-7B", "Chatbot"): Adapter(
        model_id="Baichuan-7B", service_type="Chatbot", size=0.60, accuracy=0.82
    ),
    ("Baichuan-7B", "Summarization"): Adapter(
        model_id="Baichuan-7B", service_type="Summarization", size=0.64, accuracy=0.79
    ),
    ("Baichuan-7B", "CodeGeneration"): Adapter(
        model_id="Baichuan-7B", service_type="CodeGeneration", size=0.69, accuracy=0.70
    ),
    ("Baichuan-7B", "MathReasoning"): Adapter(
        model_id="Baichuan-7B", service_type="MathReasoning", size=0.71, accuracy=0.72
    ),
    ("Baichuan-7B", "DomainQA"): Adapter(
        model_id="Baichuan-7B", service_type="DomainQA", size=0.75, accuracy=0.84
    ),
    # Qwen-4b：小模型，明显偏科到 Math / DomainQA
    ("Qwen-4b", "Chatbot"): Adapter(
        model_id="Qwen-4b", service_type="Chatbot", size=0.52, accuracy=0.68
    ),
    ("Qwen-4b", "Summarization"): Adapter(
        model_id="Qwen-4b", service_type="Summarization", size=0.55, accuracy=0.66
    ),
    ("Qwen-4b", "CodeGeneration"): Adapter(
        model_id="Qwen-4b", service_type="CodeGeneration", size=0.60, accuracy=0.62
    ),
    ("Qwen-4b", "MathReasoning"): Adapter(
        model_id="Qwen-4b", service_type="MathReasoning", size=0.70, accuracy=0.78
    ),  # 数学偏强
    ("Qwen-4b", "DomainQA"): Adapter(
        model_id="Qwen-4b", service_type="DomainQA", size=0.72, accuracy=0.80
    ),  # 领域偏强
    # Qwen-8b：中等模型，偏 Chat/Summ 强
    ("Qwen-8b", "Chatbot"): Adapter(
        model_id="Qwen-8b", service_type="Chatbot", size=0.54, accuracy=0.81
    ),
    ("Qwen-8b", "Summarization"): Adapter(
        model_id="Qwen-8b", service_type="Summarization", size=0.58, accuracy=0.80
    ),
    ("Qwen-8b", "CodeGeneration"): Adapter(
        model_id="Qwen-8b", service_type="CodeGeneration", size=0.64, accuracy=0.73
    ),
    ("Qwen-8b", "MathReasoning"): Adapter(
        model_id="Qwen-8b", service_type="MathReasoning", size=0.68, accuracy=0.78
    ),
    ("Qwen-8b", "DomainQA"): Adapter(
        model_id="Qwen-8b", service_type="DomainQA", size=0.70, accuracy=0.78
    ),
    # Qwen-14b：大模型，几乎全领域较强，尤其 Math / Code
    ("Qwen-14b", "Chatbot"): Adapter(
        model_id="Qwen-14b", service_type="Chatbot", size=0.55, accuracy=0.83
    ),
    ("Qwen-14b", "Summarization"): Adapter(
        model_id="Qwen-14b", service_type="Summarization", size=0.60, accuracy=0.82
    ),
    ("Qwen-14b", "CodeGeneration"): Adapter(
        model_id="Qwen-14b", service_type="CodeGeneration", size=0.66, accuracy=0.85
    ),
    ("Qwen-14b", "MathReasoning"): Adapter(
        model_id="Qwen-14b", service_type="MathReasoning", size=0.70, accuracy=0.86
    ),
    ("Qwen-14b", "DomainQA"): Adapter(
        model_id="Qwen-14b", service_type="DomainQA", size=0.72, accuracy=0.84
    ),
    ("CodeLlama-7b", "Chatbot"): Adapter(
        model_id="CodeLlama-7b", service_type="Chatbot", size=0.50, accuracy=0.70
    ),
    ("CodeLlama-7b", "Summarization"): Adapter(
        model_id="CodeLlama-7b", service_type="Summarization", size=0.54, accuracy=0.68
    ),
    ("CodeLlama-7b", "CodeGeneration"): Adapter(
        model_id="CodeLlama-7b",
        service_type="CodeGeneration",
        size=0.64,
        accuracy=0.84,
    ),
    ("CodeLlama-7b", "MathReasoning"): Adapter(
        model_id="CodeLlama-7b", service_type="MathReasoning", size=0.66, accuracy=0.65
    ),
    ("CodeLlama-7b", "DomainQA"): Adapter(
        model_id="CodeLlama-7b", service_type="DomainQA", size=0.69, accuracy=0.69
    ),
    # CodeLlama-13b：更强 Code/Math，其它保持中等
    ("CodeLlama-13b", "Chatbot"): Adapter(
        model_id="CodeLlama-13b", service_type="Chatbot", size=0.52, accuracy=0.79
    ),
    ("CodeLlama-13b", "Summarization"): Adapter(
        model_id="CodeLlama-13b",
        service_type="Summarization",
        size=0.56,
        accuracy=0.78,
    ),
    ("CodeLlama-13b", "CodeGeneration"): Adapter(
        model_id="CodeLlama-13b",
        service_type="CodeGeneration",
        size=0.66,
        accuracy=0.90,
    ),
    ("CodeLlama-13b", "MathReasoning"): Adapter(
        model_id="CodeLlama-13b",
        service_type="MathReasoning",
        size=0.70,
        accuracy=0.80,
    ),
    ("CodeLlama-13b", "DomainQA"): Adapter(
        model_id="CodeLlama-13b", service_type="DomainQA", size=0.72, accuracy=0.78
    ),
    # CodeLlama-34b：偏 Code/Math 的大模型，避免在通用任务上过度覆盖
    ("CodeLlama-34b", "Chatbot"): Adapter(
        model_id="CodeLlama-34b", service_type="Chatbot", size=0.62, accuracy=0.68
    ),
    ("CodeLlama-34b", "Summarization"): Adapter(
        model_id="CodeLlama-34b",
        service_type="Summarization",
        size=0.68,
        accuracy=0.67,
    ),
    ("CodeLlama-34b", "CodeGeneration"): Adapter(
        model_id="CodeLlama-34b",
        service_type="CodeGeneration",
        size=0.80,
        accuracy=0.92,
    ),
    ("CodeLlama-34b", "MathReasoning"): Adapter(
        model_id="CodeLlama-34b",
        service_type="MathReasoning",
        size=0.84,
        accuracy=0.86,
    ),
    ("CodeLlama-34b", "DomainQA"): Adapter(
        model_id="CodeLlama-34b", service_type="DomainQA", size=0.86, accuracy=0.70
    ),
}


def init_environment():
    """初始化热点场景实验所需的网络、边缘节点、模型和适配器状态。
    该函数会读取边缘服务器拓扑，构建网络图，并为每个 cloudlet
    随机分配存储、显存、计算能力以及初始缓存 and 加载状态。
    Returns:
        tuple: (G, edges, foundation_models, adapters, fm_dict)
    """
    edge_df, edge_rad = read_edge_servers()
    G = build_networkx(edge_df, edge_rad)
    print(
        f" -> 网络构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边。"
    )
    print(
        f" -> 初始化了 {len(foundation_models)} 个基座模型和 {len(adapters)} 个任务适配器。"
    )

    edges = {}
    adapter_keys = list(adapters.keys())

    for _, row in edge_df.iterrows():
        site_id = row["SITE_ID"]
        storage_cap = random.choice([24, 32, 48, 64]) * 1024  # 单位: MB
        memory_cap = random.choice([16, 24, 32, 48]) * 1024  # 单位: MB
        eta = random.uniform(107, 312)

        cloudlet = Cloudlet(
            id=site_id,
            memory_capacity=memory_cap,
            storage_capacity=storage_cap,
            eta=eta,
        )

        num_models_to_cache = random.randint(0, 1)
        sampled_models = random.sample(foundation_models, num_models_to_cache)
        for fm in sampled_models:
            if cloudlet.used_storage + fm.size <= cloudlet.storage_capacity:
                cloudlet.cached_models.add(fm.id)
                cloudlet.used_storage += fm.size

        # 从已缓存的基座模型中，挑选配套的适配器进行缓存
        valid_adapters_to_cache = [
            key for key in adapter_keys if key[0] in cloudlet.cached_models
        ]
        if valid_adapters_to_cache:
            # 随机挑选少量适配器
            num_adapters_to_cache = random.randint(
                1, min(2, len(valid_adapters_to_cache))
            )
            sampled_adp_keys = random.sample(
                valid_adapters_to_cache, num_adapters_to_cache
            )
            for adp_key in sampled_adp_keys:
                adp = adapters[adp_key]
                if cloudlet.used_storage + adp.size <= cloudlet.storage_capacity:
                    cloudlet.cached_adapters.add(adp_key)
                    cloudlet.used_storage += adp.size

        # 从已缓存的模型中，随机挑选加载到 GPU
        if cloudlet.cached_models:
            num_models_to_load = random.randint(0, len(cloudlet.cached_models))
            deterministic_model_list = sorted(list(cloudlet.cached_models))
            sampled_models_to_load = random.sample(
                deterministic_model_list, num_models_to_load
            )

            for model_id in sampled_models_to_load:
                fm_size = fm_dict[model_id].size
                if cloudlet.used_memory + fm_size <= cloudlet.memory_capacity:
                    cloudlet.loaded_models.add(model_id)
                    cloudlet.used_memory += fm_size

        # 从已缓存的适配器中，挑选其基座模型已经在 GPU 中的适配器加载到 GPU
        valid_adapters_to_load = [
            adp_key
            for adp_key in cloudlet.cached_adapters
            if adp_key[0] in cloudlet.loaded_models  # 必须保证基座模型在显存中
        ]
        if valid_adapters_to_load:
            num_adapters_to_load = random.randint(0, len(valid_adapters_to_load))
            deterministic_adapter_list = sorted(valid_adapters_to_load)
            sampled_adapters_to_load = random.sample(
                deterministic_adapter_list, num_adapters_to_load
            )

            for adp_key in sampled_adapters_to_load:
                adp = adapters[adp_key]
                if cloudlet.used_memory + adp.size <= cloudlet.memory_capacity:
                    cloudlet.loaded_adapters.add(adp_key)
                    cloudlet.used_memory += adp.size

        edges[site_id] = cloudlet
    print(f" -> 初始化了 {len(edges)} 个边缘节点，并完成了随机缓存模型和适配器。")

    # 为云数据中心配置无限容量
    if "DC" in G.nodes:
        G.nodes["DC"]["type"] = "Cloud"
        G.nodes["DC"]["cached_models"] = [fm.id for fm in foundation_models]
        G.nodes["DC"]["cached_adapters"] = adapter_keys
        G.nodes["DC"]["used_storage"] = float("inf")

    return G, edges, foundation_models, adapters, fm_dict


def init_users(number):
    """生成默认分布下的用户请求集合。
    该函数按照真实用户与最近 cloudlet 的映射关系，随机生成请求类型、
    指令长度、奖励、精度需求和时延需求，用于非热点场景实验。
    Args:
        number (int): 需要生成的用户请求数量。
    Returns:
        list[User]: 生成的用户请求对象列表。
    """
    neighbors = nearestNeighbors()
    users = []
    for _, row in neighbors[0:number].iterrows():
        home_cloudlet = row["NearestEdgeSiteID"]
        req_type = random.choice(service_types)
        instruction = random.randint(300, 2000)
        reward = random.uniform(10, 50)  # reward

        req = Request(
            homeCloudlet=home_cloudlet,
            type=req_type,
            instruction=instruction,
            reward=reward,
        )

        # 部分用户精度需求降低，形成高低混合的 QoS 分布
        r = random.random()
        if r < 0.5:
            user_acc = random.uniform(0.50, 0.65)
        elif r < 0.8:
            user_acc = random.uniform(0.65, 0.75)
        else:
            user_acc = random.uniform(0.75, 0.85)

        user_delay = random.uniform(8 * 1000, 20 * 1000)  # delay
        user = User(request=req, accuracy=user_acc, delay=user_delay)
        users.append(user)
    print(f" -> 生成了 {len(users)} 个用户请求。")
    return users


def init_users_hotspot(number, num_hot_edges=100):
    """生成热点 cloudlet 场景下的用户请求集合。
    该函数先从所有 edge 服务器中随机选出若干热点 cloudlet，再只从这些
    热点节点覆盖的用户中采样请求，以制造局部负载集中和资源竞争。
    Args:
        number (int): 需要生成的用户请求数量。
        num_hot_edges (int): 热点 cloudlet 的数量。
    Returns:
        list[User]: 生成的热点场景用户请求对象列表。
    """
    neighbors = nearestNeighbors()
    users = []

    # 先选出一批热点 cloudlet
    unique_edges = neighbors["NearestEdgeSiteID"].unique().tolist()
    if num_hot_edges > len(unique_edges):
        num_hot_edges = len(unique_edges)
    hot_edges = random.sample(unique_edges, num_hot_edges)
    hot_edge_service_mix = {
        edge_id: tuple(random.sample(service_types, 2)) for edge_id in hot_edges
    }

    # 过滤出热点 cloudlet 所在的用户，再从中采样 number 个
    hotspot_rows = neighbors[neighbors["NearestEdgeSiteID"].isin(hot_edges)]
    if len(hotspot_rows) >= number:
        selected_rows = hotspot_rows.sample(n=number, replace=False, random_state=1)
    else:
        # 如果可用用户不足，就重复采样，保持总请求数为 number
        selected_rows = hotspot_rows.sample(n=number, replace=True, random_state=1)

    for _, row in selected_rows.iterrows():
        home_cloudlet = row["NearestEdgeSiteID"]
        primary_type, secondary_type = hot_edge_service_mix[home_cloudlet]
        req_draw = random.random()
        if req_draw < 0.7:
            req_type = primary_type
        elif req_draw < 0.9:
            req_type = secondary_type
        else:
            fallback_types = [
                service
                for service in service_types
                if service not in (primary_type, secondary_type)
            ]
            req_type = random.choice(fallback_types)
        instruction = random.randint(300, 2000)
        reward = random.uniform(10, 50)

        req = Request(
            homeCloudlet=home_cloudlet,
            type=req_type,
            instruction=instruction,
            reward=reward,
        )

        r = random.random()
        if r < 0.5:
            user_acc = random.uniform(0.50, 0.65)
        elif r < 0.8:
            user_acc = random.uniform(0.65, 0.75)
        else:
            user_acc = random.uniform(0.75, 0.85)

        user_delay = random.uniform(8 * 1000, 20 * 1000)  # delay
        user = User(request=req, accuracy=user_acc, delay=user_delay)
        users.append(user)

    print(
        f" -> 生成了 {len(users)} 个用户请求，其中热点 cloudlet 数量约为 {len(hot_edges)} 个。"
    )
    return users


def init_users_for_debug(number=100):
    """生成用于调试的集中式用户请求集合。
    该函数会强制所有请求落到同一个 home cloudlet，便于压测和观察
    MHS/RAA-Greedy 在高集中度场景下的行为。
    Args:
        number (int): 需要生成的调试请求数量。
    Returns:
        list[User]: 生成的调试用户请求对象列表。
    """
    neighbors = nearestNeighbors()
    users = []
    for _, row in neighbors[0:number].iterrows():
        home_cloudlet = 134872.0
        req_type = random.choice(service_types)
        instruction = random.randint(300, 2000)

        rand_val = random.random()
        if rand_val < 0.6:
            user_acc = random.uniform(0.75, 0.80)
            reward = random.uniform(10, 20)
        elif rand_val < 0.9:
            user_acc = random.uniform(0.80, 0.86)
            reward = random.uniform(20, 40)
        else:
            user_acc = random.uniform(0.86, 0.92)
            reward = random.uniform(40, 80)

        req = Request(
            homeCloudlet=home_cloudlet,
            type=req_type,
            instruction=instruction,
            reward=reward,
        )
        user_delay = random.uniform(8 * 1000, 20 * 1000)  # delay
        user = User(request=req, accuracy=user_acc, delay=user_delay)
        users.append(user)
    print(f" -> 生成了 {len(users)} 个用户请求。")
    return users


def run_hotspot_once(
    num_hot: int,
    num_users: int = 3000,
    lambda_delay: float = 1e-3,
    seed: int = 4,
):
    """运行一次指定热点数量下的离线算法对比实验。
    该函数会重置随机种子，初始化实验环境和热点用户请求，并分别执行
    RAA-Greedy、Knapsack、BTS 和 P2P，返回各算法的接纳数、奖励和耗时。
    Args:
        num_hot (int): 热点 cloudlet 数量。
        num_users (int): 用户请求总数。
        lambda_delay (float): 奖励函数中的时延惩罚权重。
        seed (int): 随机种子。
    Returns:
        dict: 各算法的实验结果汇总。
    """
    random.seed(seed)
    np.random.seed(seed)

    G, edges, foundation_models, adapters, fm_dict = init_environment()
    users = init_users_hotspot(num_users, num_hot_edges=num_hot)

    results = {}

    start = time.time()
    admitted = raa_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        fm_dict,
        lambda_delay=lambda_delay,
    )
    results["RAA"] = {
        "admitted": len(admitted),
        "reward": sum(u.request.reward for u in admitted),
        "time": time.time() - start,
    }

    start = time.time()
    admitted = knapsack_greedy(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
    )
    results["Knapsack"] = {
        "admitted": len(admitted),
        "reward": sum(u.request.reward for u in admitted),
        "time": time.time() - start,
    }

    start = time.time()
    admitted = bts(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
    )
    results["BTS"] = {
        "admitted": len(admitted),
        "reward": sum(u.request.reward for u in admitted),
        "time": time.time() - start,
    }

    start = time.time()
    admitted = p2p(
        G,
        users,
        copy.deepcopy(edges),
        foundation_models,
        adapters,
        lambda_delay=lambda_delay,
    )
    results["P2P"] = {
        "admitted": len(admitted),
        "reward": sum(u.request.reward for u in admitted),
        "time": time.time() - start,
    }

    return results


if __name__ == "__main__":
    random.seed(4)
    np.random.seed(4)
    G, edges, foundation_models, adapters, fm_dict = init_environment()

    # 压测不同热点 cloudlet 数量下的表现：30 / 50 / 100
    for num_hot in [30, 50, 100]:
        print("\n==============================")
        print(f"Hotspot cloudlets ~= {num_hot}")
        print("==============================")

        users = init_users_hotspot(3000, num_hot_edges=num_hot)

        # RAA-Greedy
        start_time = time.time()
        admitted_requests = raa_greedy(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            fm_dict,
            lambda_delay=1e-3,
        )
        print(f" -> RAA-Greedy 耗时: {time.time() - start_time:.2f} s")
        total_reward = sum(u.request.reward for u in admitted_requests)
        print(
            f" -> RAA 接纳率: {len(admitted_requests)} / {len(users)} "
            f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
        )

        # Knapsack-Greedy
        start_time = time.time()
        admitted_requests = knapsack_greedy(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
            lambda_delay=1e-3,
        )
        print(f" -> Knapsack-Greedy 耗时: {time.time() - start_time:.2f} s")
        total_reward = sum(u.request.reward for u in admitted_requests)
        print(
            f" -> Knapsack-Greedy 接纳率: {len(admitted_requests)} / {len(users)} "
            f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
        )

        # BTS
        start_time = time.time()
        admitted_requests = bts(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
        )
        print(f" -> BTS 耗时: {time.time() - start_time:.2f} s")
        total_reward = sum(u.request.reward for u in admitted_requests)
        print(
            f" -> BTS 接纳率: {len(admitted_requests)} / {len(users)} "
            f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
        )

        # P2P
        start_time = time.time()
        admitted_requests = p2p(
            G,
            users,
            copy.deepcopy(edges),
            foundation_models,
            adapters,
        )
        print(f" -> P2P 耗时: {time.time() - start_time:.2f} s")
        total_reward = sum(u.request.reward for u in admitted_requests)
        print(
            f" -> P2P 接纳率: {len(admitted_requests)} / {len(users)} "
            f"({len(admitted_requests)/len(users)*100:.2f}%), 总 reward = {total_reward:.2f}"
        )
