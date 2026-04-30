import copy
import time


DEPLOYMENT_COST_WEIGHT = 10.0


def cold_start_edges(edges):
    """在线算法统一的冷启动：清空缓存、显存和 LRU"""
    for edge in edges.values():
        edge.cached_models = set()
        edge.cached_adapters = set()
        edge.loaded_models = set()
        edge.loaded_adapters = set()
        edge.used_storage = 0.0
        edge.used_memory = 0.0
        edge.used_computing = 0.0

        if hasattr(edge, "model_sizes"):
            edge.model_sizes.clear()
        if hasattr(edge, "adapter_sizes"):
            edge.adapter_sizes.clear()
        if hasattr(edge, "cache_fifo"):
            edge.cache_fifo.clear()
        if hasattr(edge, "load_fifo"):
            edge.load_fifo.clear()


def prepare_online_environment(edges, foundation_models):
    """补齐 online 实验依赖的兼容字段"""
    for fm in foundation_models:
        if not hasattr(fm, "model_id"):
            fm.model_id = fm.id

    for edge in edges.values():
        if not hasattr(edge, "used_computing"):
            edge.used_computing = 0.0
        if not hasattr(edge, "computing_capacity"):
            edge.computing_capacity = getattr(edge, "memory_capacity", 1.0)


def benchmark_online_algorithms(
    G,
    users,
    edges,
    foundation_models,
    adapters,
    alpha: float = 1.0,
    epsilon: float = 0.1,
):
    """统一运行各在线算法，返回包含 admitted / reward / elapsed 的结果"""
    from online.BTS import run_bts_online
    from online.P2P import run_p2p_online
    from online.eps_greedy import run_eps_greedy_online
    from online.linucb import run_linucb_online

    runners = [
        (
            "LinUCB",
            lambda snapshot: run_linucb_online(
                G, users, snapshot, foundation_models, adapters, alpha=alpha
            ),
        ),
        (
            "P2P",
            lambda snapshot: run_p2p_online(
                G, users, snapshot, foundation_models, adapters
            ),
        ),
        (
            "BTS",
            lambda snapshot: run_bts_online(
                G, users, snapshot, foundation_models, adapters
            ),
        ),
        (
            "EpsGreedy",
            lambda snapshot: run_eps_greedy_online(
                G,
                users,
                snapshot,
                foundation_models,
                adapters,
                epsilon=epsilon,
            ),
        ),
    ]

    results = []
    total_users = len(users)
    for algo, runner in runners:
        start = time.time()
        admitted, total_reward = runner(copy.deepcopy(edges))
        results.append(
            {
                "algo": algo,
                "admitted": len(admitted),
                "total_users": total_users,
                "accept_rate": (len(admitted) / total_users) if total_users else 0.0,
                "total_reward": total_reward,
                "elapsed": time.time() - start,
            }
        )

    return results


def print_online_benchmark(results, prefix: str = ""):
    label_prefix = f"{prefix} " if prefix else ""
    for row in results:
        print(
            f"{label_prefix}-> {row['algo']} 在线 接纳率: {row['admitted']} / {row['total_users']} "
            f"({row['accept_rate'] * 100:.2f}%), Total reward: {row['total_reward']:.2f} "
            f"| 耗时: {row['elapsed']:.2f}s"
        )
