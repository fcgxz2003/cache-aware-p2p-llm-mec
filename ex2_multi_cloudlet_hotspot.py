from ex1_multi_cloudlet_hotspot import init_environment, init_users_hotspot
from online.common import (
    benchmark_online_algorithms,
    prepare_online_environment,
    print_online_benchmark,
)


def main(num_users: int = 3000, num_hot_edges: int = 20):
    G, edges, foundation_models, adapters, fm_dict = init_environment()
    prepare_online_environment(edges, foundation_models)

    users = init_users_hotspot(num_users, num_hot_edges=num_hot_edges)

    print(
        f"=== Online hotspot test: num_users={num_users}, hot_edges~={num_hot_edges} ==="
    )

    results = benchmark_online_algorithms(
        G,
        users,
        edges,
        foundation_models,
        adapters,
        alpha=1.0,
        epsilon=0.1,
    )
    print_online_benchmark(results)


if __name__ == "__main__":
    main(3000, num_hot_edges=20)
