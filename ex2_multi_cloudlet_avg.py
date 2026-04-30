from ex1_multi_cloudlet_hotspot import init_environment, init_users
from online.common import (
    benchmark_online_algorithms,
    prepare_online_environment,
    print_online_benchmark,
)


def main(num_users=100):
    G, edges, foundation_models, adapters, fm_dict = init_environment()
    prepare_online_environment(edges, foundation_models)

    users = init_users(num_users)
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
    main(100)
