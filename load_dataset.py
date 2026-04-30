import pandas as pd
import os
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances


def read_edge_servers():
    """
    读取edge_servers的坐标并转换成弧度制方便通过最邻近算法NearestNeighbors算出每个用户到哪个边缘节点最近
    """
    edge = pd.read_csv("eua-dataset/edge-servers/site-optus-melbCBD.csv")
    edge_coord = edge[["LATITUDE", "LONGITUDE"]].to_numpy()
    # 弧度制
    edge_rad = np.radians(edge_coord)
    return edge, edge_rad


def read_users():
    """
    读取users的坐标并转换成弧度制方便通过最邻近算法NearestNeighbors算出每个用户到哪个边缘节点最近
    """
    # use the larger metropolitan user set to allow more samples
    users_aus = pd.read_csv("eua-dataset/users/users-melbmetro-generated.csv")
    user_coords = users_aus[["Latitude", "Longitude"]].to_numpy()
    # 弧度制
    user_rad = np.radians(user_coords)
    return users_aus, user_rad


def nearestNeighbors():
    """
    通过最邻近算法NearestNeighbors算出每个用户到哪个边缘节点最近,作为用户的homecloudlet
    """
    edge, edge_rad = read_edge_servers()
    users_aus, user_rad = read_users()
    nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(edge_rad)
    dist_rad, idx = nbrs.kneighbors(user_rad)
    # 地球平均半径6371km
    earth_r = 6371
    # 距离精确到米
    dist_km = (dist_rad[:, 0] * earth_r).round(3)
    nearest_siteid = edge.loc[idx[:, 0], "SITE_ID"].values
    users_aus["NearestEdgeSiteID"] = nearest_siteid
    users_aus["DistToEdge"] = dist_km
    return users_aus


# 回源单位传输延迟 (Back-to-Source)，极度受限
BTS_unit_data_transmission_delays = 8  # reduced (ms/MB) to make transfer delays realistic for experiments


def get_unit_data_transmission_delays(dist_km):
    """
    将距离映射到单位数据传输
    """
    if dist_km <= 1.0:
        # 0.08 ms/MB (reduced scale for experiments)
        return 0.08
    elif dist_km <= 5.0:
        # 0.8 ms/MB
        return 0.8
    else:
        # 4 ms/MB
        return 4


def build_networkx(edge_df, edge_rad):
    """
    基于地理位置，使用 NetworkX 构建 P2P 网络拓扑图
    """
    earth_r = 6371
    dist_matrix = haversine_distances(edge_rad, edge_rad) * earth_r
    site_ids = edge_df["SITE_ID"].values
    num_edges = len(site_ids)

    G = nx.Graph()
    G.add_nodes_from(site_ids)
    G.add_node("DC")

    for i in range(num_edges):
        site_a = site_ids[i]
        for j in range(i + 1, num_edges):
            site_b = site_ids[j]
            dist = dist_matrix[i][j]
            unit_delay_ms = get_unit_data_transmission_delays(dist)
            G.add_edge(site_a, site_b, distance=dist, weight=unit_delay_ms)

        G.add_edge(
            site_a,
            "DC",
            distance=float("inf"),
            weight=BTS_unit_data_transmission_delays,
        )

    return G


if __name__ == "__main__":
    # # 仅包含 Optus 运营商位于墨尔本 CBD 区域（Central Business District）内的基站
    # site_optus_melbCBD = pd.read_csv("eua-dataset/edge-servers/site-optus-melbCBD.csv")
    # # 包含澳大利亚境内所有电信运营商（Telstra、Optus、Vodafone 等）的所有基站
    # site = pd.read_csv("eua-dataset/edge-servers/site.csv")
    # # 分配给“澳大利亚”自治系统的所有 IP 地址块（CIDR 列表）
    # australia_ip_ranges = pd.read_csv("eua-dataset/users/australia-ip-ranges.csv")
    # # 澳大利亚 IP 地址（去重后共 4748 条），再通过 IP-API 服务解析出对应的经纬度、邮编、城市、省/州和国家信息，代表“真实”用户的定位数据
    # users_aus = pd.read_csv("eua-dataset/users/users-aus.csv")
    # # 在墨尔本中央商务区（CBD）内部，按均匀分布（uniform）随机生成的 816 个“用户”坐标点，用于模拟或测试对 CBD 区域的空间分析
    # users_melbcbd_generated = pd.read_csv("eua-dataset/users/users-melbcbd-generated.csv")
    # # 在墨尔本大都会区范围内，按均匀分布随机生成的 131 312 个“用户”坐标点，用于模拟或测试对整个都市区的空间分析
    # users_melbmetro_generated = pd.read_csv("eua-dataset/users/users-melbmetro-generated.csv")

    # print(site_optus_melbCBD.head())
    # print(site.head())
    # print(australia_ip_ranges.head())
    # print(users_aus.head())
    # print(users_melbcbd_generated.head())
    # print(users_melbmetro_generated.head())

    edge_df, edge_rad = read_edge_servers()
    G = build_networkx(edge_df, edge_rad)
    print(f"总节点数 (边缘节点 + DC): {G.number_of_nodes()}")
    print(f"总边数 (连接数): {G.number_of_edges()}")
    print(
        f"示例 [{edge_df["SITE_ID"].iloc[0]} <-> DC] 属性: {G.get_edge_data(edge_df["SITE_ID"].iloc[0], "DC")}"
    )

    users_with_edge = nearestNeighbors()
    print(
        users_with_edge[
            ["Latitude", "Longitude", "NearestEdgeSiteID", "DistToEdge"]
        ].head()
    )
