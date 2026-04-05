import scipy.sparse as sp
import numpy as np
from collections import Counter
from scipy.sparse import load_npz
import torch
from torch_geometric.data import Data
from reckit import randint_choice
from utils import *

def analyze_symptoms_herbs(file_path):
    # 初始化列表用于存储症状-草药对
    syn_herb_pairs = []

    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行末可能的空白字符并按逗号分割
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    symptom = int(parts[0])  # 转换为整数
                    herb = int(parts[1])  # 转换为整数

                    # 添加到列表中
                    syn_herb_pairs.append([symptom, herb])
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None, None

    # 将列表转换为NumPy数组
    syn_herb_array = np.array(syn_herb_pairs)

    # 提取唯一的症状和草药
    unique_symptoms = np.unique(syn_herb_array[:, 0])
    unique_herbs = np.unique(syn_herb_array[:, 1])

    # 获取症状和草药的数量
    num_symptoms = len(unique_symptoms)
    num_herbs = len(unique_herbs)

    return num_symptoms, num_herbs, syn_herb_array



def create_shadj_mat( num_syn, num_herbs, syn_herb, ssl_ratio,is_subgraph=False, aug_type='ed'):
    n_nodes = num_syn + num_herbs
    users_items = syn_herb
    users_np, items_np = users_items[:, 0], users_items[:, 1]  # 分别提取用户和物品的数组

    if is_subgraph and ssl_ratio > 0:
        if aug_type in ['ed', 'rw']:  # 进行 边 丢弃操作
            keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - ssl_ratio)), replace=False)
            user_np = np.array(users_np)[keep_idx]
            item_np = np.array(items_np)[keep_idx]
            # tmp_adj = build_log_scale_adjacency_matrix(user_np, item_np, num_syn, n_nodes)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + num_syn)), shape=(n_nodes, n_nodes))
    else:
        # tmp_adj = build_log_scale_adjacency_matrix(users_np, items_np, num_syn, n_nodes)
        ratings = np.ones_like(users_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + num_syn)), shape=(n_nodes, n_nodes))
    adj_mat = tmp_adj + tmp_adj.T
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)

    # adj_mat_ss = load_npz("data/symptom_matrix.npz")
    # # 确保每个都有自连接
    adj_matrix = adj_matrix.tolil()
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i, i] == 0:
            adj_matrix[i, i] = 1
    # adj_mat_ss = adj_mat_ss.tocsr()
    # adj_mat_hh = load_npz("data/herb_matrix.npz")
    #
    # adj_matrix_ss = normalize_fun(adj_mat_ss)
    # adj_matrix_hh = normalize_fun(adj_mat_hh)
    #
    # n_symptom = num_syn
    # adj_matrix[:n_symptom, :n_symptom] += adj_matrix_ss
    # adj_matrix[n_symptom:, n_symptom:] += adj_matrix_hh

    return adj_matrix

def create_shadj_mat_cauls(causal_dict, num_syn, num_herbs, syn_herb, ssl_ratio, noise_scale=0.01):
    n_nodes = num_syn + num_herbs
    users_items = syn_herb
    users_np, items_np = users_items[:, 0], users_items[:, 1]  # 分别提取用户和物品的数组

    # ---------- 使用因果效应进行边选择 ----------
    edge_list = list(zip(users_np, items_np))
    edge_list_new = get_causal_masked_edges(edge_list, causal_dict, threshold=0.005, noise_ratio=-0.005)
    # 添加小扰动（每次调用结果不同）
    causal_scores = np.array([
        causal_dict.get((int(s), int(h)), 0.0)
        # + np.random.uniform(-noise_scale, noise_scale)
        for s, h in edge_list_new
    ])

    # 保留因果效应高的前 k 个边
    k = int(len(causal_scores) * (1 - ssl_ratio))
    keep_idx = np.argsort(-causal_scores)[:k]  # 按照因果效应从高到低排序


    user_np = np.array(users_np)[keep_idx]
    item_np = np.array(items_np)[keep_idx]
    tmp_adj = build_log_scale_adjacency_matrix(user_np, item_np, num_syn, n_nodes)


    adj_mat = tmp_adj + tmp_adj.T
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)

    # adj_mat_ss = load_npz("data/symptom_matrix.npz")
    # # 确保每个症状都有自连接
    # adj_mat_ss = adj_mat_ss.tolil()
    # for i in range(adj_mat_ss.shape[0]):
    #     if adj_mat_ss[i, i] == 0:
    #         adj_mat_ss[i, i] = 1
    # adj_mat_ss = adj_mat_ss.tocsr()
    # adj_mat_hh = load_npz("data/herb_matrix.npz")
    #
    # adj_matrix_ss = normalize_fun(adj_mat_ss)
    # adj_matrix_hh = normalize_fun(adj_mat_hh)
    #
    # n_symptom = num_syn
    # adj_matrix[:n_symptom, :n_symptom] += adj_matrix_ss
    # adj_matrix[n_symptom:, n_symptom:] += adj_matrix_hh

    return adj_matrix

def create_shadj_mat_one(num_syn, num_herbs, syn_herb):
    n_nodes = num_syn + num_herbs
    users_items = syn_herb
    users_np, items_np = users_items[:, 0], users_items[:, 1]  # 分别提取用户和物品的数组
    ratings = np.ones_like(users_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + num_syn)), shape=(n_nodes, n_nodes))
    adj_mat = tmp_adj + tmp_adj.T
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)

    return adj_matrix

def build_log_scale_adjacency_matrix(users_np, items_np, num_syn, n_nodes):
        """
        使用对数缩放构建邻接矩阵，减少长尾数据的影响，保持与原代码兼容
        """
        # 计算每个(症状,草药)对在数据中出现的次
        pair_counts = Counter(zip(users_np, items_np))

        # 创建与原始数据相同大小的权重数组
        log_weights = np.zeros_like(users_np, dtype=np.float32)

        # 为每个位置分配对数缩放的权重
        for i in range(len(users_np)):
            symptom = users_np[i]
            herb = items_np[i]
            count = pair_counts[(symptom, herb)]
            log_weights[i] = np.log1p(count)  # 使用log(1+count)作为权重

        # 使用与原始代码相同的方式构建tmp_adj，但权重不同
        tmp_adj = sp.csr_matrix((log_weights, (users_np, items_np + num_syn)), shape=(n_nodes, n_nodes))

        return tmp_adj

def normalize_fun(adj_mat):
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    return adj_matrix


# 假设 adj_matrix 是您提供的稀疏COO张量
# adj_matrix = tensor(indices=..., values=..., size=(1201,1201), ...)

def sparse_tensor_to_pyg_data(adj_matrix):
    # 1. 提取边索引 (edge_index)
    edge_index = adj_matrix.indices()  # 已经是[2, E]格式

    # 2. 创建节点特征 (x) - 使用单位矩阵或随机初始化
    num_nodes = adj_matrix.size(0)
    x = torch.eye(num_nodes, device='cuda:0')  # 使用单位矩阵作为特征
    # 或者随机初始化: x = torch.randn(num_nodes, 64, device='cuda:0')

    # 3. 构建PyG的Data对象
    data = Data(x=x, edge_index=edge_index)

    return data
