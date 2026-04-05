#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong
import pickle
import parameter
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.pS_array, self.pH_array = a, b
    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        return sid, hid

    def __len__(self):
        return self.pH_array.shape[0]


def count_diff_sparse_tensors(tensor1, tensor2):
    # 转为 CPU 的稀疏 COO
    t1 = tensor1.coalesce().cpu()
    t2 = tensor2.coalesce().cpu()

    # 取坐标和值
    i1, v1 = t1.indices(), t1.values()
    i2, v2 = t2.indices(), t2.values()

    # 合并坐标为字符串方便比较
    idx1 = {tuple(i.tolist()): val.item() for i, val in zip(i1.t(), v1)}
    idx2 = {tuple(i.tolist()): val.item() for i, val in zip(i2.t(), v2)}

    # 找出所有可能的坐标
    all_keys = set(idx1.keys()).union(set(idx2.keys()))

    # 比较值是否不同（可以加入误差容忍）
    diff_count = 0
    for k in all_keys:
        v_1 = idx1.get(k, 0.0)
        v_2 = idx2.get(k, 0.0)
        if abs(v_1 - v_2) > 1e-6:
            diff_count += 1

    return diff_count

# 根据topk取负样本
def get_hard_negatives(score_matrix, positive_symptoms, positive_herbs, K=10):
    hard_negatives = []

    for s, h in zip(positive_symptoms, positive_herbs):
        # 取前 K 个最大分数的草药
        topk_indices = torch.topk(score_matrix[s], K + 1).indices.tolist()  # +1 是为了剔除正样本
        topk_filtered = [idx for idx in topk_indices if idx != h][:K]
        hard_negatives.append(random.choice(topk_filtered))  # 每个样本选一个

    return torch.tensor(hard_negatives, device=score_matrix.device)


def get_causal_masked_edges(edge_list, causal_dict, threshold=0.005, noise_ratio=-0.005):
    retained_edges = []
    for s, h in edge_list:
        key = (int(s), int(h))
        score = causal_dict.get(key, 0.0)
        if score >= threshold:
            retained_edges.append((s, h))  # 强因果，保留
        else:
            if np.random.rand() < noise_ratio:  # 弱因果，以概率扰动保留
                retained_edges.append((s, h))
    return retained_edges

# 基于频率的 新的评价指标
def compute_fri_k(predictions, top_k, herb_freqs):
    """
    :param predictions: [B, N] 模型预测分数
    :param top_k: int, Top-K 推荐
    :param herb_freqs: list or np.array of length N，记录每个中药的频率（如在处方中出现的次数）
    :return: FRI@K 平均值
    """
    batch_size, num_herbs = predictions.shape
    topk_indices = torch.topk(predictions, k=top_k, dim=1).indices  # [B, K]
    herb_freqs = np.array(herb_freqs) + 1e-6  # 避免除以零

    fri_scores = []
    for b in range(batch_size):
        topk = topk_indices[b].cpu().numpy()
        score = sum([1 / np.log(herb_freqs[h] + 1) for h in topk]) / top_k
        fri_scores.append(score)

    return sum(fri_scores) / batch_size


