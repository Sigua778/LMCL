import torch
from collections import defaultdict

import random


def _get_positive_herbs_for_symptom(edge_index_SH, symptom_idx):
    """获取与给定症状相关的草药"""
    # 在症状-草药边中找到该症状对应的草药
    mask = edge_index_SH[0] == symptom_idx
    positive_herbs = edge_index_SH[1][mask]
    return positive_herbs


def _sample_negative_herb_for_symptom(edge_index_SH, symptom_idx, hh_num):
    """为给定症状采样负样本草药（完全GPU版本）"""
    # 获取该症状的正样本草药集合（保持在GPU上）
    positive_herbs = _get_positive_herbs_for_symptom(edge_index_SH, symptom_idx) - 390
    # 创建所有草药的mask（GPU上）
    all_herbs = torch.arange(hh_num, device=edge_index_SH.device)

    # 创建负样本mask（GPU上）
    mask = torch.ones(hh_num, dtype=torch.bool, device=edge_index_SH.device)
    mask[positive_herbs] = False

    # 获取负样本候选
    negative_candidates = all_herbs[mask]

    # 随机选择一个负样本
    if len(negative_candidates) > 0:
        return negative_candidates[torch.randint(0, len(negative_candidates), (1,))]
    else:
        # 如果没有负样本候选，随机选择一个
        return torch.randint(0, hh_num, (1,), device=edge_index_SH.device)



def pos_nev(prescription, edge_index_SH, hh_num, device):
    # 为每个处方采样正负样本对
    positive_symptoms = []
    positive_herbs = []
    negative_herbs = []
    prescription_indices = []
    for prescription_idx in range(prescription.shape[0]):
        # 获取当前处方的活跃症状\
        active_symptoms = torch.nonzero(prescription[prescription_idx]).squeeze(-1)

        if len(active_symptoms) > 0:
            # 为每个活跃症状采样正负样本
            for symptom_idx in active_symptoms:
                # 正样本：从图中找到与该症状相关的草药
                positive_herb_candidates = _get_positive_herbs_for_symptom(edge_index_SH, symptom_idx)

                if len(positive_herb_candidates) > 0:
                    # 随机选择一个正样本草药
                    pos_herb = positive_herb_candidates[torch.randint(0, len(positive_herb_candidates), (1,)).item()]

                    # 负采样：选择与该症状不相关的草药
                    neg_herb = _sample_negative_herb_for_symptom(edge_index_SH, symptom_idx, hh_num)

                    positive_symptoms.append(symptom_idx)
                    positive_herbs.append(pos_herb)
                    negative_herbs.append(neg_herb)
                    prescription_indices.append(prescription_idx)

    # 转换为张量
    if len(positive_symptoms) > 0:
        positive_symptoms = torch.tensor(positive_symptoms, dtype=torch.long, device=device)
        positive_herbs = torch.tensor(positive_herbs, dtype=torch.long, device=device) - 390
        negative_herbs = torch.tensor(negative_herbs, dtype=torch.long, device=device)
        prescription_indices = torch.tensor(prescription_indices, dtype=torch.long, device=device)
        return positive_symptoms, positive_herbs, negative_herbs
    else:
        raise RuntimeError("pos_nev：当前 batch 中没有正样本，已中止运行。")

# 基于因果图取正负样本
def pos_nev_cause(prescription, edge_index_SH, hh_num, causal_dict, device, pos_threshold=0.00, neg_threshold=-0.005):
    # 预处理因果字典为查找结构（应在函数外部执行）
    pos_herb_lookup = defaultdict(list)  # symptom -> [positive herbs]
    neg_herb_lookup = defaultdict(list)  # symptom -> [negative herbs]
    all_herbs_set = set()

    for (s, herb), effect in causal_dict.items():
        all_herbs_set.add(herb)
        if effect > pos_threshold:
            pos_herb_lookup[s].append(herb)
        elif effect < neg_threshold:
            neg_herb_lookup[s].append(herb)

    all_herbs = torch.tensor(list(all_herbs_set), device=device)

    # 向量化处理处方数据
    prescription_indices = torch.nonzero(prescription).t()
    if prescription_indices.size(1) == 0:
        raise RuntimeError("No positive samples in current batch")

    batch_indices = prescription_indices[0]
    symptoms = prescription_indices[1]

    positive_symptoms = []
    positive_herbs = []
    negative_herbs = []

    # 为每个有效症状-处方对选择样本
    for batch_idx, symptom in zip(batch_indices, symptoms):
        s = symptom.item()

        # 正样本选择：从正相关草药中随机选
        pos_candidates = pos_herb_lookup.get(s, [])
        if not pos_candidates:
            continue

        pos_herb = random.choice(pos_candidates)

        # 负样本选择：优先从负相关草药中选，否则从不相关草药中选
        neg_candidates = neg_herb_lookup.get(s, [])
        if not neg_candidates:
            continue
        neg_herb = random.choice(neg_candidates)

        positive_symptoms.append(symptom)
        positive_herbs.append(pos_herb)
        negative_herbs.append(neg_herb)

    if not positive_symptoms:
        raise RuntimeError("No valid positive samples in current batch")

    return (torch.stack(positive_symptoms),
            torch.tensor(positive_herbs, device=device),
            torch.tensor(negative_herbs, device=device))