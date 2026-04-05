from scipy.sparse import load_npz
from create_adj import analyze_symptoms_herbs, create_shadj_mat, sparse_tensor_to_pyg_data, create_shadj_mat_one, \
    create_shadj_mat_cauls
from layers import causal_attention
from pos_nev import pos_nev, pos_nev_cause
from utils import *
from model import *
import sys
import os
import parameter
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
import time
import gc
import json
from torch_sparse import SparseTensor
from util.pytorch import sp_mat_to_sp_tensor
import torch.profiler

#作用是为了在使用随机数时，即使在多 GPU 环境中，也能保证每次运行得到相同的结果，方便调试和复现结果。
seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

para = parameter.para(lr=0.0007, rec=1e-3, drop=0.3, batchSize=512, epoch=500, embed_size=512, dev_ratio=0.2,
                      test_ratio=0.2, n_layers=3, ssl_reg=0.1, ssl_ratio=0.1, ssl_temp=0.2, num_negatives=1,
                      stddev=0.01, stop_cnt=15, heads=2, BCE_L=0.025)
#获取当前脚本文件所在目录的绝对路径
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print("lr: ", para.lr, " rec: ", para.rec, " dropout: ", para.drop, " batchsize: ",
      para.batchSize, " epoch: ", para.epoch)

class CustomBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets, BCE_L):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        top20_probs, top20_indices = torch.topk(probs, 20, dim=1)
        top20_loss = 0
        for i in range(logits.size(0)):
            top20_targets = targets[i, top20_indices[i]]
            top20_loss += torch.nn.functional.binary_cross_entropy(top20_probs[i], top20_targets)
        top20_loss /= logits.size(0)
        total_loss = bce_loss + BCE_L * top20_loss
        return total_loss

class WeightedTopKBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, top_k=20, base_weight=1.0, topk_weight=5.0):
        super().__init__()
        self.top_k = top_k
        self.base_weight = base_weight
        self.topk_weight = topk_weight

    def forward(self, logits, targets):
        """
        logits: Tensor, shape [B, N], raw model outputs before sigmoid
        targets: Tensor, shape [B, N], multi-label ground truth
        """
        B, N = logits.shape

        # 1. 构造基础权重
        weight = torch.full_like(logits, self.base_weight)  # shape: [B, N]

        # 2. 选出每个样本的 top-K
        probs = torch.sigmoid(logits)
        topk_indices = torch.topk(probs, self.top_k, dim=1).indices  # shape: [B, top_k]

        # 3. 把 top-K 位置的权重加大
        weight.scatter_(1, topk_indices, self.topk_weight)

        # 4. 使用 weighted BCE with logits loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction='mean'
        )
        return loss


num_syn, num_herbs, syn_herb = analyze_symptoms_herbs('data3/hr.txt')

sh_data_adj = create_shadj_mat(num_syn, num_herbs, syn_herb, ssl_ratio=para.ssl_ratio)
adj_matrix = sp_mat_to_sp_tensor(sh_data_adj).to(device)

# 用于LMCL模型的forward
num_syn, num_herbs, syn_herb = analyze_symptoms_herbs('data3/hr_one.txt')
sh_data_adj1 = create_shadj_mat_one(num_syn, num_herbs, syn_herb)
adj_matrix1 = sp_mat_to_sp_tensor(sh_data_adj1).to(device)
sh_data = sparse_tensor_to_pyg_data(adj_matrix1)


# S-S G
ss_edge = np.load('./data3/ssgraph_T.npy')
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous()).to(device)
ss_data.x = ss_data.x.squeeze()


# H-H G
hh_edge = np.load('./data3/hhgraph_T.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long)  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(805)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous()).to(device)
hh_data.x = hh_data.x.squeeze()

# data1
# 读取处方数据
prescript = pd.read_csv('data3/prescript_1195.csv', encoding='utf-8')
pLen = len(prescript)
# 症状的one-hot 矩阵
pS_list = [[0] * 390 for _ in range(pLen)]  #一共有pLen行 每行390个数据
pS_array = np.array(pS_list)
# 草药的one-hot 矩阵
pH_list = [[0] * 805 for _ in range(pLen)]
pH_array = np.array(pH_list)
# 迭代数据集， 赋值
for i in range(pLen):
    j = eval(prescript.iloc[i, 0])
    #最终得到pLen个one-hot 矩阵
    pS_array[i, j] = 1  # 每个one-hot 矩阵行为处方编号，列为该处方中的症状编号

    k = eval(prescript.iloc[i, 1])  #获取第i行第1列
    pH_array[i, k] = 1  #阵行为处方编号，列为该处方中的草药编号

# 训练集开发集测试集的下标
p_list = [x for x in range(pLen)]

# data3
x_train, x_dev = train_test_split(p_list, test_size=0.4, shuffle=False,
                                       random_state=2021)

train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=para.batchSize)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=len(x_dev))



sh_causal_df = pd.read_csv('data3/sh_causal_structured.csv')
sh_causal_df.columns = sh_causal_df.columns.str.strip()
# 转成字典：键为 (symptom, herb)，值为 causal_effect
causal_dict = {
    (int(row['sym']), int(row['herb'])): float(row['causal_effect'])
    for _, row in sh_causal_df.iterrows()
    if not isinstance(row['causal_effect'], str)
}

#hh 因果图
hh_causal_df = pd.read_csv('data3/hh_causal_structured.csv')
hh_causal_df.columns = hh_causal_df.columns.str.strip()
# 转成字典：键为 (symptom, herb)，值为 causal_effect
hh_causal_dict = {
    (int(row['herb1']), int(row['herb2'])): float(row['causal_effect'])
    for _, row in hh_causal_df.iterrows()
    if not isinstance(row['causal_effect'], str)
}

hh_causal_tensor = causal_attention(hh_causal_dict, num_herbs, heads=para.heads)

# 初始化模型
model = LMCL(num_syn, num_herbs, num_syn + num_herbs, syn_herb, adj_matrix, device, para,  hh_causal_tensor).to(device)


criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
# criterion = CustomBCEWithLogitsLoss()
# criterion = WeightedTopKBCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.rec)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)
early_stopping = EarlyStopping(patience=7, verbose=True)
print('device: ', device)

epsilon = 1e-13



for epoch in range(para.epoch):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, (sid, hid) in enumerate(train_loader):
        sub_graph1 = create_shadj_mat_cauls(causal_dict, num_syn, num_herbs, syn_herb, para.ssl_ratio)
        sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(device)
        sub_graph2 = create_shadj_mat(num_syn, num_herbs, syn_herb, para.ssl_ratio, is_subgraph=True, aug_type='ed')
        sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(device)

        sid, hid = sid.float().to(device), hid.float().to(device)
        optimizer.zero_grad()
        positive_symptoms, positive_herbs, negative_herbs = pos_nev(sid, sh_data.edge_index, num_herbs, device=device)

        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, sid,
                        sub_graph1=sub_graph1, sub_graph2=sub_graph2,
                        positive_symptoms=positive_symptoms, positive_herbs=positive_herbs, negative_herbs=negative_herbs)


        loss = criterion(outputs[0], hid) + 0.001*outputs[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print train loss per every epoch 每个epoch 的平均损失
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))

    model.eval()

    dev_loss = 0

    dev_p5 = 0
    dev_p10 = 0
    dev_p20 = 0

    dev_r5 = 0
    dev_r10 = 0
    dev_r20 = 0

    dev_f1_5 = 0
    dev_f1_10 = 0
    dev_f1_20 = 0

    with torch.no_grad():
        for tsid, thid in dev_loader:
            tsid, thid = tsid.float().to(device), thid.float().to(device)
            batch_size = thid.size(0)

            outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                            hh_data.x, hh_data.edge_index, tsid)

            dev_loss += criterion(outputs[0], thid).item()
            # dev_loss += criterion(outputs[0], thid, para.BCE_L)

            # top-k索引
            top5 = torch.topk(outputs[0], 5, dim=1)[1]  # (batch_size, 5)
            top10 = torch.topk(outputs[0], 10, dim=1)[1]  # (batch_size, 10)
            top20 = torch.topk(outputs[0], 20, dim=1)[1]  # (batch_size, 20)


            # 真实标签矩阵 thid: (batch_size, 805)

            def batch_hits(topk_idx, labels):
                # topk_idx: (batch_size, k)
                # labels: (batch_size, num_labels) 0/1矩阵

                # 取每个样本topk索引对应的真实标签值
                hits = torch.gather(labels, 1, topk_idx)  # (batch_size, k), 1表示命中，0不命中
                hits_per_sample = hits.sum(dim=1).float()  # 每个样本命中数
                return hits_per_sample


            hits5 = batch_hits(top5, thid)
            hits10 = batch_hits(top10, thid)
            hits20 = batch_hits(top20, thid)

            true_counts = thid.sum(dim=1).float()  # 每个样本真实标签数

            # 防止除0
            true_counts[true_counts == 0] = 1

            # 计算精确率和召回率
            precision5 = (hits5 / 5).sum()
            recall5 = (hits5 / true_counts).sum()

            precision10 = (hits10 / 10).sum()
            recall10 = (hits10 / true_counts).sum()

            precision20 = (hits20 / 20).sum()
            recall20 = (hits20 / true_counts).sum()

            # 累加
            dev_p5 += precision5
            dev_r5 += recall5
            dev_p10 += precision10
            dev_r10 += recall10
            dev_p20 += precision20
            dev_r20 += recall20

    # 先转cpu再转numpy，再格式化输出
    dev_p5, dev_p10, dev_p20 = dev_p5.cpu().item(), dev_p10.cpu().item(), dev_p20.cpu().item()
    dev_r5, dev_r10, dev_r20 = dev_r5.cpu().item(), dev_r10.cpu().item(), dev_r20.cpu().item()
    print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(dev_loader))
    # print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(x_dev))
    print('p5-10-20:', dev_p5 / len(x_dev), dev_p10 / len(x_dev), dev_p20 / len(x_dev))
    print('r5-10-20:', dev_r5 / len(x_dev), dev_r10 / len(x_dev), dev_r20 / len(x_dev))
    # print('f1_5-10-20: ', dev_f1_5 / len(x_dev), dev_f1_10 / len(x_dev), dev_f1_20 / len(x_dev))
    print('f1_5-10-20: ',
          2 * (dev_p5 / len(x_dev)) * (dev_r5 / len(x_dev)) / ((dev_p5 / len(x_dev)) + (dev_r5 / len(x_dev)) + epsilon),
          2 * (dev_p10 / len(x_dev)) * (dev_r10 / len(x_dev)) / (
                      (dev_p10 / len(x_dev)) + (dev_r10 / len(x_dev)) + epsilon),
          2 * (dev_p20 / len(x_dev)) * (dev_r20 / len(x_dev)) / (
                      (dev_p20 / len(x_dev)) + (dev_r20 / len(x_dev)) + epsilon))




    gc.collect()
    torch.cuda.empty_cache()
    scheduler.step()

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

