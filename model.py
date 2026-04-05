import pickle

import torch.sparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import sparse_coo_tensor
from torch_sparse import SparseTensor

from create_adj import create_shadj_mat
from layers import CausalAttGCNConv
from loss import info_nce_loss
from util.pytorch import inner_product, l2_loss, sp_mat_to_sp_tensor
from get_params import get_user_params, get_item_params

import torch.distributions as tdist
import math

from utils import get_hard_negatives

seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


class _LightGCN(nn.Module):
    def __init__(self, ss_num, hh_num, norm_adj, para):
        super(_LightGCN, self).__init__()
        self.num_users = ss_num
        self.num_items = hh_num
        self.dropout = nn.Dropout(0.3)
        # 对比学习相关
        self.n_layers = para.n_layers
        self.norm_adj = norm_adj
        self.ssl_temp = para.ssl_temp
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _forward_gcn(self, user_embeddings, item_embeddings, norm_adj):
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                neigh_emb = torch.sparse.mm(norm_adj[k], ego_embeddings)
            else:
                neigh_emb = torch.sparse.mm(norm_adj, ego_embeddings)

            # 残差连接，alpha是可学习参数，限制在0~1范围内用sigmoid
            alpha = torch.sigmoid(self.alpha)  # 保证alpha在(0,1)

            ego_embeddings = alpha * neigh_emb + (1 - alpha) * ego_embeddings

            # 可选激活
            ego_embeddings = F.relu(ego_embeddings)

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def forward(self, syn_embeddings, herb_embeddings, sub_graph1, sub_graph2, users, items, neg_items):
        # 1. 主图传播（原始 LightGCN）
        user_embeddings, item_embeddings = self._forward_gcn(syn_embeddings, herb_embeddings, self.norm_adj)

        # 2. 子图传播（对比学习增强）
        user_embeddings1, item_embeddings1 = self._forward_gcn(syn_embeddings, herb_embeddings, sub_graph1)  # 子图1
        user_embeddings2, item_embeddings2 = self._forward_gcn(syn_embeddings, herb_embeddings, sub_graph2)  # 子图2

        # 3. 归一化（对比学习要求）
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        # 找到指定用户或项目对应的向量
        user_embs = F.embedding(users.int(), user_embeddings).float()
        item_embs = F.embedding(items.int(), item_embeddings).float()
        neg_item_embs = F.embedding(neg_items.int(), item_embeddings).float()

        user_embs1 = F.embedding(users.int(), user_embeddings1).float()
        item_embs1 = F.embedding(items.int(), item_embeddings1).float()
        user_embs2 = F.embedding(users.int(), user_embeddings2).float()
        item_embs2 = F.embedding(items.int(), item_embeddings2).float()

        sup_pos_ratings = inner_product(user_embs, item_embs)  # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)  # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings  # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)  # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)  # [batch_size]

        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))
        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))  # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_users]

        loss_user = info_nce_loss(user_embs1, user_embs2, self.ssl_temp)
        loss_item = info_nce_loss(item_embs1, item_embs2)
        infonce_loss = loss_user + loss_item

        # 新增返回训练好的嵌入
        return {
            'sup_logits': sup_logits,
            'ssl_user': ssl_logits_user,
            'ssl_item': ssl_logits_item,
            'user_emb': user_embeddings,  # 主图用户嵌入
            'item_emb': item_embeddings,  # 主图物品嵌入
            'infonce_loss ': infonce_loss
        }


class GCNConv_SS_HH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)  # 线性变换
        self.tanh = nn.Tanh()  # 激活函数
        self.aggr = 'add'  # 聚合方式为求和

    def forward(self, x, edge_index):
        """
        x: 节点特征矩阵 [num_nodes, in_channels]
        edge_index: 边索引 [2, num_edges]
        """
        # Step 1: 线性变换
        x_transformed = self.lin(x)  # [num_nodes, out_channels]

        # Step 2: 构建稀疏邻接矩阵
        num_nodes = x.size(0)
        row, col = edge_index
        edge_weight = torch.ones(edge_index.size(1), device=x.device)  # 边权重全1
        adj = sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(num_nodes, num_nodes)
        )

        # Step 3: 邻居聚合（sum）
        out = torch.sparse.mm(adj, x_transformed)  # [num_nodes, out_channels]

        # Step 4: 激活函数
        return self.tanh(out)


class LMCL(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, syn_herb, norm_adj, device, para, hh_causal_tensor):
        super(LMCL, self).__init__()
        self.batchSize = para.batchSize
        self.embed_dropout = nn.Dropout(para.drop)  # 使用传入的 dropout 参数
        self.mlp_dropout = nn.Dropout(para.drop)
        self.SH_embedding = torch.nn.Embedding(sh_num, para.embed_size)
        self.ss_num = ss_num
        self.hh_num = hh_num
        self.sh_num = sh_num
        self.norm_adj = norm_adj
        self.device = device
        self.para = para
        self.embed_size = para.embed_size
        self.syn_herb = syn_herb
        self.ssl_ratio = para.ssl_ratio
        self.syn_embedding = torch.nn.Embedding(self.ss_num, self.embed_size)
        self.herb_embedding = torch.nn.Embedding(self.hh_num, self.embed_size)
        # self.reset_parameters()

        # S-H 图所需的网络
        self.SH_GCN = _LightGCN(self.ss_num, self.hh_num, norm_adj, para)

        #SH_mlp_1：全连接层，将特征从嵌入维度投影到 256 维
        self.SH_mlp_1 = torch.nn.Linear(para.embed_size, 256)
        #归一化
        self.SH_bn_1 = torch.nn.BatchNorm1d(256)
        # 非线性变换
        self.SH_tanh_1 = torch.nn.Tanh()

        self.SH_mlp_1_h = torch.nn.Linear(para.embed_size, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()

        self.hh_causal_tensor = hh_causal_tensor
        self.convHH = CausalAttGCNConv(para.embed_size, 256, causal_prior_tensor=hh_causal_tensor, heads=para.heads)
        self.convSS = CausalAttGCNConv(para.embed_size, 256, causal_prior_tensor=None, heads=para.heads)
        # self.convHH = CausalAttGCNConv(para.embed_size, 256, causal_prior_tensor=None, heads=para.heads)

        # SI诱导层
        # SUM
        self.mlp = torch.nn.Linear(256, 256)
        # cat
        # self.mlp = torch.nn.Linear(512, 512)
        self.SI_bn = torch.nn.BatchNorm1d(256)  #归一化
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()

        # 定义MLP融合网络（对症状）
        input_dim = para.embed_size * 3  # 拼接3个embed向量
        self.mlp_sym = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, para.embed_size)
        ).to(device)

        # 定义MLP融合网络（对草药）
        self.mlp_herb = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, para.embed_size)
        ).to(device)

        # 定义MLP融合网络（对症状）
        input_dim_nollm = para.embed_size * 2  # 拼接3个embed向量
        self.mlp_sym_nollm = nn.Sequential(
            nn.Linear(input_dim_nollm, 256),
            nn.ReLU(),
            nn.Linear(256, para.embed_size)
        ).to(device)

        # 定义MLP融合网络（对草药）
        self.mlp_herb_nollm = nn.Sequential(
            nn.Linear(input_dim_nollm, 256),
            nn.ReLU(),
            nn.Linear(256, para.embed_size)
        ).to(device)

        # 1. 载入LLM症状嵌入
        with open('data3/symptom_embeddings_new.pkl', 'rb') as f:
            symptom_embeddings_llm = pickle.load(f)
        self.symptom_embeddings_llm = torch.stack(
            [torch.tensor(v, device=device) for v in symptom_embeddings_llm.values()]
        )

        # 2. 载入LLM草药嵌入
        with open('data3/enhanced_item_embeddings.pkl', 'rb') as f:
            herb_embeddings_llm = pickle.load(f)
        self.herb_embeddings_llm = torch.stack(
            [torch.tensor(v, device=device) for k, v in sorted(herb_embeddings_llm.items(), key=lambda x: int(x[0]))]
        )


        # 3. 随机映射矩阵
        self.random_embedding_sym = torch.randn(50, para.embed_size, device=device)
        self.random_embedding_herbs = torch.randn(49, para.embed_size, device=device)

        # 4. 读取类别参数
        user_params_np = get_user_params('data3/user_category.txt')
        item_params_np = get_item_params('data3/item_category.txt')

        self.user_params = torch.from_numpy(user_params_np).float().to(device)
        self.item_params = torch.from_numpy(item_params_np).float().to(device)

        # 5. 可学习类别参数初始化为0，非零位置赋0.1
        self.user_params_learnable = nn.Parameter(torch.zeros_like(self.user_params))
        self.item_params_learnable = nn.Parameter(torch.zeros_like(self.item_params))

        with torch.no_grad():
            self.user_params_learnable[self.user_params != 0] = 0.1
            self.item_params_learnable[self.item_params != 0] = 0.1

        # 8. 融合权重
        self.llm_weight = 0.1
        self.class_weight = 0.3
        self.gcn_weight = 0.6

    def fuse_mlp(self):
        # 先准备三部分向量
        user_emb = self.user_params_learnable.mm(self.random_embedding_sym)  # [N, embed_size]
        item_emb = self.item_params_learnable.mm(self.random_embedding_herbs)  # [N, embed_size]

        sym_concat = torch.cat([self.symptom_embeddings_llm,
                                user_emb,
                                self.syn_embedding.weight], dim=1).float()  # [N, embed_size*3]

        herb_concat = torch.cat([self.herb_embeddings_llm,
                                 item_emb,
                                 self.herb_embedding.weight], dim=1).float()  # [N, embed_size*3]

        # 用MLP做非线性融合
        sym_emb = self.mlp_sym(sym_concat)  # [N, embed_size]
        herb_emb = self.mlp_herb(herb_concat)  # [N, embed_size]

        return sym_emb, herb_emb

    def fuse_mlp_nollm(self):
        # 先准备三部分向量
        user_emb = self.user_params_learnable.mm(self.random_embedding_sym)  # [N, embed_size]
        item_emb = self.item_params_learnable.mm(self.random_embedding_herbs)  # [N, embed_size]

        sym_concat = torch.cat([user_emb,
                                self.syn_embedding.weight], dim=1).float()  # [N, embed_size*3]

        herb_concat = torch.cat([item_emb,
                                 self.herb_embedding.weight], dim=1).float()  # [N, embed_size*3]

        # 用MLP做非线性融合
        sym_emb = self.mlp_sym_nollm(sym_concat)  # [N, embed_size]
        herb_emb = self.mlp_herb_nollm(herb_concat)  # [N, embed_size]

        return sym_emb, herb_emb

    def fuse_mlp_nocls(self):
        # 先准备三部分向量
        user_emb = self.user_params_learnable.mm(self.random_embedding_sym)  # [N, embed_size]
        item_emb = self.item_params_learnable.mm(self.random_embedding_herbs)  # [N, embed_size]

        sym_concat = torch.cat([self.symptom_embeddings_llm,
                                self.syn_embedding.weight], dim=1).float()  # [N, embed_size*3]

        herb_concat = torch.cat([self.herb_embeddings_llm,
                                 self.herb_embedding.weight], dim=1).float()  # [N, embed_size*3]

        # 用MLP做非线性融合
        sym_emb = self.mlp_sym_nollm(sym_concat)  # [N, embed_size]
        herb_emb = self.mlp_herb_nollm(herb_concat)  # [N, embed_size]

        return sym_emb, herb_emb

    def fuse_without_llm(self):
        user_emb = self.user_params_learnable.mm(self.random_embedding_sym)
        item_emb = self.item_params_learnable.mm(self.random_embedding_herbs)
        #
        sym_emb = self.class_weight * user_emb + self.gcn_weight * self.syn_embedding.weight
        herb_emb = self.class_weight * item_emb + self.gcn_weight * self.herb_embedding.weight
        # sym_emb = self.syn_embedding.weight
        # herb_emb = self.herb_embedding.weight
        return sym_emb, herb_emb

    def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH, prescription,
                sub_graph1=None, sub_graph2=None,
                positive_symptoms=None, positive_herbs=None, negative_herbs=None):

        x_ss0, x_hh0 = self.fuse_mlp()


        # x_ss0, x_hh0 = self.fuse_mlp_nollm()
        # x_ss0, x_hh0 = self.fuse_mlp_nocls()
        x_ss1 = self.convSS(x_ss0.float(), edge_index_SS)
        x_hh1 = self.convHH(x_hh0.float(), edge_index_HH)

        # 判断是训练还是测试
        is_train = sub_graph1 is not None and positive_symptoms is not None
        sym_emb_cls_only, herb_emb_cls_only = self.fuse_without_llm()

        if is_train:
            lightgcn_output = self.SH_GCN(
                sym_emb_cls_only,
                herb_emb_cls_only,
                sub_graph1=sub_graph1,
                sub_graph2=sub_graph2,
                users=positive_symptoms,
                items=positive_herbs,
                neg_items=negative_herbs
            )

            sup_logits = lightgcn_output['sup_logits']
            ssl_user = lightgcn_output['ssl_user']
            ssl_item = lightgcn_output['ssl_item']
            x_SH9 = lightgcn_output['user_emb']  # 形状: [num_users, emb_dim]
            x_SH99 = lightgcn_output['item_emb']  # 形状: [num_items, emb_dim]

            #
            x_SH9 = self.SH_mlp_1(x_SH9)
            x_SH9 = x_SH9.view(390, -1)
            x_SH9 = self.SH_bn_1(x_SH9)  # 归一化
            x_SH9 = self.SH_tanh_1(x_SH9)

            x_SH99 = self.SH_mlp_1_h(x_SH99)
            x_SH99 = x_SH99.view(805, -1)
            x_SH99 = self.SH_bn_1_h(x_SH99)
            x_SH99 = self.SH_tanh_1_h(x_SH99)

        else:
            x_SH9, x_SH99 = self.SH_GCN._forward_gcn(sym_emb_cls_only,
                                                     herb_emb_cls_only, self.norm_adj)

            # 投影
            x_SH9 = self.SH_mlp_1(x_SH9)
            x_SH99 = self.SH_mlp_1_h(x_SH99)
            # 预测阶段，只用 SH 主图
            x_SH9 = x_SH9.view(390, -1)
            x_SH9 = self.SH_bn_1(x_SH9)
            x_SH9 = self.SH_tanh_1(x_SH9)

            x_SH99 = x_SH99.view(805, -1)
            x_SH99 = self.SH_bn_1_h(x_SH99)
            x_SH99 = self.SH_tanh_1_h(x_SH99)

        # 信息融合
        # sum
        es = x_SH9 + x_ss1  # 症状特征融合
        eh = x_SH99 + x_hh1  # 药物特征融合
        e_synd = torch.mm(prescription, es)
        preSum = prescription.sum(dim=1).view(-1, 1)
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # batch*dim
        # e_synd_norm = self.gelu(e_synd_norm)  # batch*dim
        pre = torch.mm(e_synd_norm, eh.t())

        if is_train:
            # 损失项
            bpr_loss = -torch.sum(F.logsigmoid(sup_logits))
            reg_loss = l2_loss(
                sym_emb_cls_only[positive_symptoms],  # 症状直接用
                herb_emb_cls_only[positive_herbs],  # 草药 ID 加偏移量
                herb_emb_cls_only[negative_herbs]
            )
            clogits_user = torch.logsumexp(ssl_user / self.para.ssl_temp, dim=1)
            clogits_item = torch.logsumexp(ssl_item / self.para.ssl_temp, dim=1)
            infonce_loss = torch.sum(clogits_user + clogits_item)

            loss = bpr_loss + self.para.ssl_reg * infonce_loss + self.para.rec * reg_loss
        else:
            loss = torch.tensor(0.0, device=self.device)

        return pre, loss
