import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class CausalAttGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, causal_prior_tensor=None, heads=2):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.out_proj = nn.Linear(out_channels * heads, out_channels)
        self.tanh = nn.Tanh()
        self.heads = heads
        if causal_prior_tensor is not None:
            assert causal_prior_tensor.shape == (heads, out_channels)
            self.att.data = causal_prior_tensor.unsqueeze(0)  # [1, H, out]
        else:
            nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        H = self.heads
        x_proj = self.lin(x)  # [N, H*out]
        N = x.size(0)
        x_proj = x_proj.view(N, H, -1)  # [N, H, out]

        row, col = edge_index
        edge_h = x_proj[row] * self.att + x_proj[col] * self.att
        # scores = (edge_h.sum(dim=-1)).view(-1)
        scores = edge_h.sum(dim=-1)  # [E, H]

        # 聚合多头分数，例如取平均值
        scores = scores.mean(dim=1)  # [E]

        scores = torch.softmax(scores, dim=0)

        edge_weight = scores  # [E]
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N)).to(x.device)
        x_out = torch.sparse.mm(adj, x_proj.view(N, -1))  # [N, H*out]
        return self.tanh(self.out_proj(x_out))




def causal_attention(ss_causal_dict, num_nodes, heads=2, out_channels=256):
    ss_causal_mat = torch.zeros(num_nodes, num_nodes)
    for (i, j), v in ss_causal_dict.items():
        ss_causal_mat[i, j] = v  # 构建 N×N 因果强度矩阵


    # 假设我们只简单提取平均值向量作为 embedding，再 reshape 为模板
    ss_node_embedding = torch.mean(ss_causal_mat, dim=1, keepdim=True)  # [N, 1]

    # 构造一个长度为 heads * out_channels 的因果向量模板
    flatten = ss_node_embedding[:heads * out_channels].flatten()
    if flatten.shape[0] < heads * out_channels:
        # 若症状数不够，补 0 或循环填充
        repeat = (heads * out_channels + flatten.shape[0] - 1) // flatten.shape[0]
        flatten = flatten.repeat(repeat)[:heads * out_channels]

    # reshape 成注意力模板
    causal_prior_tensor = flatten.view(heads, out_channels)  # [heads, out_channels]

    return causal_prior_tensor










