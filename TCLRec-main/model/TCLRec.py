import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from mamba_ssm import Mamba
import dgl
from dgl.nn import GraphConv
import numpy as np


# ===================== InfoNCE Loss =====================
def info_nce_loss(z1, z2, temperature=0.6):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(representations, representations.T)
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)
    pos_sim = torch.sum(z1 * z2, dim=-1)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    logits = torch.exp(pos_sim / temperature)
    denom = torch.sum(torch.exp(sim_matrix / temperature), dim=1)
    return -torch.log(logits / (denom + 1e-8)).mean()


# ===================== Item Graph Convolution Network =====================
class ItemGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ItemGCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, out_dim, weight=False)
        self.gcn2 = GraphConv(out_dim, out_dim, weight=False)

    def forward(self, graph, x, edge_weight):
        x = self.gcn1(graph, x, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.gcn2(graph, x, edge_weight=edge_weight)
        return x


# ===================== User Attention =====================
class UserAttention(nn.Module):
    def __init__(self, hidden_size):
        super(UserAttention, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        	  nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, user_emb):
        att_weights = self.att_mlp(user_emb)
        return user_emb * att_weights


# ===================== Attention Fusion =====================
class AttentionFusion(nn.Module):
    def __init__(self, hidden_size, d):
        super(AttentionFusion, self).__init__()
        self.a = nn.Parameter(torch.randn(d))
        self.Wa = nn.Parameter(torch.randn(d, d))

    def forward(self, z1, z2):
        z1_weighted = torch.exp(torch.matmul(self.a, self.Wa @ z1.T))
        z2_weighted = torch.exp(torch.matmul(self.a, self.Wa @ z2.T))
        fused_emb = z1_weighted + z2_weighted
        return fused_emb

# ===================== MambaGCL Main Model =====================
class TCLRec(SequentialRecommender):
    def __init__(self, config, dataset, item_graph, item_interactions):
        super(TCLRec, self).__init__(config, dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.cl_weight = config['cl_weight']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0).to(self.device)
        self.mamba = Mamba(
            d_model=self.hidden_size,
            d_state=config['d_state'],
            d_conv=config['d_conv'],
            expand=config['expand']
        ).to(self.device)
        self.LayerNorm = nn.LayerNorm(self.hidden_size).to(self.device)
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.gnn = ItemGCN(self.hidden_size, self.hidden_size).to(self.device)
        self.user_attention = UserAttention(self.hidden_size).to(self.device)

        self.graph = item_graph
        self.item_interactions = item_interactions
        self.conformity_factor = self._calculate_conformity_factor()

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise ValueError("loss_type must be 'BPR' or 'CE'")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _calculate_conformity_factor(self, window_size=6):
        inter_feat = self.item_interactions
        item_ids = inter_feat[self.ITEM_ID].numpy()
        timestamps = inter_feat['timestamp'].numpy()

        from collections import defaultdict

        item_time_dict = defaultdict(list)
        for item_id, timestamp in zip(item_ids, timestamps):
            item_time_dict[item_id].append(timestamp)

        conformity_factors = torch.ones(self.n_items, device=self.device)

        for item_id, times in item_time_dict.items():
            days = np.array(times) // (24 * 3600 * 1000)
            unique_days, counts = np.unique(days, return_counts=True)

            if len(counts) >= window_size:
                window_vars = [
                    np.var(counts[i:i + window_size])
                    for i in range(len(counts) - window_size + 1)
                ]
                avg_variance = np.mean(window_vars)
            else:
                avg_variance = 0

            conformity_factor = 1 - torch.sigmoid(torch.tensor(avg_variance, device=self.device))
            conformity_factors[item_id] = conformity_factor

        return conformity_factors

    def forward_mamba(self, item_seq, item_seq_len):
        x = self.item_embedding(item_seq)
        x = self.LayerNorm(self.dropout(x))
        x = self.mamba(x)
        return self.gather_indexes(x, item_seq_len - 1)

    def forward_gnn(self, item_seq, item_seq_len):
        full_item_emb = self.gnn(self.graph, self.item_embedding.weight, self.graph.edata['weight'])
        seq_emb = F.embedding(item_seq, full_item_emb)
        return self.gather_indexes(seq_emb, item_seq_len - 1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        pos_items = interaction[self.POS_ITEM_ID].to(self.device)

        z1 = self.forward_mamba(item_seq, item_seq_len)
        z2 = self.forward_gnn(item_seq, item_seq_len)

        # 对比损失
        cl_loss = info_nce_loss(z1, z2)

        # 融合嵌入
        fused_emb = self.attention_fusion(z1, z2)

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID].to(self.device)
            pos_emb = self.item_embedding(pos_items) * self.conformity_factor[pos_items]
            neg_emb = self.item_embedding(neg_items) * self.conformity_factor[neg_items]
            pos_score = torch.sum(fused_emb * pos_emb, dim=-1)
            neg_score = torch.sum(fused_emb * neg_emb, dim=-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(fused_emb, self.item_embedding.weight.T) * self.conformity_factor
            rec_loss = self.loss_fct(logits, pos_items)

        return rec_loss + self.cl_weight * cl_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        test_item = interaction[self.ITEM_ID].to(self.device)

        seq_output = self.forward_mamba(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        conformity_weight = self.conformity_factor[test_item]

        att_user_emb = self.user_attention(seq_output)
        score = torch.sum(att_user_emb * test_item_emb, dim=-1) * conformity_weight
        return score


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        seq_output = self.forward_mamba(item_seq, item_seq_len)

        # 加权所有物品的评分
        scores = torch.matmul(seq_output, self.item_embedding.weight.T)
        scores *= self.conformity_factor
        return scores
