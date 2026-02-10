import torch
from torch import nn
import math
import torch.nn.functional as F
from hflayers import Hopfield


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        device = ehr_adj.device

        self.ehr_adj = self.normalize(ehr_adj + torch.eye(ehr_adj.shape[0]).to(device))
        self.ddi_adj = self.normalize(ddi_adj + torch.eye(ddi_adj.shape[0]).to(device))
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)

        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    # def normalize(self, mx):
    #     """Row-normalize sparse matrix"""
    #     rowsum = np.array(mx.sum(1))
    #     r_inv = np.power(rowsum, -1).flatten()
    #     r_inv[np.isinf(r_inv)] = 0.
    #     r_mat_inv = np.diagflat(r_inv)
    #     mx = r_mat_inv.dot(mx)
    #     return mx

    # gpu version
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        mx = mx.to_dense()
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


# %%
class PiecewiseTSL(nn.Module):
    """近 k 次 visit 用 MLP+残差，远段用普通 cross-attention（Q=near_h, K/V=far_seq），输出 [batch, 2k, emb_dim]。"""
    def __init__(self, emb_dim, k):
        super().__init__()
        self.k = k
        self.emb_dim = emb_dim
        self.lin = nn.Linear(k * emb_dim, k * emb_dim)
        self.norm = nn.LayerNorm(k * emb_dim)
        self.scale = emb_dim ** -0.5  # 缩放因子，普通注意力 QK^T/sqrt(d)

    def forward(self, seq):
        batch, seq_len, emb_dim = seq.shape[0], seq.shape[1], seq.shape[2]
        if seq_len < self.k:
            pad_len = self.k - seq_len
            seq = torch.cat((seq, torch.zeros(batch, pad_len, emb_dim, device=seq.device)), dim=1)
        near_seq = seq[:, :self.k, :]
        near_h = self.lin(self.norm(near_seq.view(batch, -1))).view(batch, self.k, emb_dim) + near_seq
        far_h = torch.zeros(batch, self.k, emb_dim, device=seq.device)
        if seq_len > self.k:
            far_seq = seq[:, self.k:, :]  # [batch, seq_len-k, emb_dim]
            # 普通注意力：Q=near_h, K=V=far_seq, attn = softmax(QK^T/sqrt(d)), out = attn @ V
            attn_scores = torch.bmm(near_h, far_seq.transpose(-2, -1)) * self.scale  # [batch, k, seq_len-k]
            attn_weights = F.softmax(attn_scores, dim=-1)
            far_h = torch.bmm(attn_weights, far_seq)  # [batch, k, emb_dim]
        return torch.cat((near_h, far_h), dim=1)


class PatientRepLearn(nn.Module):
    """病人表征：诊断/手术各自经 PiecewiseTSL（近段+远段），再与 e_h 拼接，恢复最初层级性能。"""
    def __init__(self, emb_dim, k, d_voc_size, p_voc_size, m_voc_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.tsl_d = PiecewiseTSL(emb_dim, k)
        self.tsl_p = PiecewiseTSL(emb_dim, k)
        self.d_lin = nn.Linear(d_voc_size, emb_dim)
        self.p_lin = nn.Linear(p_voc_size, emb_dim)
        self.m_lin = nn.Linear(m_voc_size, emb_dim)

    def forward(self, diags, procs, meds):
        e_d = self.d_lin(diags)
        e_p = self.p_lin(procs)
        e_h = e_d + e_p
        h_d = self.tsl_d(e_d)
        h_p = self.tsl_p(e_p)
        if diags.size(1) < h_d.size(1):
            pad_len = h_d.size(1) - diags.size(1)
            e_h = torch.cat((e_h, torch.zeros(e_h.shape[0], pad_len, e_h.shape[2], device=e_h.device)), dim=1)
        else:
            e_h = e_h[:, :h_d.size(1), :]
        h_patient = torch.cat((e_h, h_d + h_p), dim=1)
        return h_patient
class MyNet(nn.Module):
    def __init__(self, emb_dim, voc_size,k,ehr_adj,ddi_adj):
        super().__init__()
        self.d_voc_size=voc_size[0]
        self.p_voc_size=voc_size[1]
        self.m_voc_size=voc_size[2]
        self.emb_dim=emb_dim
        self.k=k
        self.patrep=PatientRepLearn(emb_dim,k,self.d_voc_size,self.p_voc_size,self.m_voc_size)
        self.lin1=nn.Linear(2*k*emb_dim,self.m_voc_size)
        self.lin2=nn.Linear(2*k*emb_dim,2*k*emb_dim)
        self.norm=torch.nn.LayerNorm(2*k*emb_dim)
        self.gcn=GCN(self.m_voc_size, emb_dim, ehr_adj, ddi_adj)
        self.lin_med_expand=nn.Linear(emb_dim,emb_dim)
        # 用于不区分k的简单线性层和归一化层（ATC3级）
        self.lin1_simple=nn.Linear(emb_dim,self.m_voc_size)
        self.lin2_simple=nn.Linear(emb_dim,emb_dim)
        self.norm_simple=torch.nn.LayerNorm(emb_dim)
        self.w=0.7
        # 预训练相关的分类头
        # Mask预训练：预测诊断、手术、诊断->药物、手术->药物（不再区分k，使用emb_dim）
        self.cls_mask_dis = nn.Linear(emb_dim, self.d_voc_size)  # 诊断预测
        self.cls_mask_pro = nn.Linear(emb_dim, self.p_voc_size)  # 手术预测
        self.cls_mask_med_dis = nn.Linear(emb_dim, self.m_voc_size)  # 诊断->ATC3药物
        self.cls_mask_med_pro = nn.Linear(emb_dim, self.m_voc_size)  # 手术->ATC3药物
        
        # 药物历史信息整合模块（使用Hopfield网络）
        # 方案1：传统的门控单元（保留作为备选）
        self.history_contact = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.Tanh(),
            nn.Linear(emb_dim * 4, self.m_voc_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        
        # 方案2：使用Hopfield网络来存储和检索历史药物模式
        # 使用Hopfield网络：将历史药物作为stored patterns，当前患者表征作为query
        # pattern_projection也使用emb_dim维度，最后再投影到m_voc_size
        self.hopfield_history = Hopfield(
            input_size=emb_dim,  # 患者表征维度
            hidden_size=emb_dim,  # 关联空间维度
            output_size=emb_dim,  # 输出维度（先输出emb_dim，再投影到m_voc_size）
            pattern_size=None,  # 使用默认pattern size
            pattern_projection_size=emb_dim,  # pattern_projection的维度（与input_size一致）
            num_heads=4,  # 使用多头注意力
            scaling=None,  # 使用默认scaling
            update_steps_max=0,  # 1 次 Hopfield 迭代：用检索结果当新 query 再查一次（论文中通常一步即收敛）；0=纯单步注意力
            normalize_stored_pattern=True,  # 归一化存储的模式
            normalize_state_pattern=True,  # 归一化查询模式
            normalize_pattern_projection=True,  # 归一化输出投影
            batch_first=True,  # batch维度在前
            dropout=0.1,  # dropout率
            input_bias=True
        )
        
        # 将历史药物（multi-hot）映射到嵌入空间（用于stored patterns）
        self.med_history_proj = nn.Linear(self.m_voc_size, emb_dim)
        
        # 将Hopfield输出从emb_dim投影到m_voc_size
        self.hopfield_output_proj = nn.Linear(emb_dim, self.m_voc_size)
        
        # 可学习的历史增强权重（用于平衡基础预测和历史增强）
        self.history_weight = nn.Parameter(torch.tensor(0.7))  # 初始权重0.5，可学习调整
        self.use_hopfield = True  # 是否使用Hopfield网络（True使用Hopfield，False使用传统门控）
    
    

    def forward(self, diags, procs, meds, mode='finetune'):
        """
        Args:
            diags: [batch, seq_len, d_voc_size] or list of lists for pretrain
            procs: [batch, seq_len, p_voc_size] or list of lists for pretrain
            meds: [batch, seq_len, m_voc_size] or list of lists for pretrain
            mode: 'finetune', 'pretrain_mask'
        """
        if mode == 'pretrain_mask':
            # Mask预训练模式：输入是list格式 [[diag_list, proc_list], ...]
            batch_size = len(diags)
            device = next(self.parameters()).device
            
            # 将list格式转换为tensor
            diags_tensor = []
            procs_tensor = []
            
            for i in range(batch_size):
                d_list, p_list = diags[i], procs[i]
                
                d_tensor = torch.zeros(self.d_voc_size, device=device)
                d_tensor[torch.tensor(d_list, device=device)] = 1
                diags_tensor.append(d_tensor)
                
                p_tensor = torch.zeros(self.p_voc_size, device=device)
                p_tensor[torch.tensor(p_list, device=device)] = 1
                procs_tensor.append(p_tensor)
            
            diags = torch.stack(diags_tensor).unsqueeze(1)  # [batch, 1, d_voc_size]
            procs = torch.stack(procs_tensor).unsqueeze(1)  # [batch, 1, p_voc_size]
            meds = torch.zeros(batch_size, 1, self.m_voc_size, device=device)
            
            pat = self.patrep(diags, procs, meds)  # 【共享模块】训练self.patrep
            # 不再区分k，直接使用最后一个时间步的表示，维度为emb_dim
            h_patient = pat[:, -1, :]  # [batch, emb_dim]
            result_dis_med = self.cls_mask_med_dis(h_patient)  # 【预训练专用】诊断->ATC3药物
            result_pro_med = self.cls_mask_med_pro(h_patient)  # 【预训练专用】手术->ATC3药物
            
            result = torch.cat([result_dis, result_pro, result_dis_med, result_pro_med], dim=1)
            
            return result
        
        else:
            batch = diags.shape[0]
            # 使用诊断+手术+药物提取患者表示（改进版）
            pat = self.patrep(diags, procs, meds)
            # 不再区分k，直接使用最后一个时间步的表示，维度为emb_dim
            q = pat[:, -1, :]  # [batch, emb_dim]

            ehr_meds, ddi_meds = self.gcn()

            h_meds = ehr_meds.unsqueeze(0).repeat(diags.shape[0], 1, 1) * 1
            # h_meds += ddi_meds.unsqueeze(0).repeat(diags.shape[0], 1, 1) * 0.5
            h_meds = self.lin_med_expand(h_meds)

            # ATC3级预测
            # 不再区分k，使用第一个时间步的表示，维度为emb_dim
            o_1_atc3 = self.lin1_simple(pat[:, 0, :])  # [batch, emb_dim] -> [batch, m_voc_size]
            o_2_atc3 = torch.cosine_similarity(
                q.unsqueeze(1).repeat(1, self.m_voc_size, 1),
                h_meds,
                dim=2,
            )
            output_atc3_base = o_1_atc3# * self.w + o_2_atc3 * (1 - self.w)  # [batch, m_voc_size]
            
            # 整合历史药物信息（使用Hopfield网络或传统门控单元）
            ...
            ...
            ...
            else:
                # 如果没有历史visit，只使用基础预测
                output_atc3 = output_atc3_base
            

            return output_atc3
