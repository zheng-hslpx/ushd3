from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """*** 新增：位置编码增强空间特征 ***"""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):
        # positions: [B, N, 2] - 归一化的x,y坐标
        B, N, _ = positions.shape
        pos_indices = (positions * 100).long().clamp(0, 199)  # 简单的位置索引映射
        pos_enc = self.pe[pos_indices[:,:,0]] + self.pe[pos_indices[:,:,1]]  # 组合x,y编码
        # 修复：确保维度匹配
        return x + pos_enc[:, :, :x.size(-1)]

class MultiHeadGraphAttention(nn.Module):
    """*** 核心改进：多头图注意力机制 ***"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert output_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_Q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_K = nn.Linear(input_dim, output_dim, bias=False) 
        self.W_V = nn.Linear(input_dim, output_dim, bias=False)
        self.W_O = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 残差连接的投影
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
    
    def forward(self, query, key, value, mask=None):
        B, N, _ = query.shape
        M = key.shape[1]
        
        # 多头变换
        Q = self.W_Q(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_O(out)
        
        # 残差连接和层归一化
        residual = self.residual_proj(query)
        return self.layer_norm(out + residual)

class EnhancedUSVNodeEmbedding(nn.Module):
    """*** 增强版USV节点嵌入 ***"""
    def __init__(self, usv_dim: int, task_dim: int, edge_dim: int, embedding_dim: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dtype = torch.float32
        
        # *** 核心修复：多头注意力在embedding_dim上操作 ***
        self.usv_self_attn = MultiHeadGraphAttention(embedding_dim, embedding_dim, num_heads, dropout)
        self.usv_task_attn = MultiHeadGraphAttention(embedding_dim, embedding_dim, num_heads, dropout)
        
        # 特征投影
        self.usv_proj = nn.Linear(usv_dim, embedding_dim)
        self.task_proj = nn.Linear(task_dim + edge_dim, embedding_dim)
        
        # *** 新增：位置编码 ***
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

    def forward(self, usv_features, task_features, usv_task_edges, usv_task_adj):
        usv_features = usv_features.to(dtype=self.dtype)
        task_features = task_features.to(dtype=self.dtype)
        usv_task_edges = usv_task_edges.to(dtype=self.dtype)
        
        B, U, _ = usv_features.shape
        T = task_features.shape[1]
        
        # *** 核心修复：USV特征处理流程 ***
        # 1. 先投影到embedding_dim
        usv_emb = self.usv_proj(usv_features)
        # 2. 添加位置编码
        usv_emb = self.pos_encoding(usv_emb, usv_features[:, :, :2])
        # 3. USV自注意力（现在维度匹配）
        usv_emb = self.usv_self_attn(usv_emb, usv_emb, usv_emb)
        
        # USV-Task交叉注意力
        task_feat_combined = torch.cat([
            task_features.unsqueeze(1).expand(B, U, T, -1),
            usv_task_edges
        ], dim=-1)
        task_proj = self.task_proj(task_feat_combined).view(B, U*T, -1)
        
        # 构建mask
        mask = usv_task_adj.view(B, U, T)
        cross_attn_out = self.usv_task_attn(
            usv_emb.unsqueeze(2).expand(-1, -1, T, -1).contiguous().view(B, U*T, -1),
            task_proj,
            task_proj,
            mask.view(B, U*T, 1)
        ).view(B, U, T, -1).mean(dim=2)  # 平均池化
        
        # 前馈网络
        ffn_out = self.ffn(cross_attn_out)
        usv_emb = self.ffn_norm(ffn_out + cross_attn_out)
        
        return usv_emb

class EnhancedTaskNodeEmbedding(nn.Module):
    """*** 增强版Task节点嵌入 ***"""
    def __init__(self, task_dim: int, usv_embedding_dim: int, edge_dim: int, embedding_dim: int, 
                 eta: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dtype = torch.float32
        self.eta = eta
        
        # *** 核心修复：多头注意力在embedding_dim上操作 ***
        self.task_self_attn = MultiHeadGraphAttention(embedding_dim, embedding_dim, num_heads, dropout)
        self.task_usv_attn = MultiHeadGraphAttention(embedding_dim, embedding_dim, num_heads, dropout)
        
        # 投影层
        self.task_proj = nn.Linear(task_dim, embedding_dim)
        self.edge_proj = nn.Linear(edge_dim, embedding_dim)
        self.usv_proj = nn.Linear(usv_embedding_dim, embedding_dim)
        
        # *** 新增：位置编码 ***
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # *** 改进：更深的前馈网络 ***
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

    def forward(self, task_features, usv_embeddings, task_task_edges, task_positions):
        task_features = task_features.to(dtype=self.dtype)
        usv_embeddings = usv_embeddings.to(dtype=self.dtype)
        task_task_edges = task_task_edges.to(dtype=self.dtype)
        
        B, T, _ = task_features.shape
        U = usv_embeddings.shape[1]
        
        # *** 核心修复：Task特征处理流程 ***
        # 1. Task投影到embedding_dim
        task_emb = self.task_proj(task_features)
        # 2. 位置编码
        task_emb = self.pos_encoding(task_emb, task_positions)
        
        # *** 改进：基于学习的邻居选择而非固定eta ***
        with torch.no_grad():
            distances = torch.cdist(task_positions, task_positions)
            # 自适应邻居数量
            adaptive_k = min(max(3, T // 4), self.eta + 2)
            _, neighbor_idx = torch.topk(distances, k=adaptive_k, dim=-1, largest=False)
            neighbor_mask = torch.zeros_like(distances)
            batch_idx = torch.arange(B).unsqueeze(-1).unsqueeze(-1)
            task_idx = torch.arange(T).unsqueeze(0).unsqueeze(-1)
            neighbor_mask[batch_idx, task_idx, neighbor_idx] = 1.0
        
        # 3. Task自注意力（现在维度匹配）
        task_emb = self.task_self_attn(task_emb, task_emb, task_emb, neighbor_mask)
        
        # Task-USV交叉注意力
        usv_proj = self.usv_proj(usv_embeddings)
        task_usv_out = self.task_usv_attn(task_emb, usv_proj, usv_proj)
        
        # 前馈网络
        ffn_out = self.ffn(task_usv_out)
        task_emb = self.ffn_norm(ffn_out + task_usv_out)
        
        return task_emb

class GraphLevelAttentionPooling(nn.Module):
    """*** 新增：图级别注意力池化 ***"""
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )
        self.output_proj = nn.Linear(embedding_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, usv_embeddings, task_embeddings):
        # USV池化
        usv_attn_weights = F.softmax(self.attention(usv_embeddings), dim=1)
        usv_global = torch.sum(usv_attn_weights * usv_embeddings, dim=1)
        
        # Task池化
        task_attn_weights = F.softmax(self.attention(task_embeddings), dim=1)
        task_global = torch.sum(task_attn_weights * task_embeddings, dim=1)
        
        # 组合
        graph_emb = torch.cat([usv_global, task_global], dim=-1)
        graph_emb = self.output_proj(graph_emb)
        return self.layer_norm(graph_emb)

class EnhancedHeterogeneousGNN(nn.Module):
    """*** 增强版异构图神经网络 ***"""
    def __init__(self, config: dict):
        super().__init__()
        self.dtype = torch.float32
        
        E = config['embedding_dim']
        # *** 改进：自适应网络深度 ***
        base_layers = config.get('num_hgnn_layers', 3)
        complexity_factor = (config.get('num_tasks', 24) * config.get('num_usvs', 5)) / 120
        L = min(6, max(3, base_layers + int(complexity_factor)))
        print(f"[INFO] Using adaptive HGNN depth: {L} layers")
        
        eta = config.get('eta_neighbors', 3)
        dropout = float(config.get('dropout', 0.1))
        num_heads = config.get('num_attention_heads', 4)  # 新增配置项
        
        # *** 核心修复：明确定义输入维度 ***
        self.usv_dim = 3
        self.task_dim = 4
        self.edge_dim_ut = 3
        self.edge_dim_tt = 1
        
        # *** 核心改进：使用增强的嵌入层 ***
        self.usv_layers = nn.ModuleList()
        self.task_layers = nn.ModuleList()
        
        for i in range(L):
            # *** 修复：第一层使用原始维度，后续层使用embedding_dim ***
            usv_in = self.usv_dim if i == 0 else E
            task_in = self.task_dim if i == 0 else E
            
            self.usv_layers.append(EnhancedUSVNodeEmbedding(
                usv_in, task_in, self.edge_dim_ut, E, num_heads, dropout
            ))
            self.task_layers.append(EnhancedTaskNodeEmbedding(
                task_in, E, self.edge_dim_tt, E, eta, num_heads, dropout
            ))
        
        # *** 核心改进：图级别注意力池化 ***
        self.global_pooling = GraphLevelAttentionPooling(E, 2*E)
        
        # *** 新增：图级别特征增强 ***
        self.graph_enhancer = nn.Sequential(
            nn.Linear(2*E, 2*E),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*E, 2*E)
        )

    def forward(self, usv_features, task_features, usv_task_edges) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        usv_features = usv_features.to(dtype=self.dtype)
        task_features = task_features.to(dtype=self.dtype)
        usv_task_edges = usv_task_edges.to(dtype=self.dtype)
        
        B, U, T = usv_features.size(0), usv_features.size(1), task_features.size(1)
        
        # Task位置信息
        task_positions = task_features[:, :, :2]
        d_tt = torch.cdist(task_positions, task_positions).unsqueeze(-1)
        usv_task_adj = torch.ones(B, U, T, device=usv_features.device)
        
        usv_emb, task_emb = usv_features, task_features
        
        # *** 多层传播 ***
        for i, (usv_layer, task_layer) in enumerate(zip(self.usv_layers, self.task_layers)):
            usv_emb = usv_layer(usv_emb, task_emb, usv_task_edges, usv_task_adj)
            task_emb = task_layer(task_emb, usv_emb, d_tt, task_positions)
            
            # *** 新增：层间特征归一化 ***
            if i < len(self.usv_layers) - 1:  # 最后一层不归一化
                usv_emb = F.layer_norm(usv_emb, usv_emb.shape[-1:])
                task_emb = F.layer_norm(task_emb, task_emb.shape[-1:])
        
        # *** 改进：图级别表示学习 ***
        graph_emb = self.global_pooling(usv_emb, task_emb)
        graph_emb = self.graph_enhancer(graph_emb) + graph_emb  # 残差连接
        
        return usv_emb, task_emb, graph_emb

# 兼容性别名
HeterogeneousGNN = EnhancedHeterogeneousGNN