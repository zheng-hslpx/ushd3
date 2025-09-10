
from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================ 基础模块 ============================

class MLP(nn.Module):
    """两层前馈（带残差），默认 GELU + Dropout + LayerNorm"""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hid = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class EdgeAwareCrossAttention(nn.Module):
    """
    边感知跨模态注意力（可选边偏置）：
      - Q 来自“查询序列”（如 USV）
      - K/V 来自“键/值序列”（如 Task）
      - 若提供边特征 edge，则投影为与 K/V 同维的可加性偏置（每个头独立）
    形状：
      Q: [B, N, E]，K/V: [B, M, E]，edge: [B, N, M, De] 或 None
    输出：
      O: [B, N, E]
    """
    def __init__(self, embed_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 use_edge_bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.scale = self.d ** 0.5
        self.use_edge_bias = use_edge_bias

        # 线性投影：标准多头注意力
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim)

        # 将边特征投影到每个头的维度，用作 K/V 的可加性偏置
        self.edge_proj = nn.Linear(edge_dim, embed_dim) if use_edge_bias else None

        self.attn_drop = nn.Dropout(dropout)
        self.res_norm = nn.LayerNorm(embed_dim)

        # 参数初始化
        for m in (self.wq, self.wk, self.wv, self.wo):
            nn.init.xavier_uniform_(m.weight)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)
            nn.init.zeros_(self.edge_proj.bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                edge: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, E = q.shape
        M = k.shape[1]

        # [B,N,E] → [B,h,N,d]
        Q = self.wq(q).view(B, N, self.h, self.d).transpose(1, 2)      # [B,h,N,d]
        K = self.wk(k).view(B, M, self.h, self.d).transpose(1, 2)      # [B,h,M,d]
        V = self.wv(v).view(B, M, self.h, self.d).transpose(1, 2)      # [B,h,M,d]

        # 将 K/V 扩展到 [B,h,N,M,d]，以便与边偏置逐对相加
        K = K.unsqueeze(2)                                             # [B,h,1,M,d]
        V = V.unsqueeze(2)                                             # [B,h,1,M,d]
        if self.use_edge_bias and edge is not None:
            # edge: [B,N,M,De] → proj→ [B,N,M,E] → [B,h,N,M,d]
            Eb = self.edge_proj(edge).view(B, N, M, self.h, self.d).permute(0, 3, 1, 2, 4)
            K = K.expand(-1, -1, N, -1, -1) + Eb
            V = V.expand(-1, -1, N, -1, -1) + Eb
        else:
            K = K.expand(-1, -1, N, -1, -1)
            V = V.expand(-1, -1, N, -1, -1)

        # 点积注意力：scores [B,h,N,M]
        # Q: [B,h,N,1,d]，K: [B,h,N,M,d]
        scores = (Q.unsqueeze(3) * K).sum(-1) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # 上下文：ctx [B,h,N,d] → [B,N,E]
        ctx = (attn.unsqueeze(-1) * V).sum(-2)
        out = ctx.transpose(1, 2).contiguous().view(B, N, E)
        out = self.wo(out)

        # 残差 + LN
        return self.res_norm(q + out)


class AttentionPool(nn.Module):
    """图级注意力池化：给出序列 → 权重 → 加权和"""
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,E]
        w = F.softmax(self.scorer(x), dim=1)  # [B,N,1]
        g = (w * x).sum(dim=1)                # [B,E]
        return self.norm(g)


# ============================ 异构层（USV↔Task） ============================

class HeteroBlock(nn.Module):
    """
    单层“USV↔Task 跨模态交互”：
      USV ← Task（带边偏置） → 残差前馈
      Task ← USV（带边偏置） → 残差前馈
    """
    def __init__(self, embed_dim: int, edge_dim: int, heads: int, dropout: float, use_edge_bias: bool):
        super().__init__()
        self.usv_from_task = EdgeAwareCrossAttention(embed_dim, edge_dim, heads, dropout, use_edge_bias)
        self.task_from_usv = EdgeAwareCrossAttention(embed_dim, edge_dim, heads, dropout, use_edge_bias)
        self.usv_ffn = MLP(embed_dim, expansion=2, dropout=dropout)
        self.task_ffn = MLP(embed_dim, expansion=2, dropout=dropout)

    def forward(self, usv_emb: torch.Tensor, task_emb: torch.Tensor,
                ut_edge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # USV ← Task：Q=USV，K/V=Task，边为 [B,U,T,De]
        usv_upd = self.usv_from_task(usv_emb, task_emb, task_emb, edge=ut_edge)
        usv_out = self.usv_ffn(usv_upd)

        # Task ← USV：Q=Task，K/V=USV，边转置 [B,T,U,De]
        tu_edge = ut_edge.transpose(1, 2).contiguous()
        task_upd = self.task_from_usv(task_emb, usv_emb, usv_emb, edge=tu_edge)
        task_out = self.task_ffn(task_upd)

        return usv_out, task_out


# ============================ 主网络 ============================

class HeterogeneousGNN(nn.Module):
    """
    异构图编码器（精简版）：
      - 若数据规模较小且 `arch_slim=True`，自动减层（L=2）；否则默认 L=3（或由 num_hgnn_layers 指定）
      - 可通过 `disable_feature_attention=True` 关闭“边偏置”（退化为普通跨注意力）
    config 关键项：
      embedding_dim:int, dropout:float, num_attention_heads:int=4, num_hgnn_layers:int|None,
      arch_slim:bool=True, slim_threshold:int=120, disable_feature_attention:bool
      （可从 env 透传 num_usvs/num_tasks 用于自适应层数）
    """
    def __init__(self, config: dict):
        super().__init__()
        E = int(config.get("embedding_dim", 128))
        drop = float(config.get("dropout", 0.1))
        heads = int(config.get("num_attention_heads", 4))
        use_edge_bias = not bool(config.get("disable_feature_attention", False))

        # 自适应层数
        L_cfg = config.get("num_hgnn_layers", None)
        if L_cfg is not None:
            L = int(L_cfg)
        else:
            n_usv = int(config.get("num_usvs", 5))
            n_task = int(config.get("num_tasks", 24))
            slim = bool(config.get("arch_slim", True))
            thr = int(config.get("slim_threshold", 120))
            L = 2 if (slim and (n_usv * n_task <= thr)) else 3

        # 明确输入维：与 train/compute_edges 对齐
        self.usv_in_dim  = 3   # [x, y, available_time]
        self.task_in_dim = 4   # [x, y, processing_time, is_active]
        self.edge_ut_dim = 3   # [dist_ut, task_prox, usv2act]

        # 首层投影
        self.usv_proj  = nn.Linear(self.usv_in_dim, E)
        self.task_proj = nn.Linear(self.task_in_dim, E)

        # 堆叠异构层
        self.layers = nn.ModuleList([
            HeteroBlock(E, self.edge_ut_dim, heads, drop, use_edge_bias)
            for _ in range(L)
        ])

        # 图级池化：分别对 USV/Task 序列做注意力池化
        self.pool_usv  = AttentionPool(E)
        self.pool_task = AttentionPool(E)

        # ★ 图级投影：把拼接的 2E 压到 E（修复 Actor 期望 3E 的总输入）
        self.graph_proj = nn.Sequential(
            nn.Linear(2 * E, E),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(E, E)
        )
        self.graph_norm = nn.LayerNorm(E)

        # 初始化
        nn.init.xavier_uniform_(self.usv_proj.weight);  nn.init.zeros_(self.usv_proj.bias)
        nn.init.xavier_uniform_(self.task_proj.weight); nn.init.zeros_(self.task_proj.bias)

    # --------------------------- 前向传播 ---------------------------

    def forward(self,
                usv_features: torch.Tensor,      # [B,U,3] 或 [U,3]
                task_features: torch.Tensor,     # [B,T,4] 或 [T,4]
                usv_task_edges: torch.Tensor     # [B,U,T,3] 或 [U,T,3]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          usv_emb  : [B,U,E]
          task_emb : [B,T,E]
          graph_emb: [B,E]   # ★ 与 Actor 侧 3E 拼接一致
        """
        # 兼容单样本（无 batch 维）情况
        squeeze_b = False
        if usv_features.dim() == 2:
            usv_features = usv_features.unsqueeze(0); squeeze_b = True
        if task_features.dim() == 2:
            task_features = task_features.unsqueeze(0); squeeze_b = True
        if usv_task_edges.dim() == 3:
            usv_task_edges = usv_task_edges.unsqueeze(0); squeeze_b = True

        # 统一 dtype
        usv_features   = usv_features.to(dtype=torch.float32)
        task_features  = task_features.to(dtype=torch.float32)
        usv_task_edges = usv_task_edges.to(dtype=torch.float32)

        # 投影到统一维度
        usv_emb  = self.usv_proj(usv_features)    # [B,U,E]
        task_emb = self.task_proj(task_features)  # [B,T,E]

        # 多层 USV↔Task 交互
        for i, layer in enumerate(self.layers):
            usv_emb, task_emb = layer(usv_emb, task_emb, usv_task_edges)
            # 中间层进行 LayerNorm，最后一层不过度正则
            if i < len(self.layers) - 1:
                usv_emb  = F.layer_norm(usv_emb,  usv_emb.shape[-1:])
                task_emb = F.layer_norm(task_emb, task_emb.shape[-1:])

        # 图级别表示：USV/Task 注意力池化后拼接 → 压到 E 维（残差+LN）
        g_usv  = self.pool_usv(usv_emb)           # [B,E]
        g_task = self.pool_task(task_emb)         # [B,E]
        graph  = torch.cat([g_usv, g_task], dim=-1)   # [B,2E]
        graph  = self.graph_proj(graph)               # [B,E]
        graph  = self.graph_norm(graph)

        if squeeze_b:
            # 保持与旧调用兼容：若输入无 batch 维，这里也可不压缩，让上层统一处理
            # 但通常 Actor 端以 batch=1 调用，因此直接返回 [1,...] 更稳妥
            pass

        return usv_emb, task_emb, graph
