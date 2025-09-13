
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================== 工具函数（数值稳定） ==============================

def _safe_masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    对 logits 做带掩码 softmax：
      - mask 中 True/1 表示“可选”，False/0 表示“屏蔽”
      - 全被屏蔽或 softmax 出现 NaN 时，退化为“等概率分布”（或对可选位等概率）
    返回概率（满足 simplex）
    """
    if mask is None:
        probs = F.softmax(torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9), dim=dim)
        if torch.isnan(probs).any():
            probs = torch.full_like(logits, 1.0 / logits.size(dim))
        return probs

    if mask.dtype != torch.bool:
        mask = mask != 0
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probs = F.softmax(masked_logits, dim=dim)

    # 全被屏蔽：均匀分布
    all_masked = (~mask).all(dim=dim, keepdim=True)
    if all_masked.any():
        uni = torch.full_like(logits, 1.0 / logits.size(dim))
        probs = torch.where(all_masked, uni, probs)

    # 数值异常：对可选位均匀
    if torch.isnan(probs).any():
        valid_cnt = mask.sum(dim=dim, keepdim=True).clamp(min=1).to(probs.dtype)
        probs = mask.to(probs.dtype) / valid_cnt
    return probs


def _build_fallback_mask_from_state(state: Dict, device: torch.device) -> torch.Tensor:
    """
    最弱约束掩码：只允许“最早可用 USV × 未调度任务”
    返回形状 [1, U*T] 的 bool
    """
    uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=device)  # [U,3]
    tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=device) # [T,4]
    U, T = uf.size(0), tf.size(0)
    avail = uf[:, 2]
    min_av = torch.min(avail) if U > 0 else torch.tensor(0.0, device=device)
    earliest = (torch.abs(avail - min_av) <= 1e-6)   # [U]
    unscheduled = tf[:, 3] > 0                       # [T]
    m2d = earliest[:, None] & unscheduled[None, :]   # [U,T]
    return m2d.view(1, U * T)


# ============================== 经验回放（按步存） ==============================

class Memory:
    """最简 Memory：逐步追加，最后 values 需补一个 bootstrap 值"""
    def __init__(self):
        self.states: List[Dict] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []
        self.values: List[float] = []          # 注意：最后需要额外 append 一个 bootstrap V_{T}
        self.usv_task_edges: List[torch.Tensor] = []

    def add(self, state: Dict, action: int, logprob: float, reward: float,
            done: bool, value: float, edges: torch.Tensor):
        self.states.append(state)
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.is_terminals.append(bool(done))
        self.values.append(float(value))
        self.usv_task_edges.append(edges.detach().cpu())

    def clear_memory(self):
        self.__init__()


# ============================== Actor / Critic ==============================

class PairwiseActor(nn.Module):
    """
    Actor：对每个 <USV,Task> 生成一个 logit
    输入：
      ue: [B,U,E], te: [B,T,E], ge: [B,E]
    处理：
      Broadcast 到 [B,U,T,E]，拼接 [u_e, t_e, g_e] → [B,U,T,3E]
      MLP 输出 [B,U,T,1] → 拉平 [B,U*T]
    """
    def __init__(self, embed_dim: int, n_hidden: int = 2, n_latent: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = 3 * embed_dim
        layers = []
        h = n_latent
        layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_hidden - 1)):
            layers += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)

        # 初始化（Xavier）
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, ue: torch.Tensor, te: torch.Tensor, ge: torch.Tensor) -> torch.Tensor:
        B, U, E = ue.shape
        T = te.size(1)
        u = ue.unsqueeze(2).expand(-1, -1, T, -1)                    # [B,U,T,E]
        t = te.unsqueeze(1).expand(-1, U, -1, -1)                    # [B,U,T,E]
        g = ge.unsqueeze(1).unsqueeze(1).expand(-1, U, T, -1)        # [B,U,T,E]
        x = torch.cat([u, t, g], dim=-1)                             # [B,U,T,3E]
        logits = self.net(x).squeeze(-1).view(B, U * T)              # [B,U*T]
        return logits


class GraphCritic(nn.Module):
    """
    Critic：对 (ue, te, ge) 评估状态价值
    简洁实现：对序列用注意力池化（或直接用 ge），这里采用 ge + 轻 MLP
    """
    def __init__(self, embed_dim: int, n_hidden: int = 2, n_latent: int = 128, dropout: float = 0.1):
        super().__init__()
        layers = []
        h = n_latent
        layers += [nn.Linear(embed_dim, h), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_hidden - 1)):
            layers += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(h, 1)]
        self.v = nn.Sequential(*layers)

        for m in self.v:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, ue: torch.Tensor, te: torch.Tensor, ge: torch.Tensor) -> torch.Tensor:
        return self.v(ge)  # [B,1]


# ============================== Agent ==============================

class EnhancedPPOAgent(nn.Module):
    """
    负责：
      - 调用 HGNN 提取 (ue, te, ge)
      - Actor 取动作、Critic 给价值
      - 训练/评估模式切换（控制探索 ε）
      - 提供探索统计
    """
    def __init__(self, hgnn: nn.Module, model_cfg: Dict[str, Any]):
        super().__init__()
        self.hgnn = hgnn
        E = int(model_cfg.get("embedding_dim", 128))
        self.actor = PairwiseActor(E,
                                   n_hidden=int(model_cfg.get("n_hidden_actor", 2)),
                                   n_latent=int(model_cfg.get("n_latent_actor", 128)),
                                   dropout=float(model_cfg.get("dropout", 0.1)))
        self.critic = GraphCritic(E,
                                  n_hidden=int(model_cfg.get("n_hidden_critic", 2)),
                                  n_latent=int(model_cfg.get("n_latent_critic", 128)),
                                  dropout=float(model_cfg.get("dropout", 0.1)))
        # 兼容字段（老代码里用 old_actor/old_critic）
        self.old_actor = self.actor
        self.old_critic = self.critic

        # 设备
        self.device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")

        # 探索相关
        self.training_mode: bool = True
        self.current_epsilon: float = float(model_cfg.get("initial_epsilon", 0.30))
        self.min_epsilon: float = float(model_cfg.get("min_epsilon", 0.05))
        self.epsilon_decay: float = float(model_cfg.get("epsilon_decay", 0.995))
        self.exploration_steps: int = int(model_cfg.get("exploration_steps", 1000))
        self._random_actions_total: int = 0

    # -------- 训练/评估模式 --------
    def set_train_mode(self, mode: bool) -> None:
        self.training_mode = bool(mode)
        if mode: self.train()
        else:    self.eval()

    def reset_exploration_episode_stats(self) -> None:
        self._random_actions_episode = 0

    def get_exploration_stats(self) -> Dict[str, Any]:
        phase = "early" if self.current_epsilon > (self.min_epsilon + 0.5 * (self.current_epsilon - self.min_epsilon)) else "late"
        return {
            "current_epsilon": float(self.current_epsilon),
            "total_random_actions": int(self._random_actions_total),
            "episode_random_actions": int(getattr(self, "_random_actions_episode", 0)),
            "exploration_phase": phase
        }

    def log_exploration_summary(self) -> None:
        s = self.get_exploration_stats()
        print(f"[Exploration] ε={s['current_epsilon']:.3f}, total_random={s['total_random_actions']} ({s['exploration_phase']})")

    # -------- 取动作（含安全 softmax + ε-greedy）--------
    @torch.no_grad()
    def get_action(self, state: Dict, usv_task_edges: torch.Tensor, epoch, max_epoch, deterministic: bool = False):
        """
        返回：action(int), logp(float), value(float)
        - 评估/确定性：不做 ε-greedy
        - 训练/随机：先用掩码 softmax 得到 probs，然后按 ε-greedy 采样
        """
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device

        # 1) batch=1 的张量
        uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=dev).unsqueeze(0)   # [1,U,Du]
        tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=dev).unsqueeze(0)  # [1,T,Dt]
        edges = usv_task_edges.to(dev).unsqueeze(0) if usv_task_edges.dim() == 3 else usv_task_edges.to(dev)
        U, T = uf.size(1), tf.size(1)

        # 2) 嵌入
        ue, te, ge = self.hgnn(uf, tf, edges)        # ue:[1,U,E], te:[1,T,E], ge:[1,E]

        # 3) Actor logits
        logits = self.old_actor(ue, te, ge)          # [1, U*T]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)

        # 4) 取/构建掩码
        raw_mask = state.get("action_mask", None)
        mask = None
        if raw_mask is not None:
            mask = torch.as_tensor(raw_mask, device=dev).view(1, U * T) != 0

        # 5) 安全 softmax
        probs = _safe_masked_softmax(logits, mask, dim=-1)
        # 兜底：若仍异常或全 0，用“最早可用×未调度”掩码
        if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any() or (mask is not None and mask.sum() == 0):
            fb_mask = _build_fallback_mask_from_state(state, dev)
            probs = _safe_masked_softmax(logits, fb_mask, dim=-1)

        # 6) ε-greedy（仅在训练&非确定性）
        use_eps = (self.training_mode and not deterministic)
        self.current_epsilon = max(1 - epoch / 1000, 0.001)
        if use_eps and random.random() < float(self.current_epsilon):
            # 输出随机选择概率
            # print("随机选择概率：", self.current_epsilon)
            # 均匀在“可选位”里随机
            m = mask
            if m is None or m.sum() == 0:
                m = _build_fallback_mask_from_state(state, dev)
            valid_idx = torch.nonzero(m.view(-1), as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(U * T, device=dev)
            a_idx = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=dev)]
            # logp = 对应概率（避免 -inf），用 probs 取值
            p = probs.view(-1)[a_idx].clamp_min(1e-12)
            logp = torch.log(p)
            self._random_actions_total += 1
            self._random_actions_episode = getattr(self, "_random_actions_episode", 0) + 1
        else:
            # 按分布采样或取 argmax
            if deterministic:
                a_idx = torch.argmax(probs, dim=-1)           # [1]
                p = probs.gather(-1, a_idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
                logp = torch.log(p)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a_idx = dist.sample()                          # [1]
                logp = dist.log_prob(a_idx)                    # [1]
            a_idx = a_idx.view(-1)

        action = int(a_idx.item())
        value = self.old_critic(ue, te, ge).squeeze(-1)        # [1]
        return action, float(logp.item()), float(value.item())


# ============================== PPO（分离优化） ==============================

class EnhancedPPO:
    """
    训练器：分离更新 Critic 与 Actor
    - 先多步 Critic（减小方差/更稳）
    - 再一步 Actor（clip-PPO）
    """
    def __init__(self, agent: EnhancedPPOAgent, cfg: Dict[str, Any]):
        self.agent = agent
        self.gamma       = float(cfg.get("gamma", 0.995))
        self.lam         = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps    = float(cfg.get("eps_clip", 0.15))
        self.vf_coeff    = float(cfg.get("vf_coeff", 0.30))
        self.ent_coeff   = float(cfg.get("entropy_coeff", 0.02))
        self.K_epochs    = int(cfg.get("K_epochs", 4))
        self.max_grad    = float(cfg.get("max_grad_norm", 0.5))
        self.minibatch   = int(cfg.get("minibatch_size", 256))  # 这里只作为“目标规模”的参考（变长样本按步处理）
        self.critic_updates = int(cfg.get("critic_updates_per_epoch", 2))
        self.actor_updates  = int(cfg.get("actor_updates_per_epoch", 1))

        lr = float(cfg.get("lr", 8e-5))
        lr_a = float(cfg.get("lr_actor", lr))
        lr_c = float(cfg.get("lr_critic", lr * 2.0))

        # 创建独立优化器
        self.opt_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=lr_a, betas=(0.9, 0.999))
        self.opt_hgnn = torch.optim.Adam(self.agent.hgnn.parameters(), lr=lr_a, betas=(0.9, 0.999))
        self.opt_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=lr_c, betas=(0.9, 0.999))

        self.lr_actor, self.lr_critic = lr_a, lr_c

        # 早停（可选）
        self.best_eval = -math.inf
        self.no_improve = 0
        self.early_patience = int(cfg.get("early_stop_patience", 999999))
        self.early_delta    = float(cfg.get("early_stop_delta", 1e-6))

    # --------- 计算 GAE & returns ---------
    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
          rewards[t], values[t]（含最后一个 bootstrap 值 values[T]）, dones[t]
        输出：
          advantages[t], returns[t]
        """
        T = len(rewards)
        assert len(values) == T + 1, "values 需要包含最后一个 bootstrap 值 V_T"
        adv = torch.zeros(T, dtype=torch.float32)
        ret = torch.zeros(T, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
            ret[t] = adv[t] + values[t]
        # 优化数值：标准化优势
        if T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # --------- 单步前向（用于 update）---------
    def _step_eval(self, state: Dict, edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：logits[1,N], value[1], mask[1,N], fallback_mask[1,N]
        """
        dev = self.agent.device if hasattr(self.agent, "device") else next(self.agent.parameters()).device
        uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=dev).unsqueeze(0)
        tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=dev).unsqueeze(0)
        e  = edges.to(dev).unsqueeze(0) if edges.dim() == 3 else edges.to(dev)
        U, T = uf.size(1), tf.size(1)

        ue, te, ge = self.agent.hgnn(uf, tf, e)
        logits = self.agent.actor(ue, te, ge)              # [1, U*T]
        raw_mask = state.get("action_mask", None)
        mask = None
        if raw_mask is not None:
            mask = torch.as_tensor(raw_mask, device=dev).view(1, U * T) != 0
        fb_mask = _build_fallback_mask_from_state(state, dev)
        value = self.agent.critic(ue, te, ge).squeeze(-1)  # [1]
        return logits, value, (mask if mask is not None else fb_mask), fb_mask

    # --------- 更新 ---------
    def update(self, memory: Memory, eval_reward: float = 0.0) -> Dict[str, float]:
        device = self.agent.device if hasattr(self.agent, "device") else next(self.agent.parameters()).device
        T = len(memory.rewards)
        if T == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "lr_critic": self.lr_critic, "epsilon": self.agent.current_epsilon}

    # 准备张量（放 CPU 即可，逐步 to(device)）
        rewards = torch.tensor(memory.rewards, dtype=torch.float32)
        values  = torch.tensor(memory.values,  dtype=torch.float32)
        dones   = torch.tensor(memory.is_terminals, dtype=torch.bool)
        old_log = torch.tensor(memory.logprobs, dtype=torch.float32)

    # 计算 GAE / returns
        adv, ret = self._compute_gae(rewards.tolist(), values.tolist(), dones.tolist())

    # ---------- Critic 多轮 ----------
        critic_loss_avg = 0.0
        for _ in range(self.critic_updates):
            loss_v = torch.zeros((), dtype=torch.float32, device=device)
            for t in range(T):
                logits, v, mask, fb_mask = self._step_eval(memory.states[t], memory.usv_task_edges[t])
                target_v = torch.as_tensor(ret[t], device=device).unsqueeze(0)
                loss_v = loss_v + 0.5 * F.mse_loss(v, target_v)

            self.opt_critic.zero_grad(set_to_none=True)
            self.opt_hgnn.zero_grad(set_to_none=True)
            loss_v.backward()
        # 🔧 增大梯度裁剪阈值
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 2.0)  # 从0.5改为2.0
            self.opt_critic.step()
            self.opt_hgnn.step()

            critic_loss_avg += float(loss_v.item())
        critic_loss_avg /= max(1, self.critic_updates)

    # ---------- Actor 改进版 ----------
        actor_loss_avg = 0.0
        entropy_avg = 0.0
    
        for _ in range(self.actor_updates):
            loss_pi = torch.zeros((), dtype=torch.float32, device=device)
            ent_sum = torch.zeros((), dtype=torch.float32, device=device)

            for t in range(T):
                logits, v, mask, fb_mask = self._step_eval(memory.states[t], memory.usv_task_edges[t])
                probs = _safe_masked_softmax(logits, mask, dim=-1)
            
                if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any() or (mask is not None and mask.sum() == 0):
                    probs = _safe_masked_softmax(logits, fb_mask, dim=-1)

            # 当前动作概率/对数概率
                a = torch.as_tensor(memory.actions[t], device=device).view(1, 1)
                new_logp = torch.log(probs.gather(-1, a).clamp_min(1e-12)).squeeze(-1)
            
            # 🔧 添加重要性采样比率限制
                old_logp_t = old_log[t].to(device)
                ratio = torch.exp(new_logp - old_logp_t)
                ratio = torch.clamp(ratio, 0.1, 10.0)  # 限制比率范围，防止过大更新

            # 优势标准化（在每个时间步）
                A = adv[t].to(device)

            # 🔧 修正的PPO目标（移除负号）
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * A
                policy_loss = -torch.min(surr1, surr2).mean()  # 注意这里是负号，因为我们要最大化目标
            
                loss_pi = loss_pi + policy_loss  # 🔥 关键修改：移除了额外的负号

            # 熵正则
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
                ent_sum = ent_sum + ent

        # 🔧 调整熵系数权重
            total_loss = loss_pi - self.ent_coeff * ent_sum  # 熵项保持负号（鼓励探索）
        
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_hgnn.zero_grad(set_to_none=True)
            total_loss.backward()
        
        # 🔧 增大Actor梯度裁剪阈值
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 2.0)  # 从0.5改为2.0
            torch.nn.utils.clip_grad_norm_(self.agent.hgnn.parameters(), 2.0)
        
            self.opt_actor.step()
            self.opt_hgnn.step()

            actor_loss_avg += float(total_loss.item())
            entropy_avg    += float(ent_sum.item())

        actor_loss_avg /= max(1, self.actor_updates)
        entropy_avg    /= max(1, self.actor_updates)

    # ε 衰减
        if self.agent.training_mode:
            self.agent.current_epsilon = max(self.agent.min_epsilon, self.agent.current_epsilon * self.agent.epsilon_decay)

        return {
            "actor_loss": actor_loss_avg,
            "critic_loss": critic_loss_avg,
            "entropy": entropy_avg,
            "lr_critic": float(self.opt_critic.param_groups[0]["lr"]),
            "epsilon": float(self.agent.current_epsilon)
    }


    # --------- 早停（可选）---------
    def check_early_stop(self, eval_reward: float) -> bool:
        """
        若 eval_reward 连续 early_stop_patience 次未提升 early_stop_delta，返回 True
        （train.py 会根据 True 触发早停）
        """
        if eval_reward > self.best_eval + self.early_delta:
            self.best_eval = eval_reward
            self.no_improve = 0
        else:
            self.no_improve += 1
        return self.no_improve >= self.early_patience
