from typing import Dict, Tuple
import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals, self.values = [],[],[],[],[],[]
        self.usv_features, self.task_features, self.action_masks = [],[],[]
        self.usv_task_edges = []

    def clear_memory(self):
        for lst in [self.states, self.actions, self.logprobs, self.rewards, self.is_terminals, 
                    self.values, self.usv_features, self.task_features, self.action_masks, self.usv_task_edges]:
            lst.clear()

    def add(self, state, action, logprob, reward, is_terminal, value, usv_task_edges):
        def _convert_and_validate(data, name):
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                tensor = data
            else:
                raise TypeError(f"{name} must be numpy array or torch tensor")
            
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values detected in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf values detected in {name}")
            return tensor

        try:
            usv_features = _convert_and_validate(state['usv_features'], 'usv_features')
            task_features = _convert_and_validate(state['task_features'], 'task_features')
            action_mask = _convert_and_validate(state['action_mask'], 'action_mask')
            
            self.states.append(state)
            self.actions.append(torch.tensor(action, dtype=torch.float32))
            self.logprobs.append(torch.tensor(logprob, dtype=torch.float32))
            self.rewards.append(float(reward))
            self.is_terminals.append(bool(is_terminal))
            self.values.append(float(value))
            self.usv_features.append(usv_features)
            self.task_features.append(task_features)
            self.action_masks.append(action_mask)
            self.usv_task_edges.append(usv_task_edges if isinstance(usv_task_edges, torch.Tensor) 
                                     else torch.from_numpy(usv_task_edges))
        except Exception as e:
            print(f"Error adding to memory: {str(e)}")
            print(f"State keys: {state.keys()}")
            if 'usv_features' in state:
                print(f"usv_features shape: {state['usv_features'].shape}")
            raise

class EnhancedActorNetwork(nn.Module):
    """*** 增强版Actor网络 ***"""
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # *** 改进：更深的网络架构 ***
        layers = []
        dim = input_dim
        
        # 输入投影层
        layers.extend([
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 使用GELU替代Tanh
            nn.Dropout(dropout)
        ])
        dim = hidden_dim
        
        # 主要隐藏层
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
            # *** 新增：残差连接（每两层） ***
            if i > 0 and i % 2 == 1:
                # 添加残差连接的标记，在forward中处理
                pass
        
        # 输出层
        layers.extend([
            nn.Linear(dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 输出层降低dropout
            nn.Linear(hidden_dim // 2, output_dim)
        ])
        
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)
        
        # *** 新增：注意力权重用于特征重要性 ***
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, usv_embeddings, task_embeddings, graph_embedding):
        B, U, E = usv_embeddings.shape
        T = task_embeddings.shape[1]
        
        # *** 改进：特征组合和注意力加权 ***
        usv_exp = usv_embeddings.unsqueeze(2).expand(B, U, T, E)
        task_exp = task_embeddings.unsqueeze(1).expand(B, U, T, E)
        graph_exp = graph_embedding.unsqueeze(1).unsqueeze(1).expand(B, U, T, graph_embedding.shape[-1])
        
        # 拼接特征
        feat = torch.cat([usv_exp, task_exp, graph_exp], dim=-1).view(B, U*T, -1)
        
        # *** 新增：特征注意力加权 ***
        attn_weights = self.feature_attention(feat)
        feat = feat * attn_weights
        
        # *** 改进：通过网络传播 ***
        x = feat
        residual = None
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i > 8 and i % 8 == 0 and residual is not None:
                # 每8层添加一个残差连接
                x = layer(x) + residual
                residual = x
            else:
                x = layer(x)
                if isinstance(layer, nn.Dropout) and i < 8:
                    residual = x
        
        return x.squeeze(-1)

class EnhancedCriticNetwork(nn.Module):
    """*** 核心改进：更强的价值网络 ***"""
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, dropout: float = 0.1):
        super().__init__()
        
        # *** 关键改进：增加网络深度和宽度 ***
        enhanced_hidden_dim = hidden_dim * 2  # 增加宽度
        enhanced_layers = max(num_hidden_layers + 2, 4)  # 增加深度
        
        layers = []
        dim = input_dim
        
        # *** 新增：多尺度特征提取 ***
        self.multi_scale_proj = nn.ModuleList([
            nn.Linear(input_dim, enhanced_hidden_dim),
            nn.Linear(input_dim, enhanced_hidden_dim // 2),
            nn.Linear(input_dim, enhanced_hidden_dim // 4)
        ])
        
        # 主网络
        layers.extend([
            nn.Linear(enhanced_hidden_dim + enhanced_hidden_dim//2 + enhanced_hidden_dim//4, enhanced_hidden_dim),
            nn.LayerNorm(enhanced_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # *** 改进：更深的隐藏层 ***
        dim = enhanced_hidden_dim
        for i in range(enhanced_layers):
            layers.extend([
                nn.Linear(dim, enhanced_hidden_dim),
                nn.LayerNorm(enhanced_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * (0.8 ** i))  # 逐层递减dropout
            ])
            
            # *** 新增：每隔两层添加残差连接 ***
            if i > 0 and i % 2 == 1:
                layers.append(ResidualBlock(enhanced_hidden_dim, dropout))
        
        # *** 改进：多层输出头 ***
        layers.extend([
            nn.Linear(enhanced_hidden_dim, enhanced_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(enhanced_hidden_dim // 2, enhanced_hidden_dim // 4),
            nn.GELU(),
            nn.Linear(enhanced_hidden_dim // 4, 1)
        ])
        
        self.net = nn.Sequential(*layers)
        
        # *** 新增：价值范围预测辅助头 ***
        self.value_range_head = nn.Sequential(
            nn.Linear(input_dim, enhanced_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(enhanced_hidden_dim // 2, 2)  # 预测价值的上下界
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, graph_embedding):
        # *** 新增：多尺度特征提取 ***
        multi_scale_feats = []
        for proj in self.multi_scale_proj:
            multi_scale_feats.append(F.gelu(proj(graph_embedding)))
        
        # 组合多尺度特征
        combined_feat = torch.cat(multi_scale_feats, dim=-1)
        
        # 主价值预测
        value = self.net(combined_feat)
        
        # *** 新增：价值范围辅助监督 ***
        value_range = self.value_range_head(graph_embedding)
        value_min, value_max = value_range[:, 0:1], value_range[:, 1:2]
        
        # 确保价值在合理范围内
        value = torch.clamp(value, value_min, value_max)
        
        return value

class ResidualBlock(nn.Module):
    """*** 新增：残差块 ***"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        return F.gelu(x + self.block(x))

class AdaptiveLearningRateScheduler:
    """*** 新增：自适应学习率调度器 ***"""
    def __init__(self, optimizer, initial_lr: float = 1e-4, patience: int = 50, factor: float = 0.8):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        
        self.best_critic_loss = float('inf')
        self.wait = 0
        self.warmup_steps = 100
        self.current_step = 0
        
    def step(self, critic_loss: float, reward_trend: float):
        self.current_step += 1
        
        # 前期warmup
        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        
        # 基于性能的自适应调整
        if critic_loss < self.best_critic_loss * 0.99:  # 显著改善
            self.best_critic_loss = critic_loss
            self.wait = 0
            # 小幅提升学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.05, self.initial_lr * 2)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # 降低学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
                self.wait = 0
                print(f"[INFO] Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.2e}")

# =============================================================================
# *** 探索机制核心实现开始 - EXPLORATION MECHANISM CORE IMPLEMENTATION ***
# =============================================================================

class EnhancedPPOAgent(nn.Module):
    """*** 增强版PPO智能体 - 集成epsilon-greedy随机探索机制 ***"""
    def __init__(self, hgnn_model: nn.Module, config: dict):
        super().__init__()
        E = int(config['embedding_dim'])
        dropout = float(config.get('dropout', 0.1))
        
        self.hgnn = hgnn_model
        
        # *** 核心改进：更强的网络架构 ***
        actor_hidden = int(config.get('n_latent_actor', 128)) * 2  # 增加Actor容量
        critic_hidden = int(config.get('n_latent_critic', 128)) * 2  # 增加Critic容量
        
        self.actor = EnhancedActorNetwork(
            4*E, actor_hidden, 
            int(config.get('n_hidden_actor', 2)) + 1,  # 增加层数
            1, dropout
        )
        self.critic = EnhancedCriticNetwork(
            2*E, critic_hidden, 
            int(config.get('n_hidden_critic', 2)) + 2,  # 显著增加Critic层数
            dropout
        )

        # =======================================================================
        # *** 探索机制参数配置 - EXPLORATION MECHANISM CONFIGURATION ***
        # =======================================================================
        self.exploration_config = {
            'initial_epsilon': config.get('initial_epsilon', 0.3),    # 初始探索率30%
            'min_epsilon': config.get('min_epsilon', 0.05),          # 最小探索率5%
            'epsilon_decay': config.get('epsilon_decay', 0.995),     # 探索率衰减系数
            'exploration_steps': config.get('exploration_steps', 1000),  # 探索阶段步数阈值
            'adaptive_epsilon': config.get('adaptive_epsilon', True), # 启用自适应探索率调整
        }
        
        # 探索机制运行时状态
        self.current_epsilon = self.exploration_config['initial_epsilon']
        self.training_step = 0
        self.train_flag = True  # 训练模式标志：True=训练模式(启用探索), False=评估模式(禁用探索)
        self.exploration_stats = {
            'total_random_actions': 0,      # 总随机动作数
            'recent_random_actions': 0,     # 最近的随机动作数
            'exploration_episodes': 0,      # 探索的回合数
        }

        # 网络初始化检查
        print("\n=== Enhanced Network Initialization Check ===")
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Actor parameters: {actor_params:,}")
        print(f"Critic parameters: {critic_params:,}")
        print(f"Total parameters: {(actor_params + critic_params):,}")
        print("=" * 50)
        print(f"🎯 EXPLORATION MECHANISM INITIALIZED")
        print(f"   Initial Epsilon: {self.current_epsilon:.3f}")
        print(f"   Min Epsilon: {self.exploration_config['min_epsilon']:.3f}")
        print(f"   Decay Rate: {self.exploration_config['epsilon_decay']:.4f}")
        print(f"   Exploration Steps Threshold: {self.exploration_config['exploration_steps']}")
        print("=" * 50)
        
        # *** 新增：梯度监控 ***
        self.grad_stats = {'actor': [], 'critic': []}
        
        self.old_hgnn = copy.deepcopy(hgnn_model)
        self.old_actor = copy.deepcopy(self.actor)
        self.old_critic = copy.deepcopy(self.critic)
        
        for p in self.old_hgnn.parameters(): p.requires_grad = False
        for p in self.old_actor.parameters(): p.requires_grad = False
        for p in self.old_critic.parameters(): p.requires_grad = False
        
        self.device = next(self.hgnn.parameters()).device

    def _to_dev(self, x, dtype=torch.float32):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.array(x))
        return x.to(self.device, dtype=dtype)

    # ==========================================================================
    # *** 探索机制核心方法 - EXPLORATION MECHANISM CORE METHODS ***
    # ==========================================================================
    
    def update_epsilon(self):
        """*** 探索率动态更新机制 - Dynamic Epsilon Update Mechanism ***"""
        self.training_step += 1
        old_epsilon = self.current_epsilon
        
        # 分阶段衰减策略：前期快速衰减，后期缓慢衰减
        if self.training_step < self.exploration_config['exploration_steps']:
            # 前期：指数衰减探索率
            self.current_epsilon = max(
                self.exploration_config['min_epsilon'],
                self.current_epsilon * self.exploration_config['epsilon_decay']
            )
        else:
            # 后期：线性衰减到最小值，保持少量探索
            decay_ratio = min(1.0, (self.training_step - self.exploration_config['exploration_steps']) / 2000)
            self.current_epsilon = max(
                self.exploration_config['min_epsilon'],
                self.exploration_config['min_epsilon'] * 2 * (1 - decay_ratio)
            )
        
        # 输出探索率变化日志
        if abs(old_epsilon - self.current_epsilon) > 0.001:
            print(f"🔄 [EXPLORATION UPDATE] Step {self.training_step}: ε {old_epsilon:.4f} → {self.current_epsilon:.4f}")
    
    def set_train_mode(self, train_flag: bool):
        """*** 训练/评估模式切换 - Training/Evaluation Mode Switch ***"""
        self.train_flag = train_flag
        if not train_flag:
            print(f"📊 [MODE SWITCH] → EVALUATION Mode (epsilon = 0, deterministic actions)")
        else:
            print(f"🎯 [MODE SWITCH] → TRAINING Mode (epsilon = {self.current_epsilon:.3f}, exploration enabled)")

    def get_exploration_stats(self):
        """*** 获取探索统计信息 - Get Exploration Statistics ***"""
        return {
            'current_epsilon': self.current_epsilon,
            'training_step': self.training_step,
            'train_mode': self.train_flag,
            'exploration_phase': 'early' if self.training_step < self.exploration_config['exploration_steps'] else 'late',
            'total_random_actions': self.exploration_stats['total_random_actions'],
            'recent_random_actions': self.exploration_stats['recent_random_actions'],
        }

    # ==========================================================================
    # *** 动作选择核心方法（含探索机制） - Action Selection with Exploration ***
    # ==========================================================================
    
    @torch.no_grad()
    def get_action(self, state: Dict, usv_task_edges: torch.Tensor, deterministic: bool=False) -> Tuple[int, float, float]:
        """*** 集成探索机制的动作选择方法 - Action Selection with Exploration Mechanism ***"""
        uf = self._to_dev(state['usv_features']).unsqueeze(0)
        tf = self._to_dev(state['task_features']).unsqueeze(0)
        am = self._to_dev(state['action_mask'], dtype=torch.bool).unsqueeze(0)
        ute = usv_task_edges.unsqueeze(0)

        # 通过HGNN获取节点嵌入和图嵌入
        ue, te, ge = self.old_hgnn(uf, tf, ute)
        scores = self.old_actor(ue, te, ge)
        
        # *** 动作掩码应用 - Apply Action Mask ***
        masked_scores = scores.masked_fill(~am, -1e8)
        
        # 检查是否有有效动作
        valid_actions = am.squeeze(0)
        if torch.all(~valid_actions):
            # 无有效动作时返回默认动作
            action, logp = torch.tensor(0, device=self.device), torch.tensor(0.0, device=self.device)
            value = self.old_critic(ge).squeeze(-1)
            return int(action.item()), float(logp.item()), float(value.item())
        
        # =======================================================================
        # *** 探索机制决策核心逻辑 - EXPLORATION MECHANISM DECISION LOGIC ***
        # =======================================================================
        
        # 只在训练模式且非确定性选择时启用探索
        use_exploration = self.train_flag and not deterministic
        
        if use_exploration:
            epsilon = self.current_epsilon
            
            # 🎲 epsilon-greedy探索策略决策点
            if torch.rand(1).item() < epsilon:
                # *** 执行随机探索动作 - EXECUTE RANDOM EXPLORATION ACTION ***
                valid_indices = torch.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    # 从有效动作中随机选择
                    random_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
                    action = valid_indices[random_idx]
                    
                    # 计算随机动作的log概率（用于策略梯度训练）
                    temperature = 1.0
                    probs = F.softmax(masked_scores / temperature, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    logp = dist.log_prob(action)
                    
                    # 统计随机动作信息
                    usv_idx = action.item() // tf.shape[1]
                    task_idx = action.item() % tf.shape[1]
                    self.exploration_stats['total_random_actions'] += 1
                    self.exploration_stats['recent_random_actions'] += 1
                    
                    # 🎯 探索动作日志输出
                    print(f"🎲 [RANDOM EXPLORATION] USV-{usv_idx} → Task-{task_idx} "
                          f"(ε={epsilon:.3f}, total_random={self.exploration_stats['total_random_actions']})")
                    
                    # *** 无人船任务规划特定的探索合理性检查 ***
                    # 检查当前USV的负载情况
                    current_usv_assignments = state.get('usv_assignments', {}).get(str(usv_idx), [])
                    if len(current_usv_assignments) > 3:  # 如果USV已有较多任务
                        print(f"⚠️  [EXPLORATION WARNING] USV-{usv_idx} has high load: {len(current_usv_assignments)} tasks")
                    
                else:
                    # 无有效动作时的回退处理
                    action = torch.tensor(0, device=self.device)
                    logp = torch.tensor(0.0, device=self.device)
            else:
                # *** 执行策略网络动作 - EXECUTE POLICY NETWORK ACTION ***
                # 使用策略网络进行动作选择（贪心或采样）
                temperature = 0.1 if deterministic else 1.0
                probs = F.softmax(masked_scores / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
                
                logp = dist.log_prob(action)
                
                # 策略动作日志（较少输出）
                if torch.rand(1).item() < 0.1:  # 10%概率输出策略动作日志
                    usv_idx = action.item() // tf.shape[1]
                    task_idx = action.item() % tf.shape[1]
                    print(f"🧠 [POLICY ACTION] USV-{usv_idx} → Task-{task_idx} (policy-guided)")
        else:
            # *** 评估模式或确定性模式：纯策略选择 ***
            temperature = 0.1 if deterministic else 1.0
            probs = F.softmax(masked_scores / temperature, dim=-1)
            
            dist = torch.distributions.Categorical(probs=probs)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
                logp = dist.log_prob(action)
            else:
                action = dist.sample()
                logp = dist.log_prob(action)
        
        # 价值函数评估
        value = self.old_critic(ge).squeeze(-1)
        
        # *** 最终动作验证 - Final Action Validation ***
        final_usv_idx = action.item() // tf.shape[1]
        final_task_idx = action.item() % tf.shape[1]
        
        # 确保动作在有效范围内
        if not valid_actions[action.item()]:
            print(f"❌ [ERROR] Selected invalid action: USV-{final_usv_idx} → Task-{final_task_idx}")
            # 回退到第一个有效动作
            valid_indices = torch.where(valid_actions)[0]
            if len(valid_indices) > 0:
                action = valid_indices[0]
                logp = dist.log_prob(action)
        
        return int(action.item()), float(logp.item()), float(value.item())
    
    # =======================================================================
    # *** 探索机制辅助方法 - EXPLORATION MECHANISM AUXILIARY METHODS ***
    # =======================================================================
    
    def reset_exploration_episode_stats(self):
        """*** 重置回合探索统计 - Reset Episode Exploration Statistics ***"""
        self.exploration_stats['recent_random_actions'] = 0
        self.exploration_stats['exploration_episodes'] += 1
    
    def log_exploration_summary(self):
        """*** 输出探索机制总结信息 - Log Exploration Summary ***"""
        stats = self.get_exploration_stats()
        print("\n" + "="*60)
        print("🎯 EXPLORATION MECHANISM SUMMARY")
        print("="*60)
        print(f"Current Epsilon: {stats['current_epsilon']:.4f}")
        print(f"Training Step: {stats['training_step']}")
        print(f"Mode: {'TRAINING' if stats['train_mode'] else 'EVALUATION'}")
        print(f"Phase: {stats['exploration_phase'].upper()}")
        print(f"Total Random Actions: {stats['total_random_actions']}")
        print(f"Recent Random Actions: {stats['recent_random_actions']}")
        print(f"Exploration Episodes: {self.exploration_stats['exploration_episodes']}")
        print("="*60 + "\n")

# =============================================================================
# *** 探索机制核心实现结束 - EXPLORATION MECHANISM CORE IMPLEMENTATION END ***
# =============================================================================

    def evaluate_actions(self, uf_b, tf_b, am_b, ute_b, actions_b):
        ue, te, ge = self.hgnn(uf_b, tf_b, ute_b)
        scores = self.actor(ue, te, ge)
        
        am_b_bool = am_b.to(torch.bool)
        
        # 检查是否有样本完全没有有效动作
        no_valid_actions = torch.all(~am_b_bool, dim=1)
        if torch.any(no_valid_actions):
            print(f"Warning: {no_valid_actions.sum().item()} samples have no valid actions!")
            am_b_bool[no_valid_actions, 0] = True
        
        # *** 改进：更稳定的概率计算 ***
        masked_scores = scores.masked_fill(~am_b_bool, -1e8)
        
        # 数值稳定性处理
        max_scores = torch.max(masked_scores, dim=-1, keepdim=True).values
        stable_scores = masked_scores - max_scores
        
        # 计算概率
        exp_scores = torch.exp(stable_scores) * am_b_bool.float()
        probs_sum = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / (probs_sum + 1e-20)
        
        # 最终合法性检查
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = torch.distributions.Categorical(probs=probs)
        return dist.log_prob(actions_b), self.critic(ge).squeeze(-1), dist.entropy()

    def update_old_policy(self):
        self.old_hgnn.load_state_dict(self.hgnn.state_dict())
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
    
    def get_gradient_stats(self):
        """*** 新增：梯度统计 ***"""
        actor_grad_norm = 0
        critic_grad_norm = 0
        
        for p in self.actor.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.data.norm(2).item() ** 2
        
        for p in self.critic.parameters():
            if p.grad is not None:
                critic_grad_norm += p.grad.data.norm(2).item() ** 2
        
        actor_grad_norm = actor_grad_norm ** 0.5
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.grad_stats['actor'].append(actor_grad_norm)
        self.grad_stats['critic'].append(critic_grad_norm)
        
        return actor_grad_norm, critic_grad_norm

class EnhancedPPO:
    """*** 增强版PPO训练器 - 支持探索机制 ***"""
    def __init__(self, agent: EnhancedPPOAgent, config: dict):
        self.agent = agent
        
        # *** 改进：更精细的超参数 ***
        self.gamma = float(config.get('gamma', 0.99))
        self.eps_clip = float(config.get('eps_clip', 0.15))  # 略微增加
        self.K_epochs = int(config.get('K_epochs', 4))  # 减少epoch数
        self.vf_coeff = float(config.get('vf_coeff', 0.3))  # 降低value loss权重
        self.entropy_coeff = float(config.get('entropy_coeff', 0.02))  # 增加探索
        self.minibatch_size = int(config.get('minibatch_size', 64))  # 减小batch size
        self.gae_lambda = float(config.get('gae_lambda', 0.95))
        self.max_grad_norm = float(config.get('max_grad_norm', 0.5))
        
        # *** 核心改进：自适应学习率 ***
        initial_lr = float(config.get('lr', 8e-5))  # 降低初始学习率
        self.optimizer = torch.optim.AdamW(  # 使用AdamW
            self.agent.parameters(), 
            lr=initial_lr, 
            eps=1e-5, 
            weight_decay=1e-4  # 添加权重衰减
        )
        
        self.scheduler = AdaptiveLearningRateScheduler(self.optimizer, initial_lr)
        
        # 早停参数
        self.early_stop_patience = int(config.get('early_stop_patience', 150))
        self.best_eval_reward = float('-inf')
        self.patience_counter = 0
        
        # *** 新增：训练统计和探索管理 ***
        self.training_stats = {
            'critic_losses': [],
            'actor_losses': [],
            'gradient_norms': {'actor': [], 'critic': []},
            'exploration_stats': []  # 新增探索统计
        }
        
        # =======================================================================
        # *** 探索机制训练调度配置 - EXPLORATION TRAINING SCHEDULE CONFIG ***
        # =======================================================================
        self.exploration_schedule = {
            'update_frequency': config.get('epsilon_update_frequency', 10),        # 每10次更新调整一次epsilon
            'performance_threshold': config.get('exploration_performance_threshold', -50),  # 性能阈值
            'adaptive_exploration': config.get('adaptive_exploration', True),       # 自适应探索开关
            'exploration_boost_episodes': config.get('exploration_boost_episodes', 100),  # 前100回合加强探索
        }
        
        self.update_count = 0
        
        print("🎯 PPO TRAINING WITH EXPLORATION MECHANISM INITIALIZED")
        print(f"   Update frequency: every {self.exploration_schedule['update_frequency']} updates")
        print(f"   Performance threshold: {self.exploration_schedule['performance_threshold']}")
        print(f"   Adaptive exploration: {self.exploration_schedule['adaptive_exploration']}")

    def update(self, memory: Memory, eval_reward: float = None):
        dev = self.agent.device
        self.update_count += 1
        
        # =======================================================================
        # *** 探索机制更新调度 - EXPLORATION MECHANISM UPDATE SCHEDULE ***
        # =======================================================================
        if self.update_count % self.exploration_schedule['update_frequency'] == 0:
            old_epsilon = self.agent.current_epsilon
            
            # 基于性能的自适应探索率调整
            if self.exploration_schedule['adaptive_exploration'] and eval_reward is not None:
                if eval_reward < self.exploration_schedule['performance_threshold']:
                    # 性能不佳，适度增加探索
                    self.agent.current_epsilon = min(
                        self.agent.current_epsilon * 1.1,
                        self.agent.exploration_config['initial_epsilon'] * 0.8
                    )
                    print(f"📈 [ADAPTIVE EXPLORATION] Performance low ({eval_reward:.1f}), "
                          f"boosted ε: {old_epsilon:.3f} → {self.agent.current_epsilon:.3f}")
                else:
                    # 性能良好，正常衰减探索率
                    self.agent.update_epsilon()
                    
                    if abs(old_epsilon - self.agent.current_epsilon) > 0.001:
                        print(f"📉 [NORMAL DECAY] Performance good ({eval_reward:.1f}), "
                              f"ε: {old_epsilon:.3f} → {self.agent.current_epsilon:.3f}")
            else:
                # 标准探索率更新
                self.agent.update_epsilon()
        
        # 学习率监控
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n🔧 Update #{self.update_count} - LR: {current_lr:.2e}, Epsilon: {self.agent.current_epsilon:.3f}")
        
        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=dev)
        values = torch.tensor(memory.values, dtype=torch.float32, device=dev)
        terminals = torch.tensor(memory.is_terminals, dtype=torch.float32, device=dev)
        
        # 奖励统计
        print(f"Rewards - min: {rewards.min().item():.2f}, max: {rewards.max().item():.2f}, mean: {rewards.mean().item():.2f}")
        
        # *** 改进：更精确的优势估计 ***
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - terminals[t]
                nextvalue = values[t]
            else:
                nextnonterminal = 1.0 - terminals[t]
                nextvalue = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
        
        returns = advantages + values[:-1]
        
        # *** 改进：更稳定的优势归一化 ***
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备批量数据
        old_uf = self.to_tensor(memory.usv_features, dev)
        old_tf = self.to_tensor(memory.task_features, dev)
        old_am = self.to_tensor(memory.action_masks, dev)
        old_ute = self.to_tensor(memory.usv_task_edges, dev)
        old_actions = self.to_tensor(memory.actions, dev, torch.long)
        old_logprobs = self.to_tensor(memory.logprobs, dev)
        
        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates = 0
        
        for epoch in range(self.K_epochs):
            indices = torch.randperm(len(old_actions), device=dev)
            
            for start_idx in range(0, len(old_actions), self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, len(old_actions))
                batch_indices = indices[start_idx:end_idx]
                
                logprobs, vals, entropy = self.agent.evaluate_actions(
                    old_uf[batch_indices], old_tf[batch_indices], 
                    old_am[batch_indices], old_ute[batch_indices], 
                    old_actions[batch_indices]
                )
                
                # *** 改进：更稳定的ratio计算 ***
                ratios = torch.exp(torch.clamp(logprobs - old_logprobs[batch_indices], -10, 10))
                surr1 = ratios * advantages[batch_indices]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[batch_indices]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # *** 改进：更鲁棒的价值损失 ***
                value_target = returns[batch_indices]
                value_loss_unclipped = F.mse_loss(vals, value_target)
                
                # 价值裁剪（可选）
                vals_clipped = old_logprobs[batch_indices] + torch.clamp(
                    vals - old_logprobs[batch_indices], -self.eps_clip, self.eps_clip
                )
                value_loss_clipped = F.mse_loss(vals_clipped, value_target)
                critic_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                
                entropy_loss = entropy.mean()
                
                # *** 新增：探索阶段自适应损失权重 ***
                exploration_factor = 1.0 + 0.5 * self.agent.current_epsilon  # 探索期增加熵权重
                
                total_loss = (actor_loss + 
                             self.vf_coeff * critic_loss - 
                             self.entropy_coeff * exploration_factor * entropy_loss)
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # *** 改进：自适应梯度裁剪 ***
                actor_grad_norm, critic_grad_norm = self.agent.get_gradient_stats()
                
                # 根据梯度历史动态调整裁剪阈值
                if len(self.training_stats['gradient_norms']['critic']) > 10:
                    recent_critic_grads = self.training_stats['gradient_norms']['critic'][-10:]
                    adaptive_clip = min(self.max_grad_norm * 2, np.percentile(recent_critic_grads, 75))
                else:
                    adaptive_clip = self.max_grad_norm
                
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), adaptive_clip)
                
                self.optimizer.step()
                
                # 统计更新
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        
        self.agent.update_old_policy()
        
        # *** 核心改进：自适应学习率调度 ***
        avg_critic_loss = total_critic_loss / num_updates
        avg_reward = rewards.mean().item()
        self.scheduler.step(avg_critic_loss, avg_reward)
        
        # *** 新增：更新训练统计（包括探索信息） ***
        self.training_stats['critic_losses'].append(avg_critic_loss)
        self.training_stats['actor_losses'].append(total_actor_loss / num_updates)
        self.training_stats['exploration_stats'].append(self.agent.get_exploration_stats())
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': avg_critic_loss,
            'entropy_loss': total_entropy_loss / num_updates,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm_actor': np.mean(self.agent.grad_stats['actor'][-10:]) if self.agent.grad_stats['actor'] else 0,
            'grad_norm_critic': np.mean(self.agent.grad_stats['critic'][-10:]) if self.agent.grad_stats['critic'] else 0,
            # *** 探索机制相关统计信息 ***
            'exploration_epsilon': self.agent.current_epsilon,
            'exploration_phase': self.training_stats['exploration_stats'][-1]['exploration_phase'],
            'total_random_actions': self.training_stats['exploration_stats'][-1]['total_random_actions'],
        }

    def check_early_stop(self, eval_reward: float) -> bool:
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stop_patience

    def to_tensor(self, data, device, dtype=None):
        return torch.tensor(np.array(data), device=device, dtype=dtype)

# 兼容性别名
PPOAgent = EnhancedPPOAgent
PPO = EnhancedPPO