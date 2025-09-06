import copy
import json
import numpy as np
from typing import Dict, List

class ProgressiveTrainingConfig:
    """渐进式训练配置管理器 - 修复版本，解决阶段转换问题"""
    
    def __init__(self):
        self.stages = [
            # 第一阶段：基础学习 (250轮) - 延长基础训练
            {
                "name": "Stage1_Foundation",
                "episodes": 250,
                "env_config": {
                    "num_usvs": 2,
                    "num_tasks": 6,
                    "map_size": [60, 60],
                    "usv_speed": 5.0,
                    "battery_capacity": 220.0,
                    "charge_time": 30.0,
                    "device": "cuda:0",
                    "min_travel_time": 1.0,
                    "min_travel_distance": 2.0,
                    "travel_constraint_mode": "time",
                    "reward_config": {
                        "use_potential_based_reward": False,
                        "makespan_normalization": "dynamic"
                    },
                    "dynamic_masking_config": {
                        "enabled": True,
                        "max_load_ratio": 1.2
                    }
                },
                "model_config": {
                    "embedding_dim": 64,
                    "num_heads": 1,
                    "num_attention_heads": 4,
                    "num_hgnn_layers": 2,
                    "eta_neighbors": 3,
                    "mlp_hidden_dim": 64,
                    "n_hidden_actor": 2,
                    "n_hidden_critic": 3,
                    "n_latent_actor": 64,
                    "n_latent_critic": 128,
                    "dropout": 0.1,
                    "device": "cuda:0",
                    "use_batch_norm": True,
                    "initial_epsilon": 0.6,  # 降低初始探索率
                    "min_epsilon": 0.08,
                    "epsilon_decay": 0.996
                },
                "train_config": {
                    "lr": 3e-4,
                    "betas": [0.9, 0.999],
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "K_epochs": 3,
                    "eps_clip": 0.2,
                    "A_coeff": 1.0,
                    "vf_coeff": 0.5,
                    "entropy_coeff": 0.02,
                    "max_grad_norm": 0.5,
                    "minibatch_size": 32,
                    "save_frequency": 50,
                    "eval_frequency": 20,
                    "lr_schedule": "cosine_warm",  # 新增：平滑学习率调度
                    "warmup_episodes": 30        # 新增：预热期
                }
            },
            
            # 第二阶段：渐进扩展 (200轮) - 缓慢增加复杂度
            {
                "name": "Stage2_Gradual", 
                "episodes": 200,
                "transition_episodes": 20,  # 新增：过渡期
                "env_config": {
                    "num_usvs": 3,
                    "num_tasks": 9,           # 从12减少到9，更平滑过渡
                    "map_size": [70, 70],     # 从80减少到70
                    "usv_speed": 5.0,
                    "battery_capacity": 220.0,
                    "charge_time": 30.0,
                    "device": "cuda:0",
                    "min_travel_time": 1.2,   # 更温和的约束增加
                    "min_travel_distance": 2.5,
                    "travel_constraint_mode": "time",
                    "reward_config": {
                        "use_potential_based_reward": False,
                        "makespan_normalization": "dynamic"
                    },
                    "dynamic_masking_config": {
                        "enabled": True,
                        "max_load_ratio": 1.25
                    }
                },
                "model_config": {
                    "embedding_dim": 64,
                    "num_heads": 1,
                    "num_attention_heads": 4,
                    "num_hgnn_layers": 2,    # 保持模型复杂度稳定
                    "eta_neighbors": 3,
                    "mlp_hidden_dim": 80,    # 轻微增加
                    "n_hidden_actor": 2,
                    "n_hidden_critic": 3,
                    "n_latent_actor": 80,
                    "n_latent_critic": 160,
                    "dropout": 0.1,
                    "device": "cuda:0",
                    "use_batch_norm": True,
                    "initial_epsilon": 0.3,  # 继承前一阶段的探索策略
                    "min_epsilon": 0.06,
                    "epsilon_decay": 0.997
                },
                "train_config": {
                    "lr": 2.5e-4,           # 更温和的学习率下降
                    "betas": [0.9, 0.999],
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "K_epochs": 3,
                    "eps_clip": 0.18,       # 轻微调整
                    "A_coeff": 1.0,
                    "vf_coeff": 0.5,
                    "entropy_coeff": 0.018,
                    "max_grad_norm": 0.5,
                    "minibatch_size": 40,
                    "save_frequency": 50,
                    "eval_frequency": 20,
                    "lr_schedule": "cosine_warm",
                    "warmup_episodes": 20
                }
            },
            
            # 第三阶段：中等复杂度 (250轮)
            {
                "name": "Stage3_Intermediate",
                "episodes": 250,
                "transition_episodes": 25,
                "env_config": {
                    "num_usvs": 4,
                    "num_tasks": 15,          # 从18减少到15
                    "map_size": [85, 85],     # 从100减少到85
                    "usv_speed": 5.0,
                    "battery_capacity": 220.0,
                    "charge_time": 30.0,
                    "device": "cuda:0",
                    "min_travel_time": 1.5,
                    "min_travel_distance": 3.5,
                    "travel_constraint_mode": "time",  # 暂时保持time约束
                    "reward_config": {
                        "use_potential_based_reward": False,
                        "makespan_normalization": "dynamic"
                    },
                    "dynamic_masking_config": {
                        "enabled": True,
                        "max_load_ratio": 1.3
                    }
                },
                "model_config": {
                    "embedding_dim": 64,
                    "num_heads": 1,
                    "num_attention_heads": 4,
                    "num_hgnn_layers": 3,
                    "eta_neighbors": 3,
                    "mlp_hidden_dim": 96,
                    "n_hidden_actor": 3,
                    "n_hidden_critic": 4,
                    "n_latent_actor": 96,
                    "n_latent_critic": 192,
                    "dropout": 0.12,
                    "device": "cuda:0",
                    "use_batch_norm": True,
                    "initial_epsilon": 0.2,
                    "min_epsilon": 0.05,
                    "epsilon_decay": 0.998
                },
                "train_config": {
                    "lr": 2e-4,             # 更平滑的下降
                    "betas": [0.9, 0.999],
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "K_epochs": 4,
                    "eps_clip": 0.16,
                    "A_coeff": 1.0,
                    "vf_coeff": 0.45,
                    "entropy_coeff": 0.015,
                    "max_grad_norm": 0.5,
                    "minibatch_size": 48,
                    "save_frequency": 60,
                    "eval_frequency": 25,
                    "lr_schedule": "cosine_warm",
                    "warmup_episodes": 25
                }
            },
            
            # 第四阶段：目标复杂度 (300轮) - 最终目标
            {
                "name": "Stage4_Target",
                "episodes": 300,
                "transition_episodes": 25,
                "env_config": {
                    "num_usvs": 4,          # 保守的最终规模
                    "num_tasks": 12,        # 从20减少到12
                    "map_size": [80, 80],   # 从100减少到80
                    "usv_speed": 5.0,
                    "battery_capacity": 220.0,
                    "charge_time": 30.0,
                    "device": "cuda:0",
                    "min_travel_time": 1.5,
                    "min_travel_distance": 3.0,
                    "travel_constraint_mode": "both",  # 最终阶段使用both约束
                    "reward_config": {
                        "use_potential_based_reward": False,
                        "makespan_normalization": "dynamic"
                    },
                    "dynamic_masking_config": {
                        "enabled": True,
                        "max_load_ratio": 1.4
                    }
                },
                "model_config": {
                    "embedding_dim": 64,
                    "num_heads": 1,
                    "num_attention_heads": 4,
                    "num_hgnn_layers": 3,    # 适度增加网络深度
                    "eta_neighbors": 3,
                    "mlp_hidden_dim": 96,
                    "n_hidden_actor": 3,
                    "n_hidden_critic": 4,
                    "n_latent_actor": 96,
                    "n_latent_critic": 192,
                    "dropout": 0.15,
                    "device": "cuda:0",
                    "use_batch_norm": True,
                    "initial_epsilon": 0.12, # 最低的探索率
                    "min_epsilon": 0.02,
                    "epsilon_decay": 0.999
                },
                "train_config": {
                    "lr": 1.5e-4,           # 最终的学习率
                    "betas": [0.9, 0.999],
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "K_epochs": 4,
                    "eps_clip": 0.15,
                    "A_coeff": 1.0,
                    "vf_coeff": 0.4,
                    "entropy_coeff": 0.015,
                    "max_grad_norm": 0.5,
                    "minibatch_size": 48,
                    "save_frequency": 80,
                    "eval_frequency": 30,
                    "lr_schedule": "cosine_warm",
                    "warmup_episodes": 25
                }
            }
        ]
        
        self.current_stage = 0
        self.convergence_threshold = 0.03  # 3%改善率作为收敛标准，更严格
        self.stability_threshold = 0.12   # 12%方差阈值
        
    def get_current_stage_config(self):
        """获取当前阶段配置"""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # 停留在最后阶段
    
    def get_stage_config(self, stage_idx):
        """获取指定阶段配置"""
        if 0 <= stage_idx < len(self.stages):
            return self.stages[stage_idx]
        return None
    
    def should_advance_stage(self, recent_makespans, recent_rewards=None, window_size=80):
        """判断是否应该进入下一阶段 - 增强的收敛检测"""
        if len(recent_makespans) < window_size:
            return False
            
        # 检查收敛性：最近episodes的改善幅度
        quarter_size = window_size // 4
        early_makespans = recent_makespans[:quarter_size*2]
        late_makespans = recent_makespans[-quarter_size*2:]
        
        if not early_makespans or not late_makespans:
            return False
            
        early_avg = np.mean(early_makespans)
        late_avg = np.mean(late_makespans)
        
        # 性能改善率
        improvement_rate = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
        
        # 稳定性检查
        late_variance = np.std(late_makespans) / np.mean(late_makespans) if np.mean(late_makespans) > 0 else 1
        
        # 奖励趋势检查（如果提供）
        reward_stable = True
        if recent_rewards and len(recent_rewards) >= window_size:
            recent_reward_std = np.std(recent_rewards[-quarter_size*2:])
            recent_reward_mean = np.mean(recent_rewards[-quarter_size*2:])
            reward_cv = recent_reward_std / abs(recent_reward_mean) if recent_reward_mean != 0 else 1
            reward_stable = reward_cv < 0.3  # 奖励变异系数小于30%
        
        # 收敛条件：改善率足够 + 方差稳定 + 奖励稳定
        converged = (
            improvement_rate >= self.convergence_threshold and 
            late_variance < self.stability_threshold and
            reward_stable
        )
        
        print(f"[STAGE] Improvement: {improvement_rate:.3f} (>{self.convergence_threshold}), "
              f"Stability: {late_variance:.3f} (<{self.stability_threshold}), "
              f"Reward Stable: {reward_stable}, Converged: {converged}")
        
        return converged
    
    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"[STAGE] Advanced to stage {self.current_stage + 1}: "
                  f"{self.stages[self.current_stage]['name']}")
            return True
        return False
    
    def get_transition_config(self, from_stage_idx, to_stage_idx):
        """获取阶段转换配置 - 用于平滑过渡"""
        if from_stage_idx >= len(self.stages) or to_stage_idx >= len(self.stages):
            return None
            
        from_config = self.stages[from_stage_idx]
        to_config = self.stages[to_stage_idx]
        
        # 创建过渡配置：插值关键参数
        transition_config = copy.deepcopy(to_config)
        
        # 学习率平滑过渡
        from_lr = from_config['train_config']['lr']
        to_lr = to_config['train_config']['lr']
        transition_lr = (from_lr + to_lr) / 2
        transition_config['train_config']['lr'] = transition_lr
        
        # Epsilon平滑过渡
        from_eps = from_config['model_config']['initial_epsilon']
        to_eps = to_config['model_config']['initial_epsilon']
        transition_eps = (from_eps + to_eps) / 2
        transition_config['model_config']['initial_epsilon'] = transition_eps
        
        # 熵系数平滑过渡
        from_entropy = from_config['train_config']['entropy_coeff']
        to_entropy = to_config['train_config']['entropy_coeff']
        transition_entropy = (from_entropy + to_entropy) / 2
        transition_config['train_config']['entropy_coeff'] = transition_entropy
        
        print(f"[TRANSITION] Created transition config: lr={transition_lr:.2e}, "
              f"epsilon={transition_eps:.3f}, entropy={transition_entropy:.3f}")
        
        return transition_config
    
    def generate_config_file(self, stage_idx=None, output_path="current_config.json"):
        """生成配置文件"""
        if stage_idx is None:
            stage_idx = self.current_stage
            
        stage_config = self.get_stage_config(stage_idx)
        if not stage_config:
            return None
            
        # 转换为兼容格式
        config = {
            "env_paras": stage_config["env_config"],
            "model_paras": stage_config["model_config"],
            "train_paras": {
                **stage_config["train_config"],
                "max_episodes": stage_config["episodes"],
                "debug_mode": True,
                "plot_metrics_every": 25,
                "report_root": "results/progressive_reports",
                "report_episodes": 25,
                "viz": True,
                "viz_name": f"USV_Progressive_{stage_config['name']}",
                "save_root": "results/progressive_models",
                "save_only_last": False,
                "run_name": f"Progressive_{stage_config['name']}"
            }
        }
        
        # 保存配置文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[CONFIG] Generated config file: {output_path}")
        return config

class ProgressiveTrainingMonitor:
    """渐进式训练监控器 - 增强版本"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.makespan_history = []
        self.reward_history = []
        self.episode_count = 0
        self.stage_start_episode = 0
        self.stage_performance_history = []  # 记录每个阶段的性能
        
    def log_episode(self, makespan, reward):
        """记录每轮训练结果"""
        self.makespan_history.append(makespan)
        self.reward_history.append(reward)
        self.episode_count += 1
        
        # 每50轮检查一次训练健康状况
        if self.episode_count % 50 == 0:
            self._check_training_health()
            
        # 每80轮检查是否收敛
        if self.episode_count % 80 == 0:
            return self._check_stage_convergence()
        
        return False
    
    def _check_training_health(self):
        """检查训练健康状况 - 增强版本"""
        if len(self.makespan_history) < 50:
            return
            
        recent_makespans = self.makespan_history[-50:]
        recent_rewards = self.reward_history[-50:]
        
        # 检查是否陷入局部最优
        makespan_std = np.std(recent_makespans)
        makespan_mean = np.mean(recent_makespans)
        
        if makespan_std / makespan_mean < 0.015:
            print("[WARNING] Possible local optimum detected - very low variance")
            
        # 检查奖励趋势
        if all(r < -20 for r in recent_rewards[-15:]):
            print("[WARNING] Consistently very negative rewards")
            
        # 检查makespan趋势
        if len(recent_makespans) >= 30:
            early_30 = recent_makespans[:15]
            late_30 = recent_makespans[-15:]
            if np.mean(late_30) > np.mean(early_30) * 1.1:
                print("[WARNING] Makespan degrading - possible overfitting")
    
    def _check_stage_convergence(self):
        """检查当前阶段是否收敛"""
        stage_episodes = self.episode_count - self.stage_start_episode
        
        # 需要足够的数据和最小训练时间
        if stage_episodes < 80:
            return False
            
        window_size = min(80, stage_episodes)
        recent_makespans = self.makespan_history[-window_size:]
        recent_rewards = self.reward_history[-window_size:]
        
        if self.config.should_advance_stage(recent_makespans, recent_rewards):
            # 记录当前阶段性能
            stage_summary = {
                'stage': self.config.current_stage,
                'episodes': stage_episodes,
                'best_makespan': min(recent_makespans),
                'final_makespan': recent_makespans[-1],
                'avg_makespan': np.mean(recent_makespans),
                'avg_reward': np.mean(recent_rewards),
                'improvement': (recent_makespans[0] - recent_makespans[-1]) / recent_makespans[0] if recent_makespans[0] > 0 else 0
            }
            self.stage_performance_history.append(stage_summary)
            
            if self.config.advance_stage():
                print(f"[STAGE] Stage advancement at episode {self.episode_count}")
                print(f"[STAGE] Stage {stage_summary['stage']} summary: "
                      f"best={stage_summary['best_makespan']:.2f}, "
                      f"improvement={stage_summary['improvement']:.1%}")
                self.stage_start_episode = self.episode_count
                return True
        return False
    
    def get_stage_summary(self):
        """获取当前阶段总结"""
        stage_episodes = self.episode_count - self.stage_start_episode
        if stage_episodes == 0:
            return None
            
        stage_makespans = self.makespan_history[-stage_episodes:]
        stage_rewards = self.reward_history[-stage_episodes:]
        
        return {
            "episodes": stage_episodes,
            "best_makespan": min(stage_makespans),
            "final_makespan": stage_makespans[-1],
            "avg_reward": np.mean(stage_rewards),
            "improvement_rate": (stage_makespans[0] - stage_makespans[-1]) / stage_makespans[0] if stage_makespans[0] > 0 else 0,
            "stability": np.std(stage_makespans[-20:]) / np.mean(stage_makespans[-20:]) if len(stage_makespans) >= 20 else 1.0
        }

def create_progressive_configs():
    """创建所有阶段的配置文件"""
    config_manager = ProgressiveTrainingConfig()
    
    for i in range(len(config_manager.stages)):
        output_path = f"stage_{i+1}_config.json"
        config_manager.generate_config_file(i, output_path)
        print(f"Created: {output_path}")

if __name__ == "__main__":
    # 生成所有阶段的配置文件
    create_progressive_configs()
    print("All progressive training configs generated!")