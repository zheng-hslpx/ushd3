import os
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端 - 必须在其他matplotlib导入之前
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式

import json, argparse, csv, re
from pathlib import Path
from datetime import datetime
from typing import Dict  
import numpy as np
import torch
from tqdm import tqdm

from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
from usv_agent.ppo_policy import PPOAgent, PPO, Memory
from utils.vis_manager import VisualizationManager

# 设置全局默认数据类型
torch.set_default_dtype(torch.float32)

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _device_from_cfg(model_cfg):
    want = str(model_cfg.get('device', 'auto')).lower()
    if want == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, falling back to CPU")
        return torch.device('cpu')
    return torch.device(want)

def _group_name(cfg):
    return cfg['train_paras'].get('run_name', 'default_run')

def _next_run_index(dir_path: Path) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in dir_path.iterdir() if p.is_dir() and re.fullmatch(r'\d{2,}', p.name)]
    return f"{(max(nums) + 1) if nums else 1:02d}"

def _short_ts():
    return datetime.now().strftime("%m%d_%H%M%S")

def compute_lookahead_features(state: Dict, device: torch.device) -> torch.Tensor:
    usv_pos = torch.from_numpy(state['usv_features'][:, :2]).float().to(device)
    task_pos = torch.from_numpy(state['task_features'][:, :2]).float().to(device)
    task_status = torch.from_numpy(state['task_features'][:, 3]).float().to(device)
    map_size = torch.tensor(state['map_size'], dtype=torch.float32, device=device)
    
    U, T = usv_pos.shape[0], task_pos.shape[0]
    dist_ut = torch.cdist(usv_pos, task_pos)
    
    unassigned_mask = task_status > 0
    unassigned_pos = task_pos[unassigned_mask]
    
    feat_task_proximity = torch.zeros(U, T, device=device)
    if unassigned_pos.shape[0] > 1:
        dist_tt_unassigned = torch.cdist(unassigned_pos, unassigned_pos)
        dist_tt_unassigned.fill_diagonal_(float('inf'))
        min_dist_tt, _ = torch.min(dist_tt_unassigned, dim=1)
        temp_prox = torch.zeros(T, device=device)
        temp_prox[unassigned_mask] = min_dist_tt
        feat_task_proximity = temp_prox.unsqueeze(0).expand(U, -1).clone()

    feat_usv_opportunity = torch.zeros(U, T, device=device)
    if unassigned_pos.shape[0] > 0:
        dist_usv_to_unassigned = torch.cdist(usv_pos, unassigned_pos)
        min_dist_usv, _ = torch.min(dist_usv_to_unassigned, dim=1)
        feat_usv_opportunity = min_dist_usv.unsqueeze(1).expand(-1, T).clone()

    map_diag = torch.norm(map_size)
    if map_diag > 0:
        dist_ut /= map_diag
        feat_task_proximity /= map_diag
        feat_usv_opportunity /= map_diag
    
    usv_task_edges = torch.stack([dist_ut, feat_task_proximity, feat_usv_opportunity], dim=-1)
    return usv_task_edges

def evaluate(env: USVEnv, agent: PPOAgent, episodes: int = 5, deterministic: bool = True):
    """*** 修复：增强的评估函数，增加 makespan异常检测 ***"""
    ms_list, rews_list, balance_list = [], [], []
    invalid_episodes = 0  # *** 新增：记录异常episode数量 ***
    
    # *** 新增：临时禁用调试模式以减少输出 ***
    original_debug = env.debug_mode
    env.set_debug_mode(False)
    
    for ep in range(episodes):
        state = env.reset()
        state['map_size'] = env.map_size
        done, total_r = False, 0.0
        step_count = 0
        
        while not done:
            with torch.no_grad():
                usv_task_edges = compute_lookahead_features(state, agent.device)
            a, _, _ = agent.get_action(state, usv_task_edges, deterministic=deterministic)
            state, r, done, info = env.step(a)
            state['map_size'] = env.map_size
            total_r += float(r)
            step_count += 1
            
            # *** 新增：防止无限循环 ***
            if step_count > env.num_tasks * 2:
                print(f"[WARN] Episode {ep} exceeded maximum steps, forcing termination")
                break
        
        makespan = info.get('makespan', 0.0)
        
        # *** 新增：makespan异常检测 ***
        min_task_time = min(t.processing_time for t in env.tasks) if env.tasks else 0
        if makespan <= 0 or makespan < min_task_time:
            print(f"[WARN] Invalid makespan in eval episode {ep}: {makespan}")
            invalid_episodes += 1
            # 使用一个合理的默认值，而不是跳过
            makespan = max(min_task_time, 1.0)
        
        ms_list.append(makespan)
        rews_list.append(total_r)
        balance_metrics = env.get_balance_metrics()
        balance_list.append(balance_metrics['jains_index'])
    
    # *** 恢复调试模式 ***
    env.set_debug_mode(original_debug)
    
    # *** 新增：报告异常情况 ***
    if invalid_episodes > 0:
        print(f"[WARN] {invalid_episodes}/{episodes} evaluation episodes had invalid makespan")
    
    return {
        'makespan': float(np.mean(ms_list)),
        'reward': float(np.mean(rews_list)),
        'jains_index': float(np.mean(balance_list)),
        'makespan_std': float(np.std(ms_list)),
        'invalid_episodes': invalid_episodes  # *** 新增：返回异常episode数量 ***
    }

class MetricsLogger:
    def __init__(self, run_dir: Path, prefix: str, plot_every: int = 40):
        self.run_dir = run_dir; self.prefix = prefix
        self.csv_path  = run_dir / f"{prefix}metrics.csv"
        self.plot_path = run_dir / f"{prefix}metrics.png"
        self.plot_every = int(plot_every)
        self.headers = ["episode","train_makespan","train_reward","actor_loss","critic_loss","entropy_loss",
                        "eval_makespan","eval_reward","jains_index","task_load_variance","current_lr"]
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)
        self.data = {h: [] for h in self.headers}

    def log(self, ep, metrics: dict):
        for h in self.headers: self.data[h].append(metrics.get(h, np.nan))
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([metrics.get(h) for h in self.headers])
        if ep % self.plot_every == 0 or ep == 1: self.plot()

    def plot(self):
        if not self.data["episode"]: return
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        plot_map = {
            'Makespan': ('train_makespan', 'eval_makespan'), 
            'Reward': ('train_reward', 'eval_reward'),
            "Jain's Fairness Index": ('jains_index',), 
            'Actor Loss': ('actor_loss',),
            'Critic Loss': ('critic_loss',), 
            'Entropy Loss': ('entropy_loss',),
            'Task Load Variance': ('task_load_variance',),
            'Learning Rate': ('current_lr',)
        }
        
        for i, (title, keys) in enumerate(plot_map.items()):
            ax = axes[i]; ax.set_title(title); ax.set_xlabel("Episode")
            ax.plot(self.data['episode'], self.data[keys[0]], label="train")
            if len(keys) > 1 and not all(np.isnan(self.data[keys[1]])):
                ax.plot(self.data['episode'], self.data[keys[1]], linestyle="--", label="eval")
            if 'makespan' in keys[0] or 'reward' in keys[0]: ax.legend()
            if 'Fairness' in title: ax.set_ylim(0, 1.05)
            if 'Learning Rate' in title: ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 确保关闭图形以释放内存

def validate_model_architecture(hgnn, agent, env, device):
    """*** 新增：模型架构验证函数 ***"""
    print("\n=== Model Architecture Validation ===")
    
    # 获取样本数据
    sample_state = env.reset()
    sample_state['map_size'] = env.map_size
    
    # 转换为torch tensors
    usv_features = torch.from_numpy(sample_state['usv_features']).float().unsqueeze(0).to(device)
    task_features = torch.from_numpy(sample_state['task_features']).float().unsqueeze(0).to(device)
    usv_task_edges = compute_lookahead_features(sample_state, device).unsqueeze(0)
    
    print(f"Input shapes:")
    print(f"  USV features: {usv_features.shape}")
    print(f"  Task features: {task_features.shape}")
    print(f"  USV-Task edges: {usv_task_edges.shape}")
    
    # 测试HGNN前向传播
    try:
        with torch.no_grad():
            usv_emb, task_emb, graph_emb = hgnn(usv_features, task_features, usv_task_edges)
        print(f"✅ HGNN forward pass successful!")
        print(f"  USV embeddings: {usv_emb.shape}")
        print(f"  Task embeddings: {task_emb.shape}")
        print(f"  Graph embeddings: {graph_emb.shape}")
    except Exception as e:
        print(f"❌ HGNN forward pass failed: {e}")
        raise e
    
    # 测试Agent前向传播
    try:
        with torch.no_grad():
            action, logp, value = agent.get_action(sample_state, usv_task_edges.squeeze(0), deterministic=False)
        print(f"✅ Agent forward pass successful!")
        print(f"  Action: {action}")
        print(f"  Log probability: {logp}")
        print(f"  Value: {value}")
    except Exception as e:
        print(f"❌ Agent forward pass failed: {e}")
        raise e
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in hgnn.parameters())
    trainable_params = sum(p.numel() for p in hgnn.parameters() if p.requires_grad)
    agent_params = sum(p.numel() for p in agent.parameters())
    
    print(f"📊 Model Statistics:")
    print(f"  HGNN total parameters: {total_params:,}")
    print(f"  HGNN trainable parameters: {trainable_params:,}")
    print(f"  Agent total parameters: {agent_params:,}")
    print(f"  Memory usage: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB" if device.type == 'cuda' else "  Memory usage: CPU mode")
    
    print("=" * 50 + "\n")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "improved_config.json"), 
                       help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    env_cfg, model_cfg, train_cfg = cfg['env_paras'], cfg['model_paras'], cfg['train_paras']
    device = _device_from_cfg(model_cfg)
    
    print(f"[INFO] Using device: {device}, Torch version: {torch.__version__}")
    print(f"[INFO] Model config: HGNN layers={model_cfg['num_hgnn_layers']}, dropout={model_cfg['dropout']}")
    print(f"[INFO] Train config: lr={train_cfg['lr']}, entropy_coeff={train_cfg['entropy_coeff']}")

    save_root = Path(train_cfg.get('save_root', "results/saved_models"))
    group_dir = save_root / _group_name(cfg)
    run_dir = group_dir / _next_run_index(group_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_short_ts()}_"
    print(f"[INFO] Artifacts will be saved in: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    env = USVEnv(env_cfg)
    
    # *** 新增：控制调试模式 ***
    debug_training = train_cfg.get('debug_mode', False)
    env.set_debug_mode(debug_training)
    if debug_training:
        print("[INFO] Debug mode enabled - detailed logs will be shown")
    
    ##############################################
    # 环境状态结构检查
    ##############################################
    print("\n=== Environment State Structure Check ===")
    sample_state = env.reset()
    print("State keys:", sample_state.keys())
    print("usv_features type/shape:", 
          type(sample_state['usv_features']), 
          sample_state['usv_features'].shape)
    print("task_features type/shape:", 
          type(sample_state['task_features']), 
          sample_state['task_features'].shape)
    print("action_mask type/shape:", 
          type(sample_state['action_mask']), 
          sample_state['action_mask'].shape)
    
    # 额外检查数据范围
    print("\nData Range Check:")
    print("usv_features min/max:", 
          np.min(sample_state['usv_features']), 
          np.max(sample_state['usv_features']))
    print("task_features min/max:", 
          np.min(sample_state['task_features']), 
          np.max(sample_state['task_features']))
    print("action_mask sum:", np.sum(sample_state['action_mask']))
    
    # *** 新增：makespan初始化检查 ***
    print(f"Initial makespan: {env.makespan}")
    print("=======================================\n")
    ##############################################
    # 环境检查代码结束
    ##############################################
    
    # *** 核心修复：创建模型并验证 ***
    try:
        hgnn = HeterogeneousGNN(model_cfg).to(device)
        agent = PPOAgent(hgnn, model_cfg).to(device)
        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise e
    
    # *** 新增：模型架构验证 ***
    validate_model_architecture(hgnn, agent, env, device)
    
    ppo = PPO(agent, train_cfg)
    memory = Memory()
    
    viz = None
    if train_cfg.get('viz', False):
        try:
            viz = VisualizationManager(viz_name=train_cfg['viz_name'], enabled=True)
        except Exception as e:
            print(f"Warning: Failed to initialize Visdom. Live plotting disabled. Error: {e}")
    
    logger = MetricsLogger(run_dir, prefix, plot_every=train_cfg.get('plot_metrics_every', 50))
    best_eval_ms = float("inf")
    best_eval_reward = float("-inf")

    training_stats = {
        'episodes_completed': 0,
        'best_makespan': float('inf'),
        'best_reward': float('-inf'),
        'early_stop_triggered': False,
        'invalid_makespan_episodes': 0  # *** 新增：记录异常episode ***
    }

    print(f"\n🚀 Starting training for {train_cfg['max_episodes']} episodes...")
    print("=" * 60)

    for ep in tqdm(range(1, train_cfg['max_episodes'] + 1), desc="Training Progress"):
        state = env.reset()
        state['map_size'] = env_cfg['map_size']
        done, ep_reward, last_value = False, 0.0, 0.0
        
        # *** 新增：episode级别的异常检测 ***
        step_count = 0
        max_steps = env.num_tasks * 3  # 设置最大步数限制

        while not done and step_count < max_steps:
            with torch.no_grad():
                usv_task_edges = compute_lookahead_features(state, agent.device)

            action, logp, value = agent.get_action(state, usv_task_edges, deterministic=False)
            next_state, r, done, info = env.step(action)
            ep_reward += r
            step_count += 1
            
            # 修改后的调用：
            memory.add(
                state,
                action,
                logp,
                float(r),  # 确保是Python float
                bool(done),  # 确保是Python bool
                float(value),  # 确保是Python float
                usv_task_edges.cpu()  # 保持为PyTorch tensor
            )
            state = next_state
            state['map_size'] = env_cfg['map_size']
            last_value = value

        # *** 新增：检查episode是否正常完成 ***
        makespan = info.get('makespan', 0.0)
        if makespan <= 0 or step_count >= max_steps:
            training_stats['invalid_makespan_episodes'] += 1
            if step_count >= max_steps:
                print(f"[WARN] Episode {ep} exceeded maximum steps ({max_steps})")
            if makespan <= 0:
                print(f"[WARN] Episode {ep} ended with invalid makespan: {makespan}")

        memory.values.append(last_value)
        losses = ppo.update(memory)
        memory.clear_memory()

        log_metrics = {
            'episode': ep, 
            'train_makespan': makespan, 
            'train_reward': ep_reward
        }
        log_metrics.update(losses)
        log_metrics.update(env.get_balance_metrics())

        training_stats['episodes_completed'] = ep
        if makespan > 0 and makespan < training_stats['best_makespan']:  # *** 修复：只有合理的makespan才更新最佳记录 ***
            training_stats['best_makespan'] = makespan
        if ep_reward > training_stats['best_reward']:
            training_stats['best_reward'] = ep_reward

        if viz and viz.enabled:
            viz.update_plots(ep, log_metrics)

        if ep % train_cfg['eval_frequency'] == 0:
            eval_results = evaluate(env, agent, episodes=5)
            log_metrics.update({
                'eval_makespan': eval_results['makespan'],
                'eval_reward': eval_results['reward']
            })
            
            # *** 修复：增强评估报告 ***
            print(f"\n[Eval Ep {ep:4d}] Makespan: {eval_results['makespan']:.2f}±{eval_results['makespan_std']:.2f}, "
                  f"Reward: {eval_results['reward']:.2f}, Jain's: {eval_results['jains_index']:.3f}")
            
            if eval_results.get('invalid_episodes', 0) > 0:
                print(f"  ⚠️  {eval_results['invalid_episodes']}/5 eval episodes had issues")

            # *** 修复：只有合理的结果才更新最佳模型 ***
            if eval_results['makespan'] > 0 and eval_results['makespan'] < best_eval_ms:
                best_eval_ms = eval_results['makespan']
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_makespan_model.pt")
                print(f"  🎯 New best makespan model saved: {best_eval_ms:.2f}")
            
            if eval_results['reward'] > best_eval_reward:
                best_eval_reward = eval_results['reward']
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_reward_model.pt")
                print(f"  🎯 New best reward model saved: {best_eval_reward:.2f}")
            
            if ppo.check_early_stop(eval_results['reward']):
                print(f"\nℹ️ Early stopping triggered at episode {ep}")
                training_stats['early_stop_triggered'] = True
                break
        
        logger.log(ep, log_metrics)

        if ep % train_cfg['save_frequency'] == 0:
            torch.save(agent.state_dict(), run_dir / f"{prefix}ep{ep:04d}.pt")
            torch.save(agent.state_dict(), run_dir / f"{prefix}latest.pt")
    
    print("\n" + "=" * 60)
    print("🎉 Training finished!")
    print(f"📊 Episodes completed: {training_stats['episodes_completed']}")
    print(f"📈 Best makespan achieved: {training_stats['best_makespan']:.2f}")
    print(f"🏆 Best reward achieved: {training_stats['best_reward']:.2f}")
    print(f"⚠️  Episodes with issues: {training_stats['invalid_makespan_episodes']}")
    if training_stats['early_stop_triggered']:
        print("ℹ️ Training stopped early due to convergence")
    
    final_model_path = run_dir / f"{prefix}final_model.pt"
    torch.save(agent.state_dict(), final_model_path)
    
    with open(run_dir / f"{prefix}training_stats.json", "w") as f:
        # *** 修复：确保所有数据都是JSON可序列化的 ***
        serializable_stats = {k: float(v) if isinstance(v, np.number) else v 
                             for k, v in training_stats.items()}
        json.dump(serializable_stats, f, indent=2)
    
    try:
        viz_final = VisualizationManager(viz_name="final_gantt", enabled=False)
        
        best_model_path = run_dir / f"{prefix}best_makespan_model.pt"
        if best_model_path.exists():
            print(f"[INFO] Loading best makespan model for final Gantt chart")
            agent.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # *** 修复：最终评估时可用调试模式 ***
        env.set_debug_mode(True)
        evaluate(env, agent, episodes=1, deterministic=True)
        gantt_path = run_dir / f"{prefix}gantt_final.png"
        summary = viz_final.generate_gantt_chart(env, save_path=str(gantt_path))
        
        table_path = run_dir / f"{prefix}gantt_table.png"
        viz_final.save_summary_table(summary, env.makespan, str(table_path))
        print(f"✅ Final Gantt chart and summary table saved.")
        
        final_balance = env.get_balance_metrics()
        print(f"📊 Final Jain's Index: {final_balance['jains_index']:.4f}")
        print(f"📊 Final Task Load Variance: {final_balance['task_load_variance']:.4f}")
        print(f"📊 Final Makespan: {env.makespan:.2f}")
        
    except Exception as e:
        print(f"[WARN] Failed to generate final report: {e}")

    print(f"✅ All artifacts saved in: {run_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()