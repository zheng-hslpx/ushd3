import os
# 8.21修改_序号1：设置非交互式后端，确保无显示环境也能作图
# 说明：在服务器/无GUI环境训练时，使用Agg后端避免matplotlib报错；plt.ioff关闭交互模式释放内存。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import json, argparse, csv, re
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import torch
from tqdm import tqdm

# 8.21修改_序号2：补充标准库导入（预留）
# 说明：当前版本未直接使用，但保留以便后续在训练/评估阶段做安全副本。
from copy import deepcopy
from typing import Optional

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
    """
    说明：基于当前状态构造USV-Task边特征，包含归一化距离、任务邻近度、USV机会度。
    """
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

# 新的一周修改+序号2：增强环境验证函数
def validate_environment_integrity(env: USVEnv, agent: PPOAgent, device: torch.device, 
                                   validation_episodes: int = 3, save_path: Optional[str] = None):
    """
    通过多次运行和甘特图可视化来验证环境的正确性
    
    Args:
        env: USV环境实例
        agent: PPO智能体
        device: 计算设备
        validation_episodes: 验证episode数量
        save_path: 甘特图保存路径
    
    Returns:
        validation_results: 验证结果字典
    """
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT INTEGRITY VALIDATION")
    print("="*60)
    
    validation_results = {
        'episodes_validated': 0,
        'successful_episodes': 0,
        'makespan_consistency': [],
        'assignment_errors': 0,
        'time_consistency_errors': 0,
        'gantt_charts_generated': 0
    }
    
    original_debug_mode = env.debug_mode
    env.set_debug_mode(True)  # 启用详细调试信息
    
    for ep in range(validation_episodes):
        print(f"\n🧪 Validation Episode {ep + 1}/{validation_episodes}")
        print("-" * 40)
        
        # 重置环境
        state = env.reset()
        state['map_size'] = env.map_size
        
        # 验证初始状态
        if not env.validate_environment_state():
            print(f"❌ Episode {ep + 1}: Initial state validation failed!")
            continue
            
        episode_steps = []
        done, step_count = False, 0
        max_steps = env.num_tasks * 2
        
        # 运行一个完整的episode
        while not done and step_count < max_steps:
            # 计算边特征
            with torch.no_grad():
                usv_task_edges = compute_lookahead_features(state, device)
            
            # 获取动作（使用确定性策略以便复现）
            action, _, _ = agent.get_action(state, usv_task_edges, deterministic=True)
            
            # 记录步骤信息
            step_info = {
                'step': step_count,
                'action': action,
                'usv_idx': action // env.num_tasks,
                'task_idx': action % env.num_tasks,
                'prev_makespan': env.makespan
            }
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            step_info.update({
                'new_makespan': env.makespan,
                'reward': reward,
                'done': done
            })
            episode_steps.append(step_info)
            
            # 中间状态验证
            if not env.validate_environment_state():
                print(f"❌ Episode {ep + 1}, Step {step_count}: State validation failed!")
                validation_results['assignment_errors'] += 1
                break
                
            state = next_state
            state['map_size'] = env.map_size
            step_count += 1
        
        # Episode完成后的验证
        final_makespan = info.get('makespan', 0.0)
        calculated_makespan = max(u.available_time for u in env.usvs) if env.usvs else 0.0
        
        makespan_diff = abs(final_makespan - calculated_makespan)
        validation_results['makespan_consistency'].append(makespan_diff)
        
        if makespan_diff > 1e-6:
            print(f"⚠️  Episode {ep + 1}: Makespan inconsistency detected!")
            print(f"    Reported: {final_makespan:.6f}, Calculated: {calculated_makespan:.6f}")
            validation_results['time_consistency_errors'] += 1
        
        # 生成甘特图进行可视化验证
        if save_path and (ep == 0 or makespan_diff > 1e-6):  # 第一个episode或有问题的episode
            try:
                viz_manager = VisualizationManager(viz_name="validation_gantt", enabled=False)
                gantt_path = f"{save_path}_ep{ep + 1}_gantt.png"
                
                gantt_summary = viz_manager.generate_gantt_chart(env, save_path=gantt_path)
                validation_results['gantt_charts_generated'] += 1
                
                print(f"📊 Gantt chart saved: {gantt_path}")
                print(f"    Final makespan: {gantt_summary['makespan']:.3f}")
                print(f"    Task assignments: {gantt_summary['total_tasks']} tasks assigned")
                print(f"    Load balance (Jain's): {gantt_summary['jains_index']:.4f}")
                
                # 保存详细的验证报告
                validation_report = {
                    'episode': ep + 1,
                    'steps_taken': step_count,
                    'final_makespan': final_makespan,
                    'calculated_makespan': calculated_makespan,
                    'makespan_difference': makespan_diff,
                    'gantt_summary': gantt_summary,
                    'episode_steps': episode_steps
                }
                
                report_path = f"{save_path}_ep{ep + 1}_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_report, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"⚠️  Failed to generate Gantt chart: {str(e)}")
        
        if done and step_count < max_steps:
            validation_results['successful_episodes'] += 1
            print(f"✅ Episode {ep + 1}: Successfully completed in {step_count} steps")
        else:
            print(f"❌ Episode {ep + 1}: Failed to complete properly")
        
        validation_results['episodes_validated'] += 1
    
    # 恢复原始调试模式
    env.set_debug_mode(original_debug_mode)
    
    # 输出验证总结
    print("\n" + "="*60)
    print("🏁 VALIDATION SUMMARY")
    print("="*60)
    print(f"Episodes validated: {validation_results['episodes_validated']}")
    print(f"Successful episodes: {validation_results['successful_episodes']}")
    print(f"Success rate: {validation_results['successful_episodes']/validation_results['episodes_validated']*100:.1f}%")
    print(f"Assignment errors: {validation_results['assignment_errors']}")
    print(f"Time consistency errors: {validation_results['time_consistency_errors']}")
    print(f"Gantt charts generated: {validation_results['gantt_charts_generated']}")
    
    if validation_results['makespan_consistency']:
        avg_makespan_error = np.mean(validation_results['makespan_consistency'])
        max_makespan_error = np.max(validation_results['makespan_consistency'])
        print(f"Makespan consistency - Avg error: {avg_makespan_error:.2e}, Max error: {max_makespan_error:.2e}")
    
    # 判断验证是否通过
    validation_passed = (
        validation_results['successful_episodes'] == validation_results['episodes_validated'] and
        validation_results['assignment_errors'] == 0 and
        validation_results['time_consistency_errors'] == 0
    )
    
    if validation_passed:
        print("🎉 ENVIRONMENT VALIDATION PASSED!")
    else:
        print("❌ ENVIRONMENT VALIDATION FAILED!")
        print("   Please check the generated reports and Gantt charts for detailed analysis.")
    
    print("="*60 + "\n")
    
    return validation_results

# 8.21修改_序号10：增强的评估逻辑（多点防护 + 统计信息）
# 说明：加入最大步数防护、makespan异常兜底、临时关闭env调试输出，返回标准差/异常计数，便于稳定性评估。
def evaluate(env: USVEnv, agent: PPOAgent, episodes: int = 5, deterministic: bool = True):
    ms_list, rews_list, balance_list = [], [], []
    invalid_episodes = 0  # 新增：记录异常episode数量

    # 临时禁用调试模式以减少评估输出噪声
    original_debug = env.debug_mode
    env.set_debug_mode(False)

    for ep in range(episodes):
        state = env.reset()
        state['map_size'] = env.map_size
        done, total_r = False, 0.0
        step_count = 0
        max_steps = env.num_tasks * 2  # 防无限循环

        while not done:
            with torch.no_grad():
                usv_task_edges = compute_lookahead_features(state, agent.device)
            a, _, _ = agent.get_action(state, usv_task_edges, deterministic=deterministic)
            state, r, done, info = env.step(a)
            state['map_size'] = env.map_size
            total_r += float(r)
            step_count += 1
            if step_count > max_steps:
                print(f"[WARN] Eval episode {ep} exceeded maximum steps, force stop")
                break

        makespan = info.get('makespan', 0.0)

        # makespan异常检测与兜底
        min_task_time = min(t.processing_time for t in env.tasks) if env.tasks else 0
        if makespan <= 0 or makespan < min_task_time:
            print(f"[WARN] Invalid makespan in eval episode {ep}: {makespan}")
            invalid_episodes += 1
            makespan = max(min_task_time, 1.0)

        ms_list.append(makespan)
        rews_list.append(total_r)
        balance_metrics = env.get_balance_metrics()
        balance_list.append(balance_metrics['jains_index'])

    # 恢复调试模式
    env.set_debug_mode(original_debug)

    if invalid_episodes > 0:
        print(f"[WARN] {invalid_episodes}/{episodes} evaluation episodes had invalid makespan")

    return {
        'makespan': float(np.mean(ms_list)),
        'reward': float(np.mean(rews_list)),
        'jains_index': float(np.mean(balance_list)),
        'makespan_std': float(np.std(ms_list)),
        'invalid_episodes': invalid_episodes
    }


class MetricsLogger:
    # 8.21修改_序号13：支持"尾段退火起点"可视化（tail_start_ep）
    # 说明：多接收一个可选参数，在所有子图上画竖虚线和淡色尾段区间，便于定位退火前后指标变化。
    def __init__(self, run_dir: Path, prefix: str, plot_every: int = 40, tail_start_ep: Optional[int] = None):
        self.run_dir = run_dir; self.prefix = prefix
        self.csv_path  = run_dir / f"{prefix}metrics.csv" 
        self.plot_path = run_dir / f"{prefix}metrics.png"
        self.plot_every = int(plot_every)
        self.tail_start_ep = tail_start_ep  # 新增：退火起点（episode）

        # 8.21修改_序号3：日志列新增 eval_makespan_ema（用于画更平的评估曲线）
        # 说明：保留原有列，额外记录"评估均值再经EMA平滑"的makespan，帮助判读长期趋势；不会影响训练与保存逻辑。
        self.headers = ["episode","train_makespan","train_reward","actor_loss","critic_loss","entropy_loss",
                        "eval_makespan","eval_makespan_ema","eval_reward","jains_index","task_load_variance","current_lr"]

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

            # 8.21修改_序号4：Makespan子图增加第三条EMA平滑评估曲线（仅展示，不影响训练）
            # 说明：eval(EMA)来自对评估均值做EMA(默认alpha=0.1)，视觉上更稳定，便于观察真实趋势。
            if title == 'Makespan':
                ax.plot(self.data['episode'], self.data['train_makespan'], label="train")
                if not all(np.isnan(self.data['eval_makespan'])):
                    ax.plot(self.data['episode'], self.data['eval_makespan'], linestyle="--", label="eval")
                if 'eval_makespan_ema' in self.data and not all(np.isnan(self.data['eval_makespan_ema'])):
                    ax.plot(self.data['episode'], self.data['eval_makespan_ema'], linestyle=":", label="eval(EMA)")
                ax.legend()
            else:
                ax.plot(self.data['episode'], self.data[keys[0]], label="train")
                if len(keys) > 1 and not all(np.isnan(self.data[keys[1]])):
                    ax.plot(self.data['episode'], self.data[keys[1]], linestyle="--", label="eval")
                if 'makespan' in keys[0] or 'reward' in keys[0]: ax.legend()

            # 8.21修改_序号15：在子图上标记"尾段退火起点"及尾段区间
            # 说明：虚线表示退火起点；尾段区间用淡色遮罩，帮助对比退火前后曲线形态变化。
            if self.tail_start_ep is not None and len(self.data['episode']) > 0:
                ax.axvline(self.tail_start_ep, linestyle='--', alpha=0.6)
                last_ep = self.data['episode'][-1]
                if last_ep >= self.tail_start_ep:
                    ax.axvspan(self.tail_start_ep, last_ep, alpha=0.06)

            if 'Fairness' in title: ax.set_ylim(0, 1.05)
            if 'Learning Rate' in title: ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 确保关闭图形以释放内存


# 8.21修改_序号11：环境状态结构快速检查（便于排查状态张量/掩码问题）
# 说明：只在启动时打印一次关键shape与数据范围，帮助定位潜在异常；不影响训练流程。
def _env_state_structure_check(env: USVEnv):
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

    # makespan初始化检查
    print(f"Initial makespan: {env.makespan}")
    print("=======================================\n")


# 8.21修改_序号12：新增"模型架构验证"函数
# 说明：在正式训练前做一次HGNN与Agent的前向/尺寸检查，便于快速定位维度/设备问题。
def validate_model_architecture(hgnn, agent, env, device):
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


# 8.21修改_序号5：新增"策略EMA"类（仅用于评估与保存best，不改变训练参数）
# 说明：经典的Polyak/EMA稳定器；用滑动平均权重进行评估/选best，可显著降低策略对场景扰动的敏感度。
class ParamEMA:
    def __init__(self, model, decay: float = 0.995):
        self.decay = float(decay)
        self.ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        cur = model.state_dict()
        for k, v in self.ema_state.items():
            v.mul_(self.decay).add_(cur[k].detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        """把EMA权重临时load进来；返回原始权重用于恢复"""
        orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_state, strict=True)
        return orig

    @torch.no_grad()
    def swap_out(self, model, orig_state):
        model.load_state_dict(orig_state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    # 8.21修改_序号6：默认配置路径保持为improved_config.json（与你提供一致，含新增键）
    # 说明：确保新加的键（n_eval_episodes/use_ema_eval/ema_decay/eval_ema_alpha/plot_metrics_every）能被读取。
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "improved_config.json"),
                        help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, model_cfg, train_cfg = cfg['env_paras'], cfg['model_paras'], cfg['train_paras']
    device = _device_from_cfg(model_cfg)

    print(f"[INFO] Using device: {device}, Torch version: {torch.__version__}")
    print(f"[INFO] Model config: HGNN layers={model_cfg.get('num_hgnn_layers')}, dropout={model_cfg.get('dropout')}")
    print(f"[INFO] Train config: lr={train_cfg.get('lr')}, entropy_coeff={train_cfg.get('entropy_coeff')}")

    save_root = Path(train_cfg.get('save_root', "results/saved_models"))
    group_dir = save_root / _group_name(cfg)
    run_dir = group_dir / _next_run_index(group_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_short_ts()}_"
    print(f"[INFO] Artifacts will be saved in: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    env = USVEnv(env_cfg)

    # 控制调试模式
    debug_training = train_cfg.get('debug_mode', False)
    env.set_debug_mode(debug_training)
    if debug_training:
        print("[INFO] Debug mode enabled - detailed logs will be shown")

    # 8.21修改_序号11（调用）：启动时进行一次环境状态结构检查（仅打印，不影响训练）
    _env_state_structure_check(env)

    # 创建模型并验证
    try:
        hgnn = HeterogeneousGNN(model_cfg).to(device)
        agent = PPOAgent(hgnn, model_cfg).to(device)
        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise e

    # 8.21修改_序号12（调用）：训练前做一次模型架构验证，提前暴露潜在shape/设备问题
    validate_model_architecture(hgnn, agent, env, device)

    # 新的一周修改+序号2：进行环境完整性验证
    # 说明：在训练开始前通过运行多个验证episode并生成甘特图来检测环境的正确性
    validation_save_path = str(run_dir / f"{prefix}validation")
    env_validation_results = validate_environment_integrity(
        env, agent, device, 
        validation_episodes=train_cfg.get('env_validation_episodes', 3),
        save_path=validation_save_path
    )
    
    # 如果环境验证失败，询问是否继续
    if env_validation_results['successful_episodes'] < env_validation_results['episodes_validated']:
        print("⚠️  Environment validation detected issues. Check the generated reports.")
        print("    Training will continue, but results may be unreliable.")
        
        # 保存验证结果
        with open(run_dir / f"{prefix}env_validation_results.json", 'w') as f:
            json.dump(env_validation_results, f, indent=2)

    ppo = PPO(agent, train_cfg)
    memory = Memory()

    viz = None
    if train_cfg.get('viz', False):
        try:
            viz = VisualizationManager(viz_name=train_cfg['viz_name'], enabled=True)
        except Exception as e:
            print(f"Warning: Failed to initialize Visdom. Live plotting disabled. Error: {e}")

    # 8.21修改_序号13（main侧）：计算"尾段退火起点"episode（优先读取tail_start_episode，否则按比例计算）
    # 说明：与ppo_policy.py中的退火逻辑对齐，便于在图上做可视化标记。
    max_eps = int(train_cfg.get('max_episodes', 1000))
    tail_start_ep = int(train_cfg.get('tail_start_episode',
                        int(max_eps * float(train_cfg.get('tail_start_frac', 0.75)))))

    # 8.21修改_序号14：将尾段起点传入 MetricsLogger 以便画虚线/阴影
    logger = MetricsLogger(run_dir, prefix,
                           plot_every=train_cfg.get('plot_metrics_every', 50),
                           tail_start_ep=tail_start_ep)

    best_eval_ms = float("inf")
    best_eval_reward = float("-inf")

    training_stats = {
        'episodes_completed': 0,
        'best_makespan': float('inf'),
        'best_reward': float('-inf'),
        'early_stop_triggered': False,
        'invalid_makespan_episodes': 0
    }

    # 8.21修改_序号7：初始化"策略EMA"与评估EMA缓存（仅用于评估/展示，不影响训练）
    # 说明：agent_ema用于swap_in/out评估；eval_ms_ema用于绘图（更平的eval线）。
    use_ema_eval = bool(train_cfg.get('use_ema_eval', True))
    ema_decay = float(train_cfg.get('ema_decay', 0.995))
    agent_ema = ParamEMA(agent, decay=ema_decay) if use_ema_eval else None

    eval_ema_alpha = float(train_cfg.get('eval_ema_alpha', 0.1))
    eval_ms_ema = None

    print(f"\n🚀 Starting training for {train_cfg['max_episodes']} episodes...")
    print("=" * 60)

    for ep in tqdm(range(1, train_cfg['max_episodes'] + 1), desc="Training Progress"):
        state = env.reset()
        state['map_size'] = env_cfg['map_size']
        done, ep_reward, last_value = False, 0.0, 0.0

        # episode级别的异常检测
        step_count = 0
        max_steps = env.num_tasks * 3  # 设置最大步数限制

        while not done and step_count < max_steps:
            with torch.no_grad():
                usv_task_edges = compute_lookahead_features(state, agent.device)

            action, logp, value = agent.get_action(state, usv_task_edges, deterministic=False)
            next_state, r, done, info = env.step(action)
            ep_reward += r
            step_count += 1

            memory.add(
                state,
                action,
                logp,
                float(r),
                bool(done),
                float(value),
                usv_task_edges.cpu()
            )
            state = next_state
            state['map_size'] = env_cfg['map_size']
            last_value = value

        # 检查episode是否正常完成
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

        # 8.21修改_序号8：在每个episode更新后，刷新EMA参数缓存
        # 说明：保持EMA追踪到最新策略；训练仍使用原策略参数，EMA仅在评估/保存best时使用。
        if use_ema_eval:
            agent_ema.update(agent)

        log_metrics = {
            'episode': ep,
            'train_makespan': makespan,
            'train_reward': ep_reward,
            # —— 新增的两个兼容键（关键）——
            'makespan': makespan,   # Visdom 用
            'reward': ep_reward     # Visdom 用
        }
        log_metrics.update(losses)
        log_metrics.update(env.get_balance_metrics())

        training_stats['episodes_completed'] = ep
        if makespan > 0 and makespan < training_stats['best_makespan']:
            training_stats['best_makespan'] = makespan
        if ep_reward > training_stats['best_reward']:
            training_stats['best_reward'] = ep_reward

        if viz and viz.enabled:
            viz.update_plots(ep, log_metrics)

        if ep % train_cfg['eval_frequency'] == 0:
            # 8.21修改_序号9：评估阶段多次取均值 + （可选）使用EMA权重评估 + 记录EMA平滑指标
            # 说明：n_eval_episodes降低统计噪声；EMA权重降低策略对场景的敏感度；eval_ms_ema用于绘图更稳定。
            n_eval = int(train_cfg.get('n_eval_episodes', 10))

            if use_ema_eval:
                _orig = agent_ema.swap_in(agent)  # EMA权重装入agent
                eval_results = evaluate(env, agent, episodes=n_eval, deterministic=True)
                agent_ema.swap_out(agent, _orig)  # 恢复训练权重
            else:
                eval_results = evaluate(env, agent, episodes=n_eval, deterministic=True)

            cur_ms = float(eval_results['makespan'])
            eval_ms_ema = cur_ms if eval_ms_ema is None else (eval_ema_alpha*cur_ms + (1.0-eval_ema_alpha)*eval_ms_ema)

            log_metrics.update({
                'eval_makespan': eval_results['makespan'],
                'eval_makespan_ema': eval_ms_ema,
                'eval_reward': eval_results['reward']
            })

            print(f"\n[Eval Ep {ep:4d}] Makespan: {eval_results['makespan']:.2f}±{eval_results['makespan_std']:.2f}, "
                  f"Reward: {eval_results['reward']:.2f}, Jain's: {eval_results['jains_index']:.3f}")

            if eval_results.get('invalid_episodes', 0) > 0:
                print(f"  ⚠️  {eval_results['invalid_episodes']}/{n_eval} eval episodes had issues")

            # 保存best（评估结果若来自EMA策略，将天然更稳定）
            if eval_results['makespan'] > 0 and eval_results['makespan'] < best_eval_ms:
                best_eval_ms = eval_results['makespan']
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_makespan_model.pt")
                print(f"  🎯 New best makespan model saved: {best_eval_ms:.2f}")

            if eval_results['reward'] > best_eval_reward:
                best_eval_reward = eval_results['reward']
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_reward_model.pt")
                print(f"  🎯 New best reward model saved: {best_eval_reward:.2f}")

            # 新的一周修改+序号2：在特定间隔生成甘特图检查训练进展
            # 说明：每100个episode或最佳性能更新时生成甘特图，确保环境状态始终正确
            if (ep % 100 == 0 or eval_results['makespan'] < best_eval_ms * 1.01):
                try:
                    gantt_check_path = str(run_dir / f"{prefix}gantt_check_ep{ep}.png")
                    viz_check = VisualizationManager(viz_name="training_check", enabled=False)
                    
                    # 临时开启调试模式进行详细检查
                    env.set_debug_mode(True)
                    check_results = evaluate(env, agent, episodes=1, deterministic=True)
                    gantt_summary = viz_check.generate_gantt_chart(env, save_path=gantt_check_path)
                    env.set_debug_mode(debug_training)
                    
                    print(f"  📊 Gantt check generated: ep{ep}, makespan={gantt_summary['makespan']:.2f}")
                    
                except Exception as e:
                    print(f"  ⚠️  Failed to generate Gantt check: {e}")

            if ppo.check_early_stop(eval_results['reward']):
                print(f"\nℹ️ Early stopping triggered at episode {ep}")
                training_stats['early_stop_triggered'] = True
                logger.log(ep, log_metrics)
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
        serializable_stats = {k: float(v) if isinstance(v, np.number) else v
                              for k, v in training_stats.items()}
        json.dump(serializable_stats, f, indent=2)

    # 新的一周修改+序号2：最终环境验证和详细甘特图生成
    # 说明：训练完成后进行最终的环境验证，生成详细的甘特图和分析报告
    try:
        print("\n" + "="*60)
        print("🏁 FINAL VALIDATION AND REPORTING")
        print("="*60)
        
        viz_final = VisualizationManager(viz_name="final_gantt", enabled=False)

        best_model_path = run_dir / f"{prefix}best_makespan_model.pt"
        if best_model_path.exists():
            print(f"[INFO] Loading best makespan model for final analysis")
            agent.load_state_dict(torch.load(best_model_path, map_location=device))

        # 最终环境验证（使用最佳模型）
        final_validation_path = str(run_dir / f"{prefix}final_validation")
        final_validation_results = validate_environment_integrity(
            env, agent, device, 
            validation_episodes=5,  # 更多验证episode
            save_path=final_validation_path
        )
        
        # 生成最终甘特图
        env.set_debug_mode(True)
        evaluate(env, agent, episodes=1, deterministic=True)
        gantt_path = run_dir / f"{prefix}gantt_final.png"
        summary = viz_final.generate_gantt_chart(env, save_path=str(gantt_path))

        table_path = run_dir / f"{prefix}gantt_table.png"
        viz_final.save_summary_table(summary, env.makespan, str(table_path))
        print(f"✅ Final Gantt chart and summary table saved.")

        # 输出最终统计
        final_balance = env.get_balance_metrics()
        print(f"📊 Final Jain's Index: {final_balance['jains_index']:.4f}")
        print(f"📊 Final Task Load Variance: {final_balance['task_load_variance']:.4f}")
        print(f"📊 Final Makespan: {env.makespan:.2f}")
        
        # 保存最终验证报告
        final_report = {
            'training_stats': training_stats,
            'final_validation': final_validation_results,
            'final_performance': {
                'makespan': env.makespan,
                'jains_index': final_balance['jains_index'],
                'task_load_variance': final_balance['task_load_variance']
            },
            'gantt_summary': summary
        }
        
        with open(run_dir / f"{prefix}final_report.json", 'w') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        print(f"📋 Final report saved: {run_dir / f'{prefix}final_report.json'}")

    except Exception as e:
        print(f"[WARN] Failed to generate final report: {e}")

    print(f"✅ All artifacts saved in: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
