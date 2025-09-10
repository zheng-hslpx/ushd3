import os, re, json, argparse, csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# 后端与作图（非交互）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# ===== 依赖模块 =====
from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
from usv_agent.ppo_policy import EnhancedPPOAgent, EnhancedPPO, Memory

# 可选 Visdom（没有也不影响）
try:
    from utils.vis_manager import VisualizationManager
except Exception:
    VisualizationManager = None

torch.set_default_dtype(torch.float32)


# -------------------- 工具方法 --------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_device(model_cfg: Dict[str, Any]) -> torch.device:
    want = str(model_cfg.get("device", "auto")).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(want)

def group_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("train_paras", {}).get("run_name", "default_run")

def next_run_index(group_dir: Path) -> str:
    group_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in group_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{2,}", p.name)]
    return f"{(max(nums) + 1) if nums else 1:02d}"

def short_ts() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


# -------------------- 边特征：对角线归一距离 + 两个轻量前瞻量 --------------------
# ★ 修复点：避免对 expand 视图做原位除法；统一改为“非原位除法”产生新张量

def compute_lookahead_edges(state: Dict[str, np.ndarray], map_size, device: torch.device) -> torch.Tensor:
    """
    输入 state：{'usv_features':[U,3], 'task_features':[T,4]}
    返回 usv_task_edges：[U,T,3] = [dist_ut_norm, task_proximity, usv_opportunity]
      - dist_ut_norm：USV-Task 欧氏距离 / 地图对角线
      - task_proximity：任务之间的最小邻距（仅未调度任务），广播到 U
      - usv_opportunity：每艘 USV 到最近未调度任务的距离，广播到 T
    """
    uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=device)   # [U,3]
    tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=device)  # [T,4]
    U, T = uf.size(0), tf.size(0)

    usv_pos = uf[:, :2]                   # [U,2]
    task_pos = tf[:, :2]                  # [T,2]
    active   = tf[:, 3] > 0               # [T]

    # U×T 距离
    dist_ut = torch.cdist(usv_pos, task_pos) if (U > 0 and T > 0) else torch.zeros(U, T, device=device)

    # 未调度任务间的最小邻距（广播到 U）
    prox = torch.zeros(T, device=device)
    if active.sum() > 1:
        pos_u = task_pos[active]
        d_tt = torch.cdist(pos_u, pos_u)
        d_tt.fill_diagonal_(float("inf"))
        min_d, _ = torch.min(d_tt, dim=1)
        prox[active] = min_d
    # 注意：expand 产生视图，后续不要原位写入
    feat_task_proximity = prox.unsqueeze(0).expand(U, -1) if T > 0 else torch.zeros(U, T, device=device)

    # 每艘 USV 到最近未调度任务的距离（广播到 T）
    if active.any():
        d_uu = torch.cdist(usv_pos, task_pos[active])  # [U, #active]
        min_du, _ = torch.min(d_uu, dim=1)             # [U]
        feat_usv_opp = min_du.unsqueeze(1).expand(-1, T) if T > 0 else torch.zeros(U, T, device=device)
    else:
        feat_usv_opp = torch.zeros(U, T, device=device)

    # 对角线归一（避免原位操作！）
    diag = torch.norm(torch.as_tensor(map_size, dtype=torch.float32, device=device))
    if diag > 0:
        dist_ut = dist_ut / diag
        feat_task_proximity = feat_task_proximity / diag
        feat_usv_opp = feat_usv_opp / diag

    return torch.stack([dist_ut, feat_task_proximity, feat_usv_opp], dim=-1)  # [U,T,3]


# -------------------- 评估（奖励=ΔMakespan） --------------------

@torch.no_grad()
def evaluate(env: USVEnv, agent: EnhancedPPOAgent, episodes: int = 5) -> Dict[str, float]:
    """
    评估（禁用探索，确定性策略）：
    返回：平均 makespan / ΔMakespan(reward) / Jain 指数 / makespan std
    """
    agent.set_train_mode(False)
    orig_debug = env.debug_mode
    env.set_debug_mode(False)

    ms_list, rew_list, j_list = [], [], []
    for _ in range(episodes):
        s = env.reset()
        done, ep_rew = False, 0.0
        max_steps = env.num_usvs * env.num_tasks + 5  # 简单上界
        steps = 0
        while not done and steps < max_steps:
            edges = compute_lookahead_edges(s, env.map_size, device=agent.device)
            a, logp, v = agent.get_action(s, edges, deterministic=True)  # 确定性
            s, r, done, info = env.step(a)
            ep_rew += float(r)
            steps += 1
        ms_list.append(float(info.get("makespan", 0.0)))
        j_list.append(float(env.get_balance_metrics()["jains_index"]))
        rew_list.append(float(ep_rew))

    env.set_debug_mode(orig_debug)
    agent.set_train_mode(True)

    return {
        "makespan": float(np.mean(ms_list)),
        "makespan_std": float(np.std(ms_list)),
        "reward": float(np.mean(rew_list)),        # ΔMakespan（越大越好）
        "jains_index": float(np.mean(j_list)),
    }


# -------------------- 简易日志器（CSV + 小图） --------------------

class MetricsLogger:
    def __init__(self, run_dir: Path, prefix: str, plot_every: int = 50):
        self.run_dir = run_dir
        self.csv_path = run_dir / f"{prefix}metrics.csv"
        self.plot_path = run_dir / f"{prefix}metrics.png"
        self.plot_every = int(plot_every)
        self.headers = [
            "episode", "train_makespan", "train_delta_ms", "actor_loss", "critic_loss", "entropy",
            "eval_makespan", "eval_delta_ms", "jains_index", "lr_critic", "epsilon"
        ]
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)
        self.data = {h: [] for h in self.headers}

    def log(self, ep: int, row: Dict[str, float]):
        for h in self.headers:
            self.data[h].append(row.get(h, np.nan))
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(h, "") for h in self.headers])
        if ep % self.plot_every == 0 or ep == 1:
            self.plot()

    def plot(self):
        if not self.data["episode"]:
            return
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        # ΔMakespan
        axes[0].plot(self.data["episode"], self.data["train_delta_ms"], label="train")
        if any(np.isfinite(self.data["eval_delta_ms"])):
            axes[0].plot(self.data["episode"], self.data["eval_delta_ms"], "--", label="eval")
        axes[0].set_title("ΔMakespan (Reward)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.data["episode"], self.data["train_makespan"], label="train")
        axes[1].plot(self.data["episode"], self.data["eval_makespan"], "--", label="eval")
        axes[1].set_title("Makespan"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.data["episode"], self.data["jains_index"])
        axes[2].set_title("Jain's Index"); axes[2].set_ylim(0, 1.05); axes[2].grid(True, alpha=0.3)

        axes[3].plot(self.data["episode"], self.data["actor_loss"])
        axes[3].set_title("Actor Loss"); axes[3].grid(True, alpha=0.3)

        axes[4].plot(self.data["episode"], self.data["critic_loss"])
        axes[4].set_title("Critic Loss"); axes[4].grid(True, alpha=0.3)

        axes[5].plot(self.data["episode"], self.data["epsilon"])
        axes[5].set_title("Epsilon"); axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


# -------------------- 主流程 --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join("config", "improved_config.json"),
                        help="配置文件路径")
    parser.add_argument("--save_root", type=str, default="results/saved_models2",
                        help="保存根目录（将创建 run 子目录）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env_paras"]; model_cfg = cfg["model_paras"]; train_cfg = cfg["train_paras"]

    device = pick_device(model_cfg)
    print(f"[INFO] Device: {device} | Torch: {torch.__version__}")

    # 结果目录：results/saved_models2/<run_name>/<01,02,...>
    save_root = Path(train_cfg.get("save_root", args.save_root))
    run_group = save_root / group_name(cfg)
    run_dir = run_group / next_run_index(run_group)
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{short_ts()}_"
    print(f"[INFO] 保存目录: {run_dir}")
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 构建环境/模型/智能体/优化器
    env = USVEnv(env_cfg)
    hgnn = HeterogeneousGNN({**model_cfg, "num_usvs": env_cfg["num_usvs"], "num_tasks": env_cfg["num_tasks"]}).to(device)
    agent = EnhancedPPOAgent(hgnn, {**model_cfg, **train_cfg}).to(device)
    ppo = EnhancedPPO(agent, train_cfg)
    memory = Memory()

    # 可选 Visdom
    viz = None
    if train_cfg.get("viz", False) and VisualizationManager is not None:
        try:
            viz = VisualizationManager(viz_name=train_cfg.get("viz_name", "usv"), enabled=True)
        except Exception as e:
            print(f"[WARN] Visdom 初始化失败：{e}")

    logger = MetricsLogger(run_dir, prefix, plot_every=int(train_cfg.get("plot_metrics_every", 50)))

    # 训练参量
    max_episodes  = int(train_cfg.get("max_episodes", 1200))
    eval_every    = int(train_cfg.get("eval_frequency", 25))
    save_every    = int(train_cfg.get("save_frequency", 100))
    best_ms       = float("inf")
    best_reward   = -float("inf")

    print("\n============================================")
    print(f"=== Training start: {max_episodes} episodes ===")

    agent.set_train_mode(True)
    for ep in tqdm(range(1, max_episodes + 1), desc="Training"):
        s = env.reset()
        done, ep_rew = False, 0.0
        steps = 0
        max_steps = env.num_usvs * env.num_tasks + 5

        # 每回合探索统计复位（仅用于信息输出，不影响算法）
        agent.reset_exploration_episode_stats()

        last_value = 0.0
        while not done and steps < max_steps:
            edges = compute_lookahead_edges(s, env.map_size, device=agent.device)
            a, logp, v = agent.get_action(s, edges, deterministic=False)  # 训练时允许 ε-greedy
            ns, r, done, info = env.step(a)
            ep_rew += float(r)
            # 记录一步
            memory.add(s, a, logp, float(r), bool(done), float(v), edges.cpu())
            s = ns
            steps += 1
            last_value = v

        # 终止后追加 bootstrap 值（终局取 0 即可）
        memory.values.append(0.0)
        # PPO 更新
        losses = ppo.update(memory, eval_reward=ep_rew)
        memory.clear_memory()

        # 日志与可视化
        log_row = {
            "episode": ep,
            "train_makespan": float(info.get("makespan", 0.0)),
            "train_delta_ms": float(ep_rew),                 # ΔMakespan（reward 的累计）
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "critic_loss": float(losses.get("critic_loss", 0.0)),
            "entropy": float(losses.get("entropy", 0.0)),
            "lr_critic": float(losses.get("lr_critic", 0.0)),
            "epsilon": float(losses.get("epsilon", 0.0)),
            "eval_makespan": np.nan, "eval_delta_ms": np.nan, "jains_index": float(env.get_balance_metrics()["jains_index"])
        }

        if viz and getattr(viz, "enabled", False):
            try:
                viz.update_plots(ep, {
                    "train_reward": log_row["train_delta_ms"],   # 复用键名
                    "train_makespan": log_row["train_makespan"],
                    "actor_loss": log_row["actor_loss"],
                    "critic_loss": log_row["critic_loss"],
                    "entropy": log_row["entropy"],
                    "jains_index": log_row["jains_index"],
                    "eval_reward": np.nan, "eval_makespan": np.nan
                })
            except Exception:
                pass

        # 周期性评估（确定性策略）
        if ep % eval_every == 0:
            eval_res = evaluate(env, agent, episodes=int(train_cfg.get("eval_episodes", 5)))
            log_row.update({
                "eval_makespan": eval_res["makespan"],
                "eval_delta_ms": eval_res["reward"],
                "jains_index": eval_res["jains_index"]
            })
            print(f"[Eval {ep:4d}] Makespan: {eval_res['makespan']:.2f} ± {eval_res['makespan_std']:.2f} | "
                  f"ΔMakespan (Reward): {eval_res['reward']:.2f} | Jain's: {eval_res['jains_index']:.3f}")

            # 保存最优
            if eval_res["makespan"] < best_ms:
                best_ms = eval_res["makespan"]
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_makespan.pt")
                print(f"  ↳ Save best makespan: {best_ms:.2f}")
            if eval_res["reward"] > best_reward:
                best_reward = eval_res["reward"]
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_reward.pt")
                print(f"  ↳ Save best ΔMakespan reward: {best_reward:.2f}")

            # 早停（可选）
            if ppo.check_early_stop(eval_res["reward"]):
                print(f"[INFO] Early stop at episode {ep}")
                logger.log(ep, log_row)
                break

        # 周期性快照
        if ep % save_every == 0:
            torch.save(agent.state_dict(), run_dir / f"{prefix}ep{ep:04d}.pt")
            torch.save(agent.state_dict(), run_dir / f"{prefix}latest.pt")

        logger.log(ep, log_row)

    # 收尾：保存最终模型与一次最终评估
    torch.save(agent.state_dict(), run_dir / f"{prefix}final.pt")
    final_eval = evaluate(env, agent, episodes=int(train_cfg.get("final_eval_episodes", 3)))
    print("\n=== Training finished ===")
    print(f"Best makespan: {best_ms:.2f} | Best ΔMakespan reward: {best_reward:.2f}")
    print(f"Final eval → Makespan: {final_eval['makespan']:.2f} ± {final_eval['makespan_std']:.2f}, "
          f"ΔMakespan: {final_eval['reward']:.2f}, Jain's: {final_eval['jains_index']:.3f}")

    # === Final Gantt === 生成并保存甘特图与汇总表（只新增这段；其他逻辑不变） ===
    try:
        if VisualizationManager is None:
            raise RuntimeError("VisualizationManager 未可用")

        agent.set_train_mode(False)  # 确保确定性
        s = env.reset()
        done = False
        steps = 0
        max_steps = env.num_usvs * env.num_tasks + 5
        while not done and steps < max_steps:
            edges = compute_lookahead_edges(s, env.map_size, device=agent.device)
            a, _, _ = agent.get_action(s, edges, deterministic=True)
            s, _, done, _ = env.step(a)
            steps += 1

        viz_final = VisualizationManager(viz_name="final_gantt", enabled=False)
        gantt_path = run_dir / f"{prefix}gantt_final.png"
        summary = viz_final.generate_gantt_chart(env, save_path=str(gantt_path))
        table_path = run_dir / f"{prefix}gantt_table.png"
        viz_final.save_summary_table(summary, env.makespan, save_path=str(table_path))
        print(f"✅ Gantt saved to: {gantt_path}")
        print(f"✅ Summary saved to: {table_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate final Gantt: {e}")

    print(f"Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
