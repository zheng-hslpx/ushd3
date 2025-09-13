
import os, json, argparse, csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 后端与作图（非交互）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# 固定随机种子（可按需调整/去除）
import random
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

# ===== 依赖模块 =====
from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
from usv_agent.ppo_policy import EnhancedPPO, EnhancedPPOAgent, Memory
from usv_agent.data_generator import USVTaskDataGenerator

# 9.11修改_序号V1：Visdom 管理器（注意导入路径）
from utils.vis_manager import VisualizationManager


# ============ 基础工具 ============
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def next_run_index(run_group: Path) -> str:
    """在 run_group 下生成 01、02、03... 的新子目录名"""
    idx = 1
    existing = {d.name for d in run_group.iterdir() if d.is_dir()}
    while True:
        cand = f"{idx:02d}"
        if cand not in existing:
            return cand
        idx += 1

def group_name(cfg: Dict[str, Any]) -> str:
    """使用配置里的 viz_name 构建分组目录（推荐：<场景>/<版本>，如 RTX2060_5x24/v28）"""
    tp = cfg.get("train_paras", {})
    return tp.get("viz_name", "RUN")

def human_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def short_ts() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


# -------------------- 轻量边特征（与评估保持一致） --------------------
def compute_lookahead_edges(state: Dict[str, np.ndarray], map_size, device: torch.device) -> torch.Tensor:
    """
    输入 state：{'usv_features':[U,3], 'task_features':[T,4]}
    返回张量 [U,T,3]：
      0) dist_norm      USV-Task 欧氏距离 / 地图对角线
      1) task_proximity 任务之间的最小邻距（仅未调度任务），广播到 U
      2) usv_opp        USV 到最近未调度任务的距离，广播到 T
    """
    uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=device)   # [U,3]
    tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=device)  # [T,4]
    U, T = uf.size(0), tf.size(0)
    if U == 0 or T == 0:
        return torch.zeros((U, T, 3), dtype=torch.float32, device=device)

    usv_pos = uf[:, :2]                   # [U,2]
    task_pos = tf[:, :2]                  # [T,2]
    active   = tf[:, 3] > 0               # [T] 1=未完成

    # U×T 距离
    dist_ut = torch.cdist(usv_pos, task_pos)  # [U,T]

    # 任务最小邻距（仅 active）
    prox = torch.zeros(T, device=device)
    if active.sum() > 1:
        pos_a = task_pos[active]
        d_tt = torch.cdist(pos_a, pos_a)
        d_tt.fill_diagonal_(float("inf"))
        min_d, _ = torch.min(d_tt, dim=1)
        prox[active] = min_d
    feat_task_prox = prox.unsqueeze(0).expand(U, -1)   # [U,T]

    # USV 到最近 active 任务的距离，广播到 T
    if active.any():
        d_ua = torch.cdist(usv_pos, task_pos[active])
        min_du, _ = torch.min(d_ua, dim=1)             # [U]
        feat_usv_opp = min_du.unsqueeze(1).expand(-1, T)
    else:
        feat_usv_opp = torch.zeros(U, T, device=device)

    # 对角线归一
    diag = torch.norm(torch.as_tensor(map_size, dtype=torch.float32, device=device))
    if diag > 0:
        dist_ut        = dist_ut / diag
        feat_task_prox = feat_task_prox / diag
        feat_usv_opp   = feat_usv_opp / diag

    return torch.stack([dist_ut, feat_task_prox, feat_usv_opp], dim=-1)  # [U,T,3]


# -------------------- 随机评估（兜底） --------------------
@torch.no_grad()
def evaluate(env: USVEnv, agent: EnhancedPPOAgent, episodes: int = 5) -> Dict[str, float]:
    """评估（禁用探索，确定性策略；场景随机）"""
    agent.set_train_mode(False)
    ms_list, rw_list, j_list = [], [], []
    device = agent.device if hasattr(agent, "device") else next(agent.parameters()).device

    orig_debug = getattr(env, "debug_mode", None)
    try: env.set_debug_mode(False)
    except Exception: pass

    for _ in range(int(episodes)):
        s = env.reset()
        done, steps, ep_rew = False, 0, 0.0
        info = {}
        while not done and steps < env.num_usvs * env.num_tasks + 5:
            edges = compute_lookahead_edges(s, env.map_size, device=device)
            a, _, _ = agent.get_action(s, edges, epoch=0, max_epoch=1, deterministic=True)
            s, r, done, info = env.step(a)
            ep_rew += float(r)
            steps += 1

        ms_list.append(float(info.get("makespan", getattr(env, "makespan", 0.0))))
        try:
            j = env.get_balance_metrics().get("jains_index", 1.0)
        except Exception:
            j = 1.0
        j_list.append(float(j))
        rw_list.append(ep_rew)

    res = {
        "makespan": float(np.mean(ms_list)) if ms_list else float("inf"),
        "makespan_std": float(np.std(ms_list)) if ms_list else 0.0,
        "reward": float(np.mean(rw_list)) if rw_list else 0.0,
        "jains_index": float(np.mean(j_list)) if j_list else 1.0,
    }
    if orig_debug is not None:
        try: env.set_debug_mode(orig_debug)
        except Exception: pass

    agent.set_train_mode(True)
    return res


# -------------------- 固定评估用例（由 seeds 决定） --------------------
def _build_fixed_eval_cases_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[dict, dict]]:
    e = (cfg.get("eval_fixed_cases") or {})
    if not e.get("enabled", False):
        return []

    envp = cfg.get("env_paras", {})
    seeds = list(e.get("seeds", []))
    if not seeds:
        return []

    gen_cfg = {
        "num_usvs": int(envp.get("num_usvs", 5)),
        "num_tasks": int(envp.get("num_tasks", 24)),
        "map_size": envp.get("map_size", [120, 120]),
        "battery_capacity": float(envp.get("battery_capacity", 220.0)),
        "min_processing_time": float(e.get("min_processing_time", envp.get("min_processing_time", 8.0))),
        "max_processing_time": float(e.get("max_processing_time", envp.get("max_processing_time", 30.0))),
        "task_distribution": e.get("task_distribution", envp.get("task_distribution", "uniform")),
    }
    gen = USVTaskDataGenerator(gen_cfg)
    return [gen.generate_instance(seed=s) for s in seeds]

@torch.no_grad()
def evaluate_fixed(env: USVEnv, agent: EnhancedPPOAgent, cases: List[Tuple[dict, dict]]) -> Dict[str, float]:
    """固定评估（禁用探索，确定性策略；用 _build_fixed_eval_cases_from_cfg 的 (usvs, tasks)）"""
    if not cases:
        return evaluate(env, agent, episodes=5)

    agent.set_train_mode(False)
    ms_list, rw_list, j_list = [], [], []
    device = agent.device if hasattr(agent, "device") else next(agent.parameters()).device

    orig_debug = getattr(env, "debug_mode", None)
    try: env.set_debug_mode(False)
    except Exception: pass

    for (usvs, tasks) in cases:
        s = env.reset(tasks_data=tasks, usvs_data=usvs)  # ★ 固定实例：坐标/时长一致
        done, steps, ep_rew = False, 0, 0.0
        info = {}
        while not done and steps < env.num_usvs * env.num_tasks + 5:
            edges = compute_lookahead_edges(s, env.map_size, device=device)
            a, _, _ = agent.get_action(s, edges, epoch=0, max_epoch=1, deterministic=True)
            s, r, done, info = env.step(a)
            ep_rew += float(r)
            steps += 1

        ms_list.append(float(info.get("makespan", getattr(env, "makespan", 0.0))))
        try:
            j = env.get_balance_metrics().get("jains_index", 1.0)
        except Exception:
            j = 1.0
        j_list.append(float(j))
        rw_list.append(ep_rew)

    res = {
        "makespan": float(np.mean(ms_list)) if ms_list else float("inf"),
        "makespan_std": float(np.std(ms_list)) if ms_list else 0.0,
        "reward": float(np.mean(rw_list)) if rw_list else 0.0,
        "jains_index": float(np.mean(j_list)) if j_list else 1.0,
    }
    if orig_debug is not None:
        try: env.set_debug_mode(orig_debug)
        except Exception: pass

    agent.set_train_mode(True)
    return res


# -------------------- 简易日志器（CSV + 本地小图） --------------------
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

    def log(self, ep: int, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                row.get("episode"),
                row.get("train_makespan"),
                row.get("train_delta_ms"),
                row.get("actor_loss"), row.get("critic_loss"), row.get("entropy"),
                row.get("eval_makespan"), row.get("eval_delta_ms"),
                row.get("jains_index"), row.get("lr_critic"), row.get("epsilon")
            ])
        # 到点出图
        if ep % self.plot_every == 0:
            self._plot()

    def _plot(self):
        """改进：对有限值过滤，避免 NaN 导致整条线不可见；对评估稀疏点连线显示"""
        try:
            xs, train_ms, eval_ms = [], [], []
            with open(self.csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(int(row["episode"]))
                    # 训练曲线：允许 NaN
                    tm = row.get("train_makespan", "")
                    train_ms.append(float(tm) if tm not in ("", "nan", "NaN") else np.nan)
                    # 评估曲线：允许 NaN（非评估行）
                    em = row.get("eval_makespan", "")
                    eval_ms.append(float(em) if em not in ("", "nan", "NaN") else np.nan)

            xs = np.asarray(xs, dtype=float)
            train_ms = np.asarray(train_ms, dtype=float)
            eval_ms  = np.asarray(eval_ms, dtype=float)

            plt.figure(figsize=(10, 4.5))

            # 训练曲线：有值就画（可能全部 NaN）
            mask_train = np.isfinite(train_ms)
            if np.any(mask_train):
                plt.plot(xs[mask_train], train_ms[mask_train], label="Train Makespan")

            # 评估曲线：只取有限值点连线
            mask_eval = np.isfinite(eval_ms)
            if np.any(mask_eval):
                plt.plot(xs[mask_eval], eval_ms[mask_eval], label="Eval Makespan")

            plt.xlabel("Episode"); plt.ylabel("Makespan")
            plt.title("Makespan Curves (Train vs Eval)")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=140)
            plt.close()
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")


# -------------------- 设备选择 --------------------
def pick_device(model_cfg: dict) -> torch.device:
    want = str(model_cfg.get("device", "auto")).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(want)


# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("config") / "improved_config.json"))
    parser.add_argument("--save_root", type=str, default=None,
                        help="保存根目录（建议仅在配置里设置 save_root；命令行为空时使用配置值）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env_paras"]; model_cfg = cfg["model_paras"]; train_cfg = cfg["train_paras"]

    device = pick_device(model_cfg)
    print(f"[INFO] Device: {device} | Torch: {torch.__version__}")

    # ==== 路径（完全由配置控制）====
    # 只要配置使用：
    #   train_paras.save_root: "results/saved_models_2"
    #   train_paras.viz_name: "RTX2060_5x24/v27"
    # 目录就会是：results/saved_models_2/RTX2060_5x24/v27/01
    save_root = Path(train_cfg.get("save_root", "results/saved_models_2")) if args.save_root is None else Path(args.save_root)
    run_group = save_root / group_name(cfg)
    ensure_dir(run_group)
    run_dir = run_group / next_run_index(run_group)
    ensure_dir(run_dir)
    prefix = f"{short_ts()}_"
    print(f"[INFO] Run dir: {run_dir}\n")

    print("===================================")
    print(f"  Start training @ {human_ts()}")
    print(f"  Episodes={int(train_cfg.get('max_episodes', 1200))} | "
          f"Eval every={int(train_cfg.get('eval_frequency', 10))} | "
          f"Save every={int(train_cfg.get('save_frequency', 100))}")
    print("===================================\n")

    # ==== 环境、模型、PPO ====
    env = USVEnv(env_cfg)
    hgnn = HeterogeneousGNN(model_cfg).to(device)
    agent = EnhancedPPOAgent(hgnn, model_cfg).to(device)
    ppo = EnhancedPPO(agent, train_cfg)

    # ==== Visdom ====
    viz_kwargs = {}
    if "viz_server" in train_cfg: viz_kwargs["server"] = train_cfg["viz_server"]
    if "viz_port"   in train_cfg: viz_kwargs["port"]   = train_cfg["viz_port"]
    vis = VisualizationManager(
        viz_name=train_cfg.get("viz_name", "RTX2060_5x24"),
        enabled=bool(train_cfg.get("viz", True)),
        **viz_kwargs
    )

    # ==== 记录器 ====
    logger = MetricsLogger(run_dir, prefix, plot_every=int(train_cfg.get("plot_metrics_every", 50)))

    # ==== 训练参量 ====
    max_episodes  = int(train_cfg.get("max_episodes", 1200))
    eval_every    = int(train_cfg.get("eval_frequency", 10))   # 默认 10
    prefer_eval   = bool(train_cfg.get("plot_eval_only", True))  # 只画评估曲线（训练仍随机）
    save_every    = int(train_cfg.get("save_frequency", 100))
    best_ms       = float("inf")
    best_reward   = -float("inf")

    # ==== 主循环 ====
    for ep in tqdm(range(1, max_episodes + 1), ncols=100, desc="Training"):
        s = env.reset()  # ★ 训练保持“随机算例”
        done, steps, ep_rew = False, 0, 0.0
        mem = Memory()   # 方式A

        # rollout
        while not done and steps < env.num_usvs * env.num_tasks + 5:
            edges = compute_lookahead_edges(s, env.map_size, device=device)
            a, logp, v = agent.get_action(s, edges, epoch=ep, max_epoch=max_episodes, deterministic=False)
            s_next, r, done, info = env.step(a)

            ep_rew += float(r)
            mem.add(s, a, logp, float(r), bool(done), float(v), edges)  # 方式A：Memory.add(...)
            s = s_next
            steps += 1

        # bootstrap（GAE/returns 必需：长度 T+1）
        with torch.no_grad():
            edges = compute_lookahead_edges(s, env.map_size, device=device)
            _, _, vT = agent.get_action(s, edges, epoch=ep, max_epoch=max_episodes, deterministic=True)
        mem.values.append(float(vT))

        losses = ppo.update(mem, eval_reward=ep_rew)

        # 当前训练期可用指标（评估发生前）
        try:
            train_jain = float(getattr(env, "get_balance_metrics", lambda: {"jains_index": 1.0})().get("jains_index", 1.0))
        except Exception:
            train_jain = 1.0

        # —— Visdom：先推训练值（若 plot_eval_only=True，训练曲线仍记 NaN 以隐藏）
        vis.update_plots(ep, {
            "train_makespan": (np.nan if prefer_eval else float(info.get("makespan", 0.0))),
            "train_reward": float(ep_rew),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "critic_loss": float(losses.get("critic_loss", 0.0)),
            "jains_index": train_jain
        })

        # 默认行（评估后会被覆盖 eval_* 字段）
        log_row = {
            "episode": ep,
            "train_makespan": (np.nan if prefer_eval else float(info.get("makespan", 0.0))),
            "train_delta_ms": float(ep_rew),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "critic_loss": float(losses.get("critic_loss", 0.0)),
            "entropy": float(losses.get("entropy", 0.0)),
            "eval_makespan": np.nan,
            "eval_delta_ms": np.nan,
            "jains_index": train_jain,
            "lr_critic": float(losses.get("lr_critic", 0.0)),
            "epsilon": float(losses.get("epsilon", 0.0))
        }

        # 周期评估——优先使用固定用例（评估固定、训练随机）
        if ep % eval_every == 0:
            _cases = _build_fixed_eval_cases_from_cfg(cfg)
            if _cases:
                eval_res = evaluate_fixed(env, agent, _cases)
            else:
                eval_res = evaluate(env, agent, episodes=int(train_cfg.get("eval_episodes", 5)))

            # 覆盖日志中的评估字段
            log_row.update({
                "eval_makespan": float(eval_res["makespan"]),
                "eval_delta_ms": float(eval_res["reward"]),
                "jains_index": float(eval_res["jains_index"])  # 评估后的公平性（可覆盖训练期值）
            })

            print(f"[Eval {ep:5d}] Makespan={eval_res['makespan']:.2f} (std={eval_res['makespan_std']:.2f})"
                  f" | ΔMs={eval_res['reward']:.2f} | Jain={eval_res['jains_index']:.3f}")

            # —— Visdom：在同一次调用里同时推 train_* 与 eval_*，这样同窗双折线
            vis.update_plots(ep, {
                "train_makespan": (np.nan if prefer_eval else float(info.get("makespan", 0.0))),
                "train_reward": float(ep_rew),
                "actor_loss": float(losses.get("actor_loss", 0.0)),
                "critic_loss": float(losses.get("critic_loss", 0.0)),
                "jains_index": train_jain,

                "eval_makespan": float(eval_res["makespan"]),
                "eval_reward": float(eval_res["reward"]),
                "eval_jains_index": float(eval_res["jains_index"])
            })

            # 最优保存（以 makespan 为主）
            if eval_res["makespan"] < best_ms:
                best_ms = eval_res["makespan"]
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_makespan.pt")

            # 奖励最佳
            if eval_res["reward"] > best_reward:
                best_reward = eval_res["reward"]
                torch.save(agent.state_dict(), run_dir / f"{prefix}best_reward.pt")

        # 周期性快照
        if ep % save_every == 0:
            torch.save(agent.state_dict(), run_dir / f"{prefix}ep{ep:04d}.pt")

        # 写 CSV & 小图（每 plot_every 集自动出图；图内会“对有限值取线”）
        logger.log(ep, log_row)

    # 收尾评估（与训练期一致）
    try:
        _cases = _build_fixed_eval_cases_from_cfg(cfg)
        final_eval = evaluate_fixed(env, agent, _cases) if _cases else evaluate(env, agent, episodes=int(train_cfg.get("eval_episodes", 5)))
        print(f"\n[Final] Makespan: {final_eval['makespan']:.2f} (std={final_eval['makespan_std']:.2f})"
              f" | ΔMs: {final_eval['reward']:.2f} | Jain: {final_eval['jains_index']:.3f}")
    except Exception as e:
        print(f"[WARN] Final eval failed: {e}")

    print(f"[DONE] {human_ts()}  Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
