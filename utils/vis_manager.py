"""
Visualization manager for optional live plots (Visdom) and static Gantt charts.
- Visdom 连接稳健(连不上自动降级,env 固定为传入的 viz_name)
- 训练曲线首次自动建窗口，后续 append,并保存面板
- 甘特图：任务段彩色（按 USV 固定色），“航行时间”统一灰色 Transit time
- 去掉尾部补灰（避免把长时间的空闲误画成航行）
- 额外导出“USV 工作负载汇总表”
- *** 新增：甘特图对短任务的自适应文本渲染 ***
- *** 新增：支持平衡指标的实时绘图 ***
"""
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import visdom
except ImportError:
    visdom = None

class VisualizationManager:
    def __init__(self, viz_name: str, enabled: bool = True, **kwargs):
        self.enabled = bool(enabled) and visdom is not None
        self.env = viz_name
        self.viz = None
        self.plots = {}
        self.colors = self._load_colors()
        if self.enabled:
            try:
                self.viz = visdom.Visdom(env=viz_name, use_incoming_socket=False, **kwargs)
                if self.viz.check_connection():
                    self._init_windows()
                    print(f"============================================\n[INFO] Visdom Environment Name: {viz_name}\n打开浏览器访问: http://{kwargs.get('server', 'localhost')}:{kwargs.get('port', 8097)} 并切换到该 Environment\n============================================")
                else:
                    self.enabled = False
                    print("⚠️ Visdom connection failed. Live plotting disabled.")
            except Exception as e:
                self.enabled = False
                print(f"⚠️ Visdom init failed: {e}")

    def _init_windows(self):
        """
        修正: 确保key和title一致,并且与train.py中传来的字典key一致,以修复图表丢失问题
        """
        if not self.viz: return
        import torch
        
        plot_configs = {
            'train_makespan': {'title': 'Makespan'},
            'train_reward': {'title': 'Reward'},
            'actor_loss': {'title': 'Actor Loss'},
            'critic_loss': {'title': 'Critic Loss'},
            'jains_index': {'title': "Fairness (Jain's Index)"},
            'task_load_variance': {'title': 'Task Load Variance'},
        }
        for key, config in plot_configs.items():
            # 使用一个图例来区分 train 和 eval
            opts_with_legend = config.copy()
            opts_with_legend['legend'] = ['train', 'eval']
            self.plots[key] = self.viz.line(
                X=np.array([0]), Y=np.array([np.nan]), name='train', opts=opts_with_legend
            )

    def update_plots(self, episode, metrics):
        if not self.enabled: return
        import torch
        for key, value in metrics.items():
            if key in self.plots:
                self.viz.line(X=torch.tensor([episode]), Y=torch.tensor([value]), 
                              win=self.plots[key], update='append', name='train')
            
            eval_key = f"eval_{key.replace('train_', '')}"
            if eval_key in metrics and key in self.plots:
                 self.viz.line(X=torch.tensor([episode]), Y=torch.tensor([metrics[eval_key]]), 
                              win=self.plots[key], update='append', name='eval')

    def _load_colors(self) -> List[str]:
        path_options = ['./color_config.json', 'usv_agent/color_config.json', 'utils/color_config.json']
        for p in path_options:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        return json.load(f).get('gantt_color', [])
                except Exception:
                    continue
        return ["#FC5E55", "#B3E159", "#2C9CFF", "#F5D43E", "#AA5FBA", "#7780FE"]

    def _color_for_usv(self, u_id: int) -> str:
        if u_id < len(self.colors): return self.colors[u_id]
        random.seed(u_id)
        return "#" + "".join(random.choice("0123456789ABCDEF") for _ in range(6))

    def _extract_tasks(self, env) -> Tuple[Dict[int, List[Dict]], int, float, float]:
        num_usvs = int(getattr(env, "num_usvs", 0))
        makespan = float(getattr(env, "makespan", 0.0))
        text_thresh = float(getattr(env, "min_task_time_visual", 5.0))
        
        per_usv: Dict[int, List[Dict]] = {u: [] for u in range(num_usvs)}
        if hasattr(env, "schedule_history"):
            for rec in env.schedule_history:
                per_usv.setdefault(rec['usv'], []).append({
                    "task": rec["task"], "start": float(rec["start_time"]), "end": float(rec["completion_time"]),
                })
        for u in per_usv:
            per_usv[u].sort(key=lambda x: x["start"])
        return per_usv, num_usvs, makespan, text_thresh

    def generate_gantt_chart(self, env, save_path: Optional[str] = None, show: bool = False):
        per_usv, num_usvs, makespan, text_thresh = self._extract_tasks(env)

        fig, ax = plt.subplots(figsize=(16, 2 + num_usvs * 0.7))
        bar_h, transit_color = 0.5, "#D0D0D0"
        summary = []

        for u in range(num_usvs):
            items, y = per_usv.get(u, []), u
            prev_end, task_time, transit_time = 0.0, 0.0, 0.0

            for it in items:
                s, e = it["start"], it["end"]
                
                if s > prev_end:
                     ax.barh(y, width=(s - prev_end), left=prev_end, height=bar_h, 
                             color=transit_color, edgecolor='grey', alpha=0.5)
                     transit_time += (s - prev_end)
                
                task_duration = e - s
                if task_duration > 1e-6: # 只有当任务时长大于一个很小的值时才绘制
                    ax.barh(y, width=task_duration, left=s, height=bar_h,
                            color=self._color_for_usv(u), edgecolor='black', alpha=0.95)
                    
                    label = f"T{it['task']}"
                    if task_duration > text_thresh:
                        ax.text(s + task_duration / 2, y, label, ha='center', va='center',
                                color='white', fontsize=8, weight='bold')
                    else:
                        ax.text(s + task_duration / 2, y + bar_h, label, ha='center', va='bottom',
                                color='black', fontsize=7)

                task_time += task_duration
                prev_end = e

            load = (task_time / makespan) if makespan > 0 else 0.0
            summary.append((u, len(items), task_time, transit_time, load))

        ax.set_xlabel("Time", fontsize=12)
        ax.set_yticks(range(num_usvs))
        ax.set_yticklabels([f"USV {s[0]}" for s in summary])
        ax.set_title("USV Scheduling Gantt Chart", fontsize=14, weight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.7)
        if makespan > 0:
            ax.axvline(makespan, color='r', linestyle='--', linewidth=1.5, label=f"Makespan: {makespan:.1f}")

        patches = [mpatches.Patch(color=transit_color, label='Transit', alpha=0.7)]
        patches += [mpatches.Patch(color=self._color_for_usv(u), label=f'USV {u} Tasks')
                    for u, s in enumerate(summary) if s[1] > 0]
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')
        
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        if save_path:
            fig.savefig(save_path, dpi=200)
        if show:
            plt.show()
        plt.close(fig)
        return summary
    
    def save_summary_table(self, summary: List[Tuple], makespan: float, save_path: str):
        fig = plt.figure(figsize=(6, 2 + 0.25*len(summary)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        headers = ["USV", "Task#", "Task Time", "Transit Time", "Load %"]
        rows = [[str(u), str(cnt), f"{t_task:.1f}", f"{t_tran:.1f}", f"{load*100:.1f}%"]
                for u, cnt, t_task, t_tran, load in summary]
        table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        ax.set_title(f"USV Workload Summary (Makespan: {makespan:.1f})", pad=15, weight='bold')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)