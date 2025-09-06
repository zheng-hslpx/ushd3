"""
Visualization manager for optional live plots (Visdom) and static Gantt charts.
- Visdom 连接稳健(连不上自动降级,env 固定为传入的 viz_name)
- 训练曲线首次自动建窗口，后续 append,并保存面板
- 甘特图：任务段彩色（按 USV 固定色），"航行时间"统一灰色 Transit time
- 去掉尾部补灰（避免把长时间的空闲误画成航行）
- 额外导出"USV 工作负载汇总表"
- *** 新增：甘特图对短任务的自适应文本渲染 ***
- *** 新增：支持平衡指标的实时绘图 ***
- *** 9.2修改_序号27: 修复中文字体显示问题 ***
"""
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 9.2修改_序号28: 设置matplotlib中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
        """
        稳健的任务信息提取，基于调试版本的成功经验
        """
        num_usvs = int(getattr(env, "num_usvs", 0))
        makespan = float(getattr(env, "makespan", 0.0))
        text_thresh = float(getattr(env, "min_task_time_visual", 5.0))
        
        task_positions = {}
        
        # 使用调试版本中验证成功的提取方法
        if hasattr(env, "tasks") and env.tasks:
            for i, task in enumerate(env.tasks):
                task_id = getattr(task, 'task_id', i)
                
                # 按调试版本中成功的方法提取位置
                position = None
                for pos_attr in ['position', 'pos', 'location', 'coord']:
                    if hasattr(task, pos_attr):
                        position = getattr(task, pos_attr)
                        break
                
                if position is None and hasattr(task, 'x') and hasattr(task, 'y'):
                    position = (getattr(task, 'x'), getattr(task, 'y'))
                
                if position is not None and len(position) >= 2:
                    task_positions[task_id] = (float(position[0]), float(position[1]))
                else:
                    task_positions[task_id] = (0.0, 0.0)
        
        elif hasattr(env, 'task_features'):
            task_features = env.task_features
            if hasattr(task_features, 'shape') and len(task_features.shape) >= 2:
                if task_features.shape[1] >= 2:
                    for i in range(task_features.shape[0]):
                        task_positions[i] = (float(task_features[i, 0]), float(task_features[i, 1]))
        
        per_usv: Dict[int, List[Dict]] = {u: [] for u in range(num_usvs)}
        if hasattr(env, "schedule_history"):
            for rec in env.schedule_history:
                task_id = rec["task"]
                task_pos = task_positions.get(task_id, (0, 0))
                
                per_usv.setdefault(rec['usv'], []).append({
                    "task": task_id, 
                    "start": float(rec["start_time"]), 
                    "end": float(rec["completion_time"]),
                    "position": task_pos
                })
        
        for u in per_usv:
            per_usv[u].sort(key=lambda x: x["start"])
        
        return per_usv, num_usvs, makespan, text_thresh

    def generate_gantt_chart(self, env, save_path: Optional[str] = None, show: bool = False):
        """
        9.2修改_序号29: 修复标签显示逻辑，使用英文避免字体问题
        """
        per_usv, num_usvs, makespan, text_thresh = self._extract_tasks(env)

        fig, ax = plt.subplots(figsize=(20, 2.5 + num_usvs * 0.8))
        bar_h, transit_color = 0.6, "#D0D0D0"
        summary = []

        for u in range(num_usvs):
            items, y = per_usv.get(u, []), u
            prev_end, task_time, transit_time = 0.0, 0.0, 0.0

            for it in items:
                s, e = it["start"], it["end"]
                task_pos = it.get("position", (0, 0))
                
                # 绘制航行时间
                if s > prev_end:
                     ax.barh(y, width=(s - prev_end), left=prev_end, height=bar_h, 
                             color=transit_color, edgecolor='grey', alpha=0.5)
                     transit_time += (s - prev_end)
                
                task_duration = e - s
                if task_duration > 1e-6:
                    # 绘制任务矩形
                    ax.barh(y, width=task_duration, left=s, height=bar_h,
                            color=self._color_for_usv(u), edgecolor='black', alpha=0.95)
                    
                    # 9.2修改_序号30: 确保所有任务都显示完整信息
                    task_id = it['task']
                    pos_x, pos_y = task_pos[0], task_pos[1]
                    
                    # 统一的标签格式，确保坐标和时间信息都显示
                    if task_duration > text_thresh * 1.5:  # 长任务：矩形内显示完整信息
                        label = f"T{task_id}\n({pos_x:.0f},{pos_y:.0f})\n{s:.0f}-{e:.0f}"
                        ax.text(s + task_duration / 2, y, label, ha='center', va='center',
                                color='white', fontsize=8, weight='bold', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.4))
                                
                    elif task_duration > text_thresh * 0.8:  # 中等任务：矩形内显示ID和坐标，时间信息在上方
                        main_label = f"T{task_id}\n({pos_x:.0f},{pos_y:.0f})"
                        time_label = f"{s:.0f}-{e:.0f}"
                        
                        ax.text(s + task_duration / 2, y, main_label, ha='center', va='center',
                                color='white', fontsize=7, weight='bold')
                        ax.text(s + task_duration / 2, y + bar_h + 0.05, time_label, 
                                ha='center', va='bottom', color='darkblue', fontsize=6,
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcyan', alpha=0.8))
                                
                    else:  # 短任务：分层显示，确保所有信息都可见
                        # 主标签在矩形内
                        main_label = f"T{task_id}"
                        ax.text(s + task_duration / 2, y, main_label, ha='center', va='center',
                                color='white', fontsize=7, weight='bold')
                        
                        # 详细信息在上方，包含坐标和时间
                        detail_label = f"({pos_x:.0f},{pos_y:.0f}) {s:.0f}-{e:.0f}"
                        ax.text(s + task_duration / 2, y + bar_h + 0.08, detail_label, 
                                ha='center', va='bottom', color='darkblue', fontsize=6,
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightyellow', 
                                         alpha=0.8, edgecolor='gray', linewidth=0.5))

                task_time += task_duration
                prev_end = e

            load = (task_time / makespan) if makespan > 0 else 0.0
            summary.append((u, len(items), task_time, transit_time, load))

        # 9.2修改_序号31: 使用英文标签避免字体问题
        ax.set_xlabel("Time", fontsize=12, weight='bold')
        ax.set_ylabel("USV ID", fontsize=12, weight='bold')
        ax.set_yticks(range(num_usvs))
        ax.set_yticklabels([f"USV {s[0]} ({s[1]} tasks)" for s in summary])
        ax.set_title("USV Task Scheduling Gantt Chart - Enhanced Labels", fontsize=16, weight='bold', pad=20)
        ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
        ax.set_ylim(-0.4, num_usvs + 0.4)
        
        # 美化时间轴
        if makespan > 0:
            ax.axvline(makespan, color='red', linestyle='--', linewidth=2, 
                      label=f"Makespan: {makespan:.1f}", alpha=0.8)
        
        # 改进图例
        patches = [mpatches.Patch(color=transit_color, label='Transit Time', alpha=0.7)]
        patches += [mpatches.Patch(color=self._color_for_usv(u), label=f'USV {u} Tasks')
                    for u, s in enumerate(summary) if s[1] > 0]
        legend = ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True)
        legend.set_title("Legend", prop={'weight': 'bold'})
        
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
        plt.close(fig)
        return summary
    
    def save_summary_table(self, summary: List[Tuple], makespan: float, save_path: str):
        """
        9.2修改_序号32: 使用英文标题避免编码问题
        """
        fig = plt.figure(figsize=(10, 2.5 + 0.3*len(summary)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        headers = ["USV", "Tasks", "Work Time", "Transit Time", "Total Time", "Load %", "Efficiency %"]
        rows = []
        
        total_task_time = sum(s[2] for s in summary)
        total_transit_time = sum(s[3] for s in summary)
        
        for u, cnt, t_task, t_tran, load in summary:
            total_time = t_task + t_tran
            efficiency = (t_task / total_time * 100) if total_time > 0 else 0
            
            rows.append([
                f"USV {u}", str(cnt), f"{t_task:.1f}", f"{t_tran:.1f}", 
                f"{total_time:.1f}", f"{load*100:.1f}%", f"{efficiency:.1f}%"
            ])
        
        # 添加汇总行
        total_usv_time = sum(s[2] + s[3] for s in summary)
        avg_efficiency = (total_task_time / total_usv_time * 100) if total_usv_time > 0 else 0
        
        rows.append([
            "Total", str(sum(s[1] for s in summary)), f"{total_task_time:.1f}", 
            f"{total_transit_time:.1f}", f"{total_usv_time:.1f}", 
            f"Avg: {np.mean([s[4]*100 for s in summary]):.1f}%", 
            f"{avg_efficiency:.1f}%"
        ])
        
        # 创建表格
        table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.3, 1.6)
        
        # 美化表格
        for i, (_, cnt, _, _, load) in enumerate(summary):
            if load > 0.8:
                color = '#ffcccb'  # 浅红色 - 高负载
            elif load > 0.6:
                color = '#fff2cc'  # 浅黄色 - 中负载  
            else:
                color = '#d4edda'  # 浅绿色 - 低负载
            
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
        
        # 汇总行使用特殊颜色
        for j in range(len(headers)):
            table[(len(summary)+1, j)].set_facecolor('#e9ecef')
            table[(len(summary)+1, j)].set_text_props(weight='bold')
        
        # 表头样式
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#6c757d')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax.set_title(f"USV Workload Summary\nMakespan: {makespan:.1f} | System Efficiency: {avg_efficiency:.1f}%", 
                    pad=20, weight='bold', fontsize=14)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Summary table saved to: {save_path}")