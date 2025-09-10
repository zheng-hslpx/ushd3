
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from gymnasium import spaces


# ============================== 基本状态结构 ==============================

@dataclass
class USVState:
    id: int
    position: np.ndarray             # 当前位置 (x, y)
    battery: float                   # 电量（占位，未使用）
    status: str                      # 'idle' / 'working'
    current_task: Optional[int] = None
    available_time: float = 0.0      # 可接新任务的时间点
    total_distance: float = 0.0
    work_time: float = 0.0
    assigned_tasks: Optional[List[int]] = None

    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []


@dataclass
class TaskState:
    id: int
    position: np.ndarray             # 任务位置 (x, y)
    processing_time: float           # 处理时长
    fuzzy_time: Tuple[float, float, float]
    status: str                      # 'unscheduled' / 'scheduled'
    assigned_usv: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


# ============================== 环境主体 ==============================

class USVEnv:
    """USV 任务调度环境（离散动作：选择 <USV, Task> 一次分配）——极简版"""

    # ---------- 初始化 ----------
    def __init__(self, env_config: Dict[str, Any]):
        self.num_usvs: int = int(env_config['num_usvs'])
        self.num_tasks: int = int(env_config['num_tasks'])
        self.map_size: np.ndarray = np.array(env_config['map_size'], dtype=np.float32)
        self.usv_speed: float = float(env_config.get('usv_speed', 5.0))  # 仅用于时间轴推进

        # 动作空间：选择 <USV, Task>
        self.action_space = spaces.Discrete(self.num_usvs * self.num_tasks)

        # 运行时状态
        self.usvs: List[USVState] = []
        self.tasks: List[TaskState] = []
        self.schedule_history: List[Dict[str, float]] = []  # 供 Gantt/汇总使用
        self.makespan: float = 0.0
        self.done: bool = False
        self.step_count: int = 0
        self.debug_mode: bool = False  # 训练时建议关闭；需要追踪时再打开

    # ---------- 环境重置 ----------
    def reset(self,
              tasks_data: Optional[List[TaskState]] = None,
              usvs_data: Optional[List[USVState]] = None) -> Dict[str, np.ndarray]:
        """重置环境，返回初始观测"""
        self.makespan, self.done, self.step_count = 0.0, False, 0
        self.schedule_history.clear()

        # USV 初始化（默认全部在原点，available_time=0）
        self.usvs = usvs_data or [
            USVState(
                id=i,
                position=np.zeros(2, dtype=np.float32),
                battery=float('inf'),
                status='idle'
            ) for i in range(self.num_usvs)
        ]

        # Task 初始化（随机分布在地图内）
        self.tasks = tasks_data or [
            TaskState(
                id=i,
                position=np.random.uniform(0, self.map_size, 2).astype(np.float32),
                processing_time=float(np.random.uniform(8.0, 30.0)),
                fuzzy_time=(0.0, 0.0, 0.0),
                status='unscheduled'
            ) for i in range(self.num_tasks)
        ]

        self._update_current_makespan()
        if self.debug_mode:
            print(f"[DEBUG] reset() makespan={self.makespan:.2f}")

        return self._get_observation()

    # ---------- 观测 ----------
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        与上层 compute_edges 对齐的观测：
        - usv_features:  [U, 3] -> [x, y, available_time]
        - task_features: [T, 4] -> [x, y, processing_time, is_active(1/0)]
        - action_mask:   [U*T]  -> 仅允许“最早可用 USV × 未调度任务”
        """
        usv_feats = np.array([[*u.position, u.available_time] for u in self.usvs], dtype=np.float32)
        task_feats = np.array([[*t.position, t.processing_time, 1 if t.status == 'unscheduled' else 0]
                               for t in self.tasks], dtype=np.float32)
        return {
            "usv_features": usv_feats,
            "task_features": task_feats,
            "action_mask": self._compute_action_mask()
        }

    # ---------- 动作掩码（仅保留最原始规则） ----------
    def _compute_action_mask(self) -> np.ndarray:
        """
        只允许：最早可用 USV × 未调度任务
        （所有“动态均衡/旅行最小约束”等策略已移除）
        """
        mask = np.zeros(self.num_usvs * self.num_tasks, dtype=np.int8)
        if self.num_usvs == 0 or self.num_tasks == 0:
            return mask

        # 找最早可用 USV（允许并列）
        avail_times = np.array([u.available_time for u in self.usvs], dtype=np.float32)
        min_avail = float(avail_times.min()) if len(avail_times) else 0.0
        is_earliest = [abs(u.available_time - min_avail) <= 1e-6 for u in self.usvs]

        for ui, u in enumerate(self.usvs):
            if not is_earliest[ui]:
                continue
            for ti, t in enumerate(self.tasks):
                if t.status == 'unscheduled':
                    mask[ui * self.num_tasks + ti] = 1
        return mask

    # ---------- 交互一步（极简版） ----------
    def step(self, action: int):
        """
        执行动作：将某个未调度任务分配给（最早可用的）某个 USV
        约定：
        - 无“无效动作”分支：动作合法性在策略侧用 action_mask 保证；
        - 若任务已全部调度（无可选动作），环境应被上层判定 done，此处直接返回 done。
        奖励：
        - 仅使用 makespan 的相邻差值：reward = prev_ms - new_ms
        """
        # 若已完成，直接返回
        if all(t.status != "unscheduled" for t in self.tasks):
            self.done = True
            return self._get_observation(), 0.0, True, {"makespan": float(self.makespan), "step_count": int(self.step_count)}

        # 映射到 <USV,Task>
        assert 0 <= action < self.num_usvs * self.num_tasks, "动作越界（应由上层掩码避免）"
        usv_idx = int(action // self.num_tasks)
        task_idx = int(action % self.num_tasks)

        prev_ms = float(self.makespan)

        # 分配 + 刷新 makespan
        self._assign_task_to_usv(usv_idx, task_idx)
        self._update_current_makespan()

        #  极简奖励：相邻 makespan 差值
        reward = float(prev_ms - self.makespan)

        self.step_count += 1
        self.done = all(t.status != "unscheduled" for t in self.tasks)

        if self.debug_mode and (self.step_count % 5 == 0 or self.done):
            print(f"[DEBUG] step={self.step_count:03d}  prev_ms={prev_ms:.1f}  ms={self.makespan:.1f}  r={reward:.3f}")

        info = {"makespan": float(self.makespan), "step_count": int(self.step_count)}
        return self._get_observation(), reward, self.done, info

    # ---------- 时间轴维护 ----------
    def _update_current_makespan(self) -> None:
        """makespan = 所有 USV 的最大 available_time"""
        if not self.usvs:
            self.makespan = 0.0
            return
        self.makespan = max(u.available_time for u in self.usvs)

    # ---------- 任务分配（保持最小可用实现） ----------
    def _assign_task_to_usv(self, usv_idx: int, task_idx: int) -> None:
        """把 task 分配给 usv，并写入时间轴与历史（最小旅行时间=直线/速度）"""
        u, t = self.usvs[usv_idx], self.tasks[task_idx]
        if t.status != 'unscheduled':
            # 按约定，上层应已屏蔽；这里直接返回（无惩罚），保证可重入安全
            return

        decision_time = u.available_time
        travel_distance = float(np.linalg.norm(u.position - t.position))
        travel_time = travel_distance / max(self.usv_speed, 1e-6)
        start_time = decision_time + travel_time
        completion_time = start_time + float(t.processing_time)

        # 更新 USV
        u.position = t.position.copy()
        u.available_time = completion_time
        u.status = 'working'
        u.assigned_tasks.append(task_idx)

        # 更新 Task
        t.status = 'scheduled'
        t.assigned_usv = usv_idx
        t.start_time = start_time
        t.completion_time = completion_time

        # 记录历史（Gantt）
        self.schedule_history.append({
            "usv": usv_idx,
            "task": task_idx,
            "start_time": float(start_time),
            "completion_time": float(completion_time),
            "travel_time": float(travel_time),
            "travel_distance": float(travel_distance),
        })

    # ---------- 指标/调试（保留以兼容外部日志/可视化） ----------
    def get_balance_metrics(self) -> Dict[str, float]:
        """负载均衡：Jain 指数 + 任务负载方差（非奖励项，仅日志/分析用）"""
        cnts = [len(u.assigned_tasks) for u in self.usvs]
        sum_sq = sum(x * x for x in cnts) if cnts else 0.0
        jain = (sum(cnts) ** 2) / (self.num_usvs * sum_sq) if sum_sq > 0 else 1.0
        var = float(np.var(cnts) if cnts else 0.0)
        return {"jains_index": float(jain), "task_load_variance": var}

    def set_debug_mode(self, enabled: bool) -> None:
        """开/关调试打印"""
        self.debug_mode = bool(enabled)
