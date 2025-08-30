from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
from gymnasium import spaces

@dataclass
class USVState:
    id: int; position: np.ndarray; battery: float; status: str
    current_task: Optional[int] = None; available_time: float = 0.0
    total_distance: float = 0.0; work_time: float = 0.0
    assigned_tasks: Optional[List[int]] = None
    def __post_init__(self):
        if self.assigned_tasks is None: self.assigned_tasks = []

@dataclass
class TaskState:
    id: int; position: np.ndarray; processing_time: float; fuzzy_time: Tuple[float, float, float]; status: str
    assigned_usv: Optional[int] = None; start_time: Optional[float] = None; completion_time: Optional[float] = None

class USVEnv:
    def __init__(self, env_config: Dict):
        self.num_usvs = int(env_config['num_usvs'])
        self.num_tasks = int(env_config['num_tasks'])
        self.map_size = env_config['map_size']
        self.usv_speed = float(env_config['usv_speed'])
        
        self.reward_config = env_config.get('reward_config', {})
        self.mask_config = env_config.get('dynamic_masking_config', {'enabled': False})
        print(f"[INFO] Reward Config: {self.reward_config}")
        print(f"[INFO] Masking Config: {self.mask_config}")

        # 其他初始化
        self.action_space = spaces.Discrete(self.num_usvs * self.num_tasks)
        self.usvs: List[USVState] = []; self.tasks: List[TaskState] = []
        self.makespan = 0.0; self.done = False
        
        # *** 新增：用于奖励计算的历史信息 ***
        self.previous_makespan = 0.0
        self.previous_unassigned_tasks = 0
        self.step_count = 0
        
        # *** 新增：调试信息 ***
        self.debug_mode = True

    def reset(self, tasks_data=None, usvs_data=None):
        """重置环境状态"""
        self.makespan = 0.0
        self.done = False
        self.schedule_history = []
        self.step_count = 0
        self.previous_makespan = 0.0
        self.previous_unassigned_tasks = self.num_tasks
        
        self.usvs = usvs_data or [USVState(id=i, position=np.zeros(2, dtype=np.float32), 
                                           battery=float('inf'), status='idle')
                                  for i in range(self.num_usvs)]
        self.tasks = tasks_data or [TaskState(id=i, position=np.random.uniform(0, self.map_size, 2).astype(np.float32),
                                               processing_time=np.random.uniform(8.0, 30.0), fuzzy_time=(0,0,0), status='unscheduled')
                                     for i in range(self.num_tasks)]
        
        # *** 修复：初始化时计算初始makespan（应该为0，但要确保逻辑正确） ***
        self._update_current_makespan()
        
        if self.debug_mode:
            print(f"[DEBUG] Environment reset - Initial makespan: {self.makespan}")
        
        return self._get_observation()

    def _get_observation(self):
        usv_feats = np.array([[*u.position, u.available_time] for u in self.usvs], dtype=np.float32)
        task_feats = np.array([[*t.position, t.processing_time, 1 if t.status=='unscheduled' else 0] for t in self.tasks], dtype=np.float32)
        return {'usv_features': usv_feats, 'task_features': task_feats, 'action_mask': self._compute_action_mask()}

    def _compute_action_mask(self):
        mask = np.zeros(self.num_usvs * self.num_tasks, dtype=np.int8)
        tasks_per_usv = [len(u.assigned_tasks) for u in self.usvs]
        avg_tasks = np.mean(tasks_per_usv) if tasks_per_usv else 0
        load_threshold = avg_tasks * self.mask_config.get('max_load_ratio', 1.5) + 1

        for ui in range(self.num_usvs):
            # *** 改进：更严格的负载均衡mask ***
            if self.mask_config.get('enabled', False):
                current_load = len(self.usvs[ui].assigned_tasks)
                min_load = min(len(u.assigned_tasks) for u in self.usvs)
                if current_load > min_load + 2:  # 不允许某个USV比最少的多超过2个任务
                    continue
            
            # 只有在USV的可用时间点，它才能接受新任务
            is_available = self.usvs[ui].available_time <= (min(u.available_time for u in self.usvs if u.status != 'idle') if any(u.status != 'idle' for u in self.usvs) else 0)

            if is_available:
                 for ti in range(self.num_tasks):
                    if self.tasks[ti].status == 'unscheduled':
                        mask[ui * self.num_tasks + ti] = 1
        return mask

    def step(self, action: int):
        """执行一个动作步骤"""
        if self._compute_action_mask()[action] == 0:
            return self._get_observation(), -10.0, True, {'error': 'invalid_action', 'makespan': self.makespan}

        usv_idx = int(action // self.num_tasks)
        task_idx = int(action % self.num_tasks)
        
        # *** 修复：保存分配前的状态用于调试 ***
        prev_makespan = self.makespan
        
        # *** 核心修复：先分配任务，再计算奖励 ***
        self._assign_task_to_usv(usv_idx, task_idx)
        
        # *** 关键修复：立即更新当前makespan ***
        self._update_current_makespan()
        
        # *** 修复：使用更合理的奖励函数 ***
        reward = self._compute_improved_reward(usv_idx, task_idx, prev_makespan)
        
        self.step_count += 1
        self.done = all(t.status != 'unscheduled' for t in self.tasks)
        
        # *** 修复：最终奖励调整 ***
        if self.done:
            final_reward = self._compute_final_reward()
            reward += final_reward
            
            # *** 重要：最终验证makespan合理性 ***
            if self.makespan <= 0 and any(t.processing_time > 0 for t in self.tasks):
                print(f"[ERROR] Invalid final makespan: {self.makespan}")
                reward -= 100.0  # 严重惩罚无效的makespan
        
        # *** 新增：makespan约束检查 ***
        if self.makespan < 0:
            print(f"[ERROR] Negative makespan detected: {self.makespan}")
            reward -= 50.0
            self.makespan = max(u.available_time for u in self.usvs) if self.usvs else 0.0
        
        if self.debug_mode and self.step_count % 5 == 0:
            print(f"[DEBUG] Step {self.step_count}: makespan={self.makespan:.2f}, reward={reward:.2f}")
        
        info = {'makespan': float(self.makespan), 'step_count': self.step_count}
        return self._get_observation(), float(reward), bool(self.done), info

    def _update_current_makespan(self):
        """*** 新增：实时更新当前makespan ***"""
        if not self.usvs:
            self.makespan = 0.0
            return
            
        # makespan = 所有USV中最大的完成时间
        all_completion_times = [u.available_time for u in self.usvs]
        self.makespan = max(all_completion_times) if all_completion_times else 0.0
        
        # *** 重要：确保makespan不会无故为0 ***
        if self.makespan <= 0 and any(len(u.assigned_tasks) > 0 for u in self.usvs):
            # 如果有任务被分配但makespan为0，这是异常情况
            print(f"[WARNING] Makespan inconsistency detected. Recalculating...")
            self._recalculate_makespan()

    def _recalculate_makespan(self):
        """*** 新增:重新计算makespan以修复异常 ***"""
        max_completion = 0.0
        for usv in self.usvs:
            if usv.assigned_tasks:
                # 找到最后一个任务的完成时间
                last_completion = 0.0
                for task_id in usv.assigned_tasks:
                    if task_id < len(self.tasks) and self.tasks[task_id].completion_time:
                        last_completion = max(last_completion, self.tasks[task_id].completion_time)
                max_completion = max(max_completion, last_completion)
            max_completion = max(max_completion, usv.available_time)
        
        self.makespan = max_completion
        print(f"[INFO] Makespan recalculated: {self.makespan}")

    def _compute_improved_reward(self, usv_idx: int, task_idx: int, prev_makespan: float) -> float:
        """*** 平衡修复版本的奖励函数 ***"""
        u = self.usvs[usv_idx]
        t = self.tasks[task_idx]
        
        # 1. *** 平衡修复：适中的基础进度奖励 ***
        progress_reward = self.reward_config.get('w_progress', 1.8)  # 提升到1.8，接近原来但不过高
        
        # 2. *** 平衡修复：温和的makespan改善奖励 ***
        makespan_improvement = prev_makespan - self.makespan
        makespan_reward = makespan_improvement * self.reward_config.get('w_makespan_improvement', 1.0)  # 降低到1.0
        
        # 3. 距离惩罚：鼓励选择近距离任务
        travel_distance = np.linalg.norm(u.position - t.position)
        max_distance = np.linalg.norm(self.map_size)
        distance_penalty = -(travel_distance / max_distance) * self.reward_config.get('w_distance', 0.4)
        
        # 4. 负载均衡奖励：鼓励任务分配均匀
        task_counts = [len(usv.assigned_tasks) for usv in self.usvs]
        current_count = task_counts[usv_idx]
        min_count = min(task_counts)
        max_count = max(task_counts)
        
        balance_reward = 0.0
        if current_count == min_count:
            balance_reward = self.reward_config.get('w_balance', 0.6)
        elif current_count == max_count and max_count > min_count:
            balance_reward = -self.reward_config.get('w_balance', 0.6)
        
        # 5. 效率奖励：鼓励短处理时间的任务优先
        efficiency_reward = -(t.processing_time / 30.0) * self.reward_config.get('w_efficiency', 0.4)
        
        # 6. *** 平衡修复：大幅减轻makespan惩罚 ***
        makespan_penalty = -(self.makespan / 200.0) * self.reward_config.get('w_makespan_penalty', 0.2)  # 从1.0降到0.2
        
        total_reward = (progress_reward + makespan_reward + distance_penalty + 
                       balance_reward + efficiency_reward + makespan_penalty)
        
        return total_reward

    def _compute_final_reward(self) -> float:
        """*** 平衡修复版本的最终奖励 ***"""
        # *** 重要：最终检查并确保makespan正确 ***
        self._update_current_makespan()
        
        # 1. *** 关键：温和的Makespan奖励 ***
        makespan_reward = -self.makespan / self.reward_config.get('makespan_normalization', 200.0) * 8.0  # 降低权重
        
        # 2. 负载均衡奖励
        task_counts = [len(u.assigned_tasks) for u in self.usvs]
        sum_sq = sum(x**2 for x in task_counts)
        jain_index = (sum(task_counts)**2) / (self.num_usvs * sum_sq) if sum_sq > 0 else 1.0
        balance_reward = jain_index * self.reward_config.get('w_final_balance', 4.0)
        
        # 3. 效率奖励（步数越少越好）
        efficiency_reward = -(self.step_count / self.num_tasks) * self.reward_config.get('w_step_efficiency', 0.3)
        
        # 4. 方差惩罚
        variance_penalty = -np.var(task_counts) * self.reward_config.get('w_variance', 0.2)
        
        # *** 平衡修复：大幅减轻validity检查惩罚 ***
        validity_bonus = 0.0
        min_possible_makespan = max(t.processing_time for t in self.tasks) if self.tasks else 0
        if self.makespan >= min_possible_makespan:
            validity_bonus = 1.0  # 小幅奖励而不是大幅
        else:
            validity_bonus = -3.0  # 从-20减轻到-3
        
        final_reward = makespan_reward + balance_reward + efficiency_reward + variance_penalty + validity_bonus
        
        if self.debug_mode:
            print(f"[DEBUG] Final rewards - makespan: {makespan_reward:.2f}, balance: {balance_reward:.2f}, "
                  f"efficiency: {efficiency_reward:.2f}, variance: {variance_penalty:.2f}, validity: {validity_bonus:.2f}")
            print(f"[DEBUG] Final makespan: {self.makespan:.2f}, Jain index: {jain_index:.3f}")
        
        return final_reward

    def _assign_task_to_usv(self, usv_idx: int, task_idx: int):
        """分配任务给USV"""
        u = self.usvs[usv_idx]
        t = self.tasks[task_idx]
        
        # 决策发生的时间点是该USV的空闲时间点
        decision_time = u.available_time
        
        travel_time = np.linalg.norm(u.position - t.position) / self.usv_speed
        start_time = decision_time + travel_time
        completion_time = start_time + t.processing_time
        
        # *** 重要：确保时间计算正确 ***
        if completion_time <= start_time or start_time < decision_time:
            print(f"[ERROR] Invalid time calculation: decision={decision_time}, start={start_time}, completion={completion_time}")
            completion_time = decision_time + travel_time + t.processing_time
        
        u.position = t.position.copy()  # 确保位置更新
        u.available_time = completion_time
        u.status = 'working'
        u.assigned_tasks.append(task_idx)
        
        t.status = 'scheduled'
        t.assigned_usv = usv_idx
        t.start_time = start_time
        t.completion_time = completion_time

        self.schedule_history.append({
            'usv': usv_idx, 
            'task': task_idx, 
            'start_time': start_time, 
            'completion_time': completion_time
        })
        
        if self.debug_mode:
            print(f"[DEBUG] Task {task_idx} assigned to USV {usv_idx}: start={start_time:.2f}, end={completion_time:.2f}")
        
    def get_balance_metrics(self):
        """获取负载均衡指标"""
        task_counts = [len(u.assigned_tasks) for u in self.usvs]
        sum_sq = sum(x**2 for x in task_counts)
        jain = (sum(task_counts)**2) / (self.num_usvs * sum_sq) if sum_sq > 0 else 1.0
        return {'jains_index': jain, 'task_load_variance': float(np.var(task_counts))}

    def set_debug_mode(self, enabled: bool):
        """*** 新增：控制调试模式 ***"""
        self.debug_mode = enabled