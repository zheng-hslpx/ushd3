from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
from gymnasium import spaces

@dataclass
class USVState:
    id: int
    position: np.ndarray
    battery: float
    status: str
    current_task: Optional[int] = None
    available_time: float = 0.0
    total_distance: float = 0.0
    work_time: float = 0.0
    assigned_tasks: Optional[List[int]] = None
    # *** 新增：记录上一个任务的位置，用于计算最小航行时间 ***
    last_task_position: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []
        if self.last_task_position is None:
            self.last_task_position = self.position.copy()

@dataclass
class TaskState:
    id: int
    position: np.ndarray
    processing_time: float
    fuzzy_time: Tuple[float, float, float]
    status: str
    assigned_usv: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

class USVEnv:
    def __init__(self, env_config: Dict):
        self.num_usvs = int(env_config['num_usvs'])
        self.num_tasks = int(env_config['num_tasks'])
        self.map_size = env_config['map_size']
        self.usv_speed = float(env_config['usv_speed'])
        # *** 新增：最小航行时间约束配置 ***
        self.min_travel_time = float(env_config.get('min_travel_time', 2.0))  # 默认最小航行时间2单位
        self.min_travel_distance = float(env_config.get('min_travel_distance', 5.0))  # 默认最小航行距离5单位
        self.travel_constraint_mode = env_config.get('travel_constraint_mode', 'time')  # 'time' 或 'distance' 或 'both'
        self.reward_config = env_config.get('reward_config', {})
        self.mask_config = env_config.get('dynamic_masking_config', {'enabled': False})
        print(f"[INFO] Reward Config: {self.reward_config}")
        print(f"[INFO] Masking Config: {self.mask_config}")
        print(f"[INFO] Travel Constraints - Min Time: {self.min_travel_time}, Min Distance: {self.min_travel_distance}, Mode: {self.travel_constraint_mode}")
        
        # 其他初始化
        self.action_space = spaces.Discrete(self.num_usvs * self.num_tasks)
        self.usvs: List[USVState] = []
        self.tasks: List[TaskState] = []
        self.makespan = 0.0
        self.done = False
        
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
        
        self.usvs = usvs_data or [
            USVState(
                id=i,
                position=np.zeros(2, dtype=np.float32),
                battery=float('inf'),
                status='idle'
            )
            for i in range(self.num_usvs)
        ]
        
        self.tasks = tasks_data or [
            TaskState(
                id=i,
                position=np.random.uniform(0, self.map_size, 2).astype(np.float32),
                processing_time=np.random.uniform(8.0, 30.0),
                fuzzy_time=(0, 0, 0),
                status='unscheduled'
            )
            for i in range(self.num_tasks)
        ]
        
        # *** 重置时初始化last_task_position ***
        for usv in self.usvs:
            usv.last_task_position = usv.position.copy()
            
        # *** 修复：初始化时计算初始makespan（应该为0，但要确保逻辑正确） ***
        self._update_current_makespan()
        
        if self.debug_mode:
            print(f"[DEBUG] Environment reset - Initial makespan: {self.makespan}")
            
        return self._get_observation()

    def _get_observation(self):
        usv_feats = np.array([
            [*u.position, u.available_time] for u in self.usvs
        ], dtype=np.float32)
        
        task_feats = np.array([
            [*t.position, t.processing_time, 1 if t.status == 'unscheduled' else 0] 
            for t in self.tasks
        ], dtype=np.float32)
        
        return {
            'usv_features': usv_feats,
            'task_features': task_feats,
            'action_mask': self._compute_action_mask()
        }

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
            is_available = self.usvs[ui].available_time <= (
                min(u.available_time for u in self.usvs if u.status != 'idle') 
                if any(u.status != 'idle' for u in self.usvs) else 0
            )
            
            if is_available:
                for ti in range(self.num_tasks):
                    if self.tasks[ti].status == 'unscheduled':
                        # *** 新增：检查是否满足最小航行时间约束 ***
                        if self._check_travel_constraint(ui, ti):
                            mask[ui * self.num_tasks + ti] = 1
                            
        return mask

    def _check_travel_constraint(self, usv_idx: int, task_idx: int) -> bool:
        """*** 新增：检查USV到任务的航行是否满足最小时间/距离约束 ***"""
        usv = self.usvs[usv_idx]
        task = self.tasks[task_idx]
        
        # 如果USV还没有执行过任务，不需要检查约束
        if len(usv.assigned_tasks) == 0:
            return True
            
        # 计算从上一个任务位置到新任务的距离和时间
        travel_distance = np.linalg.norm(usv.last_task_position - task.position)
        travel_time = travel_distance / self.usv_speed
        
        # 根据约束模式检查
        if self.travel_constraint_mode == 'time':
            return travel_time >= self.min_travel_time
        elif self.travel_constraint_mode == 'distance':
            return travel_distance >= self.min_travel_distance
        elif self.travel_constraint_mode == 'both':
            return travel_time >= self.min_travel_time and travel_distance >= self.min_travel_distance
        else:
            return True  # 如果模式未知，默认允许

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
        """重点优化makespan的奖励函数"""
        u = self.usvs[usv_idx]
        t = self.tasks[task_idx]
        
        # 1. **核心修改：强化makespan改善奖励**
        makespan_improvement = prev_makespan - self.makespan
        # 大幅提升makespan改善的权重
        makespan_reward = makespan_improvement * self.reward_config.get('w_makespan_improvement', 5.0)  # 从1.0提升到5.0
        
        # 2. **新增：makespan相对改善奖励**
        if prev_makespan > 0:
            relative_improvement = makespan_improvement / prev_makespan
            relative_reward = relative_improvement * 10.0  # 相对改善奖励
        else:
            relative_reward = 0.0
        
        # 3. 适度的基础进度奖励
        progress_reward = self.reward_config.get('w_progress', 0.8)  # 从1.8降低到0.8
        
        # 4. 轻微的距离和平衡考虑
        travel_distance = np.linalg.norm(u.position - t.position)
        max_distance = np.linalg.norm(self.map_size)
        distance_penalty = -(travel_distance / max_distance) * 0.2  # 降低权重
        
        # 5. **新增：提前奖励机制** - 越早完成越好
        completion_time = u.available_time + travel_distance/self.usv_speed + t.processing_time
        early_completion_bonus = max(0, (200 - completion_time) / 200) * 0.5
        
        # 6. **新增：关键路径奖励** - 奖励能缩短关键路径的分配
        is_on_critical_path = self._is_critical_path_task(usv_idx, task_idx)
        critical_path_bonus = 1.0 if is_on_critical_path else 0.0
        
        total_reward = (makespan_reward + relative_reward + progress_reward + 
                       distance_penalty + early_completion_bonus + critical_path_bonus)
        
        return total_reward

    def _is_critical_path_task(self, usv_idx: int, task_idx: int) -> bool:
        """判断是否在关键路径上的任务"""
        # 简化版：检查是否是当前makespan的瓶颈USV
        usv_completion_times = [u.available_time for u in self.usvs]
        max_completion_usv = np.argmax(usv_completion_times)
        return usv_idx == max_completion_usv

    def _compute_final_reward(self) -> float:
        """重点关注makespan的最终奖励"""
        self._update_current_makespan()
        
        # 1. **主导奖励：makespan**（占总奖励的60-70%）
        # 使用负指数奖励，makespan越小奖励越大
        target_makespan = 100.0  # 期望的目标makespan
        if self.makespan > 0:
            makespan_ratio = self.makespan / target_makespan
            makespan_reward = -20.0 * makespan_ratio  # 主导性奖励
        else:
            makespan_reward = -50.0  # 严重惩罚无效makespan
        
        # 2. **次要奖励：负载均衡**（占20%）
        task_counts = [len(u.assigned_tasks) for u in self.usvs]
        if len(set(task_counts)) <= 2:  # 如果任务分配比较均匀
            balance_reward = 3.0
        else:
            balance_reward = -np.var(task_counts) * 0.5
        
        # 3. **效率奖励**（占10%）
        efficiency_reward = -(self.step_count / self.num_tasks) * 0.5
        
        # 4. **有效性检查**
        validity_bonus = 2.0 if self.makespan > 0 else -20.0
        
        final_reward = makespan_reward + balance_reward + efficiency_reward + validity_bonus
        
        return final_reward

    def _assign_task_to_usv(self, usv_idx: int, task_idx: int):
        """分配任务给USV"""
        u = self.usvs[usv_idx]
        t = self.tasks[task_idx]
        
        # 决策发生的时间点是该USV的空闲时间点
        decision_time = u.available_time
        
        # *** 修改：计算航行时间时考虑最小航行时间约束 ***
        travel_distance = np.linalg.norm(u.position - t.position)
        base_travel_time = travel_distance / self.usv_speed
        
        # 应用最小航行时间约束
        actual_travel_time = base_travel_time
        if len(u.assigned_tasks) > 0:  # 如果不是第一个任务
            if self.travel_constraint_mode in ['time', 'both']:
                actual_travel_time = max(actual_travel_time, self.min_travel_time)
            # 如果是距离约束，可能需要调整速度或等待时间
            if self.travel_constraint_mode in ['distance', 'both']:
                if travel_distance < self.min_travel_distance:
                    # 如果距离不足，增加等待时间来满足最小航行时间
                    additional_time = (self.min_travel_distance - travel_distance) / self.usv_speed
                    actual_travel_time = max(actual_travel_time, base_travel_time + additional_time)
                    
        start_time = decision_time + actual_travel_time
        completion_time = start_time + t.processing_time
        
        # *** 重要：确保时间计算正确 ***
        if completion_time <= start_time or start_time < decision_time:
            print(f"[ERROR] Invalid time calculation: decision={decision_time}, start={start_time}, completion={completion_time}")
            completion_time = decision_time + actual_travel_time + t.processing_time
            
        # *** 重要：更新USV的last_task_position ***
        u.last_task_position = u.position.copy()  # 保存当前位置作为上一个任务位置
        u.position = t.position.copy()  # 更新到新任务位置
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
            'completion_time': completion_time,
            'travel_time': actual_travel_time,  # *** 新增：记录实际航行时间 ***
            'travel_distance': travel_distance  # *** 新增：记录航行距离 ***
        })
        
        if self.debug_mode:
            print(f"[DEBUG] Task {task_idx} assigned to USV {usv_idx}: start={start_time:.2f}, end={completion_time:.2f}, "
                  f"travel_time={actual_travel_time:.2f}, travel_distance={travel_distance:.2f}")

    def get_balance_metrics(self):
        """获取负载均衡指标"""
        task_counts = [len(u.assigned_tasks) for u in self.usvs]
        sum_sq = sum(x**2 for x in task_counts)
        jain = (sum(task_counts)**2) / (self.num_usvs * sum_sq) if sum_sq > 0 else 1.0
        return {'jains_index': jain, 'task_load_variance': float(np.var(task_counts))}

    def get_travel_constraint_metrics(self):
        """*** 新增：获取航行约束相关指标 ***"""
        total_violations = 0
        total_transitions = 0
        violation_details = []
        
        for usv in self.usvs:
            if len(usv.assigned_tasks) <= 1:
                continue
                
            for i in range(len(usv.assigned_tasks) - 1):
                task1_id = usv.assigned_tasks[i]
                task2_id = usv.assigned_tasks[i + 1]
                
                if task1_id < len(self.tasks) and task2_id < len(self.tasks):
                    task1_pos = self.tasks[task1_id].position
                    task2_pos = self.tasks[task2_id].position
                    travel_distance = np.linalg.norm(task1_pos - task2_pos)
                    travel_time = travel_distance / self.usv_speed
                    
                    violation = False
                    if self.travel_constraint_mode in ['time', 'both'] and travel_time < self.min_travel_time:
                        violation = True
                    if self.travel_constraint_mode in ['distance', 'both'] and travel_distance < self.min_travel_distance:
                        violation = True
                        
                    if violation:
                        total_violations += 1
                        violation_details.append({
                            'usv': usv.id,
                            'from_task': task1_id,
                            'to_task': task2_id,
                            'travel_time': travel_time,
                            'travel_distance': travel_distance
                        })
                    total_transitions += 1
                    
        compliance_rate = (total_transitions - total_violations) / total_transitions if total_transitions > 0 else 1.0
        return {
            'travel_constraint_compliance_rate': compliance_rate,
            'total_violations': total_violations,
            'total_transitions': total_transitions,
            'violation_details': violation_details
        }

    def set_debug_mode(self, enabled: bool):
        """*** 新增：控制调试模式 ***"""
        self.debug_mode = enabled