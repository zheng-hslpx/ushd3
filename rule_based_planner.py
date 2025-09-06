# rule_based_planner.py

import numpy as np
from typing import List, Tuple, Optional
from usv_agent.usv_env import USVEnv, USVState, TaskState

class RuleBasedPlanner:
    """
    一个基于规则的无人船任务规划器，模拟 COLREGS 理念。
    策略：
    1. 优先分配：优先考虑距离远或可能导致约束冲突的任务。
    2. USV选择：选择满足约束且对 makespan 影响最小的 USV。
    3. 负载均衡：在满足前两者基础上，考虑 USV 负载。
    """

    def __init__(self, env: USVEnv, config: dict = None):
        """
        初始化规划器。
        :param env: USVEnv 实例。
        :param config: 规划器配置。
        """
        self.env = env
        self.config = config or {}
        # 可配置的权重，用于排序
        self.distance_weight = self.config.get('distance_weight', 1.0)
        self.constraint_weight = self.config.get('constraint_weight', 1.5) # 优先处理可能违反约束的任务
        self.load_balance_weight = self.config.get('load_balance_weight', 0.5)

    def plan_step(self) -> Optional[int]:
        """
        规划并返回一个动作。
        :return: 动作 (action index) 或 None (如果无有效动作或已完成)。
        """
        if self.env.done:
            print("[Planner] Environment is done.")
            return None

        # 1. 获取当前环境状态和动作掩码
        obs = self.env._get_observation() # 直接调用内部方法获取当前状态
        mask = obs['action_mask']
        
        if not np.any(mask):
            print("[Planner] No valid actions available according to mask.")
            # 这可能意味着任务都已分配但环境未检测到 done，或约束过严
            return None 

        # 2. 获取 USV 和任务的当前状态
        usvs: List[USVState] = self.env.usvs
        tasks: List[TaskState] = self.env.tasks

        # 3. 筛选未调度的任务
        unscheduled_tasks = [t for t in tasks if t.status == 'unscheduled']
        if not unscheduled_tasks:
             print("[Planner] No unscheduled tasks found.")
             return None # 理论上 env.done 应该为 True

        # 4. 筛选“可用”的 USV (这里简化为 earliest available)
        #    更复杂的规则可以考虑负载均衡
        available_usvs = sorted(usvs, key=lambda u: u.available_time)

        # 5. 对任务进行排序 (优先级)
        #    策略：优先处理可能导致航行约束问题或距离远的任务
        def task_priority(task: TaskState) -> float:
            score = 0.0
            min_distance = float('inf')
            constraint_risk = 0 # 假设风险与距离负相关

            for usv in usvs:
                dist = np.linalg.norm(task.position - usv.position)
                min_distance = min(min_distance, dist)
                # 简单模拟约束风险：距离越近，风险越高
                if dist < max(self.env.min_travel_distance, self.env.usv_speed * self.env.min_travel_time):
                     constraint_risk += 1
            
            # 距离越远，优先级可能越高 (鼓励分散)，但也要考虑约束风险
            score = self.distance_weight * min_distance - self.constraint_weight * constraint_risk
            return score

        sorted_tasks = sorted(unscheduled_tasks, key=task_priority, reverse=True)

        # 6. 对 USV 进行排序 (适合度)
        #    策略：选择分配后对 makespan 影响最小，且负载较轻的 USV
        def usv_suitability(usv: USVState, task: TaskState) -> float:
            # 计算分配此任务给此 USV 后的预计完成时间
            # 这是一个简化的估计，不完全精确，但足以用于排序
            travel_distance = np.linalg.norm(task.position - usv.position)
            base_travel_time = travel_distance / self.env.usv_speed
            
            # 应用环境中的约束检查逻辑来估算实际航行时间
            actual_travel_time = base_travel_time
            if len(usv.assigned_tasks) > 0: # 如果不是第一个任务，则需要检查约束
                temp_env = self.env # 为了调用 _check_travel_constraint，我们直接使用 env
                # 注意：这里直接模拟检查，不修改 env 状态
                temp_travel_time = base_travel_time
                temp_travel_distance = travel_distance
                temp_mode = self.env.travel_constraint_mode

                if temp_mode in ['time', 'both'] and temp_travel_time < self.env.min_travel_time:
                    temp_travel_time = self.env.min_travel_time
                if temp_mode in ['distance', 'both'] and temp_travel_distance < self.env.min_travel_distance:
                    # 简化处理：如果距离不足，认为需要等待或调整
                    temp_travel_time = max(temp_travel_time, self.env.min_travel_time) 
                
                actual_travel_time = temp_travel_time

            estimated_completion_time = usv.available_time + actual_travel_time + task.processing_time
            
            # 计算对 makespan 的潜在影响 (越小越好)
            current_makespan_estimate = self.env.makespan # 使用当前环境的 makespan 估计
            potential_makespan_impact = max(0, estimated_completion_time - current_makespan_estimate)

            # 考虑负载均衡 (任务数越少越好)
            load_penalty = len(usv.assigned_tasks) * self.load_balance_weight

            # 适合度分数：影响小 + 负载轻 (都是负向影响，所以总分越小越好，但我们要的是“适合度”高，所以取负)
            suitability_score = -(potential_makespan_impact + load_penalty)
            return suitability_score

        # 7. 寻找最佳有效动作
        # 遍历排序后的任务和 USV，找到第一个有效的动作
        for task in sorted_tasks:
            sorted_usvs_for_task = sorted(available_usvs, key=lambda u: usv_suitability(u, task), reverse=True)
            for usv in sorted_usvs_for_task:
                action = usv.id * self.env.num_tasks + task.id
                if mask[action] == 1: # 检查动作是否被掩码允许
                    if self.env.debug_mode:
                        print(f"[Planner] Selected Action: Assign Task {task.id} to USV {usv.id} (Action Index: {action})")
                    return action
        
        # 如果循环结束都没找到，说明 mask 和我们的逻辑有出入
        print("[Planner] Warning: Could not find a valid action despite mask allowing some. This might indicate a logic mismatch.")
        # 可以选择返回第一个允许的动作作为 fallback
        fallback_action = np.argmax(mask)
        if mask[fallback_action] == 1:
             print(f"[Planner] Fallback to first valid action: {fallback_action}")
             return fallback_action

        return None # 真的没有有效动作


# --- 示例运行代码 (在 VSCode 中作为脚本运行) ---
if __name__ == "__main__":
    # 1. 定义环境和规划器配置
    env_config = {
        'num_usvs': 3,
        'num_tasks': 10,
        'map_size': [100, 100],
        'usv_speed': 5.0,
        'min_travel_time': 3.0,
        'min_travel_distance': 10.0,
        'travel_constraint_mode': 'both', # 同时启用时间和距离约束
        'reward_config': {'w_makespan_improvement': 5.0, 'w_progress': 0.8},
        'dynamic_masking_config': {'enabled': True, 'max_load_ratio': 1.5}
    }

    planner_config = {
        'distance_weight': 1.0,
        'constraint_weight': 2.0, # 更强调处理约束风险
        'load_balance_weight': 0.3
    }

    # 2. 创建环境实例
    env = USVEnv(env_config)
    env.set_debug_mode(True) # 启用调试信息

    # 3. 生成任务数据 (可选，也可以让 env.reset() 自动生成)
    # from data_generator import USVTaskDataGenerator
    # gen_config = {
    #     'num_usvs': env_config['num_usvs'],
    #     'num_tasks': env_config['num_tasks'],
    #     'map_size': env_config['map_size'],
    #     'battery_capacity': float('inf'),
    #     'min_processing_time': 5.0,
    #     'max_processing_time': 25.0
    # }
    # generator = USVTaskDataGenerator(gen_config)
    # usvs_data, tasks_data = generator.generate_instance(seed=42)
    # obs = env.reset(tasks_data=tasks_data, usvs_data=usvs_data) # 使用生成的数据

    # 4. 重置环境 (使用随机数据)
    obs = env.reset()

    # 5. 创建规划器实例
    planner = RuleBasedPlanner(env, planner_config)

    # 6. 运行规划循环
    step = 0
    max_steps = env.num_tasks * 2 # 预估最大步数
    try:
        while not env.done and step < max_steps:
            print(f"\n--- Step {step} ---")
            action = planner.plan_step()

            if action is None:
                print("[Main] Planner returned no action. Ending simulation.")
                break

            print(f"[Main] Executing action: {action}")
            obs, reward, done, info = env.step(action)
            print(f"[Main] Step Result - Reward: {reward:.2f}, Done: {done}, Info: {info}")
            step += 1

    except KeyboardInterrupt:
        print("\n[Main] Simulation interrupted by user.")

    # 7. 输出最终结果
    if env.done:
        print("\n--- Simulation Completed ---")
        print(f"Final Makespan: {env.makespan:.2f}")
        print("USV Assignments:")
        for i, usv in enumerate(env.usvs):
            print(f"  USV {i}: {usv.assigned_tasks} (Total: {len(usv.assigned_tasks)})")
        
        # 输出航行约束检查结果
        travel_metrics = env.get_travel_constraint_metrics()
        print(f"Travel Constraint Compliance Rate: {travel_metrics['travel_constraint_compliance_rate']:.2%}")
        print(f"Total Violations: {travel_metrics['total_violations']}")

        balance_metrics = env.get_balance_metrics()
        print(f"Jain's Balance Index: {balance_metrics['jains_index']:.4f}")
        print(f"Task Load Variance: {balance_metrics['task_load_variance']:.2f}")

    else:
        print("\n--- Simulation Stopped (Incomplete) ---")
        print(f"Reason: Reached max steps ({max_steps}) or no valid action.")
        print(f"Current Makespan Estimate: {env.makespan:.2f}")
        unassigned_count = sum(1 for t in env.tasks if t.status == 'unscheduled')
        print(f"Unassigned Tasks: {unassigned_count}/{env.num_tasks}")
