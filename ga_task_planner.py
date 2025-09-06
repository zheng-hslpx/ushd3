# ga_task_planner.py

import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from usv_agent.usv_env import USVEnv, USVState, TaskState
# 如果需要保存/加载实例
# from data_generator import USVTaskDataGenerator

class GATaskPlanner:
    """
    基于遗传算法 (GA) 的无人船任务规划器。
    """

    def __init__(self, env_config: Dict, ga_config: Dict = None):
        """
        初始化 GA 规划器。
        :param env_config: 传递给 USVEnv 的配置。
        :param ga_config: GA 算法的配置。
        """
        self.env_config = env_config
        self.ga_config = ga_config or {}
        
        # GA 参数
        self.population_size = self.ga_config.get('population_size', 50)
        self.num_generations = self.ga_config.get('num_generations', 100)
        self.crossover_rate = self.ga_config.get('crossover_rate', 0.8)
        self.mutation_rate = self.ga_config.get('mutation_rate', 0.1)
        self.elitism_count = self.ga_config.get('elitism_count', 2) # 保留精英个体数
        self.tournament_size = self.ga_config.get('tournament_size', 3) # 锦标赛选择大小

        # 从 env_config 获取问题规模
        self.num_usvs = int(env_config['num_usvs'])
        self.num_tasks = int(env_config['num_tasks'])

        # 存储最优解
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf') # 因为我们是最小化 makespan

        # 用于存储初始的 USV 和任务数据，以便每次评估时使用相同的数据
        self.initial_usvs_data: List[USVState] = []
        self.initial_tasks_data: List[TaskState] = []

    def _evaluate_individual(self, individual: np.ndarray) -> float:
        """
        评估一个个体（染色体）的适应度。
        适应度定义为该分配方案在 USVEnv 中的最终 makespan。
        :param individual: 一个 numpy 数组，长度为 num_tasks，元素为 USV ID。
        :return: 适应度值 (makespan)。越小越好。
        """
        # 创建一个临时环境实例用于评估，使用初始数据
        # 注意：需要深拷贝数据以避免修改原始数据
        temp_usvs = [USVState(
            id=u.id, position=np.copy(u.position), battery=u.battery,
            status=u.status, current_task=u.current_task,
            available_time=u.available_time, total_distance=u.total_distance,
            work_time=u.work_time, assigned_tasks=list(u.assigned_tasks),
            last_task_position=np.copy(u.last_task_position)
        ) for u in self.initial_usvs_data]

        temp_tasks = [TaskState(
            id=t.id, position=np.copy(t.position), processing_time=t.processing_time,
            fuzzy_time=t.fuzzy_time, status=t.status, assigned_usv=t.assigned_usv,
            start_time=t.start_time, completion_time=t.completion_time
        ) for t in self.initial_tasks_data]

        env = USVEnv(self.env_config)
        env.reset(tasks_data=temp_tasks, usvs_data=temp_usvs)
        # 为了评估效率，可以关闭调试模式
        env.set_debug_mode(False) 

        # 按照个体的顺序执行任务分配
        # 为了更符合实际，我们按 USV 的 available_time 顺序来分配任务
        # 这里我们简化处理：按任务索引顺序分配
        for task_id, usv_id in enumerate(individual):
            if usv_id < 0 or usv_id >= self.num_usvs:
                # 无效分配，返回一个非常差的适应度
                return float('inf')
            
            action = usv_id * self.num_tasks + task_id
            obs, reward, done, info = env.step(action)
            
            # 如果动作无效，返回一个非常差的适应度
            if 'error' in info:
                # print(f"[GA Eval] Invalid action for individual {individual}, task {task_id} -> USV {usv_id}")
                return float('inf')
            
            # 如果所有任务都完成了，可以提前结束
            if done:
                break

        final_makespan = env.makespan
        # 检查约束合规性，如果不合规，可以增加惩罚
        metrics = env.get_travel_constraint_metrics()
        if metrics['total_violations'] > 0:
            penalty = metrics['total_violations'] * 100.0 # 可调的惩罚系数
            final_makespan += penalty
            # print(f"[GA Eval] Penalty applied: {penalty} for {metrics['total_violations']} violations.")

        return final_makespan

    def _create_initial_population(self) -> List[np.ndarray]:
        """创建初始种群。"""
        population = []
        for _ in range(self.population_size):
            # 随机分配：每个任务随机分配给一个 USV
            individual = np.random.randint(0, self.num_usvs, size=self.num_tasks)
            population.append(individual)
        return population

    def _tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """锦标赛选择。"""
        selected_indices = random.sample(range(self.population_size), self.tournament_size)
        selected_fitnesses = [fitnesses[i] for i in selected_indices]
        winner_index = selected_indices[np.argmin(selected_fitnesses)] # 选择适应度最好的 (最小 makespan)
        return np.copy(population[winner_index])

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均匀交叉 (Uniform Crossover)。"""
        if random.random() > self.crossover_rate:
            return np.copy(parent1), np.copy(parent2)
        
        child1, child2 = np.copy(parent1), np.copy(parent2)
        for i in range(self.num_tasks):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """均匀变异 (Uniform Mutation)。"""
        mutated_individual = np.copy(individual)
        for i in range(self.num_tasks):
            if random.random() < self.mutation_rate:
                # 随机选择一个新的 USV ID (不等于当前的)
                new_usv_id = random.randint(0, self.num_usvs - 1)
                # 确保变异后不与原值相同
                if new_usv_id != mutated_individual[i]:
                    mutated_individual[i] = new_usv_id
        return mutated_individual

    def _get_best_individual(self, population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, float]:
        """获取种群中适应度最好的个体。"""
        best_idx = np.argmin(fitnesses)
        return np.copy(population[best_idx]), fitnesses[best_idx]

    def train(self, seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        运行遗传算法进行训练。
        :param seed: 随机种子。
        :return: 最优个体 (染色体) 和其适应度。
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print("[GA] Initializing training...")
        # 1. 获取初始环境数据
        env_for_init = USVEnv(self.env_config)
        obs = env_for_init.reset()
        # 保存初始数据用于评估
        self.initial_usvs_data = [USVState(
            id=u.id, position=np.copy(u.position), battery=u.battery,
            status=u.status, current_task=u.current_task,
            available_time=u.available_time, total_distance=u.total_distance,
            work_time=u.work_time, assigned_tasks=list(u.assigned_tasks),
            last_task_position=np.copy(u.last_task_position)
        ) for u in env_for_init.usvs]

        self.initial_tasks_data = [TaskState(
            id=t.id, position=np.copy(t.position), processing_time=t.processing_time,
            fuzzy_time=t.fuzzy_time, status=t.status, assigned_usv=t.assigned_usv,
            start_time=t.start_time, completion_time=t.completion_time
        ) for t in env_for_init.tasks]
        
        # 2. 初始化种群
        population = self._create_initial_population()
        print(f"[GA] Initial population created with size {self.population_size}.")

        # 3. 主循环
        for generation in range(self.num_generations):
            # 评估适应度
            fitnesses = [self._evaluate_individual(ind) for ind in population]
            
            # 更新全局最优解
            current_best_ind, current_best_fit = self._get_best_individual(population, fitnesses)
            if current_best_fit < self.best_fitness:
                self.best_individual = np.copy(current_best_ind)
                self.best_fitness = current_best_fit
                print(f"[GA] Generation {generation}: New best fitness = {self.best_fitness:.2f}")

            # 选择、交叉、变异生成新种群
            new_population = []

            # 精英主义：保留最好的个体
            sorted_indices = np.argsort(fitnesses)
            for i in range(self.elitism_count):
                new_population.append(np.copy(population[sorted_indices[i]]))

            # 生成其余的个体
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        print(f"[GA] Training finished after {self.num_generations} generations.")
        print(f"[GA] Final best fitness: {self.best_fitness:.2f}")
        return self.best_individual, self.best_fitness

    def execute_best_plan(self, env: USVEnv, debug: bool = True) -> Tuple[float, Dict]:
        """
        在给定的环境上执行找到的最佳计划。
        :param env: 已经 reset 过的 USVEnv 实例。
        :param debug: 是否开启环境调试模式。
        :return: 最终 makespan 和 info 字典。
        """
        if self.best_individual is None:
            raise ValueError("No best individual found. Please run train() first.")

        env.set_debug_mode(debug)
        print("\n--- Executing Best Plan Found by GA ---")
        
        # 按照最佳个体的顺序执行任务分配
        for task_id, usv_id in enumerate(self.best_individual):
            action = usv_id * self.num_tasks + task_id
            print(f"[GA Execute] Assigning Task {task_id} to USV {usv_id} (Action: {action})")
            obs, reward, done, info = env.step(action)
            print(f"[GA Execute] Step Result - Reward: {reward:.2f}, Done: {done}, Info: {info}")
            if done:
                break
        
        final_makespan = env.makespan
        print(f"[GA Execute] Final Makespan: {final_makespan:.2f}")
        return final_makespan, info


# --- 示例运行代码 (在 VSCode 中作为脚本运行) ---
if __name__ == "__main__":
    # 1. 定义环境和 GA 配置
    env_config = {
        'num_usvs': 3,
        'num_tasks': 10,
        'map_size': [100, 100],
        'usv_speed': 5.0,
        'min_travel_time': 3.0,
        'min_travel_distance': 10.0,
        'travel_constraint_mode': 'both',
        'reward_config': {'w_makespan_improvement': 5.0, 'w_progress': 0.8},
        'dynamic_masking_config': {'enabled': True, 'max_load_ratio': 1.5}
    }

    ga_config = {
        'population_size': 30,      # 种群大小
        'num_generations': 50,      # 迭代代数
        'crossover_rate': 0.8,      # 交叉概率
        'mutation_rate': 0.1,       # 变异概率
        'elitism_count': 2,         # 精英个体数
        'tournament_size': 3        # 锦标赛选择大小
    }

    # 2. 创建规划器实例
    planner = GATaskPlanner(env_config, ga_config)

    # 3. 运行训练
    best_individual, best_fitness = planner.train(seed=42)

    # 4. 在一个新的环境实例上执行最佳计划以查看详细结果
    print("\n" + "="*40)
    print("Evaluating the best plan in detail...")
    eval_env = USVEnv(env_config)
    eval_env.reset() # 使用相同的随机种子初始化的数据
    final_makespan, _ = planner.execute_best_plan(eval_env, debug=True)

    # 5. 输出最终评估指标
    if eval_env.done:
        print("\n--- GA Execution Completed ---")
        print(f"Final Makespan (from env): {eval_env.makespan:.2f}")
        print("USV Assignments:")
        for i, usv in enumerate(eval_env.usvs):
            print(f"  USV {i}: {usv.assigned_tasks} (Total: {len(usv.assigned_tasks)})")
        
        travel_metrics = eval_env.get_travel_constraint_metrics()
        print(f"Travel Constraint Compliance Rate: {travel_metrics['travel_constraint_compliance_rate']:.2%}")
        print(f"Total Violations: {travel_metrics['total_violations']}")

        balance_metrics = eval_env.get_balance_metrics()
        print(f"Jain's Balance Index: {balance_metrics['jains_index']:.4f}")
        print(f"Task Load Variance: {balance_metrics['task_load_variance']:.2f}")

    else:
        print("\n--- GA Execution Incomplete ---")
        print("The environment did not reach 'done' state. This might indicate an issue with the plan or environment.")
