import json
import numpy as np
from typing import List, Tuple, Dict
from .usv_env import USVState, TaskState

class USVTaskDataGenerator:
    def __init__(self, config: Dict):
        self.num_usvs = int(config['num_usvs'])
        self.num_tasks = int(config['num_tasks'])
        self.map_size = config['map_size']
        self.battery_capacity = float(config['battery_capacity'])
        # 允许从配置覆盖最小/最大处理时间
        self.min_processing_time = float(config.get('min_processing_time', 8.0))  # ← 最小执行时间
        self.max_processing_time = float(config.get('max_processing_time', 30.0))
        self.task_distribution = config.get('task_distribution', 'uniform')

    def generate_instance(self, seed: int = None) -> Tuple[List[USVState], List[TaskState]]:
        if seed is not None:
            np.random.seed(seed)
        usvs = [USVState(id=i, position=np.array([0.0, 0.0], dtype=np.float32),
                         battery=self.battery_capacity, status='idle')
                for i in range(self.num_usvs)]

        positions = np.random.uniform([0,0], self.map_size, size=(self.num_tasks, 2)).astype(np.float32)
        tasks = []
        for i in range(self.num_tasks):
            base = float(np.random.uniform(self.min_processing_time, self.max_processing_time))
            # 模糊三角取值，且期望也 ≥ 最小执行时间
            fuzzy = (max(self.min_processing_time*0.8, 0.8*base),
                     max(self.min_processing_time, base),
                     max(self.min_processing_time*1.2, 1.2*base))
            expected = max(self.min_processing_time, (fuzzy[0] + 2*fuzzy[1] + fuzzy[2]) / 4.0)
            tasks.append(TaskState(id=i, position=positions[i],
                                   processing_time=expected, fuzzy_time=fuzzy, status='unscheduled'))
        return usvs, tasks

    def save_instance(self, usvs: List[USVState], tasks: List[TaskState], filename: str):
        data = {
            'num_usvs': len(usvs),
            'num_tasks': len(tasks),
            'usvs': [{'id': u.id, 'position': u.position.tolist(), 'battery': u.battery} for u in usvs],
            'tasks': [{'id': t.id, 'position': t.position.tolist(),
                       'fuzzy_time': t.fuzzy_time, 'processing_time': t.processing_time} for t in tasks]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
