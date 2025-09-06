# dqn_task_planner.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, Tuple, Optional, List
import json
import os

# 假设 usv_env.py 和 data_generator.py 在同一目录下
from usv_agent.usv_env import USVEnv, USVState, TaskState
# from data_generator import USVTaskDataGenerator # 如果需要保存/加载实例

# --- 1. 定义 Q-Network ---
class DQN(nn.Module):
    """
    深度 Q 网络模型。
    输入: 拼接的 USV 和任务特征向量。
    输出: 每个动作的 Q 值。
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # 输出原始 Q 值，不加激活函数

# --- 2. 定义经验回放缓冲区 ---
Experience = namedtuple('Experience', ('state_flat', 'action', 'reward', 'next_state_flat', 'done', 'mask'))

class ReplayBuffer:
    """固定大小的循环缓冲区，用于存储经验。"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_flat, action, reward, next_state_flat, done, mask):
        """保存一条经验。"""
        self.buffer.append(Experience(state_flat, action, reward, next_state_flat, done, mask))

    def sample(self, batch_size: int) -> Tuple:
        """随机采样一个批次的经验。"""
        experiences = random.sample(self.buffer, k=batch_size)
        # 使用 zip(*) 将元组列表转置为元组的列表
        batch = Experience(*zip(*experiences))
        
        # 转换为 PyTorch 张量
        state_flat = torch.FloatTensor(batch.state_flat)
        action = torch.LongTensor(batch.action).unsqueeze(1) # [batch_size, 1]
        reward = torch.FloatTensor(batch.reward).unsqueeze(1) # [batch_size, 1]
        next_state_flat = torch.FloatTensor(batch.next_state_flat)
        done = torch.BoolTensor(batch.done).unsqueeze(1) # [batch_size, 1]
        mask = torch.BoolTensor(batch.mask) # [batch_size, action_dim]

        return state_flat, action, reward, next_state_flat, done, mask

    def __len__(self):
        return len(self.buffer)

# --- 3. 定义 DQN 智能体 ---
class DQNAgent:
    """DQN 智能体，负责动作选择和学习。"""
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN Agent] Using device: {self.device}")

        # 网络
        self.q_network = DQN(state_dim, action_dim, config.get('hidden_dim', 128)).to(self.device)
        self.target_network = DQN(state_dim, action_dim, config.get('hidden_dim', 128)).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))

        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 超参数
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        self.tau = config.get('tau', 1e-3) # 软更新参数

        # 经验回放
        self.memory = ReplayBuffer(config.get('buffer_size', 10000))
        self.batch_size = config.get('batch_size', 32)

    def _flatten_obs(self, obs: Dict) -> np.ndarray:
        """将 USVEnv 的 observation 字典扁平化为向量。"""
        # 简单拼接 USV 特征和任务特征
        # 注意：这可能不是最优表示，但作为起点
        usv_flat = obs['usv_features'].flatten()
        task_flat = obs['task_features'].flatten()
        return np.concatenate([usv_flat, task_flat], axis=0)

    def act(self, obs: Dict, evaluate: bool = False) -> int:
        """
        根据当前观察选择一个动作。
        :param obs: USVEnv 的 observation。
        :param evaluate: 如果为 True，则不使用 epsilon 贪婪，直接用网络选择。
        :return: 动作索引。
        """
        state_flat = self._flatten_obs(obs)
        mask_tensor = torch.BoolTensor(obs['action_mask']).unsqueeze(0).to(self.device) # [1, action_dim]

        if not evaluate and random.random() < self.epsilon:
            # 探索：随机选择一个被允许的动作
            valid_actions = np.where(obs['action_mask'] == 1)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)
            else:
                # 如果没有有效动作，随机选一个（理论上 env 会处理）
                return random.randrange(self.action_dim)

        # 利用：使用 Q 网络选择最佳动作
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device) # [1, state_dim]
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor) # [1, action_dim]
            # 将被掩码的动作的 Q 值设为 -inf
            q_values = q_values.masked_fill(~mask_tensor, float('-inf'))
            action = q_values.max(1)[1].item() # 选择 Q 值最大的动作索引
        
        return action

    def remember(self, obs: Dict, action: int, reward: float, next_obs: Dict, done: bool):
        """将经验存储到回放缓冲区。"""
        state_flat = self._flatten_obs(obs)
        next_state_flat = self._flatten_obs(next_obs)
        mask = obs['action_mask']
        self.memory.push(state_flat, action, reward, next_state_flat, done, mask)

    def learn(self):
        """从经验回放中学习。"""
        if len(self.memory) < self.batch_size:
            return

        # 采样一批经验
        states, actions, rewards, next_states, dones, masks = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        masks = masks.to(self.device) # [batch_size, action_dim]

        # 计算当前 Q 值
        current_q_values = self.q_network(states).gather(1, actions) # [batch_size, 1]

        # 计算下一个状态的最大 Q 值 (Double DQN 风格)
        with torch.no_grad():
            # 使用当前 Q 网络选择下一个动作
            next_q_values = self.q_network(next_states) # [batch_size, action_dim]
            # 应用掩码
            next_q_values = next_q_values.masked_fill(~masks, float('-inf'))
            next_actions = next_q_values.max(1)[1].unsqueeze(1) # [batch_size, 1]
            # 使用目标网络评估这些动作的 Q 值
            target_next_q_values = self.target_network(next_states).gather(1, next_actions) # [batch_size, 1]
            target_q_values = rewards + (self.gamma * target_next_q_values * ~dones) # [batch_size, 1]

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_network(self):
        """软更新目标网络参数。"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        """衰减 epsilon。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """保存模型。"""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"[DQN Agent] Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型。"""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"[DQN Agent] Model loaded from {filepath}")


# --- 4. 定义训练器 ---
class DQNTrainer:
    """管理 DQN 训练流程。"""
    def __init__(self, env_config: Dict, dqn_config: Dict):
        self.env_config = env_config
        self.dqn_config = dqn_config

        # 创建环境
        self.env = USVEnv(self.env_config)
        self.env.set_debug_mode(False) # 训练时关闭调试信息

        # 计算状态和动作维度
        # 注意：这里我们假设 reset 后可以获取到 observation 的形状
        # 一个更健壮的方法是在 agent 初始化时传入维度
        dummy_obs = self.env.reset()
        self.state_dim = dummy_obs['usv_features'].size + dummy_obs['task_features'].size
        self.action_dim = self.env.action_space.n
        
        print(f"[DQN Trainer] State Dim: {self.state_dim}, Action Dim: {self.action_dim}")

        # 创建智能体
        self.agent = DQNAgent(self.state_dim, self.action_dim, self.dqn_config)

        # 训练参数
        self.num_episodes = self.dqn_config.get('num_episodes', 500)
        self.max_steps_per_episode = self.dqn_config.get('max_steps_per_episode', 200)
        self.update_target_every = self.dqn_config.get('update_target_every', 100) # 每 N 步更新一次目标网络
        self.learn_every = self.dqn_config.get('learn_every', 4) # 每 N 步学习一次
        self.save_every = self.dqn_config.get('save_every', 100) # 每 N 回合保存一次模型

    def train(self, seed: Optional[int] = None, model_save_path: str = "dqn_model.pth"):
        """运行训练循环。"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        scores = deque(maxlen=100) # 记录最近100回合的得分
        scores_window = [] # 记录所有回合得分
        makespans = deque(maxlen=100) # 记录最近100回合的 makespan
        makespans_window = []

        for i_episode in range(1, self.num_episodes + 1):
            obs = self.env.reset()
            score = 0
            steps = 0

            while not self.env.done and steps < self.max_steps_per_episode:
                # 选择动作
                action = self.agent.act(obs, evaluate=False)

                # 执行动作
                next_obs, reward, done, info = self.env.step(action)
                score += reward
                steps += 1

                # 存储经验
                self.agent.remember(obs, action, reward, next_obs, done)

                # 学习
                if steps % self.learn_every == 0:
                    self.agent.learn()

                # 更新目标网络
                if steps % self.update_target_every == 0:
                    self.agent.update_target_network()

                obs = next_obs

                if done:
                    break

            # 回合结束
            scores.append(score)
            scores_window.append(score)
            makespans.append(self.env.makespan)
            makespans_window.append(self.env.makespan)

            # 衰减 epsilon
            self.agent.decay_epsilon()

            # 打印进度
            if i_episode % 10 == 0:
                avg_score = np.mean(scores)
                avg_makespan = np.mean(makespans)
                print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tAverage Makespan: {avg_makespan:.2f}\tEpsilon: {self.agent.epsilon:.3f}')

            # 保存模型
            if i_episode % self.save_every == 0:
                self.agent.save(f"{model_save_path}_ep_{i_episode}.pth")

        # 训练结束，保存最终模型
        self.agent.save(model_save_path)
        print(f'\nTraining completed. Final model saved to {model_save_path}')
        print(f'Final 100 Episode Average Score: {np.mean(scores):.2f}')
        print(f'Final 100 Episode Average Makespan: {np.mean(makespans):.2f}')

        # 可选：保存训练历史
        history = {
            'scores': scores_window,
            'makespans': makespans_window,
            'final_epsilon': self.agent.epsilon
        }
        with open('training_history.json', 'w') as f:
            json.dump(history, f)
        print("Training history saved to training_history.json")

        return scores_window, makespans_window

    def evaluate(self, model_path: str, num_episodes: int = 10, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """评估训练好的模型。"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.agent.load(model_path)
        self.env.set_debug_mode(True) # 评估时开启调试信息

        scores = []
        makespans = []

        for i_episode in range(1, num_episodes + 1):
            obs = self.env.reset()
            score = 0
            steps = 0

            print(f"\n--- Evaluation Episode {i_episode} ---")
            while not self.env.done and steps < self.max_steps_per_episode:
                # 评估模式，不探索
                action = self.agent.act(obs, evaluate=True)
                print(f"[Eval] Step {steps}: Selected action {action}")

                next_obs, reward, done, info = self.env.step(action)
                score += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            scores.append(score)
            makespans.append(self.env.makespan)
            print(f"[Eval] Episode {i_episode} finished. Score: {score:.2f}, Makespan: {self.env.makespan:.2f}")
            
            # 输出最终指标
            if self.env.done:
                print("\n--- Evaluation Episode Completed ---")
                print(f"Final Makespan: {self.env.makespan:.2f}")
                print("USV Assignments:")
                for i, usv in enumerate(self.env.usvs):
                    print(f"  USV {i}: {usv.assigned_tasks} (Total: {len(usv.assigned_tasks)})")
                
                travel_metrics = self.env.get_travel_constraint_metrics()
                print(f"Travel Constraint Compliance Rate: {travel_metrics['travel_constraint_compliance_rate']:.2%}")
                print(f"Total Violations: {travel_metrics['total_violations']}")

                balance_metrics = self.env.get_balance_metrics()
                print(f"Jain's Balance Index: {balance_metrics['jains_index']:.4f}")
                print(f"Task Load Variance: {balance_metrics['task_load_variance']:.2f}")

        avg_score = np.mean(scores)
        avg_makespan = np.mean(makespans)
        print(f"\n--- Evaluation Summary ---")
        print(f"Average Score over {num_episodes} episodes: {avg_score:.2f}")
        print(f"Average Makespan over {num_episodes} episodes: {avg_makespan:.2f}")

        return scores, makespans


# --- 示例运行代码 (在 VSCode 中作为脚本运行) ---
if __name__ == "__main__":
    # 1. 定义环境和 DQN 配置
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

    dqn_config = {
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'tau': 1e-3, # 软更新
        'buffer_size': 10000,
        'batch_size': 32,
        'num_episodes': 300, # 减少回合数以便快速测试
        'max_steps_per_episode': 200,
        'update_target_every': 100,
        'learn_every': 4,
        'save_every': 100
    }

    # 2. 创建训练器实例
    trainer = DQNTrainer(env_config, dqn_config)

    # 3. 运行训练
    print("Starting DQN training...")
    scores, makespans = trainer.train(seed=42, model_save_path="dqn_model_final.pth")

    # 4. 运行评估
    print("\n" + "="*40)
    print("Starting evaluation of the trained model...")
    eval_scores, eval_makespans = trainer.evaluate("dqn_model_final.pth", num_episodes=3, seed=100)
