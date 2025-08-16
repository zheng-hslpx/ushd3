
import json, argparse, numpy as np, torch
from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
from usv_agent.ppo_policy import PPOAgent

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to agent .pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = USVEnv(cfg['env_paras'])
    model = HeterogeneousGNN(cfg['model_paras'])
    agent = PPOAgent(model, cfg['model_paras'])
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    # quick deterministic rollouts
    ms, rews = [], []
    for _ in range(5):
        s = env.reset()
        done = False
        tot = 0.0
        info = {}
        while not done:
            a, _, _ = agent.get_action(s, deterministic=True)
            s, r, done, info = env.step(a)
            tot += float(r)
        ms.append(info.get('makespan', 0.0))
        rews.append(tot)
    print(f"Avg makespan: {np.mean(ms):.3f}, Avg reward: {np.mean(rews):.3f}")

if __name__ == "__main__":
    main()
