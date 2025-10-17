import torch
from tqdm import trange
from pokers import Game, ActionEnum, Stage
from src.training.deep_cfr import DeepCFRNet   # 按你的项目路径修改
from src.training.poker_env_wrapper import PokerEnvWrapper  # 按实际导入路径修改
import numpy as np

def evaluate_vs_random(model_path, num_games=1000, device="cpu"):
    print(f"Loading model from {model_path}")
    model = DeepCFRNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = PokerEnvWrapper()
    total_profit = np.zeros(env.num_players)
    print(f"Evaluating {num_games} games...")

    for g in trange(num_games):
        env.reset()
        while not env.is_done():
            player = env.current_player
            legal_actions = env.get_legal_actions()
            if player == 0:
                # DeepCFR 玩家
                state = env.get_state_tensor(player).to(device)
                with torch.no_grad():
                    logits = model(state)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                action = np.random.choice(len(legal_actions), p=probs[:len(legal_actions)]/sum(probs[:len(legal_actions)]))
            else:
                # 随机玩家
                action = np.random.choice(len(legal_actions))
            env.step(legal_actions[action])

        total_profit += env.get_profits()

    print("Average profit per player:")
    for pid, p in enumerate(total_profit / num_games):
        print(f"Player {pid}: {p:.2f}")

    print(f"\nDeepCFR average profit vs randoms: {total_profit[0]/num_games:.2f}")

if __name__ == "__main__":
    evaluate_vs_random("models/deepcfr/deep_cfr_iter_100.pt", num_games=1000, device="cuda")

