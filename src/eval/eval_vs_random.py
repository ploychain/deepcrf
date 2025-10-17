import torch
import numpy as np
from tqdm import trange

from pokers import ActionEnum, Stage, StateStatus
from src.training.deep_cfr import DeepCFRNet
from src.training.poker_env_wrapper import PokerEnvWrapper

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
                # DeepCFR agent
                state = env.get_state_tensor(player).to(device)
                with torch.no_grad():
                    logits = model(state)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                probs = probs[:len(legal_actions)]
                probs /= probs.sum()
                action = np.random.choice(len(legal_actions), p=probs)
            else:
                # Random agent
                action = np.random.choice(len(legal_actions))

            env.step(legal_actions[action])

        total_profit += env.get_profits()

    print("\nAverage profit per player:")
    for pid, p in enumerate(total_profit / num_games):
        print(f"Player {pid}: {p:.2f}")

    print(f"\nDeepCFR average profit vs randoms: {total_profit[0]/num_games:.2f}")

if __name__ == "__main__":
    evaluate_vs_random("models/deepcfr/deep_cfr_iter_100.pt", num_games=1000, device="cuda")
