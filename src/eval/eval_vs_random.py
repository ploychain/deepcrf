import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import trange
import numpy as np
import pokers as pkrs

from src.core.deep_cfr import DeepCFRAgent
from src.agents.random_agent import RandomAgent
from src.utils.settings import set_strict_checking

def evaluate_vs_random(checkpoint_path, num_games=500, num_players=6, device=None, strict=False):
    """
    Evaluate a trained DeepCFR agent against random opponents.
    """
    set_strict_checking(strict)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    agent = DeepCFRAgent(player_id=0, num_players=num_players, device=device)
    agent.advantage_net.load_state_dict(checkpoint['advantage_net'])
    agent.strategy_net.load_state_dict(checkpoint['strategy_net'])

    random_agents = [RandomAgent(i) for i in range(num_players)]

    total_profit = 0
    completed_games = 0

    for game in trange(num_games):
        try:
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=game % num_players,
                sb=1,
                bb=2,
                stake=200.0,
                seed=game
            )

            while not state.final_state:
                current_player = state.current_player
                if current_player == agent.player_id:
                    action = agent.choose_action(state)
                else:
                    action = random_agents[current_player].choose_action(state)

                new_state = state.apply_action(action)
                if new_state.status != pkrs.StateStatus.Ok:
                    print(f"WARNING: Invalid state ({new_state.status}) in game {game}")
                    break
                state = new_state

            if state.final_state:
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1

        except Exception as e:
            print(f"Error in game {game}: {e}")

    if completed_games == 0:
        print("No valid games completed!")
        return 0.0

    avg_profit = total_profit / completed_games
    print(f"\nAverage profit per game: {avg_profit:.2f} ({completed_games}/{num_games} completed)")
    return avg_profit


if __name__ == "__main__":
    ckpt = "models/checkpoint_iter_100.pt"  # 你可以改成任意 checkpoint 文件路径
    evaluate_vs_random(ckpt, num_games=1000, num_players=6, strict=False)
