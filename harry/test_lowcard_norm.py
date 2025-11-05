# harry/test_lowcard_norm.py
import pokers as pkrs
from src.core.lowcard_on_board_norm import lowcard_on_board_norm

def card_to_str(card):
    rank_str = str(card.rank).split('.')[-1].replace('R', '')
    suit_str = str(card.suit).split('.')[-1]
    return f"{rank_str}-{suit_str}"

def show(cards):
    print(f"Board: {[card_to_str(c) for c in cards]} -> lowcard_on_board_norm = {lowcard_on_board_norm(cards):.4f}")

def main():
    # 模拟几组公共牌
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=33)
    for n in (3, 4, 5):
        state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
        while len(state.public_cards) < n and not state.final_state:
            state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
        show(state.public_cards)

if __name__ == "__main__":
    main()
