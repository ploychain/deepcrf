# coding: utf-8
import pokers as pkrs
from src.core.lowcard_on_board_norm import lowcard_on_board_norm

def pick_naive_action(state: pkrs.State) -> pkrs.Action:
    """极简推进策略：优先 CHECK，其次 CALL；否则最小 BET/RAISE；否则 FOLD。"""
    AE = pkrs.ActionEnum
    la = list(state.legal_actions)

    def has(ae):
        try:
            return any((a is ae) or (int(a) == int(ae)) for a in la)
        except Exception:
            return ae in la

    if has(AE.Check):
        return pkrs.Action(AE.Check)
    if has(AE.Call):
        return pkrs.Action(AE.Call)
    if has(AE.Bet):
        return pkrs.Action(AE.Bet, max(1, int(state.min_bet)))
    if has(AE.Raise):
        return pkrs.Action(AE.Raise, max(1, int(state.min_bet)))
    if hasattr(AE, "AllIn") and has(AE.AllIn):
        return pkrs.Action(AE.AllIn, 0)
    return pkrs.Action(AE.Fold)

def advance_until_board_len(state: pkrs.State, target_len: int) -> pkrs.State:
    """执行最朴素动作，直到公共牌张数达到 target_len（3/4/5）或终局。"""
    guard = 2000
    while (not state.final_state) and len(state.public_cards) < target_len and guard > 0:
        act = pick_naive_action(state)
        nxt = state.apply_action(act)
        if nxt.status != pkrs.StateStatus.Ok:
            raise RuntimeError(f"状态无效: {nxt.status}, 动作: {act}")
        state = nxt
        guard -= 1
    return state

def card_to_str(card):
    rank_str = str(card.rank).split('.')[-1].replace('R', '')
    suit_str = str(card.suit).split('.')[-1]
    return f"{rank_str}-{suit_str}"

def show_stage(name: str, state: pkrs.State):
    board = state.public_cards
    board_str = [card_to_str(c) for c in board]
    val = lowcard_on_board_norm(board)
    print(f"\n=== {name} ===")
    print(f"Board cards ({len(board_str)}): {board_str}")
    print(f"lowcard_on_board_norm = {val:.4f}")

def main():
    # 固定 seed 确保可复现
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=33
    )

    # Preflop
    show_stage("Preflop", state)

    # Flop (3 张公共牌)
    state = advance_until_board_len(state, 3)
    show_stage("Flop", state)

    # Turn (4 张)
    state = advance_until_board_len(state, 4)
    show_stage("Turn", state)

    # River (5 张)
    state = advance_until_board_len(state, 5)
    show_stage("River", state)

if __name__ == "__main__":
    main()
