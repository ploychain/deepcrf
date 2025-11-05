# coding: utf-8
# harry/test_highcard_norm.py
#
# 不再自己造 Card，直接用 pokers.State.from_seed 发出来的公共牌，
# 在 Flop / Turn / River 上计算 highcard_on_board_norm。

import numpy as np
import pokers as pkrs
from src.core.highcard_on_board_norm import highcard_on_board_norm


def pick_naive_action(state: pkrs.State) -> pkrs.Action:
    """极简推进策略：优先 CHECK，其次 CALL；否则用最小 BET/RAISE；最后才 FOLD。"""
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
    """把 pokers.Card 转成 'A♠' 这样的字符串，方便人类看"""
    try:
        rank_str = str(card.rank).split('.')[-1].replace('R', '')
        suit_str = str(card.suit).split('.')[-1]

        suit_symbol = {
            'Spades':   '♠',
            'Hearts':   '♥',
            'Diamonds': '♦',
            'Clubs':    '♣'
        }.get(suit_str, '?')

        rank_symbol = {
            'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
        }.get(rank_str, rank_str)

        return f"{rank_symbol}{suit_symbol}"
    except Exception:
        return str(card)


def show_stage(name: str, state: pkrs.State):
    board = state.public_cards
    board_str = [card_to_str(c) for c in board]
    val = highcard_on_board_norm(board)

    print(f"\n=== {name} ===")
    print(f"Board cards ({len(board_str)}): {board_str}")
    print(f"highcard_on_board_norm = {val:.4f}")


def main():
    # 固定 seed，确保可复现（随便挑一个，你可以换着玩）
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=44
    )

    # Preflop（无公共牌，高牌归一化应该是 0）
    show_stage("Preflop", state)

    # Flop（3 张公共牌）
    state = advance_until_board_len(state, 3)
    if state.final_state:
        print("\n[WARN] 未到 Flop 就终局了，换个 seed 再试，例如 42 或 12345。")
        return
    show_stage("Flop", state)

    # Turn（4 张公共牌）
    state = advance_until_board_len(state, 4)
    if state.final_state:
        print("\n[WARN] 未到 Turn 就终局了，换个 seed 再试。")
        return
    show_stage("Turn", state)

    # River（5 张公共牌）
    state = advance_until_board_len(state, 5)
    show_stage("River", state)


if __name__ == "__main__":
    main()
