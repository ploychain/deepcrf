# coding: utf-8
# harry/test_royal_flush_potential.py
#
# 扫描一批随机牌局，在 Flop / Turn / River 上
# 打印 hand_royal_flush_potential > 0 的场景。

import pokers as pkrs
from src.core.hand_royal_flush_potential import hand_royal_flush_potential


def pick_naive_action(state: pkrs.State) -> pkrs.Action:
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


def card_to_str(card):
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

from contextlib import contextmanager
import sys
import os
@contextmanager
def suppress_stdout():
    """临时屏蔽 stdout（pokers 里那些 Winner id / Ranks 的调试输出）"""
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def play_one_hand(seed: int, n_players=6):
    boards = []

    # ✅ 把整段跟引擎交互的逻辑放到静音区间里
    with suppress_stdout():
        state = pkrs.State.from_seed(
            n_players=n_players, button=0, sb=1, bb=2, stake=200.0, seed=seed
        )
        guard = 2000
        while not state.final_state and guard > 0:
            if len(state.public_cards) in (3, 4, 5):
                if len(state.public_cards) == 3:
                    street = "Flop"
                elif len(state.public_cards) == 4:
                    street = "Turn"
                else:
                    street = "River"
                boards.append((street, state))

            act = pick_naive_action(state)
            nxt = state.apply_action(act)
            if nxt.status != pkrs.StateStatus.Ok:
                break
            state = nxt
            guard -= 1

    return boards


def calc_royal_on_state(state: pkrs.State, hero_id=0) -> float:
    hero_cards = state.players_state[hero_id].hand
    board_cards = state.public_cards

    n_opponents = 0
    for i, ps in enumerate(state.players_state):
        if i == hero_id:
            continue
        if getattr(ps, "active", False):
            n_opponents += 1

    return hand_royal_flush_potential(hero_cards, board_cards, n_opponents)


def main():
    found = 0
    max_show = 1000

    for seed in range(1, 100000):
        boards = play_one_hand(seed)
        if not boards:
            continue

        for street, st in boards:
            if len(st.public_cards) < 3:
                continue

            p = calc_royal_on_state(st)
            # 皇家同花顺本来就极罕见，概率会非常小，这里只要 > 0 就打印出来
            if p > 0.006:
                found += 1
                hero_hand = [card_to_str(c) for c in st.players_state[0].hand]
                board_cards = [card_to_str(c) for c in st.public_cards]

                print(f"\n=== Seed {seed} | {street} ===")
                print("Hero hand:", hero_hand)
                print(f"Board cards ({len(board_cards)}):", board_cards)
                print("hand_royal_flush_potential =", p)

                if found >= max_show:
                    return

    if found == 0:
        print("扫描到的牌局里，始终没有 royal_flush_prob > 0 的情况，这就不正常了。")


if __name__ == "__main__":
    main()
