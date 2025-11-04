# coding: utf-8
# harry/test_flushy_scan.py
#
# 测试 hand_flushy_potential：
#   - 用 State.from_seed 随机生成牌局
#   - 在 Flop / Turn / River 上计算 “别人成同花的概率”
#   - 打印出若干个 flushy_prob > 0 的场景

import pokers as pkrs
from src.core.hand_flushy_potential import hand_flushy_potential


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


def card_to_str(card):
    """把 pokers.Card 转成 '6♠' 这样的字符串，方便人类看"""
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


def play_one_hand(seed: int, n_players=6):
    """
    从指定 seed 开一局牌，返回在 Flop / Turn / River 的若干状态。
    """
    state = pkrs.State.from_seed(
        n_players=n_players, button=0, sb=1, bb=2, stake=200.0, seed=seed
    )
    boards = []

    guard = 2000
    while not state.final_state and guard > 0:
        # 当公共牌是 3/4/5 张时，记录一下当前状态
        if len(state.public_cards) in (3, 4, 5):
            if len(state.public_cards) == 3:
                street = "Flop"
            elif len(state.public_cards) == 4:
                street = "Turn"
            else:
                street = "River"
            boards.append((street, state))
        # 往前推进一手
        act = pick_naive_action(state)
        nxt = state.apply_action(act)
        if nxt.status != pkrs.StateStatus.Ok:
            break
        state = nxt
        guard -= 1

    return boards


def calc_flushy_on_state(state: pkrs.State, hero_id=0) -> float:
    """
    直接用当前 state 里的牌，算 hand_flushy_potential。
    """
    hero_cards = state.players_state[hero_id].hand
    board_cards = state.public_cards

    # 当前仍在局里的对手数
    n_opponents = 0
    for i, ps in enumerate(state.players_state):
        if i == hero_id:
            continue
        if getattr(ps, "active", False):
            n_opponents += 1

    return hand_flushy_potential(hero_cards, board_cards, n_opponents)


def main():
    found = 0
    max_show = 10   # 最多展示 10 个有同花威胁的场景

    for seed in range(1, 5000):
        boards = play_one_hand(seed)
        if not boards:
            continue

        for street, st in boards:
            # Flop 之前没公共牌，同花概率没意义
            if len(st.public_cards) < 3:
                continue

            p = calc_flushy_on_state(st)
            if p > 0.0:
                found += 1
                hero_hand = [card_to_str(c) for c in st.players_state[0].hand]
                board_cards = [card_to_str(c) for c in st.public_cards]

                print(f"\n=== Seed {seed} | {street} ===")
                print("Hero hand:", hero_hand)
                print(f"Board cards ({len(board_cards)}):", board_cards)
                print("hand_flushy_potential =", p)

                if found >= max_show:
                    return

    if found == 0:
        print("扫描到的牌局里，始终没有 flushy>0 的情况，这就不正常了。")


if __name__ == "__main__":
    main()
