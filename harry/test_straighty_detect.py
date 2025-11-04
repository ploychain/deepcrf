# coding: utf-8
# harry/test_straighty_scan.py
#
# 不自己 new Card，全程用 pkrs.State 里自带的牌，
# 扫描一堆随机牌局，打印出 hand_straighty_potential > 0 的情况。

import pokers as pkrs
from src.core.model import encode_state, set_verbose
from src.core.hand_straighty_potential import hand_straighty_potential

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

def play_one_hand(seed: int, n_players=6):
    """从 seed 开一局牌，把 flop/turn/river 状态都返回出来。"""
    state = pkrs.State.from_seed(
        n_players=n_players, button=0, sb=1, bb=2, stake=200.0, seed=seed
    )
    hero_id = 0
    boards = []

    guard = 2000
    while not state.final_state and guard > 0:
        # 在每个状态检查一下当前公共牌长度
        if len(state.public_cards) in (3, 4, 5):
            # 拷贝一个状态引用 + 当前街道名
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

def calc_straighty_on_state(state: pkrs.State, hero_id=0) -> float:
    """直接用当前 state 里的牌，算 hand_straighty_potential。"""
    hero_cards = state.players_state[hero_id].hand
    board_cards = state.public_cards

    # 统计当前仍在局里的对手数
    n_opponents = 0
    for i, ps in enumerate(state.players_state):
        if i == hero_id:
            continue
        if getattr(ps, "active", False):
            n_opponents += 1

    return hand_straighty_potential(hero_cards, board_cards, n_opponents)

def main():
    set_verbose(False)

    found = 0
    max_show = 10   # 最多展示 10 个有顺子威胁的场景

    for seed in range(1, 5000):
        boards = play_one_hand(seed)
        if not boards:
            continue

        for street, st in boards:
            if len(st.public_cards) < 3:
                continue

            p = calc_straighty_on_state(st)
            if p > 0.0:
                found += 1
                print(f"\n=== Seed {seed} | {street} ===")
                print("Board cards:", st.public_cards)
                print("Hero hand:", st.players_state[0].hand)
                print("hand_straighty_potential =", p)
                # 也可以顺便看一下 encode_state 那个槽位是不是同一个值
                x = encode_state(st, player_id=0)
                # 你之前测出来 eq_flop/eq_turn/eq_river/straighty/preflop_eq 在 156~160 一带
                idx_straighty = 159
                print("encoded straighty slot =", x[idx_straighty])
                if found >= max_show:
                    return

    if found == 0:
        print("扫描到的牌局里，始终没有 straighty>0 的情况，这就不正常了。")

if __name__ == "__main__":
    main()
