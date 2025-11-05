# coding: utf-8
# harry/test_input_slots.py

import numpy as np
import pokers as pkrs
from src.core.model import encode_state, set_verbose

"""
说明：
  这段脚本的目的：
    1. 检查 encode_state 输出向量长度是否固定（pad 到 500）
    2. 看几个关键槽位的值是否符合预期：
       - eq_flop  / eq_turn / eq_river
       - hand_straighty_potential（你新加的“别人已成顺概率”）
       - preflop_equity（起手权益）

  索引约定（0-based）：
    假设在 eq_flop / eq_turn / eq_river 后面
    插入了 hand_straighty_potential，然后才是 preflop_equity：

      eq_flop_index              = BASE_EQ_SLOT - 3
      eq_turn_index              = BASE_EQ_SLOT - 2
      eq_river_index             = BASE_EQ_SLOT - 1
      hand_straighty_potential   = BASE_EQ_SLOT
      preflop_equity             = BASE_EQ_SLOT + 1

    原来测出来 preflop_equity 在 159，现在多加一个槽位，就变成：

      BASE_EQ_SLOT              = 159   # hand_straighty_potential
      PREFLOP_EQ_SLOT           = 160
"""

# hand_straighty_potential 的槽位
BASE_EQ_SLOT        = 159   # straighty_potential 在这里
PREFLOP_EQ_SLOT     = BASE_EQ_SLOT + 1
SLOT_WINDOW         = 8     # 打印附近几个槽位


def pick_naive_action(state: pkrs.State) -> pkrs.Action:
    """
    极简推进策略：优先 CHECK，其次 CALL；否则用最小 BET/RAISE；最后才 FOLD。
    关键点：不要把 legal_actions 放进 set（ActionEnum 不可哈希）。
    """
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
    """把 pokers.Card 转成 '6♠' 这样的字符串，方便人类看"""
    try:
        # rank: R2/R3/.../RT/RJ/RQ/RK/RA 取后半段去掉 R
        rank_str = str(card.rank).split('.')[-1].replace('R', '')
        # suit: Spades/Hearts/Diamonds/Clubs
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
    x = encode_state(state, player_id=0)
    print(f"\n=== {name} ===")
    print(f"向量长度: {len(x)}")

    # 打印 hero 手牌和公共牌
    hero_hand = [card_to_str(c) for c in state.players_state[0].hand]
    board_cards = [card_to_str(c) for c in state.public_cards]
    print(f"Hero hand: {hero_hand}")
    print(f"Board cards ({len(board_cards)}): {board_cards}")

    # 槽位窗口
    lo = max(0, BASE_EQ_SLOT - 5)
    hi = BASE_EQ_SLOT + SLOT_WINDOW
    window_vals = list(np.round(x[lo:hi], 6))
    print(f"槽位窗口 [{lo}:{hi}) =")
    print(window_vals)

    # 关键槽位索引（BASE_EQ_SLOT 代表 straighty 的索引）
    idx_eq_flop = BASE_EQ_SLOT - 3
    idx_eq_turn = BASE_EQ_SLOT - 2
    idx_eq_river = BASE_EQ_SLOT - 1
    idx_straighty = BASE_EQ_SLOT
    idx_flushy = BASE_EQ_SLOT + 1
    idx_highcard = BASE_EQ_SLOT + 2
    idx_preflop = BASE_EQ_SLOT + 3

    print("\n关键槽位（来自 encode_state 向量）:")
    print(f"  eq_flop     (idx {idx_eq_flop}):    {x[idx_eq_flop]:.6f}")
    print(f"  eq_turn     (idx {idx_eq_turn}):    {x[idx_eq_turn]:.6f}")
    print(f"  eq_river    (idx {idx_eq_river}):   {x[idx_eq_river]:.6f}")
    print(f"  straighty   (idx {idx_straighty}):  {x[idx_straighty]:.6f}")
    print(f"  flushy      (idx {idx_flushy}):     {x[idx_flushy]:.6f}")
    print(f"  highcard    (idx {idx_highcard}):   {x[idx_highcard]:.6f}")
    print(f"  preflop_eq  (idx {idx_preflop}):    {x[idx_preflop]:.6f}")

    print(f"\n末尾10个: {list(np.round(x[-10:], 6))}")


def main():
    # 关掉详细 debug
    set_verbose(False)

    # 固定 seed，确保可复现（你可以改成 4、42 等自己玩）
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=33
    )

    # === Preflop ===
    show_stage("Preflop", state)

    # === Flop (3 张公共牌) ===
    state = advance_until_board_len(state, 3)
    if state.final_state:
        print("\n[WARN] 未到 Flop 就终局了，换个 seed 再试，例如把 seed 改成 12345。")
        return
    show_stage("Flop", state)

    # === Turn (第 4 张公共牌) ===
    state = advance_until_board_len(state, 4)
    if state.final_state:
        print("\n[WARN] 未到 Turn 就终局了，换个 seed 再试。")
        return
    show_stage("Turn", state)

    # === River (第 5 张公共牌) ===
    state = advance_until_board_len(state, 5)
    show_stage("River", state)


if __name__ == "__main__":
    main()
