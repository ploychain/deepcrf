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
       - eq_flop / eq_turn / eq_river
       - hand_straighty_potential / hand_flushy_potential
       - 各种 board hint（high/low/board_gap/avg_rank/paired/straight/flush/flush_board/monotone/two_tone/rainbow）
       - spr / board_flush_possible_suits / implied_pot_odds_hint
       - preflop_equity
       - hero_position_index / num_active_players

  槽位索引（0-based），对应当前 encode_state 实现：

    0   -  51   手牌 one-hot (52)
    52  - 103   公共牌 one-hot (52)
    104 - 108   阶段 one-hot (5)
    109 - 109   彩池归一化 (1)
    110 - 115   庄位 btn (6)
    116 - 121   当前行动者 cur (6)
    122 - 122   hero_position_index
    123 - 123   num_active_players
    124 - 147   每个玩家状态 6×4
    148 - 148   min_bet
    149 - 152   legal_actions one-hot
    153 - 157   上一步动作 prev_enc

    158         eq_flop
    159         eq_turn
    160         eq_river
    161         hand_straighty_potential
    162         hand_flushy_potential
    163         highcard_on_board_norm
    164         lowcard_on_board_norm
    165         board_gap_norm
    166         avg_rank_on_board_norm
    167         paired_level
    168         straighty_hint
    169         flushy_hint
    170         flush_on_board
    171         monotone
    172         two_tone
    173         rainbow
    174         spr
    175         board_flush_possible_suits
    176         implied_pot_odds_hint
    177         preflop_equity
"""

# 关键槽位常量
IDX_HERO_POS             = 122
IDX_NUM_ACTIVE           = 123

IDX_EQ_FLOP              = 158
IDX_EQ_TURN              = 159
IDX_EQ_RIVER             = 160
IDX_HAND_STRAIGHTY       = 161
IDX_HAND_FLUSHY          = 162
IDX_HIGHCARD             = 163
IDX_LOWCARD              = 164
IDX_BOARDGAP             = 165
IDX_AVG_RANK             = 166
IDX_PAIRED_LEVEL         = 167
IDX_SHINT                = 168
IDX_FHINT                = 169
IDX_FBOARD               = 170
IDX_MONOTONE             = 171
IDX_TWO_TONE             = 172
IDX_RAINBOW              = 173
IDX_SPR                  = 174
IDX_FLUSH_SUIT_COUNT     = 175
IDX_IMPLIED_ODDS_HINT    = 176
IDX_PREFLOP_EQ           = 177

# 以 eq_flop 为中心做一个窗口查看
BASE_SLOT    = IDX_EQ_FLOP
SLOT_WINDOW  = 24   # 打印附近这么多槽位


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

    # 打印位置 & 在局人数
    print(f"hero_position_index (idx {IDX_HERO_POS}): {x[IDX_HERO_POS]:.3f}")
    print(f"num_active_players (idx {IDX_NUM_ACTIVE}): {x[IDX_NUM_ACTIVE]:.3f}")

    # 槽位窗口（围绕 eq_flop 一段）
    lo = max(0, BASE_SLOT - 5)
    hi = BASE_SLOT + SLOT_WINDOW
    window_vals = list(np.round(x[lo:hi], 6))
    print(f"\n槽位窗口 [{lo}:{hi}) =")
    print(window_vals)

    print("\n关键槽位（来自 encode_state 向量）:")
    print(f"  eq_flop              (idx {IDX_EQ_FLOP:3d}): {x[IDX_EQ_FLOP]:.6f}")
    print(f"  eq_turn              (idx {IDX_EQ_TURN:3d}): {x[IDX_EQ_TURN]:.6f}")
    print(f"  eq_river             (idx {IDX_EQ_RIVER:3d}): {x[IDX_EQ_RIVER]:.6f}")
    print(f"  hand_straighty_pot   (idx {IDX_HAND_STRAIGHTY:3d}): {x[IDX_HAND_STRAIGHTY]:.6f}")
    print(f"  hand_flushy_pot      (idx {IDX_HAND_FLUSHY:3d}): {x[IDX_HAND_FLUSHY]:.6f}")
    print(f"  highcard_on_board    (idx {IDX_HIGHCARD:3d}): {x[IDX_HIGHCARD]:.6f}")
    print(f"  lowcard_on_board     (idx {IDX_LOWCARD:3d}): {x[IDX_LOWCARD]:.6f}")
    print(f"  board_gap_norm       (idx {IDX_BOARDGAP:3d}): {x[IDX_BOARDGAP]:.6f}")
    print(f"  avg_rank_on_board    (idx {IDX_AVG_RANK:3d}): {x[IDX_AVG_RANK]:.6f}")
    print(f"  paired_level         (idx {IDX_PAIRED_LEVEL:3d}): {x[IDX_PAIRED_LEVEL]:.6f}")
    print(f"  straighty_hint       (idx {IDX_SHINT:3d}): {x[IDX_SHINT]:.6f}")
    print(f"  flushy_hint          (idx {IDX_FHINT:3d}): {x[IDX_FHINT]:.6f}")
    print(f"  flush_on_board       (idx {IDX_FBOARD:3d}): {x[IDX_FBOARD]:.6f}")
    print(f"  monotone             (idx {IDX_MONOTONE:3d}): {x[IDX_MONOTONE]:.6f}")
    print(f"  two_tone             (idx {IDX_TWO_TONE:3d}): {x[IDX_TWO_TONE]:.6f}")
    print(f"  rainbow              (idx {IDX_RAINBOW:3d}): {x[IDX_RAINBOW]:.6f}")
    print(f"  spr                  (idx {IDX_SPR:3d}): {x[IDX_SPR]:.6f}")
    print(f"  flush_suit_count     (idx {IDX_FLUSH_SUIT_COUNT:3d}): {x[IDX_FLUSH_SUIT_COUNT]:.6f}")
    print(f"  implied_pot_odds_hint(idx {IDX_IMPLIED_ODDS_HINT:3d}): {x[IDX_IMPLIED_ODDS_HINT]:.6f}")
    print(f"  preflop_equity       (idx {IDX_PREFLOP_EQ:3d}): {x[IDX_PREFLOP_EQ]:.6f}")

    print(f"\n末尾10个: {list(np.round(x[-10:], 6))}")


def main():
    # 关掉详细 debug
    set_verbose(False)

    # 固定 seed，确保可复现
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=44
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
