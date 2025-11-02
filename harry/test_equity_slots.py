# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state


def make_card(rank_char, suit_char):
    """根据 Rank/Suit 枚举安全创建 Card"""
    rank_enum = getattr(pkrs.Rank, f"R{rank_char.upper()}")
    suit_enum = {
        "s": pkrs.Suit.Spades,
        "h": pkrs.Suit.Hearts,
        "d": pkrs.Suit.Diamonds,
        "c": pkrs.Suit.Clubs,
    }[suit_char.lower()]
    return pkrs.Card(rank_enum, suit_enum)


def print_vec(name, vec):
    print(f"\n{name}:")
    print("向量长度:", len(vec))
    print("末尾数值:", np.round(vec[-10:], 6))


def main():
    # === 初始化状态 ===
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=42,
    )

    # ---- 1️⃣ Preflop ----
    x_pre = encode_state(state, 0)
    print_vec("Preflop", x_pre)

    # ---- 2️⃣ Flop ----
    state.public_cards = [
        make_card("A", "h"),
        make_card("K", "d"),
        make_card("Q", "s"),
    ]
    state.stage = 1
    x_flop = encode_state(state, 0)
    print_vec("Flop", x_flop)

    # ---- 3️⃣ Turn ----
    state.public_cards.append(make_card("2", "s"))
    state.stage = 2
    x_turn = encode_state(state, 0)
    print_vec("Turn", x_turn)

    # ---- 4️⃣ River ----
    state.public_cards.append(make_card("9", "c"))
    state.stage = 3
    x_river = encode_state(state, 0)
    print_vec("River", x_river)

    # ---- 检查胜率槽位变化 ----
    print("\n==== 胜率槽位变化检测 ====")
    stages = [
        ("pre→flop", x_pre, x_flop),
        ("flop→turn", x_flop, x_turn),
        ("turn→river", x_turn, x_river),
    ]
    for name, a, b in stages:
        diff_idx = np.where(np.abs(a - b) > 1e-6)[0]
        print(f"{name}: 变化 {len(diff_idx)} 个索引")
        if len(diff_idx) > 0:
            print(" 变化索引示例:", diff_idx[:10])


if __name__ == "__main__":
    main()
