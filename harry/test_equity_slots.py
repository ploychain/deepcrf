# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

# === 通用安全的 Card 构造函数 ===
def make_card(rank_char: str, suit_char: str):
    """通用兼容构造，支持 enum 型 pokers.Card"""
    try:
        # 优先尝试 from_string
        if hasattr(pkrs.Card, "from_string"):
            return pkrs.Card.from_string(rank_char + suit_char)
        elif hasattr(pkrs.Card, "from_str"):
            return pkrs.Card.from_str(rank_char + suit_char)
        else:
            # 枚举方式构造
            rank_enum = getattr(pkrs.Rank, f"R{rank_char.upper()}")
            suit_map = {"s": pkrs.Suit.Spades, "h": pkrs.Suit.Hearts, "d": pkrs.Suit.Diamonds, "c": pkrs.Suit.Clubs}
            suit_enum = suit_map[suit_char.lower()]
            return pkrs.Card(rank_enum, suit_enum)
    except Exception as e:
        raise RuntimeError(f"❌ 构造 Card 失败: {rank_char}{suit_char} | {e}")

def print_diff_indices(pre, flop, turn, river, tol=1e-6):
    diffs = {}
    for name, a, b in [
        ("preflop→flop", pre, flop),
        ("flop→turn", flop, turn),
        ("turn→river", turn, river),
    ]:
        diff = np.where(np.abs(a - b) > tol)[0]
        diffs[name] = diff
        print(f"\n{name} 变化数量: {len(diff)}")
        if len(diff):
            print(" 变化索引示例:", diff[:20])
    return diffs

def main():
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=42
    )

    # ---- 1️⃣ Preflop ----
    x_pre = encode_state(state, 0)
    print("stage:", state.stage, "→ Preflop")
    print("向量长度:", len(x_pre))
    print("末尾数值:", np.round(x_pre[-10:], 6))

    # ---- 2️⃣ Flop ----
    state.public_cards = [
        make_card("A", "h"),
        make_card("K", "d"),
        make_card("Q", "s"),
    ]
    state.stage = 1
    x_flop = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Flop")
    print("末尾数值:", np.round(x_flop[-10:], 6))

    # ---- 3️⃣ Turn ----
    state.public_cards.append(make_card("2", "s"))
    state.stage = 2
    x_turn = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Turn")
    print("末尾数值:", np.round(x_turn[-10:], 6))

    # ---- 4️⃣ River ----
    state.public_cards.append(make_card("9", "c"))
    state.stage = 3
    x_river = encode_state(state, 0)
    print("\nstage:", state.stage, "→ River")
    print("末尾数值:", np.round(x_river[-10:], 6))

    # ---- 对比不同阶段 ----
    print("\n==== 寻找胜率所在位置 ====")
    diff_indices = print_diff_indices(x_pre, x_flop, x_turn, x_river)
    all_changed = set(diff_indices["preflop→flop"]) | set(diff_indices["flop→turn"]) | set(diff_indices["turn→river"])
    if not all_changed:
        print("\n⚠️ 未检测到变化，请确认 encode_state 是否加入 flop/turn/river 胜率逻辑。")
    else:
        print("\n>>> 疑似胜率索引:", sorted(all_changed))
        for idx in sorted(all_changed):
            print(f"idx {idx:3d} | pre={x_pre[idx]:.6f}  flop={x_flop[idx]:.6f}  turn={x_turn[idx]:.6f}  river={x_river[idx]:.6f}")

if __name__ == "__main__":
    main()
