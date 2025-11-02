# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

# ---- 兼容 Card.from_string / from_str 两种实现 ----
if hasattr(pkrs.Card, "from_str"):
    CARD_CONVERT = pkrs.Card.from_str
elif hasattr(pkrs.Card, "from_string"):
    CARD_CONVERT = pkrs.Card.from_string
else:
    raise AttributeError("未找到 Card.from_str 或 Card.from_string，请检查 pokers 库版本。")

def print_diff_indices(pre, flop, turn, river, tol=1e-6):
    diffs = {}
    for name, a, b in [
        ("preflop→flop", pre, flop),
        ("flop→turn", flop, turn),
        ("turn→river", turn, river),
    ]:
        diff = np.where(np.abs(a - b) > tol)[0]
        diffs[name] = diff
        print(f"\n{name} 发生变化的索引数量: {len(diff)}")
        if len(diff):
            print(" 变化位置示例:", diff[:20])
    return diffs

def main():
    # 创建游戏状态
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=42
    )

    # ---- 1️⃣ preflop ----
    x_pre = encode_state(state, 0)
    print("stage:", state.stage, "→ Preflop")
    print("向量长度:", len(x_pre))
    print("末尾数值:", np.round(x_pre[-10:], 6))

    # ---- 2️⃣ flop ----
    state.public_cards = [
        CARD_CONVERT("Ah"),
        CARD_CONVERT("Kd"),
        CARD_CONVERT("Qs"),
    ]
    state.stage = 1
    x_flop = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Flop")
    print("末尾数值:", np.round(x_flop[-10:], 6))

    # ---- 3️⃣ turn ----
    state.public_cards.append(CARD_CONVERT("2s"))
    state.stage = 2
    x_turn = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Turn")
    print("末尾数值:", np.round(x_turn[-10:], 6))

    # ---- 4️⃣ river ----
    state.public_cards.append(CARD_CONVERT("9c"))
    state.stage = 3
    x_river = encode_state(state, 0)
    print("\nstage:", state.stage, "→ River")
    print("末尾数值:", np.round(x_river[-10:], 6))

    # ---- 对比不同阶段变化 ----
    print("\n==== 寻找胜率所在位置 ====")
    diff_indices = print_diff_indices(x_pre, x_flop, x_turn, x_river)
    all_changed = set(diff_indices["preflop→flop"]) | set(diff_indices["flop→turn"]) | set(diff_indices["turn→river"])
    if not all_changed:
        print("\n⚠️ 未检测到明显变化，请确认 encode_state 是否已加入 flop/turn/river 胜率。")
    else:
        print("\n>>> 疑似胜率槽位 indices:", sorted(all_changed))
        print("这些位置变化最大，很可能是各阶段胜率。")
        for idx in sorted(all_changed):
            print(f"idx {idx:3d} | pre={x_pre[idx]:.6f}  flop={x_flop[idx]:.6f}  turn={x_turn[idx]:.6f}  river={x_river[idx]:.6f}")

if __name__ == "__main__":
    main()
