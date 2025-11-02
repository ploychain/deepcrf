# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

def print_diff_indices(pre, flop, turn, river, tol=1e-6):
    diffs = {}
    for name, a, b in [
        ("preflop→flop", pre, flop),
        ("flop→turn", flop, turn),
        ("turn→river", turn, river),
    ]:
        diff = np.where(np.abs(a - b) > tol)[0]
        diffs[name] = diff
        print(f"\n{name} changed {len(diff)} positions:")
        if len(diff):
            print(" ", diff[:20], "...")
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
    print("preflop equity 估计:", np.round(x_pre[-5:], 6))

    # ---- 2️⃣ flop ----
    # 模拟发三张公共牌
    state.public_cards = [
        pkrs.Card.from_str("Ah"),
        pkrs.Card.from_str("Kd"),
        pkrs.Card.from_str("Qs"),
    ]
    state.stage = 1
    x_flop = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Flop")
    print("flop equity 估计:", np.round(x_flop[-5:], 6))

    # ---- 3️⃣ turn ----
    state.public_cards.append(pkrs.Card.from_str("2s"))
    state.stage = 2
    x_turn = encode_state(state, 0)
    print("\nstage:", state.stage, "→ Turn")
    print("turn equity 估计:", np.round(x_turn[-5:], 6))

    # ---- 4️⃣ river ----
    state.public_cards.append(pkrs.Card.from_str("9c"))
    state.stage = 3
    x_river = encode_state(state, 0)
    print("\nstage:", state.stage, "→ River")
    print("river equity 估计:", np.round(x_river[-5:], 6))

    # ---- 对比不同阶段，找出胜率位置 ----
    print("\n==== 寻找胜率所在位置 ====")
    diff_indices = print_diff_indices(x_pre, x_flop, x_turn, x_river)
    all_changed = set(diff_indices["preflop→flop"]) | set(diff_indices["flop→turn"]) | set(diff_indices["turn→river"])
    print("\n>>> 疑似胜率槽位 indices:", sorted(all_changed))
    print("这些位置的值变化最大，很可能对应 pre/flop/turn/river 各阶段的胜率。")

    # 打印这些槽位的值演化
    print("\n值变化轨迹：")
    for idx in sorted(all_changed):
        print(f"idx {idx:3d} | pre={x_pre[idx]:.6f}  flop={x_flop[idx]:.6f}  turn={x_turn[idx]:.6f}  river={x_river[idx]:.6f}")

if __name__ == "__main__":
    main()
