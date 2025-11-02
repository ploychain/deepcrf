# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

def print_vec(name, vec):
    print(f"\n{name}:")
    print("末尾数值:", np.round(vec[-10:], 6))

def main():
    # 创建6人局
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=42
    )

    # ---- Preflop ----
    x_pre = encode_state(state, 0)
    print_vec("Preflop", x_pre)

    # ---- Flop ----
    state = state.proceed_to_next_stage()   # 自动发三张公共牌
    x_flop = encode_state(state, 0)
    print_vec("Flop", x_flop)

    # ---- Turn ----
    state = state.proceed_to_next_stage()   # 自动发 turn
    x_turn = encode_state(state, 0)
    print_vec("Turn", x_turn)

    # ---- River ----
    state = state.proceed_to_next_stage()   # 自动发 river
    x_river = encode_state(state, 0)
    print_vec("River", x_river)

    # ---- 比较差异 ----
    print("\n==== 胜率值变化 ====")
    diffs = {}
    stages = [("pre→flop", x_pre, x_flop), ("flop→turn", x_flop, x_turn), ("turn→river", x_turn, x_river)]
    for name, a, b in stages:
        diff_idx = np.where(np.abs(a - b) > 1e-6)[0]
        print(f"{name}: 变化 {len(diff_idx)} 个索引")
        if len(diff_idx) > 0:
            print("变化索引示例:", diff_idx[:10])

if __name__ == "__main__":
    main()
