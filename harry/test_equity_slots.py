# coding: utf-8
# harry/test_equity_slots.py
import numpy as np
import pokers as pkrs
from src.core.model import encode_state, set_verbose

# ====== 配置：根据你当前 encode_state 的布局计算 equity 槽位基准 ======
# 你给的 encode_state（只含 preflop_equity）模块长度之和：
# 52(hand) + 52(board) + 5(stage) + 1(pot) + 6(button) + 6(current) +
# 24(6人*4) + 1(min_bet) + 4(legal) + 5(prev_action) = 156
BASE_EQ_SLOT = 156  # preflop_equity 的 index
SLOT_WINDOW = 8     # 打印 8 个槽位窗口，便于你看紧邻的 flop/turn/river 如果有的话

def pick_naive_action(state: pkrs.State) -> pkrs.Action:
    """
    极简推进策略：优先 CHECK，其次 CALL；必要时用最小 BET/RAISE。
    目的只是把牌局推进到下一阶段（发公共牌），不追求策略合理性。
    """
    AE = pkrs.ActionEnum
    la = set(state.legal_actions)

    if AE.Check in la:
        return pkrs.Action(AE.Check)
    if AE.Call in la:
        return pkrs.Action(AE.Call)
    # 为了避免整手牌直接结束，尽量不 Fold
    if AE.Bet in la:
        return pkrs.Action(AE.Bet, state.min_bet)
    if AE.Raise in la:
        # 粗暴选一个最小加注（很多实现接受 min_bet 作为 raise 增量）
        return pkrs.Action(AE.Raise, state.min_bet)
    # 实在没办法才 Fold
    return pkrs.Action(AE.Fold)

def advance_until_board_len(state: pkrs.State, target_len: int) -> pkrs.State:
    """
    一直执行最朴素动作，直到 public_cards 达到 target_len（3/4/5）或终局。
    """
    guard = 2000  # 防止意外死循环
    while not state.final_state and len(state.public_cards) < target_len and guard > 0:
        act = pick_naive_action(state)
        nxt = state.apply_action(act)
        if nxt.status != pkrs.StateStatus.Ok:
            raise RuntimeError(f"状态无效: {nxt.status}, 动作: {act}")
        state = nxt
        guard -= 1
    return state

def show_stage(name: str, state: pkrs.State):
    x = encode_state(state, player_id=0)
    print(f"\n{name}:")
    print(f"向量长度: {len(x)}")
    # 打印 BASE_EQ_SLOT 附近的窗口，观察 preflop 以及紧邻的 flop/turn/river 槽位
    lo = max(0, BASE_EQ_SLOT - 2)
    hi = BASE_EQ_SLOT + SLOT_WINDOW
    win = x[lo:hi]
    print(f"索引 [{lo}:{hi}) 槽位窗口 = {list(np.round(win, 6))}")
    # 也打印末尾 10 个，避免你误以为 equity 在末尾（你的实现里不是在末尾）
    print(f"末尾10个: {list(np.round(x[-10:], 6))}")

def main():
    set_verbose(False)  # 如需看 encode_state 内部 debug，可设 True

    # 固定一个 seed，保证可复现
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42
    )

    # === Preflop ===
    show_stage("Preflop", state)

    # === Flop (3张公共牌) ===
    state = advance_until_board_len(state, 3)
    if state.final_state:
        print("\n[WARN] 还没发到 Flop 就终局了，换个 seed 再试。")
        return
    show_stage("Flop", state)

    # === Turn (第4张公共牌) ===
    state = advance_until_board_len(state, 4)
    if state.final_state:
        print("\n[WARN] 还没发到 Turn 就终局了，换个 seed 再试。")
        return
    show_stage("Turn", state)

    # === River (第5张公共牌) ===
    state = advance_until_board_len(state, 5)
    if state.final_state:
        # 正常会发到 River 才摊牌并结算
        pass
    show_stage("River", state)

if __name__ == "__main__":
    main()
