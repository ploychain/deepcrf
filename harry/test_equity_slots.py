# coding: utf-8
# harry/test_equity_slots.py
import numpy as np
import pokers as pkrs
from src.core.model import encode_state, set_verbose

# === 依据你刚才的实际输出，preflop equity 在 159 ===
BASE_EQ_SLOT = 159          # 你的 encode_state 当前版本里 preflop equity 的索引
SLOT_WINDOW  = 8            # 打印附近几个槽位

def pick_naive_action(state: pkrs.State) -> pkrs.Action:
    """
    极简推进策略：优先 CHECK，其次 CALL；否则用最小 BET/RAISE；最后才 FOLD。
    关键点：不要把 legal_actions 放进 set（ActionEnum 不可哈希）。
    """
    AE = pkrs.ActionEnum
    la = list(state.legal_actions)  # ✅ 不用 set
    def has(ae):  # 更稳的包含判断
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
    # 某些实现存在 AllIn
    if hasattr(AE, "AllIn") and has(AE.AllIn):
        return pkrs.Action(AE.AllIn, 0)
    return pkrs.Action(AE.Fold)

def advance_until_board_len(state: pkrs.State, target_len: int) -> pkrs.State:
    """执行最朴素动作，直到公共牌张数达到 target_len（3/4/5）或终局。"""
    guard = 2000
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
    lo = max(0, BASE_EQ_SLOT - 5)
    hi = BASE_EQ_SLOT + SLOT_WINDOW
    print(f"索引 [{lo}:{hi}) 槽位窗口 = {list(np.round(x[lo:hi], 6))}")
    print(f"末尾10个: {list(np.round(x[-10:], 6))}")

def main():
    set_verbose(False)

    # 固定 seed，确保可复现
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42
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
