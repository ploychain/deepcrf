# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state  # 用你已经实现的 encode_state

# 只用 Card.from_string，尝试几种常见写法
_SUITS = {"s": "S", "h": "H", "d": "D", "c": "C"}
_UNICODE = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

def make_card(r: str, s: str):
    r = r.upper()
    s = s.lower()
    Card = pkrs.Card
    assert hasattr(Card, "from_string"), "你的 pokers.Card 没有 from_string 方法"

    candidates = [
        f"{r}{s}",                 # Ah / As
        f"{r}{s.upper()}",         # AH / AS
        f"{r}{_SUITS[s]}",         # A + S/H/D/C
        f"{r}{_UNICODE[s]}",       # A♠ A♥ A♦ A♣
    ]
    last_err = None
    for txt in candidates:
        try:
            c = Card.from_string(txt)
            if c is not None:
                return c
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Card.from_string 解析失败，试过: {candidates}，最后错误: {last_err}")

def tail(name, vec, k=10):
    print(f"\n{name}:")
    print("向量长度:", len(vec))
    print("末尾数值:", np.round(vec[-k:], 6))

def main():
    # 固定一局 6 人桌
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)

    # 指定 Hero 手牌（避免与公共牌冲突即可）
    hero = state.players_state[0]
    hero.hand = [make_card("A","s"), make_card("A","h")]

    # ------- Preflop -------
    state.stage = 0
    state.public_cards = []
    x_pre = encode_state(state, 0)
    tail("Preflop", x_pre)

    # ------- Flop -------
    state.stage = 1
    state.public_cards = [make_card("K","d"), make_card("Q","s"), make_card("2","c")]
    x_flop = encode_state(state, 0)
    tail("Flop", x_flop)

    # ------- Turn -------
    state.stage = 2
    state.public_cards = state.public_cards + [make_card("9","c")]
    x_turn = encode_state(state, 0)
    tail("Turn", x_turn)

    # ------- River -------
    state.stage = 3
    # River 这里不强制指定第5张，encode_state 若在 river 槽写入值，向量应变化
    x_river = encode_state(state, 0)
    tail("River", x_river)

    # 简单对比各阶段向量是否变化（主要看靠尾部是否有改动）
    for a_name, a, b_name, b in [
        ("Preflop", x_pre, "Flop", x_flop),
        ("Flop", x_flop, "Turn", x_turn),
        ("Turn", x_turn, "River", x_river),
    ]:
        diff_idx = np.where(np.abs(a - b) > 1e-12)[0]
        print(f"\n{a_name} → {b_name} 变化的索引数: {len(diff_idx)}")
        if len(diff_idx):
            tail_idx = [i for i in diff_idx if i >= 380]  # 偏向尾部
            show = tail_idx[-10:] if len(tail_idx) else diff_idx[-10:]
            print(f"示例索引: {show}")
            print(f"{b_name} 这些索引的值:", np.round(b[show], 6))

if __name__ == "__main__":
    main()
