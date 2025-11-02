# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

# ---------- 安全造牌：优先 (rank_int, suit_int)；失败再尝试 (suit_int, rank_int)；最后尝试 from_id ----------
# 约定：rank_int: 1..13 => 2..A； suit_int: 0,1,2,3 => ♠, ♥, ♦, ♣
_RANKS = "23456789TJQKA"
_SUIT2IDX = {"s": 0, "h": 1, "d": 2, "c": 3}

def make_card(rank_char: str, suit_char: str):
    rank_char = rank_char.upper()
    suit_char = suit_char.lower()
    assert rank_char in _RANKS and suit_char in _SUIT2IDX, f"bad card {rank_char}{suit_char}"

    r = _RANKS.index(rank_char) + 1         # 1..13
    s = _SUIT2IDX[suit_char]                # 0..3

    Card = pkrs.Card
    # 1) 尝试 (rank, suit)
    try:
        c = Card(r, s)
        return c
    except Exception:
        pass
    # 2) 尝试 (suit, rank)
    try:
        c = Card(s, r)
        return c
    except Exception:
        pass
    # 3) 尝试 from_id（按 encode_state 用的下标规则 suit*13 + rank）
    if hasattr(Card, "from_id"):
        try:
            cid = s * 13 + r
            return Card.from_id(cid)
        except Exception:
            pass
    raise RuntimeError("无法构造 pkrs.Card，请检查你本地 pokers 版本的 Card 构造签名")

def print_tail(name, vec, k=12):
    print(f"\n{name}:")
    print("向量长度:", len(vec))
    tail = np.round(vec[-k:], 6)
    print("末尾数值:", tail)

def stage_name(i):
    return {0:"Preflop", 1:"Flop", 2:"Turn", 3:"River"}.get(i, str(i))

def locate_changed_slot(a, b, prefer_tail_from=380, eps=1e-9, topn=10):
    """找出 a→b 的变化索引；优先展示靠近向量末尾的变化位（通常就是胜率槽位）"""
    diff = np.abs(a - b)
    idx = np.where(diff > eps)[0]
    if idx.size == 0:
        return []
    # 先按是否在尾部，再按变化幅度降序
    scored = []
    for i in idx:
        score = (i >= prefer_tail_from, float(diff[i]), int(i))
        scored.append((score, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:topn]]

def main():
    # 造一个固定状态：6 人桌
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42
    )

    # 为 hero(0) 指定两张手牌（不与公共牌冲突）
    # 这里用 AsAh，公共牌用 Kd Qs 2s 9c
    hero = state.players_state[0]
    hero.hand = [make_card("A","s"), make_card("A","h")]

    # --------------- Preflop ---------------
    state.stage = 0
    state.public_cards = []          # 必须是 pkrs.Card 列表
    x_pre = encode_state(state, 0)
    print_tail("Preflop", x_pre)

    # --------------- Flop ---------------
    state.stage = 1
    state.public_cards = [make_card("K","d"), make_card("Q","s"), make_card("2","s")]
    x_flop = encode_state(state, 0)
    print_tail("Flop", x_flop)

    # --------------- Turn ---------------
    state.stage = 2
    state.public_cards = list(state.public_cards) + [make_card("9","c")]
    x_turn = encode_state(state, 0)
    print_tail("Turn", x_turn)

    # --------------- River ---------------
    state.stage = 3
    # River 再加一张不会再加牌（5 张已满），这里只是保持流程一致
    x_river = encode_state(state, 0)
    print_tail("River", x_river)

    # --------------- 自动定位“胜率槽位”索引 ---------------
    pairs = [
        ("pre→flop", x_pre, x_flop),
        ("flop→turn", x_flop, x_turn),
        ("turn→river", x_turn, x_river),
    ]
    print("\n==== 变化槽位定位（优先靠尾部） ====")
    for name, a, b in pairs:
        idxs = locate_changed_slot(a, b, prefer_tail_from=380, eps=1e-12, topn=5)
        if idxs:
            print(f"{name}: 变化 {len(idxs)} 个；Top 索引：{idxs[:5]}")
            # 打印这些索引对应的值（b 阶段值）
            vals = [float(b[i]) for i in idxs[:5]]
            print(f"对应数值: {np.round(vals, 6)}")
        else:
            print(f"{name}: 未检测到变化（可能 encode_state 还没把该阶段胜率写入向量）")

if __name__ == "__main__":
    main()
