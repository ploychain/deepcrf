# coding: utf-8
import numpy as np
import pokers as pkrs
from src.core.model import encode_state

# ---------- 动态造牌：优先用字符串工厂；失败再尝试多种构造 ----------
_SUIT2IDX = {"s": 0, "h": 1, "d": 2, "c": 3}
_RANKS = "23456789TJQKA"  # 映射: '2'..'A'

def _try_module_level_factory(s):
    """尝试模块级的工厂函数（有些库把解析函数放在模块上而不是类方法）"""
    candidates = ["card_from_string", "parse_card", "parse"]
    for name in candidates:
        fn = getattr(pkrs, name, None)
        if callable(fn):
            try:
                c = fn(s)
                if c is not None:
                    return c
            except Exception:
                pass
    return None

def make_card(rank_char: str, suit_char: str):
    """把 'A','s' 这样的输入稳妥地变成 pkrs.Card 实例"""
    rank_char = rank_char.upper()
    suit_char = suit_char.lower()
    assert rank_char in _RANKS and suit_char in _SUIT2IDX, f"bad card {rank_char}{suit_char}"
    s = f"{rank_char}{suit_char}"
    Card = pkrs.Card

    # 1) 类方法（最常见）
    for m in ("from_string", "from_str", "parse", "make", "new", "from_repr"):
        if hasattr(Card, m):
            try:
                c = getattr(Card, m)(s)
                if c is not None:
                    return c
            except Exception:
                pass

    # 2) 直接构造 Card("As")
    try:
        return Card(s)
    except Exception:
        pass

    # 3) 模块级工厂
    c = _try_module_level_factory(s)
    if c is not None:
        return c

    # 4) (rank,suit) / (suit,rank) 两种序，rank 试 0..12 和 1..13 两套
    r0 = _RANKS.index(rank_char)              # 0..12
    r1 = r0 + 1                               # 1..13
    su = _SUIT2IDX[suit_char]                 # 0..3
    for a, b in [(r0, su), (su, r0), (r1, su), (su, r1)]:
        try:
            return Card(a, b)
        except Exception:
            pass

    # 5) from_id(suit*13 + rankX)
    for rr in (r0, r1):
        if hasattr(Card, "from_id"):
            try:
                return Card.from_id(su * 13 + rr)
            except Exception:
                pass

    # 6) 实在不行就抛清晰报错，打印可用 API 帮你定位
    raise RuntimeError(
        "无法构造 pkrs.Card —— 你的 pokers 版本构造签名不在常见集合里。\n"
        f"Card 可用属性: {dir(Card)}\n"
        "若类方法名不同（例如 `Card.fromText` 之类），把上面的 make_card 里候选列表加上即可。"
    )

# ---------- 小工具 ----------
def print_tail(name, vec, k=12):
    print(f"\n{name}:")
    print("向量长度:", len(vec))
    print("末尾数值:", np.round(vec[-k:], 6))

def locate_changed_slot(a, b, prefer_tail_from=380, eps=1e-9, topn=10):
    """找 a→b 发生变化的索引；优先靠末尾"""
    diff = np.abs(a - b)
    idx = np.where(diff > eps)[0]
    if idx.size == 0:
        return []
    scored = []
    for i in idx:
        score = (i >= prefer_tail_from, float(diff[i]), int(i))
        scored.append((score, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:topn]]

def main():
    # 固定局面：6 人桌
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)

    # 指定 Hero 手牌（不与公共牌冲突）
    hero = state.players_state[0]
    hero.hand = [make_card("A","s"), make_card("A","h")]

    # -------- Preflop --------
    state.stage = 0
    state.public_cards = []
    x_pre = encode_state(state, 0)
    print_tail("Preflop", x_pre)

    # -------- Flop --------
    state.stage = 1
    state.public_cards = [make_card("K","d"), make_card("Q","s"), make_card("2","c")]
    x_flop = encode_state(state, 0)
    print_tail("Flop", x_flop)

    # -------- Turn --------
    state.stage = 2
    state.public_cards = list(state.public_cards) + [make_card("9","c")]
    x_turn = encode_state(state, 0)
    print_tail("Turn", x_turn)

    # -------- River --------
    state.stage = 3
    # Public cards 已经 4 张；encode_state 内部 river 槽通常在 stage==3 时写入
    # 如需 5 张展示，可继续添加一张但不影响“槽位定位”的检查
    x_river = encode_state(state, 0)
    print_tail("River", x_river)

    # -------- 自动定位各阶段“胜率槽位” --------
    print("\n==== 变化槽位定位（优先靠尾部） ====")
    for name, a, b in [("pre→flop", x_pre, x_flop),
                       ("flop→turn", x_flop, x_turn),
                       ("turn→river", x_turn, x_river)]:
        idxs = locate_changed_slot(a, b, prefer_tail_from=380, eps=1e-12, topn=5)
        if idxs:
            vals = [float(b[i]) for i in idxs]
            print(f"{name}: 索引 {idxs}  对应值 {np.round(vals, 6)}")
        else:
            print(f"{name}: 未检测到变化（可能该阶段胜率还没写入向量）")

if __name__ == "__main__":
    main()
