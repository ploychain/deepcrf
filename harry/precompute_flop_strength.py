import itertools
import random
import pandas as pd
from treys import Card, Deck, Evaluator

evaluator = Evaluator()
random.seed(42)

# ---------------- 花色规范化 ---------------- #
def canonize_flop(board_ints):
    """将3张翻牌规范化为同构代表形（忽略具体花色名称，只保留花色结构模式）"""
    triples = []
    for c in board_ints:
        r = Card.get_rank_int(c)  # 2..14
        s = Card.get_suit_int(c)  # 0..3
        triples.append((r, s, c))
    triples.sort(key=lambda x: (-x[0], x[1]))  # rank降序

    suit_seen = {}
    next_label = 0
    parts = []
    for (r, s, _) in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label
            next_label += 1
        parts.append(f"{r}{chr(ord('a') + suit_seen[s])}")
    fp = "|".join(parts)
    return fp, suit_seen


def remap_card(card_int, suit_map):
    """根据 flop 的花色映射表重新映射一张牌"""
    r = Card.get_rank_int(card_int)
    s = Card.get_suit_int(card_int)
    if s in suit_map:
        ns = suit_map[s]
    else:
        ns = max(suit_map.values(), default=-1) + 1
        suit_map[s] = ns
    return (r, ns)


def hand_type_169(hand_ints, suit_map):
    """根据花色同构分类生成 169 手牌类别名（如 AKs, QJo, 99）"""
    rc = [remap_card(c, dict(suit_map)) for c in hand_ints]
    rc.sort(key=lambda x: (-x[0], x[1]))
    r1, s1 = rc[0]
    r2, s2 = rc[1]
    rank_map = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
                9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}
    if r1 == r2:
        return f"{rank_map[r1]}{rank_map[r2]}"
    return f"{rank_map[r1]}{rank_map[r2]}{'s' if s1 == s2 else 'o'}"

# ---------------- Flop 收集 ---------------- #
def collect_canonical_flops():
    """生成 1755 个规范化 flop 代表"""
    deck = Deck()
    seen = {}
    reps = []
    for a, b, c in itertools.combinations(deck.cards, 3):
        fp, _ = canonize_flop([a, b, c])
        if fp not in seen:
            seen[fp] = (a, b, c)
            reps.append((fp, [a, b, c]))
    print(f"Collected {len(reps)} canonical flops.")
    return reps

# ---------------- 计算牌力 ---------------- #
def hs_6max(hand, board, n_sim=150):
    """计算6人桌下 hand 在 board 上的平均胜率"""
    deck = Deck()
    used = set(hand + board)
    deck.cards = [c for c in deck.cards if c not in used]

    win = tie = 0
    need = 2  # flop后要补 turn + river
    for _ in range(n_sim):
        opp_hands = [deck.draw(2) for _ in range(5)]  # 5个对手
        fill = deck.draw(need)
        full_board = board + fill
        my_score = evaluator.evaluate(full_board, hand)
        opp_scores = [evaluator.evaluate(full_board, h) for h in opp_hands]
        best_opp = min(opp_scores)
        if my_score < best_opp:
            win += 1
        elif my_score == best_opp:
            tie += 1
        # 回收卡
        deck.cards += sum(opp_hands, []) + fill
        random.shuffle(deck.cards)
    return (win + 0.5 * tie) / n_sim

# ---------------- 枚举手牌类型 ---------------- #
def list_hand_types():
    ranks = "23456789TJQKA"
    types = []
    for i, a in enumerate(ranks):
        for j, b in enumerate(ranks):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types

# ---------------- 实例化具体手牌 ---------------- #
def realize_hand_from_type(htype, board):
    deck = Deck()
    used = set(board)
    deck.cards = [c for c in deck.cards if c not in used]

    ranks = "23456789TJQKA"
    r1, r2 = htype[0], htype[1]
    suited = len(htype) == 3 and htype[2] == "s"
    offsuit = len(htype) == 3 and htype[2] == "o"

    def list_rank(r):
        return [c for c in deck.cards if Card.int_to_str(c)[0] == r]

    if r1 == r2:
        pool = list_rank(r1)
        if len(pool) < 2:
            return None
        return random.sample(pool, 2)

    pool1 = list_rank(r1)
    pool2 = list_rank(r2)
    if suited:
        for s in "cdhs":
            c1 = Card.new(r1 + s)
            c2 = Card.new(r2 + s)
            if c1 in deck.cards and c2 in deck.cards and c1 != c2:
                return [c1, c2]
        return None
    else:
        pairs = []
        for s1 in "cdhs":
            for s2 in "cdhs":
                if s1 == s2:
                    continue
                c1 = Card.new(r1 + s1)
                c2 = Card.new(r2 + s2)
                if c1 in deck.cards and c2 in deck.cards:
                    pairs.append([c1, c2])
        return random.choice(pairs) if pairs else None

# ---------------- 主程序 ---------------- #
def build_flop_table():
    reps = collect_canonical_flops()
    hand_types = list_hand_types()

    rows = []
    for htype in hand_types:
        for fp, board in reps:
            hand = realize_hand_from_type(htype, board)
            if not hand:
                continue
            s = hs_6max(hand, board, n_sim=150)
            rows.append({
                "hand_type": htype,
                "canon_flop": fp,
                "strength_mean": round(s, 6)
            })
    df = pd.DataFrame(rows)
    df.to_csv("flop_strength_6max.csv", index=False)
    print(f"✅ Saved flop_strength_6max.csv with {len(df)} rows.")

if __name__ == "__main__":
    build_flop_table()
