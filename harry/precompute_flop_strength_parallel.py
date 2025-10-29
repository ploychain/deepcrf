import argparse, itertools, random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
from treys import Card, Deck, Evaluator

# -------------------- 全局参数 --------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # 用来生成新牌
evaluator = Evaluator()
random.seed(42)

# 1) 建立 treys 的 suit_int -> 实际花色字符 映射
#    treys 的 suit_int 不是 0/1/2/3，而是 bitmask-ish (1,2,4,8)
SUIT_INT_TO_CHAR = {}
for ch in SUITS:  # 'c','d','h','s'
    test_card = Card.new("2" + ch)
    si = Card.get_suit_int(test_card)
    SUIT_INT_TO_CHAR[si] = ch
# 现在，比如 SUIT_INT_TO_CHAR[1] = 'c', SUIT_INT_TO_CHAR[2] = 'd', etc.

def suit_int_to_char(si):
    return SUIT_INT_TO_CHAR[si]

def new(rank_char, suit_char):
    return Card.new(rank_char + suit_char)

def card_rank_int(c):
    return Card.get_rank_int(c)   # 2..14

def card_suit_int(c):
    return Card.get_suit_int(c)   # e.g. 1,2,4,8 (NOT 0..3)

# -------------------- 起手牌 169 --------------------
def hand_types_169():
    types = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types  # 169

# -------------------- flop 规范化成 key --------------------
def canonicalize_flop(cards3):
    """
    给一个真实 flop (三张 treys 牌 int)，输出 canonical key:
    - rank_part: "14,13,12"
    - suit_part: "aab"/"aba"/...
    注意：排序和重标花色只用于“给这个flop一个桶的key”，不影响胜率。
    """
    triples = [(card_rank_int(c), card_suit_int(c), c) for c in cards3]
    # rank 降序, suit_int 升序只是为了稳定排序
    triples.sort(key=lambda t: (-t[0], t[1]))

    ranks_sorted = [t[0] for t in triples]

    # 第一次出现的花色 -> 'a','b','c'
    suit_seen = {}
    next_label = 0
    labels = []
    for _, s_int, _ in triples:
        if s_int not in suit_seen:
            suit_seen[s_int] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s_int]))

    rank_key = ",".join(str(r) for r in ranks_sorted)
    suit_key = "".join(labels)
    return f"{rank_key}|{suit_key}"

def enumerate_canonical_flops():
    """
    穷举所有真实flop (C(52,3)=22100)，
    对每个flop生成 canonical key，去重，保存代表。
    返回列表: [(key, (c1,c2,c3), suit_pattern)]
    """
    deck = [new(r, s) for r in RANKS for s in SUITS]
    seen = {}
    for c1, c2, c3 in itertools.combinations(deck, 3):
        flop_cards = (c1, c2, c3)
        key = canonicalize_flop(flop_cards)
        if key not in seen:
            seen[key] = flop_cards

    rows = []
    for key, flop_cards in seen.items():
        suit_pat = key.split("|")[1]  # 'aaa','aab','aba','abb','abc', etc.
        rows.append((key, flop_cards, suit_pat))
    return rows  # ~1911 项

# -------------------- suit-signature 生成 --------------------
def all_target_signatures(flop_pattern, htype_kind):
    """
    给定 flop 的花色pattern（'aaa','aab','aba','abb','abc'）
    和 手牌类别 ('pair','suited','offsuit')，
    返回这类组合理论上可能出现的 signature 集合（如 'aa','ax','xx','ab','ba',...）
    """
    # 哪些花色标签在 flop 里存在
    avail = {
        "aaa": ['a'],
        "aab": ['a','b'],
        "aba": ['a','b'],
        "abb": ['a','b'],
        "abc": ['a','b','c'],
    }
    L = avail[flop_pattern]
    Lx = L + ['x']  # 'x' = 不在flop里的花色

    if htype_kind == 'suited':  # AKs: 两张同花 => token必须一样
        return {t+t for t in Lx}  # {'aa','bb','cc','xx'} (裁剪到存在的字母)

    elif htype_kind == 'offsuit':  # AKo: 两张不同花 => token必须不一样
        sigs = set()
        for a in Lx:
            for b in Lx:
                if a != b:
                    sigs.add(a+b)  # 有序对
        return sigs

    else:  # pair: AA 口袋对子 -> 两张不同花，但同rank
        # 近似：允许所有不同花组合 + 'xx'
        sigs = set()
        for a in Lx:
            for b in Lx:
                if a != b:
                    sigs.add(a+b)
        sigs.add("xx")
        return sigs

def realize_hand_for_signature(htype, flop, target_sig):
    """
    根据:
      - htype: 'AKs','AKo','AA' 这种
      - flop: 三张真实牌 (treys ints)
      - target_sig: 比如 'aa','ax','ba','xx'，表示手牌两张的花色和翻牌花色关系
    构造一手具体两张牌(hand=[c0,c1])，不与flop冲突。
    如果构造不出来，返回 None。
    """
    used = set(flop)

    # 把 flop 按 rank 降序并依次给 suit_int -> 'a','b','c' 做映射
    triples = [(card_rank_int(c), card_suit_int(c), c) for c in flop]
    triples.sort(key=lambda t: (-t[0], t[1]))
    suit_seen = {}
    next_label = 0
    for _, s_int, _ in triples:
        if s_int not in suit_seen:
            suit_seen[s_int] = next_label
            next_label += 1
    # suit_seen: {suit_int: 0/1/2}, 0->'a',1->'b',2->'c'

    # flop 使用的实际花色集合（treys suit_int）
    flop_suit_set = set(card_suit_int(c) for c in flop)

    # 给定 signature 中的一个 token ('a','b','c','x')，返回可用的 treys suit_int 列表
    def suit_candidates_for_token(token):
        if token == 'x':
            # 'x' = 不在 flop 的花色
            return [si for si in SUIT_INT_TO_CHAR.keys() if si not in flop_suit_set]
        else:
            # token 是 'a'/'b'/'c' -> 找所有 suit_int 映射到这个label
            desired_idx = ord(token) - ord('a')  # 0,1,2
            return [si for si, idx in suit_seen.items() if idx == desired_idx]

    # 解析 htype
    r_hi = htype[0]
    r_lo = htype[1]
    is_pair = (r_hi == r_lo)
    is_suited = (len(htype) == 3 and htype[2] == 's')
    is_offsuit = (len(htype) == 3 and htype[2] == 'o')

    # 排序手牌rank，保证第一张是高rank（因为我们在signature里假设顺序是按rank高到低）
    if is_pair:
        ranks_ordered = [r_hi, r_lo]  # same rank anyway
    else:
        # 比较 r_hi vs r_lo 的大小
        idx_hi = RANKS.index(r_hi)
        idx_lo = RANKS.index(r_lo)
        if idx_hi < idx_lo:
            # r_hi 的位置更靠右 => r_hi 实际更大 (因为 RANKS是从2到A递增)
            # 注意：RANKS = "23456789TJQKA" -> index越大牌越大
            # 所以要用 index 比较大小
            # 这里我们重新定义一下更清楚：
            pass
        # 我们直接重新写干净：
        if RANKS.index(r_hi) > RANKS.index(r_lo):
            ranks_ordered = [r_hi, r_lo]
        else:
            ranks_ordered = [r_lo, r_hi]

    # signature 例如 'ax'，表示第一张高牌用'a'的花色，第二张低牌用'x'的花色
    t0 = target_sig[0]
    t1 = target_sig[1]

    cand0_suits = suit_candidates_for_token(t0)  # treys suit_int列表
    cand1_suits = suit_candidates_for_token(t1)

    # 辅助函数：在允许的 suit_int 里挑一张具体牌( rank + suit_char )，不能冲突
    def pick_card(rank_char, allowed_suit_ints):
        for s_int in allowed_suit_ints:
            suit_char = suit_int_to_char(s_int)  # <-- 修正点：用映射，不再 SUITS[s_int]
            c = new(rank_char, suit_char)
            if c not in used:
                used.add(c)
                return c
        return None

    # 先拿第一张
    c0 = pick_card(ranks_ordered[0], cand0_suits)
    if c0 is None:
        return None

    # 构建第二张时，必须考虑同花/不同花约束
    def suits_not_equal_filter(suit_int_list, suit_int_for_c0):
        return [si for si in suit_int_list if si != suit_int_for_c0]

    s0_int = card_suit_int(c0)

    if is_suited:
        # 要求两张是同一实际花色
        cand1_same = [s0_int] if s0_int in cand1_suits else []
        c1 = pick_card(ranks_ordered[1], cand1_same)
        if c1 is None:
            return None
        return [c0, c1]

    else:
        # offsuit 或 pair: 要求不同花色
        cand1_diff = suits_not_equal_filter(cand1_suits, s0_int)
        c1 = pick_card(ranks_ordered[1], cand1_diff)
        if c1 is None:
            return None
        return [c0, c1]

# -------------------- Monte Carlo 胜率 (6人桌) --------------------
def hs_6max_monte(hand, flop, n_sim=150, n_players=6):
    """
    hand: [2 treys ints]
    flop: (3 treys ints)
    估算 Hero 在6人桌的胜率 (赢+平半)。
    我们每次随机发 turn/river 和对手手牌。
    """
    deck = Deck()
    used = set(hand + list(flop))
    deck.cards = [c for c in deck.cards if c not in used]

    win = tie = 0
    for _ in range(n_sim):
        # 发对手的牌
        opp_hands = [deck.draw(2) for _ in range(n_players - 1)]

        # 发 turn / river
        board = list(flop) + deck.draw(2)  # flop三张 + turn + river

        hero_score = evaluator.evaluate(board, hand)
        opp_scores = [evaluator.evaluate(board, oh) for oh in opp_hands]
        best_score = min([hero_score] + opp_scores)

        # 胜利/平局计分
        winners = [s for s in [hero_score] + opp_scores if s == best_score]
        if hero_score == best_score:
            if len(winners) == 1:
                win += 1
            else:
                tie += 1

        # 把发出去的牌放回去洗一下 deck
        deck.cards += [c for oh in opp_hands for c in oh]
        deck.cards += board[3:]  # turn/river
        random.shuffle(deck.cards)

    return (win + 0.5 * tie) / n_sim

# -------------------- 每个任务 (hand_type, flop_class) --------------------
def worker_task(args):
    htype, flop_key, flop_cards, flop_pat, n_sim, n_players = args

    # 确定手牌类型类别
    if len(htype) == 2:
        kind = 'pair'
    else:
        kind = 'suited' if htype[2] == 's' else 'offsuit'

    # 列出这个 (hand_type, flop_pattern) 理论上允许的所有 signature
    sigs = all_target_signatures(flop_pat, kind)

    out_rows = []
    for sig in sorted(sigs):
        # 尝试构建一副符合这个 signature 的具体两张手牌，且不与这三张 flop 冲突
        hand_cards = realize_hand_for_signature(htype, flop_cards, sig)
        if hand_cards is None:
            continue  # 这种signature在这个具体flop上实现不了，就跳过

        # Monte Carlo 求胜率
        strength = hs_6max_monte(hand_cards, flop_cards,
                                 n_sim=n_sim, n_players=n_players)

        out_rows.append({
            "hand_type": htype,
            "canon_flop": flop_key,
            "flop_pattern": flop_pat,
            "suit_signature": sig,
            "strength_mean": round(strength, 6),
            "n_sim": n_sim
        })

    return out_rows

# -------------------- 主流程 --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="flop_strength_table.csv")
    ap.add_argument("--n_sim", type=int, default=150)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--quick", action="store_true",
                    help="快速模式: 只取前100个flop，用来测试流程/格式")
    args = ap.parse_args()

    hand_types = hand_types_169()               # 169 种起手牌类别
    flops_info = enumerate_canonical_flops()    # ~1911 个翻牌类
                                                # [(key, flop_cards, flop_pat), ...]

    if args.quick:
        flops_info = flops_info[:100]
        print(f"[Quick] using only {len(flops_info)} flop classes to test.")

    # 组装任务：(hand_type, flop)
    tasks = []
    for flop_key, flop_cards, flop_pat in flops_info:
        for htype in hand_types:
            tasks.append((htype, flop_key, flop_cards, flop_pat,
                          args.n_sim, args.players))

    print(f"Total tasks: {len(tasks)} | workers: {args.workers}")

    rows_collected = []
    with Pool(processes=args.workers) as pool:
        # chunksize 调大会更快，内存也更稳
        for chunk in pool.imap_unordered(worker_task, tasks, chunksize=32):
            if chunk:
                rows_collected.extend(chunk)

    df = pd.DataFrame(rows_collected)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
