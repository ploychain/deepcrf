import itertools
from collections import Counter, defaultdict
from treys import Card


# =========================
# 1. 枚举起手牌的 169 类
# =========================
def list_hand_types_169():
    """
    经典169起手牌类别:
    - 13个对子: AA, KK, ..., 22
    - 78个同花: AKs, AQs, ..., 32s
    - 78个非同花: AKo, AQo, ..., 32o
    """
    ranks = "23456789TJQKA"
    types = []
    for i, a in enumerate(ranks):
        for j, b in enumerate(ranks):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types  # len == 169 目标


# 小工具：拿 treys 的 rank/suit
def card_rank_int(card_int):
    return Card.get_rank_int(card_int)  # 2..14


def card_suit_int(card_int):
    return Card.get_suit_int(card_int)  # 0..3 (c,d,h,s，但内部顺序是treys自己的)


# =========================
# 2. flop 规范化
# =========================
def canonicalize_flop(cards3):
    """
    输入: 3张真实牌 (例如 [As,Ks,Qd] 的 treys int)
    输出: 一个canonical key，用来表示这个flop的“结构”，忽略具体花色名称，但保留花色模式。

    规则:
    - 先按牌面大小(ranks)从大到小排序，如果同rank再按suit排序，保证一致顺序
    - 对花色做重新命名:
        第一种花色 -> 'a'
        第二种花色 -> 'b'
        第三种花色 -> 'c'
      这样像 ♠♠♦ 就会变成 aab；♠♥♦ 就是 abc；♠♠♠ 是 aaa。
    - key 形式: "R1,R2,R3|PATTERN"
      比如 "14,13,12|aab"
      14=Ace,13=King,12=Queen
      aab=前两张同花，第三张不同花
    """
    triples = [(card_rank_int(c), card_suit_int(c)) for c in cards3]
    # 排序: rank降序, suit升序稳定
    triples.sort(key=lambda t: (-t[0], t[1]))

    ranks_sorted = [t[0] for t in triples]

    # 给花色做第一次出现式的重标
    suit_seen = {}
    next_label = 0  # 0->'a',1->'b',...
    pattern_labels = []
    for _, s in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label
            next_label += 1
        label_char = chr(ord('a') + suit_seen[s])
        pattern_labels.append(label_char)

    rank_key = ",".join(str(r) for r in ranks_sorted)  # 例如 "14,13,12"
    suit_key = "".join(pattern_labels)  # 例如 "aab" / "abc" / "aaa"
    return f"{rank_key}|{suit_key}"


def enumerate_canonical_flops():
    """
    穷举所有真实flop (从整副牌52张里选3张，共 C(52,3)=22100)
    每个flop做 canonicalize_flop
    用set去重 => 得到我们真正区分的"翻牌形状类"个数
    顺便统计花色pattern分布, rank多重集分布
    """
    deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "cdhs"]
    canon_set = set()
    suit_pattern_counter = Counter()
    rank_mult_counter = Counter()

    for c1, c2, c3 in itertools.combinations(deck, 3):
        key = canonicalize_flop([c1, c2, c3])
        canon_set.add(key)

    # 我们再扫一遍 set，做统计
    for key in canon_set:
        rank_part, suit_part = key.split("|")
        r1, r2, r3 = map(int, rank_part.split(","))

        # rank 结构（是否有对子、三条）
        # 注意：我们排序后 r1>=r2>=r3
        if r1 == r2 == r3:
            rank_mult = "trips"  # 三条
        elif r1 == r2 or r2 == r3:
            rank_mult = "paired"  # 一对+踢脚
        else:
            rank_mult = "unpaired"  # 全不一样

        suit_pattern_counter[suit_part] += 1  # 'aaa','aab','abc' 等
        rank_mult_counter[rank_mult] += 1

    return canon_set, suit_pattern_counter, rank_mult_counter


# =========================
# 3. suit_signature 计数逻辑
# =========================
def signature_counts_per_flop_pattern():
    """
    我们关心：我的两张手牌相对于翻牌的花色，可能出现哪些"签名"。
    用字符标记:
      'a','b','c' = 这张手牌的花色在翻牌中出现过，对应那一类花色
      'x'         = 这张手牌的花色在翻牌里根本没出现
    比如:
      - 'ax'  = 第一张牌花色跟翻牌某花色匹配，第二张是没出现的花色
      - 'aa'  = 两张牌用同一个翻牌花色（比如同花听同花）
      - 'xx'  = 两张牌都是翻牌外的花色

    但是，不是所有签名都合法：
    - 同花起手(suited)：两张底牌必须同花 -> 两个字母必须一样，比如"aa"或"xx"，不能"ab"
    - 非同花起手(offsuit)：两张底牌必须不同花 -> 两个字母必须不一样，比如"ab","ax","ba","xa"
    - 口袋对子(pair)：两张同rank，但花色必须不同。对子不可能是"aa"（同一具体花色），
      但可以是"ax"（一张跟翻牌同花色，另一张不同花色）或"xx"等等

    另外，flop 有三种花色形态：
      - "aaa": 翻牌只有一种花色类别
      - "aab": 翻牌用到两种花色类别
      - "abc": 翻牌用到三种花色类别

    这个函数，我们粗略枚举允许的 signature 组合数量，分三类手牌：
      pair      手牌是对子，比如 AA
      suited    手牌是同花非对子，比如 AKs
      offsuit   手牌是不同花非对子，比如 AKo
    """
    # flop 花色覆盖的类别字母集
    flop_color_sets = {
        "aaa": ['a'],  # 只有1种花色出现在翻牌
        "aab": ['a', 'b'],  # 两种花色
        "abc": ['a', 'b', 'c']  # 三种花色
    }

    result = defaultdict(dict)

    for flop_pat, letters in flop_color_sets.items():
        L = list(letters)  # e.g. ['a','b'] 对于"aab"
        # 加上 'x'，表示"不在翻牌里的新花色"
        L_plus_x = L + ['x']

        # --- suited（如 AKs）：两张同花
        suited_sigs = set()
        # 两张同花 => 两个位置的标签必须一样
        for token in L_plus_x:
            # "aa","bb","cc","xx" 都可能，取决于 flop 花色有多少种
            suited_sigs.add(token + token)
        # 注意：如果 flop 只有 'a'，那就可能有 "aa" 和 "xx"
        # 如果 flop 有 'a','b'，就可能有 "aa","bb","xx"
        # 如果 flop 有 'a','b','c'，就 "aa","bb","cc","xx"

        # --- offsuit（如 AKo）：两张不同花
        offsuit_sigs = set()
        for t1 in L_plus_x:
            for t2 in L_plus_x:
                if t1 != t2:
                    offsuit_sigs.add(t1 + t2)
        # 这里包括 "ab","ba","ax","xa","bx","xb","bc","cb"...等
        # 注意 offsuit 我们保留顺序，因为高牌那张 vs 另一张区分是合理的输入特征

        # --- pair（如 AA）：两张同rank但花色不同
        # 对子不允许两张牌实际上同一花色，但它们可以都来自"翻牌外花色"不同的两张 -> 我们保留保守近似:
        # 我们允许所有 offsuit_sigs，再加 'xx'（两张都外花色但不同具体花色）
        # 但我们不允许像 "aa" 这种实际上同一花色的同花对 (AA同花不可能)
        pair_sigs = set()
        for sig in offsuit_sigs:
            pair_sigs.add(sig)
        # 另外，"xx" 实际是两张都不在 flop 花色集合里，但仍可以是不同具体花色
        # 由于我们不细分 x1,x2，这里我们把 "xx" 也当成一个可能的签名
        pair_sigs.add("xx")

        result[flop_pat]["suited"] = len(suited_sigs)
        result[flop_pat]["offsuit"] = len(offsuit_sigs)
        result[flop_pat]["pair"] = len(pair_sigs)

    return result


# =========================
# 4. 用这些数字估算总“局面类”规模
# =========================
def estimate_total_scenarios(num_flop_classes, sig_stats):
    """
    我们最后关心的是：
      (起手牌类别) × (翻牌结构类别) × (花色交互签名)
    其中起手牌类别拆成三类:
        pair      13 种: AA, KK, ..., 22
        suited    78 种: AKs, AQs, ..., 32s
        offsuit   78 种: AKo, AQo, ..., 32o

    sig_stats 是 signature_counts_per_flop_pattern() 的输出:
    {
      "aaa": {"suited": X, "offsuit": Y, "pair": Z},
      "aab": {...},
      "abc": {...}
    }

    但每个翻牌类并不是只用 "aaa" 这种pattern，它会是 "aaa"/"aab"/"abc" 之一。
    我们不知道你的环境下这三类各占多少比例之前，
    我们可以先取平均（粗估），或者你跑完可以用真实占比做加权。
    """

    # 起手牌三类数量
    num_pair = 13
    num_suited = 78
    num_offsuit = 78

    # 对每种 flop 花色模式计算“这个模式能产生的平均签名组合数”
    # suited手牌的可能签名个数、offsuit的可能签名个数、pair的可能签名个数
    avg_suited = (sig_stats["aaa"]["suited"] +
                  sig_stats["aab"]["suited"] +
                  sig_stats["abc"]["suited"]) / 3.0
    avg_offsuit = (sig_stats["aaa"]["offsuit"] +
                   sig_stats["aab"]["offsuit"] +
                   sig_stats["abc"]["offsuit"]) / 3.0
    avg_pair = (sig_stats["aaa"]["pair"] +
                sig_stats["aab"]["pair"] +
                sig_stats["abc"]["pair"]) / 3.0

    # 一个flop类下，所有169种手牌能分化成多少 (hand_type, signature) 子类？
    # 约 = pair类手牌数量 * pair签名数
    #   + suited类手牌数量 * suited签名数
    #   + offsuit类手牌数量 * offsuit签名数
    avg_states_per_flop = (
            num_pair * avg_pair +
            num_suited * avg_suited +
            num_offsuit * avg_offsuit
    )

    # 最终总类（粗估）= flop类数 * 每个flop平均产生的(hand_type, signature)子类数
    est_total_classes = int(round(num_flop_classes * avg_states_per_flop))

    return {
        "avg_suited_sigs": avg_suited,
        "avg_offsuit_sigs": avg_offsuit,
        "avg_pair_sigs": avg_pair,
        "avg_states_per_flop": avg_states_per_flop,
        "estimated_total_classes": est_total_classes,
    }


# =========================
# 主流程: 打印所有关键数字
# =========================
if __name__ == "__main__":
    # 1. 起手牌 169
    hand_types = list_hand_types_169()
    print("起手牌类型数量 (不看具体花色):", len(hand_types))  # 169

    # 2. canonical flop 类数
    canon_flops, suit_pat_counter, rank_mult_counter = enumerate_canonical_flops()
    num_flop_classes = len(canon_flops)
    print("翻牌结构类别数量 (canonical flops):", num_flop_classes)
    print("  花色模式分布统计 (suit pattern 例如 'aaa','aab','abc',以及更多变体):")
    print(" ", dict(suit_pat_counter))
    print("  牌面重复结构统计 (rank multiplicity):")
    print(" ", dict(rank_mult_counter))
    # 注意: suit_pat_counter 可能不止 'aaa','aab','abc'，因为排序后还可能出现 'aba' 等等
    # 但策略意义上常用的经典标签是"aaa"(纯同花), "aab"(两同花一异花), "abc"(三花色)

    # 3. 理论签名数量 (每种 flop 花色覆盖下，我的两张牌可能的花色交互类型数量)
    sig_stats = signature_counts_per_flop_pattern()
    print("理论 signature 数量 (不同手牌类别在不同flop花色结构下可出现的花色关系组合数):")
    for pat, d in sig_stats.items():
        print(f"  flop花色模式={pat}:  suited={d['suited']}, offsuit={d['offsuit']}, pair={d['pair']}")

    # 4. 估算最终“局面类”规模
    estimate = estimate_total_scenarios(num_flop_classes, sig_stats)
    print("平均每个flop可以细分出的(hand_type, signature)子类数量 ≈", estimate["avg_states_per_flop"])
    print("粗略估计：总共需要覆盖的不同局面类（手牌种类×翻牌种类×花色互动）≈",
          estimate["estimated_total_classes"])
    print("细节: ", estimate)
