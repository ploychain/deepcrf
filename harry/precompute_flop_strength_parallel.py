import argparse, itertools, math, os, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
from treys import Card, Deck, Evaluator

# -------------------- 全局参数 --------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # treys 的顺序：c(梅) d(方) h(红) s(黑)
evaluator = Evaluator()
random.seed(42)

# -------------------- 工具 --------------------
def hand_types_169():
    types = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types  # 169

def card_rank_int(c): return Card.get_rank_int(c)   # 2..14
def card_suit_int(c): return Card.get_suit_int(c)   # 0..3
def new(rank_char, suit_char): return Card.new(rank_char + suit_char)

def canonicalize_flop(cards3):
    """生成 canonical key，仅用于分桶/查表；不影响胜率。"""
    triples = [(card_rank_int(c), card_suit_int(c), c) for c in cards3]
    triples.sort(key=lambda t: (-t[0], t[1]))  # rank降序, suit升序稳定
    ranks_sorted = [t[0] for t in triples]
    suit_seen = {}
    next_label = 0
    labels = []
    for _, s, _ in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s]))
    return f"{','.join(map(str, ranks_sorted))}|{''.join(labels)}"

def enumerate_canonical_flops():
    """真实 52 选 3 → canonical 去重，返回代表实例（3张真实牌）和 pattern。"""
    deck = [new(r, s) for r in RANKS for s in SUITS]
    seen = {}
    for c1, c2, c3 in itertools.combinations(deck, 3):
        key = canonicalize_flop([c1, c2, c3])
        if key not in seen:
            seen[key] = (c1, c2, c3)  # 选作该类的代表实例
    # 拆出 suit pattern（'aaa','aab','aba','abb','abc' 之一）
    rows = []
    for key, flop in seen.items():
        suit_pat = key.split("|")[1]
        rows.append((key, flop, suit_pat))
    return rows  # 列表长度≈1911（与你前面一致）

# -------------------- suit-signature 相关 --------------------
def suit_letters_for_flop(flop):
    """返回 flop 中实际出现的 suit 集合（以 treys suit int 表示）"""
    return {card_suit_int(c) for c in flop}

def suit_signature_for_hand_vs_flop(hand, flop):
    """把手牌两张的花色映射为 a/b/c 或 x（x=不在 flop 花色集合里）。
       注意：signature 保持按 rank 从大到小的手牌顺序。"""
    # 先得到 flop 的 a/b/c 映射
    triples = [(card_rank_int(c), card_suit_int(c), c) for c in flop]
    triples.sort(key=lambda t: (-t[0], t[1]))
    suit_seen = {}
    next_label = 0
    for _, s, _ in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label; next_label += 1
    # hand 排序（高->低）
    rr = sorted([(card_rank_int(c), c) for c in hand], key=lambda t: -t[0])
    tokens = []
    for _, c in rr:
        s = card_suit_int(c)
        if s in suit_seen:
            tokens.append(chr(ord('a') + suit_seen[s]))
        else:
            tokens.append('x')
    return "".join(tokens)

def all_target_signatures(flop_pattern, htype_kind):
    """根据 flop 花色模式 和 手牌类别，枚举理论上可达的 signature 集合。
       htype_kind ∈ {'pair','suited','offsuit'}。
    """
    avail = {
        "aaa": ['a'],
        "aab": ['a','b'],
        "aba": ['a','b'],
        "abb": ['a','b'],
        "abc": ['a','b','c'],
    }
    L = avail[flop_pattern]
    Lx = L + ['x']

    if htype_kind == 'suited':
        return {t+t for t in Lx}  # aa,bb,cc,xx（按可用字母裁剪）
    elif htype_kind == 'offsuit':
        return {a+b for a in Lx for b in Lx if a != b}  # 有序对
    else:  # pair
        S = {a+b for a in Lx for b in Lx if a != b}
        S.add('xx')  # 两张都不在 flop 花色集合里（具体不同色），用一个 'xx' 槽近似
        return S

# -------------------- 手牌实例化（生成与 signature 一致的具体两张牌） --------------------
def realize_hand_for_signature(htype, flop, target_sig):
    """给定 htype（如 'AKs','AKo','AA'）、flop（三张真实牌）、目标 signature（如 'aa','ax','ba','xx'），
       构造一副不与 flop 冲突的具体两张牌（treys int 列表）。若不可构造则返回 None。"""
    used = set(flop)
    flop_suits = [card_suit_int(c) for c in flop]
    # 建 flop suit 映射: 出现顺序 -> a,b,c
    triples = sorted([(card_rank_int(c), card_suit_int(c), c) for c in flop], key=lambda t: (-t[0], t[1]))
    suit_seen = {}
    next_label = 0
    for _, s, _ in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label; next_label += 1
    # a/b/c -> 实际 suit 集
    label2suits = {}
    for lab, idx in [('a',0),('b',1),('c',2)]:
        label2suits[lab] = [s for s,i in suit_seen.items() if i == idx]
    # 可用的 x（不在 flop 的 suit）
    flop_suit_set = set(flop_suits)
    x_suits = [si for si in range(4) if si not in flop_suit_set]
    # htype 解析
    r1, r2 = htype[0], htype[1]
    is_pair = (r1 == r2)
    is_suited = (len(htype) == 3 and htype[2] == 's')
    is_offsuit = (len(htype) == 3 and htype[2] == 'o')

    # 目标 signature 的两个 token 按手牌 rank 高->低顺序
    t0, t1 = target_sig[0], target_sig[1]

    # 选 rank：对子特殊；非对子按 r1>r2 排
    if is_pair:
        ranks = [r1, r2]  # same char; 不要求同花
    else:
        # 高牌在前
        idx1 = RANKS.index(r1); idx2 = RANKS.index(r2)
        if idx1 < idx2:
            ranks = [r2, r1]  # 把高的放前
        else:
            ranks = [r1, r2]

    def suit_candidates(token):
        if token in ('a','b','c'):
            base = []
            for s,int_idx in suit_seen.items():
                if int_idx == ord(token)-ord('a'):
                    base.append(s)
            return base
        else:
            return x_suits[:]  # 'x'

    def pick_card(rank_char, allowed_suits):
        # 找到一张不与 flop 冲突也不与另一张冲突的牌
        for s in allowed_suits:
            # treys suit 映射到字符
            suit_char = SUITS[s]
            c = new(rank_char, suit_char)
            if c not in used:
                used.add(c)
                return c
        return None

    # 同花/非同花约束
    # - suited: 两张牌必须同 suit（同具体花色）
    # - offsuit: 两张牌必须不同 suit（具体花色不同）
    # - pair: 两张牌 suit 必须不同（AA同花不存在）
    s0_cands = suit_candidates(t0)
    s1_cands = suit_candidates(t1)

    if is_suited:
        # 要求两张牌实际 suit 相同：在交集里挑
        common = [s for s in s0_cands if s in s1_cands]
        if not common: return None
        s = common  # 允许多个可选
        c0 = pick_card(ranks[0], s)
        if not c0: return None
        # 第二张与第一张同 suit
        c1 = pick_card(ranks[1], [card_suit_int(c0)])
        if not c1: return None
        return [c0, c1]

    else:
        # 先挑第一张
        c0 = pick_card(ranks[0], s0_cands)
        if not c0: return None
        # 第二张需不同 suit（offsuit/pair），且在 s1_cands 里
        s_for_c1 = [s for s in s1_cands if s != card_suit_int(c0)]
        if not s_for_c1: return None
        c1 = pick_card(ranks[1], s_for_c1)
        if not c1: return None
        # 对子：再确认不同 suit（已保证）
        return [c0, c1]

# -------------------- 胜率（6人桌） --------------------
def hs_6max_monte(hand, flop, n_sim=150, n_players=6):
    deck = Deck()
    used = set(hand + list(flop))
    deck.cards = [c for c in deck.cards if c not in used]
    win = tie = 0
    need = 5 - 3  # turn+river 两张
    for _ in range(n_sim):
        opps = [deck.draw(2) for _ in range(n_players - 1)]
        board = list(flop) + deck.draw(need)  # 注意：这里不排序，真实集合
        my_score = evaluator.evaluate(board, hand)
        best_opp = min(evaluator.evaluate(board, h) for h in opps)
        if my_score < best_opp: win += 1
        elif my_score == best_opp: tie += 1
        deck.cards += [c for h in opps for c in h] + board[3:]
        random.shuffle(deck.cards)
    return (win + 0.5 * tie) / n_sim

# -------------------- 任务切片 --------------------
def worker_task(args):
    htype, flop_key, flop_cards, flop_pat, target_sigs, n_sim, n_players = args
    out = []
    # 判定 htype 类别
    if len(htype) == 2: kind = 'pair'
    else: kind = 'suited' if htype[2] == 's' else 'offsuit'
    for sig in sorted(target_sigs):
        hand = realize_hand_for_signature(htype, flop_cards, sig)
        if hand is None:
            continue
        strength = hs_6max_monte(hand, flop_cards, n_sim=n_sim, n_players=n_players)
        out.append({
            "hand_type": htype,
            "canon_flop": flop_key,
            "flop_pattern": flop_pat,
            "suit_signature": sig,
            "strength_mean": round(strength, 6),
            "n_sim": n_sim
        })
    return out

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="flop_strength_table.csv")
    ap.add_argument("--n_sim", type=int, default=150)          # 可调：300 更稳
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--quick", action="store_true", help="快速模式：仅抽样少量 flop")
    args = ap.parse_args()

    hand_types = hand_types_169()
    canon_flops = enumerate_canonical_flops()  # [(key, flop3, pattern)]

    # 快速模式：只取前 100 个 flop 先看格式、速度
    if args.quick:
        canon_flops = canon_flops[:100]
        print(f"[Quick] Using {len(canon_flops)} flops for a smoke test.")

    # 组装任务
    tasks = []
    for key, flop3, pat in canon_flops:
        for htype in hand_types:
            kind = 'pair' if len(htype) == 2 else ('suited' if htype[2] == 's' else 'offsuit')
            sigs = all_target_signatures(pat, kind)
            tasks.append((htype, key, flop3, pat, sigs, args.n_sim, args.players))

    print(f"Total tasks (htype × flop): {len(tasks)}  | workers: {args.workers}")

    # 并行跑
    rows = []
    with Pool(processes=args.workers) as pool:
        for chunk in pool.imap_unordered(worker_task, tasks, chunksize=32):
            if chunk:
                rows.extend(chunk)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
