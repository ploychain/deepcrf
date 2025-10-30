# coding: utf-8
import argparse
import itertools
import random
from multiprocessing import Pool, cpu_count

import pandas as pd
from treys import Card, Deck, Evaluator

# -------------------- 全局配置 --------------------
RANKS = "23456789TJQKA"   # 2..A
SUITS = "cdhs"            # treys 用 'c','d','h','s'
evaluator = Evaluator()
random.seed(42)

# ====== 基础解析：只用 Card.int_to_str(c) 拿 rank/suit，避免 treys bit 花色坑 ======
def card_rank_int(c: int) -> int:
    """把 treys int 牌转换成 rank 数值 2..14 (A=14)"""
    rs = Card.int_to_str(c)  # e.g. 'As'
    rch = rs[0]
    return RANKS.index(rch) + 2

def card_suit_char(c: int) -> str:
    """返回 'c'/'d'/'h'/'s'"""
    return Card.int_to_str(c)[1]

# -------------------- 起手牌 169 类 --------------------
def hand_types_169():
    types = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types  # 169

# -------------------- flop 规范化：桶键 --------------------
def canonicalize_flop(cards3):
    """
    三张真实 flop -> (key, pattern)
    key 形如 '14,13,12|abb'
    pattern 即 'abb'/'aab'/'abc'/'aaa'/...
    """
    triples = [(card_rank_int(c), card_suit_char(c), c) for c in cards3]
    # rank 降序，suit_char 升序（只为稳定）
    triples.sort(key=lambda t: (-t[0], t[1]))

    ranks_sorted = [t[0] for t in triples]
    # 强校验：rank 不允许为 0 或越界
    for r in ranks_sorted:
        assert 2 <= r <= 14, f"[BUG] invalid rank={r} for flop={list(map(Card.int_to_str, cards3))}"

    # 首次出现的花色 -> 'a','b','c'
    suit_seen = {}   # suit_char -> idx (0/1/2)
    next_label = 0
    labels = []
    for _, s_ch, _ in triples:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s_ch]))

    rank_key = ",".join(str(r) for r in ranks_sorted)
    suit_key = "".join(labels)
    return f"{rank_key}|{suit_key}", suit_key

def enumerate_canonical_flops():
    """
    穷举 C(52,3) 真实 flop，canonical 去重，返回代表：
    [(key, flop3_tuple, suit_pattern), ...]  期望 ~1911 项
    """
    deck = [Card.new(r + s) for r in RANKS for s in SUITS]
    seen = {}
    for c1, c2, c3 in itertools.combinations(deck, 3):
        flop = (c1, c2, c3)
        key, pat = canonicalize_flop(flop)
        if key not in seen:
            seen[key] = (flop, pat)
    rows = []
    for key, (flop3, pat) in seen.items():
        rows.append((key, flop3, pat))
    return rows

# -------------------- suit-signature 枚举 --------------------
def all_target_signatures(flop_pattern: str, htype_kind: str):
    """
    flop_pattern: 'aaa','aab','aba','abb','abc'
    htype_kind: 'pair' | 'suited' | 'offsuit'
    返回该类下理论可能出现的 signature 集合：{'aa','ax','ba','xx',...}
    """
    avail = {
        "aaa": ['a'],
        "aab": ['a', 'b'],
        "aba": ['a', 'b'],
        "abb": ['a', 'b'],
        "abc": ['a', 'b', 'c'],
    }
    L = avail[flop_pattern]
    Lx = L + ['x']  # 'x' 表示不在 flop 的花色

    if htype_kind == "suited":      # 两张同花
        return {t + t for t in Lx}
    elif htype_kind == "offsuit":   # 两张不同花（有序）
        return {a + b for a in Lx for b in Lx if a != b}
    else:                           # pair（两张不同花，同时允许 'xx'）
        S = {a + b for a in Lx for b in Lx if a != b}
        S.add('xx')
        return S

# -------------------- 构造符合 signature 的具体两张手牌 --------------------
def realize_hand_for_signature(htype: str, flop, target_sig: str):
    """
    htype: 'AKs','AKo','AA'
    flop: (c1,c2,c3) treys ints
    target_sig: 'aa','ax','ba','xx'
    返回 [c0, c1] 或 None （c0 为点数较大的那张）
    """
    used = set(flop)

    # flop 花色字符集合 & 重标 a/b/c 映射
    triples = [(card_rank_int(c), card_suit_char(c), c) for c in flop]
    triples.sort(key=lambda t: (-t[0], t[1]))
    suit_seen = {}   # suit_char -> idx(0/1/2)
    next_label = 0
    for _, s_ch, _ in triples:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
    flop_suits = {card_suit_char(c) for c in flop}

    def suit_candidates_for_token(token):
        if token == 'x':
            # 不在 flop 的花色
            return [s for s in SUITS if s not in flop_suits]
        desired_idx = ord(token) - ord('a')
        # 取所有在 flop 中被标记为该 label 的实际花色字符
        return [s for s, idx in suit_seen.items() if idx == desired_idx]

    # 解析 htype
    r_hi, r_lo = htype[0], htype[1]
    is_pair    = (r_hi == r_lo)
    is_suited  = (len(htype) == 3 and htype[2] == 's')
    is_offsuit = (len(htype) == 3 and htype[2] == 'o')

    # 高牌在前（RANKS 越靠右越大）
    if not is_pair:
        if RANKS.index(r_hi) < RANKS.index(r_lo):
            r_hi, r_lo = r_lo, r_hi
    ranks_ordered = [r_hi, r_lo]

    t0, t1 = target_sig[0], target_sig[1]
    cand0_suits = suit_candidates_for_token(t0)
    cand1_suits = suit_candidates_for_token(t1)

    def pick_card(rank_char: str, suit_char_list):
        for sch in suit_char_list:
            c = Card.new(rank_char + sch)
            if c not in used:
                used.add(c)
                return c
        return None

    # 第一张（高牌）
    c0 = pick_card(ranks_ordered[0], cand0_suits)
    if c0 is None:
        return None

    s0 = card_suit_char(c0)

    # 第二张（低牌）- 满足同花/不同花约束
    if is_suited:
        if s0 not in cand1_suits:
            return None
        c1 = pick_card(ranks_ordered[1], [s0])
        if c1 is None:
            return None
    else:
        # offsuit 或 pair：第二张必须不同花
        cand1_diff = [sch for sch in cand1_suits if sch != s0]
        c1 = pick_card(ranks_ordered[1], cand1_diff)
        if c1 is None:
            return None
        # pair 再防一手（不能同花）
        if is_pair and card_suit_char(c1) == s0:
            return None

    return [c0, c1]

# -------------------- Monte Carlo 胜率（6人桌） --------------------
def hs_6max_monte(hand, flop, n_sim=150, n_players=6):
    """
    hand: [2 treys ints]
    flop: (3 treys ints)
    返回 Hero 在 6 人桌（随机对手 + 随机 turn/river）下的胜率（赢=1, 平=0.5）
    """
    deck = Deck()
    used = set(hand + list(flop))
    deck.cards = [c for c in deck.cards if c not in used]

    win = tie = 0
    for _ in range(n_sim):
        # 对手手牌
        opp_hands = [deck.draw(2) for _ in range(n_players - 1)]
        # turn + river
        board = list(flop) + deck.draw(2)

        hero_score = evaluator.evaluate(board, hand)
        opp_scores = [evaluator.evaluate(board, oh) for oh in opp_hands]
        best = min([hero_score] + opp_scores)

        winners = [s for s in [hero_score] + opp_scores if s == best]
        if hero_score == best:
            if len(winners) == 1:
                win += 1
            else:
                tie += 1

        # 还牌+洗牌
        deck.cards += [c for oh in opp_hands for c in oh] + board[3:]
        random.shuffle(deck.cards)

    return (win + 0.5 * tie) / n_sim

# -------------------- 单任务 --------------------
def worker_task(args):
    htype, flop_key, flop_cards, flop_pat, n_sim, n_players = args
    kind = 'pair' if len(htype) == 2 else ('suited' if htype[2] == 's' else 'offsuit')

    out_rows = []
    for sig in sorted(all_target_signatures(flop_pat, kind)):
        hand = realize_hand_for_signature(htype, flop_cards, sig)
        if hand is None:
            continue
        strength = hs_6max_monte(hand, flop_cards, n_sim=n_sim, n_players=n_players)
        out_rows.append({
            "hand_type": htype,
            "canon_flop": flop_key,
            "flop_pattern": flop_pat,
            "suit_signature": sig,
            "strength_mean": round(float(strength), 6),
            "n_sim": int(n_sim),
        })
    return out_rows

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="flop_strength_table.csv")
    ap.add_argument("--n_sim", type=int, default=150)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--quick", action="store_true", help="只取前100个 flop 做冒烟测试")
    args = ap.parse_args()

    hand_types = hand_types_169()                  # 169
    flops_info = enumerate_canonical_flops()       # ~1911
    if args.quick:
        flops_info = flops_info[:100]
        print(f"[Quick] using only {len(flops_info)} flop classes.")

    tasks = []
    for key, flop3, pat in flops_info:
        for htype in hand_types:
            tasks.append((htype, key, flop3, pat, args.n_sim, args.players))

    print(f"Total tasks: {len(tasks)} | workers: {args.workers}")

    rows = []
    with Pool(processes=args.workers) as pool:
        for chunk in pool.imap_unordered(worker_task, tasks, chunksize=32):
            if chunk:
                rows.extend(chunk)

    df = pd.DataFrame(rows)

    # ----------- 写盘前强校验 -----------
    if df.empty:
        raise AssertionError("[BUG] result dataframe is empty.")

    bad0 = df['canon_flop'].astype(str).str.startswith('0,0,0|').sum()
    if bad0:
        bad = df[df['canon_flop'].astype(str).str.startswith('0,0,0|')].head(5)
        raise AssertionError(f"[BUG] Detected bad canon_flop like '0,0,0|*'. Samples:\n{bad}")

    if not ((df['strength_mean'] >= 0.0) & (df['strength_mean'] <= 1.0)).all():
        raise AssertionError("[BUG] strength_mean out of [0,1] range!")

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(f"Distinct canon_flop: {df['canon_flop'].nunique()} (expected ~1911)")
    print(df.head(8))

if __name__ == "__main__":
    main()
