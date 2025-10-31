# test_flop_lookup.py
import argparse
import itertools
import random
from collections import Counter

import pandas as pd
from treys import Card, Deck, Evaluator

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # treys使用的花色字符
random.seed(42)

# treys 的 suit_int 是 1/2/4/8（位掩码），不是 0..3，这里做映射
SUIT_INT_TO_CHAR = {Card.get_suit_int(Card.new("2"+s)): s for s in SUITS}

def suit_int_to_char(si):
    return SUIT_INT_TO_CHAR[si]

def rank_int(c):
    return Card.get_rank_int(c)  # 2..14

def suit_int(c):
    return Card.get_suit_int(c)  # 1/2/4/8

def hand_type_169(c1, c2):
    """把两张treys牌映射成 169 类的hand_type：如 'AKs' / 'AKo' / 'AA'"""
    r1, r2 = rank_int(c1), rank_int(c2)
    rc1, rc2 = Card.STR_RANKS[r1], Card.STR_RANKS[r2]

    # 让第一张是更大的rank（使用 RANKS 的顺序来比较）
    if RANKS.index(rc1) < RANKS.index(rc2):
        rc1, rc2, c1, c2 = rc2, rc1, c2, c1

    if rc1 == rc2:
        return f"{rc1}{rc2}"  # 对子无 s/o
    else:
        suited = suit_int(c1) == suit_int(c2)
        return f"{rc1}{rc2}{'s' if suited else 'o'}"

def canonicalize_flop(cards3):
    """
    把真实flop(三张treys牌)规范化为：
      - canon_flop: "rank_key|suit_key"，如 "14,13,12|aab"
      - flop_pattern: 即 suit_key
      - suit_seen: {treys_suit_int: 0/1/2} 记录哪门映射到了 a/b/c
    """
    triples = sorted([(rank_int(c), suit_int(c), c) for c in cards3],
                     key=lambda t: (-t[0], t[1]))
    ranks_sorted = [t[0] for t in triples]

    suit_seen = {}
    next_label = 0
    labels = []
    for _, si, _ in triples:
        if si not in suit_seen:
            suit_seen[si] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[si]))

    rank_key = ",".join(map(str, ranks_sorted))
    suit_key = "".join(labels)
    return f"{rank_key}|{suit_key}", suit_key, suit_seen

def suit_signature_vs_flop(hole, flop_suit_key, suit_seen):
    """
    计算两张底牌相对flop的 suit_signature（按底牌高到低的顺序）：
      - 若底牌某张的花色在flop中出现，则映射为 'a'/'b'/'c'
      - 否则为 'x'
    """
    # 先把hole按rank从大到小排序，保证签名顺序稳定
    h_sorted = sorted(hole, key=lambda c: RANKS.index(Card.STR_RANKS[rank_int(c)]), reverse=True)

    # 反向映射：a/b/c -> 实际的 treys suit_int 集（通常每个只有1个）
    inv = {}
    for si, idx in suit_seen.items():
        inv.setdefault(idx, set()).add(si)  # 安全起见用 set
    label_to_suits = {chr(ord('a')+idx): inv[idx] for idx in inv}

    board_suits = set().union(*label_to_suits.values()) if label_to_suits else set()

    def token_for_card(card):
        si = suit_int(card)
        # 查它属于 a/b/c 哪一类
        for lab, sset in label_to_suits.items():
            if si in sset:
                return lab
        # 不在board里
        return 'x'

    t0 = token_for_card(h_sorted[0])
    t1 = token_for_card(h_sorted[1])
    return t0 + t1

def build_lookup(csv_path):
    """
    读取 flop_strength CSV，建立查询字典：
      key = (hand_type, canon_flop, flop_pattern, suit_signature)
      val = strength_mean
    """
    df = pd.read_csv(csv_path)
    # 统一字符串格式，避免空格/大小写问题
    for col in ["hand_type", "canon_flop", "flop_pattern", "suit_signature"]:
        df[col] = df[col].astype(str).str.strip()

    lut = {}
    misses = 0
    for _, row in df.iterrows():
        key = (row["hand_type"], row["canon_flop"], row["flop_pattern"], row["suit_signature"])
        lut[key] = float(row["strength_mean"])
    return lut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="你的 flop_strength_table CSV 路径，例如 flop_strength_table_nsim500.csv")
    ap.add_argument("--trials", type=int, default=1000, help="随机测试次数")
    args = ap.parse_args()

    lut = build_lookup(args.csv)

    hits = 0
    misses = 0
    miss_examples = []
    seen_keys = Counter()

    for _ in range(args.trials):
        d = Deck()
        # 随机发两张底牌 + 三张flop
        hole = d.draw(2)
        flop = d.draw(3)

        htype = hand_type_169(hole[0], hole[1])
        canon_flop, flop_pattern, suit_seen = canonicalize_flop(flop)
        sig = suit_signature_vs_flop(hole, flop_pattern, suit_seen)

        key = (htype, canon_flop, flop_pattern, sig)
        seen_keys[key] += 1

        if key in lut:
            hits += 1
        else:
            misses += 1
            if len(miss_examples) < 15:
                miss_examples.append({
                    "hand_type": htype,
                    "canon_flop": canon_flop,
                    "flop_pattern": flop_pattern,
                    "suit_signature": sig
                })

    total = hits + misses
    hit_rate = hits / total if total else 0.0

    print(f"Total trials: {total}")
    print(f"Hits: {hits} | Misses: {misses} | Hit rate: {hit_rate:.3%}")

    if miss_examples:
        print("\n--- First few misses (up to 15) ---")
        for i, m in enumerate(miss_examples, 1):
            print(f"{i:02d}. {m}")

    # 也可以打印出现最频繁的 key（帮助你抽查）
    print("\n--- Top seen keys (sample of 10) ---")
    for key, cnt in seen_keys.most_common(10):
        print(f"{cnt:4d} × {key}")

if __name__ == "__main__":
    main()
