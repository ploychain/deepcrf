# coding: utf-8
import argparse
import random
from collections import Counter

import pandas as pd
from treys import Card, Deck

RANKS = "23456789TJQKA"
SUITS = "cdhs"
random.seed(42)

# -------- 与生成脚本完全一致的工具函数 --------
def card_rank_int(c: int) -> int:
    rs = Card.int_to_str(c)  # e.g. 'As'
    rch = rs[0]
    return RANKS.index(rch) + 2  # 2..14

def card_suit_char(c: int) -> str:
    return Card.int_to_str(c)[1]  # 'c','d','h','s'

def hand_type_169(c1: int, c2: int) -> str:
    """与生成脚本同逻辑：高牌在前；对子无 s/o；非对子按是否同花加 s/o"""
    r1, r2 = card_rank_int(c1), card_rank_int(c2)
    rc1, rc2 = Card.STR_RANKS[r1], Card.STR_RANKS[r2]
    # 让高牌在前（RANKS 越靠右越大）
    if RANKS.index(rc1) < RANKS.index(rc2):
        rc1, rc2, c1, c2 = rc2, rc1, c2, c1
    if rc1 == rc2:
        return f"{rc1}{rc2}"
    suited = (card_suit_char(c1) == card_suit_char(c2))
    return f"{rc1}{rc2}{'s' if suited else 'o'}"

def canonicalize_flop(cards3):
    """
    与生成脚本一致：返回 (key, pattern)
    key 形如 '14,13,12|abb'
    pattern 即 'abb'/'aab'/'abc'/'aaa'/...
    """
    triples = [(card_rank_int(c), card_suit_char(c), c) for c in cards3]
    triples.sort(key=lambda t: (-t[0], t[1]))  # rank 降序 + suit_char 升序稳定
    ranks_sorted = [t[0] for t in triples]
    # 首现花色 -> a,b,c
    suit_seen = {}
    next_label = 0
    labels = []
    for _, s_ch, _ in triples:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s_ch]))
    rank_key = ",".join(str(r) for r in ranks_sorted)
    suit_key = "".join(labels)
    return f"{rank_key}|{suit_key}", suit_key, suit_seen  # 顺便返回 suit_seen

def suit_signature_vs_flop(hole2, suit_seen):
    """
    根据 flop 的 a/b/c 标号，给 hole 两张牌生成相对花色签名（高牌在前）：
    返回 'aa','ax','ba','xx' 等
    """
    # 先按点数把两张底牌排出高->低（与 hand_type_169 的顺序保持一致）
    c_high, c_low = sorted(hole2, key=lambda c: RANKS.index(Card.STR_RANKS[card_rank_int(c)]), reverse=True)

    # suit_seen: {'c':0,'d':1,...} 这样的映射（由 canonicalize_flop 得出）
    inv = {idx: set() for idx in suit_seen.values()}
    for sch, idx in suit_seen.items():
        inv[idx].add(sch)
    label_to_suits = {chr(ord('a')+idx): inv[idx] for idx in inv}
    board_suits = set().union(*label_to_suits.values()) if label_to_suits else set()

    def token_for(card):
        sch = card_suit_char(card)
        for lab, suits in label_to_suits.items():
            if sch in suits:
                return lab
        return 'x'  # 不在 flop 的花色

    return token_for(c_high) + token_for(c_low)

# -------- 构建查表（强匹配 4 键）--------
def build_strong_lookup(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in ["hand_type", "canon_flop", "flop_pattern", "suit_signature"]:
        df[col] = df[col].astype(str).str.strip()
    lut = {}
    for _, row in df.iterrows():
        key4 = (row["hand_type"], row["canon_flop"], row["flop_pattern"], row["suit_signature"])
        lut[key4] = float(row["strength_mean"])
    return lut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="路径：flop_strength_table.csv / *_nsim500.csv")
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()

    lut = build_strong_lookup(args.csv)

    exact_hits = miss = 0
    miss_examples = []
    seen = Counter()

    for _ in range(args.trials):
        d = Deck()
        hole = d.draw(2)
        flop = d.draw(3)

        htype = hand_type_169(hole[0], hole[1])
        canon_key, pattern, suit_seen = canonicalize_flop(flop)
        sig = suit_signature_vs_flop(hole, suit_seen)

        key4 = (htype, canon_key, pattern, sig)
        seen[key4] += 1
        if key4 in lut:
            exact_hits += 1
        else:
            miss += 1
            if len(miss_examples) < 20:
                miss_examples.append({
                    "hand_type": htype,
                    "canon_flop": canon_key,
                    "flop_pattern": pattern,
                    "suit_signature": sig
                })

    total = exact_hits + miss
    print(f"Total trials: {total}")
    print(f"Exact hits (4-key): {exact_hits}")
    print(f"Misses: {miss}")
    print(f"Exact hit rate: {exact_hits/total:.2%}")

    if miss_examples:
        print("\n-- Miss examples (up to 20) --")
        for i, m in enumerate(miss_examples, 1):
            print(f"{i:02d}. {m}")

if __name__ == "__main__":
    main()
