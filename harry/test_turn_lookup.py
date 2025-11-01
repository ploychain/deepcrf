# coding: utf-8
import argparse
import random
from collections import Counter

import pandas as pd
from treys import Card, Deck

RANKS = "23456789TJQKA"
SUITS = "cdhs"
random.seed(42)

def rank_char(c): return Card.int_to_str(c)[0]
def suit_char(c): return Card.int_to_str(c)[1]

def hand_type_169(c1, c2):
    r1, r2 = rank_char(c1), rank_char(c2)
    # 高牌在前
    if RANKS.index(r1) < RANKS.index(r2):
        c1, c2 = c2, c1
        r1, r2 = r2, r1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if suit_char(c1)==suit_char(c2) else 'o'}"

def canonicalize_turn(cards4):
    quads = [(RANKS.index(rank_char(c))+2, suit_char(c), c) for c in cards4]
    quads.sort(key=lambda t: (-t[0], t[1]))
    ranks_sorted = [t[0] for t in quads]
    suit_seen = {}
    next_label = 0
    labels = []
    for _, s, _ in quads:
        if s not in suit_seen:
            suit_seen[s] = next_label
            next_label += 1
        labels.append(chr(ord('a')+suit_seen[s]))
    rank_key = ",".join(str(r) for r in ranks_sorted)
    suit_key = "".join(labels)  # 长度 4
    return f"{rank_key}|{suit_key}", suit_key, suit_seen

def suit_signature_vs_turn(hole2, suit_seen):
    # 与 hand_type_169 的排序一致（高牌在前）
    c_high, c_low = sorted(hole2, key=lambda c: RANKS.index(rank_char(c)), reverse=True)
    inv = {idx:set() for idx in suit_seen.values()}
    for sch, idx in suit_seen.items():
        inv[idx].add(sch)
    label_to_suits = {chr(ord('a')+k): v for k,v in inv.items()}
    board_suits = set().union(*label_to_suits.values()) if label_to_suits else set()

    def token_for(card):
        sch = suit_char(card)
        for lab, suits in label_to_suits.items():
            if sch in suits:
                return lab
        return 'x'  # 不在 turn 花色集合
    return token_for(c_high) + token_for(c_low)

def build_lookup(csv_path):
    df = pd.read_csv(csv_path)
    # 只把 hand_type 统一成大写；其余键保持小写（与生成表一致）
    df["hand_type"] = df["hand_type"].astype(str).str.strip().str.upper()
    for col in ["canon_turn", "turn_pattern", "suit_signature"]:
        df[col] = df[col].astype(str).str.strip()  # 不要 upper()

    lut = {}
    for _, row in df.iterrows():
        key = (row["hand_type"], row["canon_turn"], row["turn_pattern"], row["suit_signature"])
        lut[key] = float(row["strength_mean"])
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--show_miss", type=int, default=20)
    args = ap.parse_args()

    lut = build_lookup(args.csv)

    hits = miss = 0
    miss_examples = []
    for _ in range(args.trials):
        d = Deck()
        hole = d.draw(2)
        board4 = d.draw(4)

        htype = hand_type_169(hole[0], hole[1]).upper()
        key, pat, suit_seen = canonicalize_turn(board4)
        sig = suit_signature_vs_turn(hole, suit_seen)

        k = (htype, key, pat, sig)
        if k in lut:
            hits += 1
        else:
            miss += 1
            if len(miss_examples) < args.show_miss:
                miss_examples.append({
                    "hand_type": htype, "canon_turn": key, "turn_pattern": pat,
                    "suit_signature": sig,
                    "hole": [Card.int_to_str(hole[0]), Card.int_to_str(hole[1])],
                    "turn": [Card.int_to_str(c) for c in board4],
                })

    total = hits + miss
    print(f"Total trials: {total}")
    print(f"Exact hits (4-key): {hits}")
    print(f"Misses: {miss}")
    print(f"Exact hit rate: {hits/total:.2%}")
    if miss_examples:
        print("\n-- Miss examples --")
        for i, m in enumerate(miss_examples, 1):
            print(f"{i:02d}. {m}")

if __name__ == "__main__":
    main()
