# coding: utf-8
import argparse
import random
from collections import Counter

import pandas as pd
from treys import Card, Deck

# -------------------- 全局配置 --------------------
RANKS = "23456789TJQKA"   # 点数顺序由小到大
SUITS = "cdhs"            # treys 用 'c','d','h','s'
random.seed(42)


# ============== 基础工具：从 treys int 取字符 ==============
def rank_char(c: int) -> str:
    """返回牌面点数字符 '2'..'A'"""
    return Card.int_to_str(c)[0]


def suit_char(c: int) -> str:
    """返回花色字符 'c'/'d'/'h'/'s'"""
    return Card.int_to_str(c)[1]


# ============== 生成 169 类起手牌标签（与CSV一致） ==============
def hand_type_169(c1: int, c2: int) -> str:
    """
    生成 169 类起手牌标签：
      - 对子: 'AA'、'TT'...
      - 同花非对子: 'AKs'、'T9s'...
      - 杂色非对子: 'AKo'、'T9o'...
    规则：高牌在前（按 RANKS 的索引比较）
    """
    r1, r2 = rank_char(c1), rank_char(c2)

    # 高牌在前：RANKS 越靠右越大
    if RANKS.index(r1) < RANKS.index(r2):
        c1, c2 = c2, c1
        r1, r2 = r2, r1

    if r1 == r2:
        return f"{r1}{r2}"

    suited = (suit_char(c1) == suit_char(c2))
    return f"{r1}{r2}{'s' if suited else 'o'}"


# ============== 把真实 flop 规范化为 CSV 的桶键 ==============
def canonicalize_flop(cards3):
    """
    返回 (canon_key, flop_pattern, suit_seen)
      - canon_key 形如 '14,13,12|abb' （点数降序 + 首现花色重标）
      - flop_pattern 即 'abb'/'aab'/'abc'/'aaa'/...（与 canon_key 的花色段一致）
      - suit_seen 是实际花色到标签索引的映射，如 {'c':0,'d':1}，便于生成签名
    注意：仅用于“给这3张牌一个标准桶”，不改变胜率本身。
    """
    triples = [(rank_char(c), suit_char(c), c) for c in cards3]
    # 先按点数降序，再按花色字符排序（只是稳定排序用）
    triples.sort(key=lambda t: (-RANKS.index(t[0]), t[1]))

    # 点数写成 2..14 的数字串
    ranks_sorted_int = [RANKS.index(t[0]) + 2 for t in triples]

    # 首现花色 -> a/b/c
    suit_seen = {}   # suit_char -> idx(0/1/2)
    next_label = 0
    labels = []
    for _, s_ch, _ in triples:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s_ch]))

    rank_key = ",".join(str(r) for r in ranks_sorted_int)
    suit_key = "".join(labels)
    return f"{rank_key}|{suit_key}", suit_key, suit_seen


# ============== 生成与 flop 的相对花色签名（与CSV一致） ==============
def suit_signature_vs_flop(hole2, suit_seen):
    """
    用 flop 的 a/b/c 标号给底牌生成签名（高牌在前）：
      - 若某张底牌的花色在 flop 中出现过，则映射为对应 'a'/'b'/'c'
      - 否则映射为 'x'
    """
    # 与 hand_type_169 的排序保持一致：高牌在前
    c_high, c_low = sorted(hole2, key=lambda c: RANKS.index(rank_char(c)), reverse=True)

    def token_for(card):
        sch = suit_char(card)
        if sch in suit_seen:
            # suit_seen: {'c':0,'d':1,'h':2}  => 'a','b','c'
            return chr(ord('a') + suit_seen[sch])
        return 'x'

    return token_for(c_high) + token_for(c_low)


# ============== 构建查表（严格四键匹配） ==============
def build_strong_lookup(csv_path: str):
    df = pd.read_csv(csv_path)
    # 统一清洗
    for col in ["hand_type", "canon_flop", "flop_pattern", "suit_signature"]:
        df[col] = df[col].astype(str).str.strip()
    # 有些生成脚本把 hand_type 大写/小写不一致，这里强制大写
    df["hand_type"] = df["hand_type"].str.upper()

    lut = {}
    for _, row in df.iterrows():
        key4 = (row["hand_type"], row["canon_flop"], row["flop_pattern"], row["suit_signature"])
        lut[key4] = float(row["strength_mean"])
    return lut


# ============== 主程序：随机抽牌 → 查表命中率 ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="flop_strength_table.csv / *_nsim500.csv")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--show_miss", type=int, default=20, help="最多展示多少条 miss 样例")
    args = ap.parse_args()

    lut = build_strong_lookup(args.csv)

    exact_hits = miss = 0
    miss_examples = []
    seen = Counter()

    for _ in range(args.trials):
        d = Deck()
        hole = d.draw(2)
        flop = d.draw(3)

        htype = hand_type_169(hole[0], hole[1]).upper()
        canon_key, pattern, suit_seen = canonicalize_flop(flop)
        sig = suit_signature_vs_flop(hole, suit_seen)

        key4 = (htype, canon_key, pattern, sig)
        seen[key4] += 1

        if key4 in lut:
            exact_hits += 1
        else:
            miss += 1
            if len(miss_examples) < args.show_miss:
                miss_examples.append({
                    "hand_type": htype,
                    "canon_flop": canon_key,
                    "flop_pattern": pattern,
                    "suit_signature": sig,
                    "hole": [Card.int_to_str(hole[0]), Card.int_to_str(hole[1])],
                    "flop": [Card.int_to_str(flop[0]), Card.int_to_str(flop[1]), Card.int_to_str(flop[2])],
                })

    total = exact_hits + miss
    print(f"Total trials: {total}")
    print(f"Exact hits (4-key): {exact_hits}")
    print(f"Misses: {miss}")
    print(f"Exact hit rate: {exact_hits/total:.2%}")

    if miss_examples:
        print("\n-- Miss examples (up to {n}) --".format(n=args.show_miss))
        for i, m in enumerate(miss_examples, 1):
            print(f"{i:02d}. {m}")


if __name__ == "__main__":
    main()
