# coding: utf-8
import argparse
import itertools
import random
from multiprocessing import Pool, cpu_count

import pandas as pd
from treys import Card, Deck, Evaluator

# -------------------- 全局配置 --------------------
RANKS = "23456789TJQKA"   # 2..A
SUITS = "cdhs"
evaluator = Evaluator()
random.seed(42)

# ====== 基础字符解析（避免 treys 花色整数陷阱） ======
def rank_char(c: int) -> str:
    return Card.int_to_str(c)[0]  # 'A'
def suit_char(c: int) -> str:
    return Card.int_to_str(c)[1]  # 's','h','d','c'

# -------------------- 169 起手牌 --------------------
def hand_types_169():
    types = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i < j:
                types += [f"{b}{a}s", f"{b}{a}o"]
            elif i == j:
                types += [f"{a}{a}"]
    return types  # 169

# -------------------- Turn 规范化：桶键 --------------------
def canonicalize_turn(cards4):
    """
    四张真实公共牌 -> (canon_key, pattern)
    canon_key: "14,13,11,7|aabb"  (点数降序 + 首现花色重标)
    pattern:   'aaaa'/'aaab'/'aabb'/'abac'/'abcd'
    说明：我们不关心原始顺序，只要“等价类”一致即可。
    """
    quads = [(RANKS.index(rank_char(c)) + 2, suit_char(c), c) for c in cards4]
    # 点数降序 + 花色字符升序（仅用于稳定排序）
    quads.sort(key=lambda t: (-t[0], t[1]))

    ranks_sorted_int = [t[0] for t in quads]

    suit_seen = {}  # 首现花色 -> a,b,c,d
    next_label = 0
    labels = []
    for _, s_ch, _ in quads:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
        labels.append(chr(ord('a') + suit_seen[s_ch]))  # a,b,c,d...

    rank_key = ",".join(str(r) for r in ranks_sorted_int)
    suit_key = "".join(labels)  # 长度 4：aaaa/aaab/aabb/abac/abcd...
    return f"{rank_key}|{suit_key}", suit_key, suit_seen

def enumerate_canonical_turns():
    """
    穷举 C(52,4) = 270,725 个 4 公共牌组合，
    对每个组合做 canonical 去重，返回代表：
    [(key, turn4_tuple, pattern), ...]
    期望唯一类远小于 270,725（通常数万 ~ 十来万量级）。
    """
    deck = [Card.new(r + s) for r in RANKS for s in SUITS]
    seen = {}
    for c1, c2, c3, c4 in itertools.combinations(deck, 4):
        board4 = (c1, c2, c3, c4)
        key, pat, _ = canonicalize_turn(board4)
        if key not in seen:
            seen[key] = (board4, pat)
    rows = []
    for key, (b4, pat) in seen.items():
        rows.append((key, b4, pat))
    return rows

# -------------------- suit-signature 枚举 --------------------
def all_target_signatures(turn_pattern: str, htype_kind: str):
    """
    turn_pattern: 4 长度的 'a'...'d' 串（如 'aabb','abcd' 等）
    htype_kind: 'pair' | 'suited' | 'offsuit'
    返回签名集合：{'aa','ax','ba','xx', ...}，高牌在前两位（与 hand_type 一致）
    关键：offsuit 需要 **包含 'xx'**，避免查表 miss。
    """
    # 从 pattern 里提取可用标签（不含 'x'）
    labels = sorted(set(turn_pattern))  # e.g. ['a','b'] 或 ['a','b','c','d']
    Lx = labels + ['x']  # 'x' 表示“非 turn 花色”

    if htype_kind == "suited":
        return {t + t for t in Lx}  # aa, bb, cc, dd, xx（无则自动被 realize 过滤）
    elif htype_kind == "offsuit":
        S = {a + b for a in Lx for b in Lx if a != b}
        S.add('xx')  # 关键：两张底牌都不在 turn 花色集合
        return S
    else:  # pair
        S = {a + b for a in Lx for b in Lx if a != b}
        S.add('xx')
        return S

# -------------------- 构造符合签名的具体底牌 --------------------
def realize_hand_for_signature(htype: str, turn4, target_sig: str):
    """
    htype: 'AKs','AKo','AA'
    turn4: 4 张 treys int
    target_sig: 'aa','ax','ba','xx' 等
    返回 [c0,c1]（c0 为高牌）或 None（无法构造）
    """
    used = set(turn4)

    # 首现花色映射（与 canonicalize_turn 保持一致）
    quads = [(RANKS.index(rank_char(c)) + 2, suit_char(c), c) for c in turn4]
    quads.sort(key=lambda t: (-t[0], t[1]))
    suit_seen = {}
    next_label = 0
    for _, s_ch, _ in quads:
        if s_ch not in suit_seen:
            suit_seen[s_ch] = next_label
            next_label += 1
    board_suits = set(suit_char(c) for c in turn4)

    def suit_candidates_for_token(token):
        if token == 'x':
            return [s for s in SUITS if s not in board_suits]
        desired_idx = ord(token) - ord('a')  # 0..3
        return [s for s, idx in suit_seen.items() if idx == desired_idx]

    # 解析 htype
    r_hi, r_lo = htype[0], htype[1]
    is_pair    = (r_hi == r_lo)
    is_suited  = (len(htype) == 3 and htype[2] == 's')
    # offsuit = len(htype) == 3 and htype[2] == 'o'   # 不需要显式用

    # 高牌在前
    if not is_pair and RANKS.index(r_hi) < RANKS.index(r_lo):
        r_hi, r_lo = r_lo, r_hi
    ranks_ordered = [r_hi, r_lo]

    t0, t1 = target_sig[0], target_sig[1]
    cand0 = suit_candidates_for_token(t0)
    cand1 = suit_candidates_for_token(t1)

    def pick_card(rank_ch, suit_list):
        for sch in suit_list:
            c = Card.new(rank_ch + sch)
            if c not in used:
                used.add(c)
                return c
        return None

    # 先高牌
    c0 = pick_card(ranks_ordered[0], cand0)
    if c0 is None:
        return None
    s0 = suit_char(c0)

    # 再低牌（处理同花/不同花约束）
    if is_suited:
        if s0 not in cand1:
            return None
        c1 = pick_card(ranks_ordered[1], [s0])
        if c1 is None:
            return None
    else:
        cand1_diff = [sch for sch in cand1 if sch != s0]
        c1 = pick_card(ranks_ordered[1], cand1_diff)
        if c1 is None:
            return None
        if is_pair and suit_char(c1) == s0:
            return None

    return [c0, c1]

# -------------------- Monte Carlo（6 人桌，随机 River） --------------------
def hs_6max_monte_on_turn(hand, turn4, n_sim=150, n_players=6):
    deck = Deck()
    used = set(hand + list(turn4))
    deck.cards = [c for c in deck.cards if c not in used]

    win = tie = 0
    for _ in range(n_sim):
        # 对手手牌
        opp_hands = [deck.draw(2) for _ in range(n_players - 1)]
        # 发 River
        river = deck.draw(1)
        board5 = list(turn4) + river

        hero = evaluator.evaluate(board5, hand)
        opps = [evaluator.evaluate(board5, oh) for oh in opp_hands]
        best = min([hero] + opps)

        winners = [s for s in [hero] + opps if s == best]
        if hero == best:
            if len(winners) == 1:
                win += 1
            else:
                tie += 1

        # 回收+洗牌
        deck.cards += [c for oh in opp_hands for c in oh] + river
        random.shuffle(deck.cards)

    return (win + 0.5 * tie) / n_sim

# -------------------- 单任务 --------------------
def worker_task(args):
    htype, key, turn4, pat, n_sim, n_players = args
    kind = 'pair' if len(htype) == 2 else ('suited' if htype[2] == 's' else 'offsuit')

    out = []
    for sig in sorted(all_target_signatures(pat, kind)):
        hand = realize_hand_for_signature(htype, turn4, sig)
        if hand is None:
            continue
        strength = hs_6max_monte_on_turn(hand, turn4, n_sim=n_sim, n_players=n_players)
        out.append({
            "hand_type": htype,
            "canon_turn": key,
            "turn_pattern": pat,
            "suit_signature": sig,
            "strength_mean": round(float(strength), 6),
            "n_sim": int(n_sim),
        })
    return out

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="turn_strength_table.csv")
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--quick", action="store_true",
                    help="冒烟：仅取前 300 个 turn 类（显著加快）")
    args = ap.parse_args()

    hand_types = hand_types_169()
    turns_info = enumerate_canonical_turns()

    if args.quick:
        turns_info = turns_info[:300]
        print(f"[Quick] using only {len(turns_info)} canonical turn classes.")

    tasks = []
    for key, t4, pat in turns_info:
        for htype in hand_types:
            tasks.append((htype, key, t4, pat, args.n_sim, args.players))

    print(f"Total tasks: {len(tasks)} | workers: {args.workers}")

    rows = []
    with Pool(processes=args.workers) as pool:
        for chunk in pool.imap_unordered(worker_task, tasks, chunksize=32):
            if chunk:
                rows.extend(chunk)

    df = pd.DataFrame(rows)
    if df.empty:
        raise AssertionError("[BUG] empty dataframe")

    # 强校验
    if not ((df['strength_mean'] >= 0.0) & (df['strength_mean'] <= 1.0)).all():
        raise AssertionError("[BUG] strength out of [0,1]")

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(f"Distinct canonical turns: {df['canon_turn'].nunique()}")
    print(df.head(8))

if __name__ == "__main__":
    main()
