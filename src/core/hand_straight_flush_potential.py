# coding: utf-8
"""
hand_straight_flush_potential: 在已知 Hero 手牌 + 公共牌的前提下，
估计“至少有一个对手在当前牌面上已经是同花顺（任意 5 连同花）”的概率（0~1）。

思路与 hand_straighty_potential / hand_flushy_potential 完全一致：
  - 从整副牌中扣掉 Hero 手牌 + 公共牌 → 未见牌堆
  - 枚举所有可能的对手两张牌组合 (a,b)
  - 看这些组合里，有多少可以和当前公共牌组成同花顺（5 连 + 同花）
  - 用组合公式：P(至少一人同花顺) = 1 - C(T-S, n) / C(T, n)
"""

from itertools import combinations
from math import comb

# suit 映射方式要和项目内一致
SUIT_CHAR_MAP = {
    'Spades':   's',
    'Hearts':   'h',
    'Diamonds': 'd',
    'Clubs':    'c',
}

_ALL_RANK_CHARS = "23456789TJQKA"   # 2..A
_ALL_SUIT_CHARS = "shdc"            # s,h,d,c


def _card_rank_char(card):
    """
    从 pkrs.Card 提取 rank 字符（'2'...'9','T','J','Q','K','A'）
    与项目里 LUT / straight 模块保持同一风格：
      str(card.rank) -> 'R2'..'RA' → 去掉前缀 'R'
    """
    rkey = str(card.rank).split('.')[-1]  # e.g. 'R4','RA'
    if rkey.startswith('R'):
        base = rkey[1:]  # 去掉 'R'
    else:
        base = rkey[-1]  # 防御式退化
    return base


def _card_suit_char(card):
    """
    从 pkrs.Card 提取花色字符：s/h/d/c
    """
    skey = str(card.suit).split('.')[-1]  # 'Spades','Hearts',...
    return SUIT_CHAR_MAP.get(skey, '?' )


def _cards_to_rs_list(cards):
    """
    pkrs.Card 列表 -> ['4d','5c','As'] 这样的 rs 串
    """
    rs_list = []
    for c in cards:
        rch = _card_rank_char(c)
        sch = _card_suit_char(c)
        rs_list.append(rch + sch)
    return rs_list


def _unseen_deck_rs(hero_cards, board_cards):
    """
    从整副 52 张牌中，去掉 hero 手牌 + 公共牌，返回剩余牌的 rs 列表。
    """
    seen = set(_cards_to_rs_list(hero_cards) + _cards_to_rs_list(board_cards))
    deck = []
    for r in _ALL_RANK_CHARS:
        for s in _ALL_SUIT_CHARS:
            rs = r + s
            if rs not in seen:
                deck.append(rs)
    return deck


def _rank_to_int(rch: str) -> int:
    """
    '2'..'9','T','J','Q','K','A' -> 2..14
    """
    if rch.isdigit():
        return int(rch)
    if rch == 'T':
        return 10
    if rch == 'J':
        return 11
    if rch == 'Q':
        return 12
    if rch == 'K':
        return 13
    if rch == 'A':
        return 14
    return 0


def _has_straight_flush_rs(board_rs, hole_rs):
    """
    当前牌面（board_rs + hole_rs）是否已经有同花顺（任意 5 连同花）。
    board_rs: 公共牌 rs 列表
    hole_rs:  对手两张牌 rs 列表
    """
    cards = board_rs + hole_rs

    # 按花色分组
    by_suit = {s: set() for s in _ALL_SUIT_CHARS}
    for rs in cards:
        if len(rs) < 2:
            continue
        r, s = rs[0], rs[1]
        if s in by_suit:
            by_suit[s].add(_rank_to_int(r))

    # 对每个花色，判断是否存在 5 连
    for s in _ALL_SUIT_CHARS:
        ranks = sorted(by_suit[s])
        if not ranks:
            continue

        # A 既可以当 14 也可以当 1，处理 wheel（A2345）
        if 14 in ranks:
            ranks = ranks + [1]
            ranks = sorted(set(ranks))

        # 找长度 >=5 的连续序列
        if len(ranks) < 5:
            continue

        # sliding window 找 5 连
        # 例如 ranks = [1,2,3,4,5,7,8,...]
        for i in range(len(ranks) - 4):
            window = ranks[i:i+5]
            if window[-1] - window[0] == 4 and len(set(window)) == 5:
                # 说明是 k, k+1, k+2, k+3, k+4
                return True

    return False


def _board_already_straight_flush(board_rs):
    """
    仅看公共牌，是否已经自带同花顺（极罕见，但逻辑上可以处理）。
    """
    return _has_straight_flush_rs(board_rs, [])


def _count_straight_flush_pairs_on_board(hero_cards, board_cards):
    """
    在当前公共牌 board_cards（pkrs.Card 列表）下：
      - 从剩余牌堆里选任意两张 (a,b)
      - 判断这些两张和公共牌能否组成同花顺
    返回:
      S = 成同花顺的两张组合数量
      T = 所有两张组合总数 C(U,2)
    """
    deck_rs = _unseen_deck_rs(hero_cards, board_cards)
    U = len(deck_rs)
    if U < 2:
        return 0, 0
    T = comb(U, 2)

    board_rs = _cards_to_rs_list(board_cards)
    S = 0
    for a, b in combinations(deck_rs, 2):
        if _has_straight_flush_rs(board_rs, [a, b]):
            S += 1
    return S, T


def hand_straight_flush_potential(hero_cards, board_cards, n_opponents: int) -> float:
    """
    计算当前街道【至少一个对手已经是同花顺】的概率（0~1）：

      - hero_cards: 我方两张 pkrs.Card
      - board_cards: 公共牌 pkrs.Card 列表（Flop 之后 len >= 3 才有意义）
      - n_opponents: 当前还在局里的对手人数（active=True 且不是 Hero）

    处理逻辑：
      - Preflop / 对手数 0 / Hero 牌不完整 → 0
      - 公共牌本身就是同花顺 → 1
      - 否则用组合精确公式：
            P(至少一人同花顺)
              = 1 - C(T-S, n) / C(T, n)
        其中：
            T = 所有可行的两张手牌组合数（在 Hero+Board 已知条件下）
            S = 其中能与当前牌面组成同花顺的组合数
    """
    # 基本防御
    if n_opponents <= 0 or len(board_cards) < 3 or len(hero_cards) != 2:
        return 0.0

    board_rs = _cards_to_rs_list(board_cards)
    # 公共牌本身如果就构成同花顺，必有人有（至少公共牌那条）
    if _board_already_straight_flush(board_rs):
        return 1.0

    S, T = _count_straight_flush_pairs_on_board(hero_cards, board_cards)
    if S == 0 or T == 0:
        return 0.0
    if T < n_opponents:
        return 0.0

    try:
        no_sf = comb(T - S, n_opponents) / comb(T, n_opponents)
        return float(1.0 - no_sf)
    except OverflowError:
        # 极少见溢出时退回独立近似：1 - (1-p)^n
        p_single = S / T
        return float(1.0 - (1.0 - p_single) ** n_opponents)
