# coding: utf-8
"""
hand_straighty_potential: 计算当前街道，
在给定公共牌 + Hero 手牌的前提下，
至少有一个对手已经成顺子的概率（0~1）。
"""

from itertools import combinations
from math import comb

# 注意：这里的 rank/suit 映射，要和你主工程里保持一致
RANK_CHAR_MAP = {
    'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6', 'R7': '7',
    'R8': '8', 'R9': '9', 'RT': 'T', 'RJ': 'J', 'RQ': 'Q', 'RK': 'K', 'RA': 'A'
}
SUIT_CHAR_MAP = {
    'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'
}

# 所有可能的顺子窗口（A 既当 1 也当 14）
_STRAIGHT_WINDOWS = [
    {1, 2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    {3, 4, 5, 6, 7},
    {4, 5, 6, 7, 8},
    {5, 6, 7, 8, 9},
    {6, 7, 8, 9, 10},
    {7, 8, 9, 10, 11},
    {8, 9, 10, 11, 12},
    {9, 10, 11, 12, 13},
    {10, 11, 12, 13, 14},
]

_ALL_RANK_CHARS = "23456789TJQKA"
_ALL_SUIT_CHARS = "shdc"  # 和 SUIT_CHAR_MAP 的 value 对应：s,h,d,c


def _rchar_to_nums(ch: str):
    """rank 字符 -> 数值集合（A→{1,14}，其它单值）"""
    if ch == 'A':
        return {1, 14}
    if ch == 'K':
        return {13}
    if ch == 'Q':
        return {12}
    if ch == 'J':
        return {11}
    if ch == 'T':
        return {10}
    # '2'..'9'
    return {int(ch)}


def _cards_to_rs_list(cards):
    """
    pkrs.Card 列表 -> ['4d','5c','6s'] 这样的 rs 串
    这里用 str(card.rank).split('.')[-1] 拿到 'R4'，
    和主工程是一致的。
    """
    rs_list = []
    for c in cards:
        rkey = str(c.rank).split('.')[-1]   # e.g. 'R4'
        skey = str(c.suit).split('.')[-1]   # e.g. 'Diamonds'
        rch = RANK_CHAR_MAP[rkey]           # '4'
        sch = SUIT_CHAR_MAP[skey]           # 'd'
        rs_list.append(rch + sch)
    return rs_list


def _unseen_deck_rs(hero_cards, board_cards):
    """
    从整副 52 张牌中，去掉 hero 手牌 + 公共牌，返回剩余牌的 rs 列表
    """
    seen = set(_cards_to_rs_list(hero_cards) + _cards_to_rs_list(board_cards))
    deck = []
    for r in _ALL_RANK_CHARS:
        for s in _ALL_SUIT_CHARS:
            rs = r + s
            if rs not in seen:
                deck.append(rs)
    return deck


def _has_straight_rs(board_rs, hole_rs):
    """
    当前牌面（board_rs + hole_rs）是否存在任意 5 连点数顺子
    board_rs: 公共牌 rs 列表
    hole_rs:  某个对手手牌 rs 列表，长度 2
    """
    nums = set()
    for rs in board_rs + hole_rs:
        rch = rs[0]
        nums |= _rchar_to_nums(rch)
    for w in _STRAIGHT_WINDOWS:
        if w.issubset(nums):
            return True
    return False


def _count_straight_pairs_on_board(hero_cards, board_cards):
    """
    在当前公共牌 board_cards（pkrs.Card 列表）下：
      - 从剩余牌堆里选任意两张(a,b)
      - 判断这些两张和公共牌能否组成顺子
    返回:
      S = 成顺的两张组合数量
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
        if _has_straight_rs(board_rs, [a, b]):
            S += 1
    return S, T


def hand_straighty_potential(hero_cards, board_cards, n_opponents: int) -> float:
    """
    计算当前街道【别人成顺的概率】（0~1）：
      - hero_cards: 我方两张 pkrs.Card
      - board_cards: 公共牌 pkrs.Card 列表（>= Flop）
      - n_opponents: 当前还在局里的对手人数
    定义为：在所有未知牌随机发给 n_opponents 个对手的情况下，
            至少有一个对手已经在当前牌面上有顺子的概率。
    """
    # 没公共牌 / 没对手，直接 0
    if n_opponents <= 0 or len(board_cards) < 3 or len(hero_cards) != 2:
        return 0.0

    S, T = _count_straight_pairs_on_board(hero_cards, board_cards)
    if S == 0 or T == 0:
        return 0.0
    if T < n_opponents:
        return 0.0

    # 组合精确公式：
    # P(至少一人顺子) = 1 - C(T-S, n) / C(T, n)
    try:
        no_straight = comb(T - S, n_opponents) / comb(T, n_opponents)
        return float(1.0 - no_straight)
    except OverflowError:
        # 极少见溢出时退回到独立近似：1 - (1-p)^n
        p_single = S / T
        return float(1.0 - (1.0 - p_single) ** n_opponents)
