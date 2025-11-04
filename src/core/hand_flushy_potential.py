# coding: utf-8
"""
hand_flushy_potential: 计算当前街道，
在已知 Hero 手牌 + 公共牌的前提下，
至少有一个对手已经成同花（5+ 张同花）的概率（0~1）。
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
    这里和顺子那个文件保持一致（用 R2..RA → 2..A）
    """
    rkey = str(card.rank).split('.')[-1]  # e.g. 'R4','RA'
    if not rkey.startswith('R'):
        # 防御式：如果库格式变了，退回原字符串最后一位
        base = rkey[-1]
    else:
        base = rkey[1:]  # 去掉前面的 'R'
    if base in ['T', 'J', 'Q', 'K', 'A']:
        return base
    return base  # '2'..'9'


def _card_suit_char(card):
    """
    从 pkrs.Card 提取花色字符：s/h/d/c
    """
    skey = str(card.suit).split('.')[-1]  # 'Spades','Hearts',...
    return SUIT_CHAR_MAP.get(skey, '?')


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
    与 straight 模块一致：全牌空间 = 13 * 4
    """
    seen = set(_cards_to_rs_list(hero_cards) + _cards_to_rs_list(board_cards))
    deck = []
    for r in _ALL_RANK_CHARS:
        for s in _ALL_SUIT_CHARS:
            rs = r + s
            if rs not in seen:
                deck.append(rs)
    return deck


def _has_flush_rs(board_rs, hole_rs):
    """
    当前牌面（board_rs + hole_rs）是否已经有任意花色的同花（>=5 张）
    board_rs: 公共牌 rs 列表，如 ['4d','5c','6s']
    hole_rs:  对手的两张牌 rs 列表
    """
    suit_cnt = {}
    for rs in board_rs + hole_rs:
        s = rs[-1]  # 最后一位是花色字符
        suit_cnt[s] = suit_cnt.get(s, 0) + 1
    # 任意花色 >=5 就算有同花
    return any(cnt >= 5 for cnt in suit_cnt.values())


def _board_already_flush(board_rs):
    """
    仅看公共牌，是否已经自带同花（>=5 张同花）
    这种情况：不管对手拿什么牌，都“有人有同花”（至少是公共牌本身的同花），
    对于“有人有同花”的事件，概率可以视作 1。
    """
    suit_cnt = {}
    for rs in board_rs:
        s = rs[-1]
        suit_cnt[s] = suit_cnt.get(s, 0) + 1
    return any(cnt >= 5 for cnt in suit_cnt.values())


def _count_flush_pairs_on_board(hero_cards, board_cards):
    """
    在当前公共牌 board_cards（pkrs.Card 列表）下：
      - 从剩余牌堆里选任意两张 (a,b)
      - 判断这些两张和公共牌能否组成同花（>=5 张同花）
    返回:
      S = 成同花的两张组合数量
      T = 所有两张组合总数 C(U,2)
    """
    deck_rs = _unseen_deck_rs(hero_cards, board_cards)
    U = len(deck_rs)
    if U < 2:
        return 0, 0
    T = comb(U, 2)

    board_rs = _cards_to_rs_list(board_cards)
    # 如果公共牌本身就已经成同花（>=5），
    # 那么任意对手拿任意两张牌，都算“有人有同花”
    # 在 hand_flushy_potential 里会直接返回 1，这里不特殊处理 S。
    S = 0
    for a, b in combinations(deck_rs, 2):
        if _has_flush_rs(board_rs, [a, b]):
            S += 1
    return S, T


def hand_flushy_potential(hero_cards, board_cards, n_opponents: int) -> float:
    """
    计算当前街道【别人成同花的概率】（0~1）：

      - hero_cards: 我方两张 pkrs.Card
      - board_cards: 公共牌 pkrs.Card 列表（>= Flop 才有意义，len >= 3）
      - n_opponents: 当前还在局里的对手人数（active=True 且不是 Hero）

    定义为：在所有未知牌随机发给 n_opponents 个对手的情况下，
            至少有一个对手已经在当前牌面上有同花（5+ 张同花）的概率。
    """
    # 没公共牌 / 没对手 / Hero 不是两张牌，直接 0
    if n_opponents <= 0 or len(board_cards) < 3 or len(hero_cards) != 2:
        return 0.0

    board_rs = _cards_to_rs_list(board_cards)
    # 若公共牌本身就有同花（>=5），可视为“必有人有同花”
    # （虽然谁赢要看牌面强度，但“同花存在”这个事件概率为 1）
    if _board_already_flush(board_rs):
        return 1.0

    S, T = _count_flush_pairs_on_board(hero_cards, board_cards)
    if S == 0 or T == 0:
        return 0.0
    if T < n_opponents:
        return 0.0

    # 组合精确公式：
    # P(至少一人同花) = 1 - C(T-S, n) / C(T, n)
    try:
        no_flush = comb(T - S, n_opponents) / comb(T, n_opponents)
        return float(1.0 - no_flush)
    except OverflowError:
        # 极少见溢出时退回独立近似：1 - (1-p)^n
        p_single = S / T
        return float(1.0 - (1.0 - p_single) ** n_opponents)
