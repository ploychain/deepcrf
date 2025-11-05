# coding: utf-8
"""
hand_royal_flush_potential: 在已知 Hero 手牌 + 公共牌的前提下，
估计“至少有一个对手在当前牌面上已经是皇家同花顺”的概率（0~1）。

风格与 hand_straighty_potential / hand_flushy_potential 完全一致：
  - 先从整副牌 52 张里扣掉 Hero 手牌和公共牌 → 未见牌堆
  - 枚举所有可能的对手两张牌组合 (a,b)
  - 统计其中有多少组合与当前公共牌能构成皇家同花顺
  - 用组合公式算：P(至少一人皇家同花顺) = 1 - C(T-S, n) / C(T, n)
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
_ROYAL_RANK_SET = set("TJQKA")      # 皇家同花顺必须包含的点数


def _card_rank_char(card):
    """
    从 pkrs.Card 提取 rank 字符（'2'...'9','T','J','Q','K','A'）
    与你项目里 LUT / straight 模块保持同一风格：
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


def _has_royal_flush_rs(board_rs, hole_rs):
    """
    当前牌面（board_rs + hole_rs）是否已经有皇家同花顺：
      - 某个花色 s，包含 T,J,Q,K,A 这五张牌。
    board_rs: 公共牌 rs 列表
    hole_rs:  对手的两张牌 rs 列表
    """
    cards = board_rs + hole_rs
    by_suit = {s: set() for s in _ALL_SUIT_CHARS}
    for rs in cards:
        r, s = rs[0], rs[1]
        if s in by_suit:
            by_suit[s].add(r)
    # 任意花色包含所有皇家点数，就算有皇家同花顺
    for s in _ALL_SUIT_CHARS:
        if _ROYAL_RANK_SET.issubset(by_suit[s]):
            return True
    return False


def _board_already_royal(board_rs):
    """
    仅看公共牌，是否已经自带皇家同花顺（很罕见，但逻辑上要处理）。
    这种情况下，不管对手拿什么牌，“有人有皇家同花顺”这个事件视作概率 1。
    """
    return _has_royal_flush_rs(board_rs, [])


def _count_royal_pairs_on_board(hero_cards, board_cards):
    """
    在当前公共牌 board_cards（pkrs.Card 列表）下：
      - 从剩余牌堆里选任意两张 (a,b)
      - 判断这些两张和公共牌能否组成皇家同花顺
    返回:
      S = 成皇家同花顺的两张组合数量
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
        if _has_royal_flush_rs(board_rs, [a, b]):
            S += 1
    return S, T


def hand_royal_flush_potential(hero_cards, board_cards, n_opponents: int) -> float:
    """
    计算当前街道【至少一个对手已经是皇家同花顺】的概率（0~1）：

      - hero_cards: 我方两张 pkrs.Card
      - board_cards: 公共牌 pkrs.Card 列表（Flop 之后 len >= 3 才有意义）
      - n_opponents: 当前还在局里的对手人数（active=True 且不是 Hero）

    处理逻辑：
      - Preflop / 对手数 0 / Hero 牌不完整 → 0
      - 公共牌本身就是皇家同花顺 → 1
      - 否则用组合精确公式：
            P(至少一人皇家同花顺)
              = 1 - C(T-S, n) / C(T, n)
        其中：
            T = 所有可行的两张手牌组合数（在 Hero+Board 已知条件下）
            S = 其中能与当前牌面组成皇家同花顺的组合数
    """
    # 基本防御
    if n_opponents <= 0 or len(board_cards) < 3 or len(hero_cards) != 2:
        return 0.0

    board_rs = _cards_to_rs_list(board_cards)
    # 公共牌本身如果就构成皇家同花顺，必有人有
    if _board_already_royal(board_rs):
        return 1.0

    S, T = _count_royal_pairs_on_board(hero_cards, board_cards)
    if S == 0 or T == 0:
        return 0.0
    if T < n_opponents:
        return 0.0

    try:
        no_royal = comb(T - S, n_opponents) / comb(T, n_opponents)
        return float(1.0 - no_royal)
    except OverflowError:
        # 极少见溢出时退回独立近似：1 - (1-p)^n
        p_single = S / T
        return float(1.0 - (1.0 - p_single) ** n_opponents)
