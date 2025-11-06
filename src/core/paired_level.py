# coding: utf-8
"""
paired_level
------------
统计公共牌上的“配对层级”：
0=无对, 1=一对, 2=两对, 3=三条, 4=四条。
"""

from collections import Counter

RANK_VALUE_MAP = {
    'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    'R8': 8, 'R9': 9, 'RT': 10, 'RJ': 11, 'RQ': 12, 'RK': 13, 'RA': 14,
}

def paired_level(board_cards) -> int:
    """
    :param board_cards: list[pokers.Card]
    :return: int in [0,4]
    """
    if not board_cards:
        return 0

    ranks = []
    for c in board_cards:
        rstr = str(c.rank).split('.')[-1]
        if rstr in RANK_VALUE_MAP:
            ranks.append(RANK_VALUE_MAP[rstr])

    if not ranks:
        return 0

    cnt = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)

    # 映射规则：
    #  [1,1,1,1,1] → 0
    #  [2,1,1,1]   → 1
    #  [2,2,1]     → 2
    #  [3,1,1]     → 3
    #  [4,1]       → 4
    #  [3,2]       → 3 （葫芦算三条层级）
    if counts[0] == 4:
        return 4
    if counts[0] == 3:
        return 3
    if counts.count(2) == 2:
        return 2
    if counts[0] == 2:
        return 1
    return 0
