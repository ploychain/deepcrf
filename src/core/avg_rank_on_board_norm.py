# coding: utf-8
"""
avg_rank_on_board_norm
----------------------
公共牌平均等级（归一化到 [0,1]）。
用于判断牌面是偏低、中、还是偏高。
"""

RANK_VALUE_MAP = {
    'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    'R8': 8, 'R9': 9, 'RT': 10, 'RJ': 11, 'RQ': 12, 'RK': 13, 'RA': 14,
}

def avg_rank_on_board_norm(board_cards) -> float:
    """
    :param board_cards: list[pokers.Card]
    :return: float in [0,1]
    """
    if not board_cards:
        return 0.0

    ranks = []
    for c in board_cards:
        rstr = str(c.rank).split('.')[-1]
        v = RANK_VALUE_MAP.get(rstr)
        if v:
            ranks.append(v)
    if not ranks:
        return 0.0

    avg = sum(ranks) / len(ranks)
    norm = (avg - 2.0) / 12.0  # 2~14 → 0~1
    if norm < 0:
        norm = 0.0
    elif norm > 1:
        norm = 1.0
    return float(norm)
