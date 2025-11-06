# coding: utf-8
import numpy as np

RANK_MAP = {
    'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    'R8': 8, 'R9': 9, 'RT': 10, 'RJ': 11, 'RQ': 12, 'RK': 13, 'RA': 14
}

def board_gap_norm(board_cards) -> float:
    """
    高牌-低牌差距（归一化）。小差距 = 连性强。
    :param board_cards: list[pokers.Card]
    :return: float, 0~1
    """
    if not board_cards:
        return 0.0
    try:
        ranks = []
        for c in board_cards:
            r = str(c.rank).split('.')[-1]
            if r in RANK_MAP:
                ranks.append(RANK_MAP[r])
        if not ranks:
            return 0.0
        gap = max(ranks) - min(ranks)
        return min(1.0, gap / 12.0)
    except Exception:
        return 0.0
