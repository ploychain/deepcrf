# coding: utf-8
"""
lowcard_on_board_norm
---------------------
计算公共牌中最低 rank 的归一化值 (0~1)，A=1.0 表示最低牌是 A（即 rank 最低）。
用于辅助判断公共牌的连性（低端是否紧密）。
"""

import numpy as np

def lowcard_on_board_norm(board_cards):
    """
    参数:
        board_cards: list[pokers.Card] —— 公共牌列表，可为空

    返回:
        float ∈ [0,1]
        - 无公共牌时返回 0.0
        - 最高 rank = A (14)，最低 rank = 2
          归一化公式： (min_rank - 2) / 12
          例如：
            [2♠, 5♥, 9♦] ->  (2 - 2)/12 = 0.0000
            [4♣, 7♦, K♠] ->  (4 - 2)/12 = 0.1667
            [A♣, 9♥, Q♠] ->  (1 - 2)/12 = 0.0000 （A当1时更低）
    """
    if not board_cards:
        return 0.0

    try:
        ranks = []
        for c in board_cards:
            r_str = str(c.rank).split('.')[-1]
            if r_str == 'RA':
                ranks.append(14)  # A作为14
                ranks.append(1)   # 也考虑A作为1，取更小
            elif r_str == 'RK':
                ranks.append(13)
            elif r_str == 'RQ':
                ranks.append(12)
            elif r_str == 'RJ':
                ranks.append(11)
            elif r_str == 'RT':
                ranks.append(10)
            else:
                ranks.append(int(r_str.replace('R', '')))

        # 取最小的 rank（考虑 A=1）
        min_rank = min(ranks)

        # 归一化：2 → 0, 14 → 1，但我们是“低牌归一化”，所以 2 越小越接近 0
        norm = (min_rank - 2) / 12.0
        return float(np.clip(norm, 0.0, 1.0))

    except Exception:
        return 0.0
