# coding: utf-8
# src/core/highcard_on_board_norm.py

RANK_VALUE_MAP = {
    'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    'R8': 8, 'R9': 9, 'RT': 10, 'RJ': 11, 'RQ': 12, 'RK': 13, 'RA': 14
}

def highcard_on_board_norm(board_cards) -> float:
    """
    输入: board_cards = state.public_cards (list of pkrs.Card)
    输出: float (0~1), 表示公共牌中最高 rank 的归一化值。
      A -> 1.0
      K -> 13/14 ≈ 0.9286
      2 -> 2/14 ≈ 0.1429
    """
    if not board_cards:
        return 0.0

    max_rank = 0
    for c in board_cards:
        try:
            rstr = str(c.rank).split('.')[-1]  # e.g. 'R5','RA'
            if rstr in RANK_VALUE_MAP:
                rv = RANK_VALUE_MAP[rstr]
            else:
                rv = int(rstr.replace('R', ''))
            max_rank = max(max_rank, rv)
        except Exception:
            continue

    # 归一化：A(14)->1.0
    return float(max_rank) / 14.0
