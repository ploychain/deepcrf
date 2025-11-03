def compute_straighty_hint(board_cards):
    """
    评估公共牌的顺子潜力（0~1）
    - A 可作高牌或低牌
    - 返回值越高，说明连性越强（容易成顺）
    """
    if not board_cards:
        return 0.0

    rank_map = {'R2':2,'R3':3,'R4':4,'R5':5,'R6':6,'R7':7,'R8':8,
                'R9':9,'RT':10,'RJ':11,'RQ':12,'RK':13,'RA':14}
    ranks = []
    for c in board_cards:
        r = str(c.rank).split('.')[-1]
        if r in rank_map:
            ranks.append(rank_map[r])

    ranks = sorted(set(ranks))
    if not ranks:
        return 0.0

    # Ace 既可作 14 也可作 1
    if 14 in ranks:
        ranks.append(1)
        ranks = sorted(set(ranks))

    # 找最长连续段
    max_run = run = 1
    for i in range(1, len(ranks)):
        if ranks[i] - ranks[i-1] == 1:
            run += 1
            max_run = max(max_run, run)
        elif ranks[i] != ranks[i-1]:
            run = 1

    # 连续长度映射到 [0,1]
    hint = min(1.0, max_run / 5.0)
    return hint
