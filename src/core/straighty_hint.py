# coding: utf-8
"""
straighty_hint
--------------
公共牌顺子潜力启发式，返回 0~1。

设计目标：
1. 必须在“5 连窗口”里才算有顺子味：max(rank) - min(rank) <= 4
2. 不仅看覆盖了多少种点数，还要看点数之间的“紧密程度”：
   - 相邻差 1（比如 5-6）：记 1.0 分
   - 相邻差 2（比如 5-7）：记 0.5 分
   - 差 >=3：记 0 分（断层太大）
3. 对去重后的有序点数 ranks = [r1 < r2 < ... < rk]:
   - 紧密度得分 score_raw = sum(gap_score(ri+1 - ri))
   - 理论紧密度满分 max_score = (k - 1) * 1.0 （所有 gap 都是 1 的时候）
   - tightness = score_raw / max_score  （若 max_score == 0 则为 0）
4. 5 连窗口最多覆盖 5 种不同点数，用 coverage = k / 5.0 表示覆盖率（上限封顶为 1.0）
   - 最终 straighty_hint = tightness * coverage
     - 这样完整 5 连（覆盖 5 张点数且完全紧凑）才会是 1.0
     - 像 [5,6,7] 这种只有 3 张连牌，顺子味会低于完整 5 连
5. A 特判：A 既可以当 14，也可以当 1（轮顺 A2345），取两种视角里更大的 straighty_hint。
"""

from typing import List
import math

# 跟你项目其他地方统一的 rank 映射
RANK_VALUE_MAP = {
    'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    'R8': 8, 'R9': 9, 'RT': 10, 'RJ': 11, 'RQ': 12, 'RK': 13, 'RA': 14,
}


def _extract_ranks(board_cards) -> (List[int], bool):
    """
    把 pokers.Card 列表转成点数列表 + 是否含 A
    返回:
        vals: List[int]  (2..14)
        has_ace: bool
    """
    vals = []
    has_ace = False
    for c in board_cards:
        rstr = str(c.rank).split('.')[-1]  # 'R2'...'RA'
        v = RANK_VALUE_MAP.get(rstr)
        if v is None:
            continue
        vals.append(v)
        if v == 14:
            has_ace = True
    return vals, has_ace


def _straighty_hint_single_variant(ranks: List[int]) -> float:
    """
    在“某一种视角”（比如正常视角或 A→1 视角）下计算 straighty_hint。
    ranks: 已去重 & 排序 的点数列表，如 [5,6,8]。
    """
    # 少于 2 种点数，不可能有顺子结构
    if len(ranks) < 2:
        return 0.0

    rmin, rmax = ranks[0], ranks[-1]
    # 必须在某个 5 连窗口内，否则顺子味直接视为 0
    if rmax - rmin > 4:
        return 0.0

    # 1）计算相邻 gap 的“紧密度得分”
    score_raw = 0.0
    for i in range(len(ranks) - 1):
        d = ranks[i + 1] - ranks[i]
        if d == 1:
            score_raw += 1.0
        elif d == 2:
            score_raw += 0.5
        else:
            # 差 3+ 视为断层，不加分
            pass

    max_score = float(len(ranks) - 1)
    if max_score <= 0.0:
        return 0.0

    # 连线紧密度：0~1
    tightness = score_raw / max_score
    tightness = max(0.0, min(1.0, tightness))

    # 2）覆盖率：5 连窗口最多 5 种点数
    coverage = min(1.0, len(ranks) / 5.0)

    # 3）综合评分
    return float(tightness * coverage)


def straighty_hint(board_cards) -> float:
    """
    公共牌顺子潜力提示（0~1）。

    关键性质（大致趋势示例）：
      - [5,6,7,8,9]        → 1.0   （完整 5 连）
      - [5,6,7]            → 0.6   （3 张紧连，有顺子味但不如完整 5 连）
      - [5,6,8]            → 0.45  （比 5,7,9 更紧）
      - [5,7,9]            → 0.3
      - [2,2,2,6]          → 0.0   （虽然 max-min=4，但实际上没顺子结构）
      - [9,T,J,Q,K]        → 1.0   （9~K 完整 5 连）
      - [A,2,3,4,5]        → 1.0   （轮顺视角 A2345）
      - [2,8,K]            → 0.0   （完全没顺子味）
    """
    if not board_cards:
        return 0.0

    vals, has_ace = _extract_ranks(board_cards)
    if not vals:
        return 0.0

    best_hint = 0.0

    # 视角 1：正常 2..14
    ranks_norm = sorted(set(vals))
    best_hint = max(best_hint, _straighty_hint_single_variant(ranks_norm))

    # 视角 2：A 当 1（轮顺 A2345）
    if has_ace:
        vals_wheel = [1 if v == 14 else v for v in vals]
        ranks_wheel = sorted(set(vals_wheel))
        best_hint = max(best_hint, _straighty_hint_single_variant(ranks_wheel))

    return float(best_hint)
