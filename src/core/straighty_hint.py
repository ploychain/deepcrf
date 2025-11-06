# coding: utf-8
"""
straighty_hint
--------------
公共牌顺子潜力启发式，返回 0~1。

设计目标：
1. 必须在“5 连窗口”里才算有顺子味：max(rank) - min(rank) <= 4
2. 看三层信息：
   (1) 点数之间的“紧密程度”（tightness）
   (2) 当前覆盖了多少种点数（coverage）
   (3) 这些牌可以嵌入多少种 5 连序列（window_factor，区分两头顺 / 单头顺）

   - 相邻差 1（比如 5-6）：记 1.0 分
   - 相邻差 2（比如 5-7）：记 0.5 分
   - 差 >=3：记 0 分（断层太大）

   对去重后的有序点数 ranks = [r1 < r2 < ... < rk]:

   紧密度：
      score_raw = sum(gap_score(ri+1 - ri))
      max_score = (k - 1) * 1.0 （所有 gap 都是 1 的时候）
      tightness = score_raw / max_score   （0~1）

   覆盖率：
      coverage = k / 5.0 （最多 5 种点数，上限封为 1.0）

   多窗口因子（两头顺/单头顺）：
      - 在允许的点数范围 [variant_min, variant_max] 内，
        统计所有长度为 5 的窗口 [x, x+1, x+2, x+3, x+4]，
        其中必须满足 ranks 的所有点数都在这个窗口里。
      - 实际窗口数 = num_windows
      - 理论最大窗口数 = 5 - (rmax - rmin)
      - window_factor = num_windows / (5 - (rmax - rmin))  （0~1）

   最终：
      straighty_hint_single = tightness * coverage * window_factor

3. A 特判：A 既可以当 14，也可以当 1（轮顺 A2345），取两种视角里更大的 straighty_hint。
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


def _straighty_hint_single_variant(ranks: List[int],
                                   variant_min: int,
                                   variant_max: int) -> float:
    ranks = sorted(set(ranks))

    if len(ranks) < 2:
        return 0.0

    rmin, rmax = ranks[0], ranks[-1]

    # ==========================================================
    # 原逻辑：要求整体在 5 连窗口内，否则 0
    # ==========================================================
    if rmax - rmin > 4:
        # ✅ 新增补丁：检测局部连续段（比如 9-T-J）
        cont_max = 1
        cur = 1
        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i - 1] + 1:
                cur += 1
                cont_max = max(cont_max, cur)
            else:
                cur = 1

        # 如果发现至少 3 张连续，则给一个弱顺子味（0.3~0.5）
        if cont_max >= 3:
            base = (cont_max - 2) / 3.0  # 3连=0.33, 4连=0.66, 5连=1
            return 0.3 + 0.2 * min(base, 1.0)
        return 0.0
    # ==========================================================

    # 以下保持你原有逻辑不变
    if variant_max - variant_min + 1 < 5:
        return 0.0

    score_raw = 0.0
    for i in range(len(ranks) - 1):
        d = ranks[i + 1] - ranks[i]
        if d == 1:
            score_raw += 1.0
        elif d == 2:
            score_raw += 0.5

    max_score = float(len(ranks) - 1)
    if max_score <= 0.0:
        return 0.0

    tightness = score_raw / max_score
    tightness = max(0.0, min(1.0, tightness))
    coverage = min(1.0, len(ranks) / 5.0)

    start_low = max(variant_min, rmax - 4)
    start_high = min(variant_max - 4, rmin)
    num_windows = max(0, start_high - start_low + 1)

    span = rmax - rmin
    denom = 5 - span
    window_factor = num_windows / float(denom) if denom > 0 else 0.0
    window_factor = max(0.0, min(1.0, window_factor))

    return float(tightness * coverage * window_factor)



def straighty_hint(board_cards) -> float:
    """
    公共牌顺子潜力提示（0~1）。

    引入“两头顺 / 单头顺”的区分，并考虑 A 的两种视角。

    大致趋势（示例）：
      - [9,T,J,Q,K]        → ~1.0   （完整 5 连）
      - [T,J,Q]            → ~0.6   （三张紧连，居中，两头开放，多窗口）
      - [J,Q,K]            → ~0.4   （三张紧连，靠近上边界，窗口略少）
      - [Q,K,A]            → ~0.2   （三张紧连，最上方，几乎单头顺）
      - [5,6,7]            → ~0.6   （三张紧连，中间偏下）
      - [5,6,8]            → ~0.45
      - [5,7,9]            → ~0.3
      - [2,2,2,6]          → 0.0    （虽然 max-min=4，但实际上没顺子结构）
      - [9,T,J,Q,K]        → 1.0    （9~K 完整 5 连）
      - [A,2,3,4,5]        → 1.0    （轮顺视角 A2345）
      - [2,8,K]            → 0.0    （完全没顺子味）
    """
    if not board_cards:
        return 0.0

    vals, has_ace = _extract_ranks(board_cards)
    if not vals:
        return 0.0

    best_hint = 0.0

    # 视角 1：正常 2..14
    ranks_norm = sorted(set(vals))
    best_hint = max(best_hint, _straighty_hint_single_variant(ranks_norm, 2, 14))

    # 视角 2：A 当 1（轮顺 A2345..9TJQK）
    if has_ace:
        vals_wheel = [1 if v == 14 else v for v in vals]
        ranks_wheel = sorted(set(vals_wheel))
        best_hint = max(best_hint, _straighty_hint_single_variant(ranks_wheel, 1, 13))

    return float(best_hint)
