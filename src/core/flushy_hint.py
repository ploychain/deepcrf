# coding: utf-8
"""
flushy_hint
-----------
公共牌同花潜力启发式，返回 0~1。

简单规则（按最大同花色数量 n）：
  n <= 1  → 0.0
  n == 2  → 0.3
  n == 3  → 0.7
  n >= 4  → 1.0

注意：这里只看公共牌本身的同花倾向，不考虑玩家手牌。
"""

from typing import List
import math
from collections import Counter

def flushy_hint(board_cards) -> float:
    """
    公共牌同花潜力提示（0~1）。

    约定：
      - 输入为 pokers.Card 列表
      - 只统计公共牌中，某一花色出现的最大次数 n
      - 映射关系：
          n <= 1 → 0.0
          n == 2 → 0.3
          n == 3 → 0.7
          n >= 4 → 1.0
    """
    if not board_cards:
        return 0.0

    # 统计各花色出现次数
    suit_counts = Counter()
    for c in board_cards:
        # 假定 c.suit 打印类似 "Suit.Spades"
        s = str(c.suit).split('.')[-1]
        suit_counts[s] += 1

    if not suit_counts:
        return 0.0

    max_same_suit = max(suit_counts.values())

    if max_same_suit <= 1:
        return 0.0
    elif max_same_suit == 2:
        return 0.3
    elif max_same_suit == 3:
        return 0.7
    else:  # max_same_suit >= 4（包括已经成同花 5 张）
        return 1.0
