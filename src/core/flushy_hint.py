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


def flush_on_board(board_cards) -> int:
    """
    是否公共牌已有同花（0/1）。
    """
    if not board_cards or len(board_cards) < 5:
        return 0

    # 统计各花色出现次数
    suit_counts = Counter()
    for c in board_cards:
        s = str(c.suit).split('.')[-1]
        suit_counts[s] += 1

    # 若任一花色数量 ≥ 5，则说明公共牌已有同花
    for cnt in suit_counts.values():
        if cnt >= 5:
            return 1

    return 0


def monotone(board_cards) -> int:
    """
    是否单花面（monotone，翻牌阶段3张全同花）。
    """
    # 仅在 flop 阶段定义（必须恰好3张）
    if not board_cards or len(board_cards) != 3:
        return 0

    suits = [str(c.suit).split('.')[-1] for c in board_cards]
    return 1 if len(set(suits)) == 1 else 0


def two_tone(board_cards) -> int:
    """
    是否两同花面（two-tone，翻牌阶段恰好两种花色）。
    """
    if not board_cards or len(board_cards) != 3:
        return 0

    suits = [str(c.suit).split('.')[-1] for c in board_cards]
    return 1 if len(set(suits)) == 2 else 0


def rainbow(board_cards) -> int:
    """
    是否三花面（rainbow，翻牌阶段三种花色）。
    """
    if not board_cards or len(board_cards) != 3:
        return 0

    suits = [str(c.suit).split('.')[-1] for c in board_cards]
    return 1 if len(set(suits)) == 3 else 0


# ----------------------------------------------------------------------
# 🔹 筹码压力类指标
# ----------------------------------------------------------------------

def spr(hero_stack: float, villain_stack: float, pot_size: float) -> float:
    """
    Stack-to-Pot Ratio（0~10）
    表示剩余筹码相对底池的压力程度。

    定义：
      SPR = 有效筹码量 / 当前底池大小
      有效筹码量 = min(hero_stack, villain_stack)

    范围：
      0 → all-in / 无操作空间
      1~3 → 小筹码局（高压力）
      4~6 → 中筹码局（标准压力）
      7~10 → 深筹码局（操作空间大）
    """
    if pot_size <= 0:
        return 0.0

    effective_stack = min(hero_stack, villain_stack)
    value = effective_stack / pot_size

    # 限制最大值 10
    return float(min(value, 10.0))

def board_flush_possible_suits(board_cards) -> int:
    """
    当前花面类型数量（1~4）。
    统计公共牌中出现的不同花色数。
    """
    if not board_cards:
        return 0
    suits = [str(c.suit).split('.')[-1] for c in board_cards]
    return len(set(suits))
