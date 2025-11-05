# coding: utf-8
# harry/test_highcard_norm.py

import pokers as pkrs
from src.core.highcard_on_board_norm import highcard_on_board_norm

def get_enums():
    """自动从 pokers 模块中提取 rank / suit 枚举对象"""
    CardRank = getattr(pkrs, "CardRank", None)
    CardSuit = getattr(pkrs, "CardSuit", None)
    if CardRank is None or CardSuit is None:
        # 有些版本是 Rank/Suit
        CardRank = getattr(pkrs, "Rank")
        CardSuit = getattr(pkrs, "Suit")
    ranks = list(CardRank)
    suits = list(CardSuit)
    return ranks, suits

def make_card(rank_idx, suit_idx, ranks, suits):
    """根据枚举下标创建 Card"""
    return pkrs.Card(rank=ranks[rank_idx], suit=suits[suit_idx])

def card_to_str(card):
    """人类可读格式"""
    rank_str = str(card.rank).split('.')[-1].replace('R', '')
    suit_str = str(card.suit).split('.')[-1]
    suit_symbol = {
        'Spades': '♠',
        'Hearts': '♥',
        'Diamonds': '♦',
        'Clubs': '♣'
    }.get(suit_str, '?')
    rank_symbol = {
        'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
    }.get(rank_str, rank_str)
    return f"{rank_symbol}{suit_symbol}"

def main():
    ranks, suits = get_enums()

    # 创建三组公共牌（不同的 rank）
    boards = [
        [make_card(0, 0, ranks, suits), make_card(3, 1, ranks, suits), make_card(7, 2, ranks, suits)],   # 2♠ 5♥ 9♦
        [make_card(8, 0, ranks, suits), make_card(10, 3, ranks, suits), make_card(11, 1, ranks, suits)], # 10♠ Q♣ K♥
        [make_card(12, 3, ranks, suits), make_card(2, 2, ranks, suits), make_card(6, 1, ranks, suits)],  # A♣ 4♦ 8♥
    ]

    for i, b in enumerate(boards):
        val = highcard_on_board_norm(b)
        cards_str = [card_to_str(c) for c in b]
        print(f"Board{i+1}: {cards_str} → highcard_on_board_norm={val:.4f}")

if __name__ == "__main__":
    main()
