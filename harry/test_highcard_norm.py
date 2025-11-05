# coding: utf-8
# harry/test_highcard_norm.py

import pokers as pkrs
from src.core.highcard_on_board_norm import highcard_on_board_norm

# suit 编号: Spades=0, Hearts=1, Diamonds=2, Clubs=3
# rank 编号: R2=1, R3=2, ..., RK=12, RA=13
def make_card(rank_num, suit_num):
    """快速创建 pkrs.Card"""
    return pkrs.Card(rank=rank_num, suit=suit_num)

def card_to_str(card):
    """将卡牌转成 'A♠' 样式方便打印"""
    rank_map = {
        1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7',
        7: '8', 8: '9', 9: '10', 10: 'J', 11: 'Q', 12: 'K', 13: 'A'
    }
    suit_symbol = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
    try:
        r = rank_map[int(card.rank)]
        s = suit_symbol[int(card.suit)]
        return f"{r}{s}"
    except Exception:
        return str(card)

def main():
    boards = [
        [make_card(1,0), make_card(4,1), make_card(8,2)],   # 2♠ 5♥ 9♦
        [make_card(9,0), make_card(11,3), make_card(12,1)], # 10♠ Q♣ K♥
        [make_card(13,3), make_card(3,2), make_card(7,1)],  # A♣ 4♦ 8♥
    ]

    for i, b in enumerate(boards):
        val = highcard_on_board_norm(b)
        cards_str = [card_to_str(c) for c in b]
        print(f"Board{i+1}: {cards_str} → highcard_on_board_norm={val:.4f}")

if __name__ == "__main__":
    main()
