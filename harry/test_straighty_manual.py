# coding: utf-8
# harry/test_straighty_manual.py

import pokers as pkrs
from src.core.hand_straighty_potential import hand_straighty_potential

def make_card(rank_num, suit_num):
    """
    rank_num: 2~14 （A=14）
    suit_num: 0~3   (Spades=0, Hearts=1, Diamonds=2, Clubs=3)
    """
    return pkrs.Card(rank=rank_num, suit=suit_num)

def main():
    # Hero 手牌随便两张（A♠ K♣）
    hero_cards = [
        make_card(14, 0),  # A♠
        make_card(13, 3),  # K♣
    ]

    # 固定公共牌：4♦ (rank=4, suit=2), 5♣ (rank=5, suit=3), 6♠ (rank=6, suit=0)
    board_cards = [
        make_card(4, 2),  # 4♦
        make_card(5, 3),  # 5♣
        make_card(6, 0),  # 6♠
    ]

    n_opponents = 5

    p = hand_straighty_potential(hero_cards, board_cards, n_opponents)

    print("Hero cards:", hero_cards)
    print("Board cards:", board_cards)
    print("Opponents:", n_opponents)
    print("hand_straighty_potential =", p)

if __name__ == "__main__":
    main()
