# coding: utf-8
# harry/test_straighty_manual.py

import pokers as pkrs
from src.core.hand_straighty_potential import hand_straighty_potential

def main():
    # 构造一个假的 hero 手牌（随便两张即可，这里用 As Kc）
    hero_cards = [
        pkrs.Card(pkrs.Rank.RA, pkrs.Suit.Spades),
        pkrs.Card(pkrs.Rank.RK, pkrs.Suit.Clubs),
    ]

    # 人为写死 Flop = 4♦ 5♣ 6♠
    board_cards = [
        pkrs.Card(pkrs.Rank.R4, pkrs.Suit.Diamonds),  # 4♦
        pkrs.Card(pkrs.Rank.R5, pkrs.Suit.Clubs),     # 5♣
        pkrs.Card(pkrs.Rank.R6, pkrs.Suit.Spades),    # 6♠
    ]

    # 假设桌上还有 5 个对手（总共 6 人桌，hero + 5）
    n_opponents = 5

    p = hand_straighty_potential(hero_cards, board_cards, n_opponents)

    print("Hero:", hero_cards)
    print("Board:", board_cards)
    print("n_opponents:", n_opponents)
    print("hand_straighty_potential (当前别人已成顺的概率) =", p)

if __name__ == "__main__":
    main()
