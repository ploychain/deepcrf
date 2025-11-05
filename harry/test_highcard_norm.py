# coding: utf-8
# harry/test_highcard_norm.py

import pokers as pkrs
from src.core.highcard_on_board_norm import highcard_on_board_norm

def make_card(rank, suit):
    return pkrs.Card(rank=pkrs.Rank(rank), suit=pkrs.Suit(suit))

def main():
    # 手工造几组公共牌
    boards = [
        [pkrs.Card(pkrs.Rank.R2, pkrs.Suit.Spades),
         pkrs.Card(pkrs.Rank.R5, pkrs.Suit.Hearts),
         pkrs.Card(pkrs.Rank.R9, pkrs.Suit.Diamonds)],
        [pkrs.Card(pkrs.Rank.RT, pkrs.Suit.Spades),
         pkrs.Card(pkrs.Rank.RQ, pkrs.Suit.Clubs),
         pkrs.Card(pkrs.Rank.RK, pkrs.Suit.Hearts)],
        [pkrs.Card(pkrs.Rank.RA, pkrs.Suit.Clubs),
         pkrs.Card(pkrs.Rank.R3, pkrs.Suit.Diamonds),
         pkrs.Card(pkrs.Rank.R7, pkrs.Suit.Hearts)],
    ]

    for i, b in enumerate(boards):
        val = highcard_on_board_norm(b)
        print(f"Board{i+1}: {[str(c) for c in b]} → highcard_on_board_norm={val:.4f}")

if __name__ == "__main__":
    main()
