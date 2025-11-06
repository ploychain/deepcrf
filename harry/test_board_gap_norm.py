# coding: utf-8
import pokers as pkrs
from src.core.board_gap_norm import board_gap_norm

def card(rank, suit):
    return pkrs.Card(rank=rank, suit=suit)

def make_card(rank_str, suit_str):
    ranks = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    suits = {'s': pkrs.CardSuit.Spades, 'h': pkrs.CardSuit.Hearts, 'd': pkrs.CardSuit.Diamonds, 'c': pkrs.CardSuit.Clubs}
    return pkrs.Card(rank=ranks[rank_str], suit=suits[suit_str])

def main():
    boards = [
        [make_card('5','s'), make_card('6','h'), make_card('7','d')],     # 连性强
        [make_card('2','s'), make_card('9','h'), make_card('K','d')],     # 很散
        [make_card('T','s'), make_card('J','h'), make_card('Q','d')],     # 顺连
        [make_card('3','s'), make_card('8','h'), make_card('Q','d')]      # 中等
    ]

    for b in boards:
        print([f"{str(c.rank).split('.')[-1]}-{str(c.suit).split('.')[-1]}" for c in b],
              "→ board_gap_norm =", round(board_gap_norm(b), 4))

if __name__ == "__main__":
    main()
