# coding: utf-8
import pokers as pkrs
from src.core.board_gap_norm import board_gap_norm

def make_card(rank_num, suit_num):
    """
    rank_num: 2~14  (2=2, 14=A)
    suit_num: 0~3   (0=♠, 1=♥, 2=♦, 3=♣)
    """
    return pkrs.Card(rank=rank_num, suit=suit_num)

def main():
    boards = [
        # 连性强
        [make_card(5, 0), make_card(6, 1), make_card(7, 2)],
        # 很散
        [make_card(2, 0), make_card(9, 1), make_card(13, 2)],
        # 顺连 (T,J,Q)
        [make_card(10, 0), make_card(11, 1), make_card(12, 2)],
        # 中等
        [make_card(3, 0), make_card(8, 1), make_card(12, 2)]
    ]

    for b in boards:
        desc = " ".join([f"{c.rank}-{c.suit}" for c in b])
        print(f"Board: {desc} -> board_gap_norm = {board_gap_norm(b):.4f}")

if __name__ == "__main__":
    main()
