# coding: utf-8
import pokers as pkrs
from src.core.board_gap_norm import board_gap_norm

def get_enums():
    """自动获取 CardRank / CardSuit（兼容各种 pokers 版本）"""
    rank_enum = None
    suit_enum = None
    for name in dir(pkrs):
        if name.lower() in ("cardrank", "rank"):
            rank_enum = getattr(pkrs, name)
        if name.lower() in ("cardsuit", "suit"):
            suit_enum = getattr(pkrs, name)
    if rank_enum is None or suit_enum is None:
        raise RuntimeError("未找到 CardRank / CardSuit 枚举，请检查 pokers 包结构。")
    return rank_enum, suit_enum


def make_card(rank_num, suit_num):
    """通过枚举安全创建 Card"""
    rank_enum, suit_enum = get_enums()
    ranks = list(rank_enum)
    suits = list(suit_enum)
    # rank_num: 2~14  → index 从 0 开始偏移 +2
    rank_obj = next((r for r in ranks if int(r) == rank_num), None)
    suit_obj = suits[suit_num % len(suits)]
    return pkrs.Card(rank=rank_obj, suit=suit_obj)


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
        desc = " ".join([f"{str(c.rank).split('.')[-1]}-{str(c.suit).split('.')[-1]}" for c in b])
        print(f"Board: {desc} -> board_gap_norm = {board_gap_norm(b):.4f}")


if __name__ == "__main__":
    main()
