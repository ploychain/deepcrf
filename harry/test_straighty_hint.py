import pokers as pkrs
from features import compute_straighty_hint  # 改成你的真实路径

def make_card(rank_char, suit_char):
    """兼容 pokers 的 Card 构造方式"""
    try:
        # 优先用枚举构造
        rank = getattr(pkrs.Rank, f"R{rank_char.upper()}")
        suit = getattr(pkrs.Suit, {
            "s": "Spades",
            "h": "Hearts",
            "d": "Diamonds",
            "c": "Clubs"
        }[suit_char.lower()])
        return pkrs.Card(rank, suit)
    except Exception:
        # 部分版本可直接 from_string
        try:
            return pkrs.Card.from_string(f"{rank_char.upper()}{suit_char.lower()}")
        except Exception as e:
            raise RuntimeError(f"无法创建牌 {rank_char}{suit_char}: {e}")

def show(board_desc, *cards):
    board = [make_card(c[0], c[1]) for c in cards]
    val = compute_straighty_hint(board)
    print(f"{board_desc:20s} → straighty_hint = {val:.3f}")

# ====== 测试用例 ======
show("A 2 3", "As", "2c", "3h")
show("K Q J", "Ks", "Qh", "Jd")
show("9 7 5", "9s", "7c", "5h")
show("A 9 4", "As", "9h", "4d")
show("Q K T", "Qh", "Ks", "Td")
show("A K Q J", "Ah", "Ks", "Qc", "Jd")
