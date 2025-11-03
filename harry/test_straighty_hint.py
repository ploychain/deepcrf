# harry/test_straighty_hint.py
import pokers as pkrs
from harry.features import compute_straighty_hint  # 你的实现里导出的函数

RANK_CHAR = {
    "2":"R2","3":"R3","4":"R4","5":"R5","6":"R6","7":"R7",
    "8":"R8","9":"R9","T":"RT","J":"RJ","Q":"RQ","K":"RK","A":"RA"
}
SUIT_NAME = {"s":"Spades","h":"Hearts","d":"Diamonds","c":"Clubs"}

def make_card(txt):
    """极度兼容的 Card 构造：依次尝试多种 API，失败就报错，不会返回 None。"""
    txt = txt.strip()
    if len(txt) != 2:
        raise ValueError(f"牌面必须两字符如 'As'，收到: {txt}")
    r, s = txt[0].upper(), txt[1].lower()

    tried = []

    # 1) 先试 from_string("As")
    if hasattr(pkrs.Card, "from_string"):
        try:
            c = pkrs.Card.from_string(f"{r}{s}")
            if hasattr(c, "rank") and hasattr(c, "suit"):
                return c
        except Exception as e:
            tried.append(f"Card.from_string('{r}{s}') -> {e}")

        # 有些版本支持 "A♠" 这样的符号
        try:
            sym = {"s":"♠","h":"♥","d":"♦","c":"♣"}[s]
            c = pkrs.Card.from_string(f"{r}{sym}")
            if hasattr(c, "rank") and hasattr(c, "suit"):
                return c
        except Exception as e:
            tried.append(f"Card.from_string('{r}{sym}') -> {e}")

    # 2) 再试枚举 (Rank/Suit)
    rank_keys = [
        ("Rank", f"R{r}"),          # Rank.RA
        ("CardRank", f"R{r}"),      # CardRank.RA
    ]
    suit_keys = [
        ("Suit", SUIT_NAME[s]),     # Suit.Spades
        ("CardSuit", SUIT_NAME[s]), # CardSuit.Spades
    ]
    for rk_cls, rk_name in rank_keys:
        for st_cls, st_name in suit_keys:
            if hasattr(pkrs, rk_cls) and hasattr(pkrs, st_cls):
                try:
                    rank_enum = getattr(getattr(pkrs, rk_cls), rk_name)
                    suit_enum = getattr(getattr(pkrs, st_cls), st_name)
                    c = pkrs.Card(rank_enum, suit_enum)
                    if hasattr(c, "rank") and hasattr(c, "suit"):
                        return c
                except Exception as e:
                    tried.append(f"Card({rk_cls}.{rk_name}, {st_cls}.{st_name}) -> {e}")

    # 3) 极端兜底：某些版本有 Card.collect 可解析字符串
    if hasattr(pkrs.Card, "collect"):
        try:
            # 有的实现 Card.collect('As') 返回 list/iterable
            got = pkrs.Card.collect(f"{r}{s}")
            # 把第一个对象拿出来
            c = None
            for x in got:
                c = x
                break
            if c is not None and hasattr(c, "rank") and hasattr(c, "suit"):
                return c
        except Exception as e:
            tried.append(f"Card.collect('{r}{s}') -> {e}")

    # 全部失败 -> 明确报错
    msg = "无法构造 Card('{}{}')，尝试路径：\n  - ".format(r, s) + "\n  - ".join(tried or ["(无可用构造API)"])
    raise RuntimeError(msg)

def show(title, *cards_txt):
    board = [make_card(t) for t in cards_txt]
    # 防御式检查，杜绝 None
    assert all(hasattr(c, "rank") and hasattr(c, "suit") for c in board), "board 内含非法牌对象"
    val = compute_straighty_hint(board)
    print(f"{title:20s} → straighty_hint = {val:.3f} | board = {[str(c) for c in board]}")

if __name__ == "__main__":
    # ====== 核心用例 ======
    show("A 2 3", "As", "2c", "3h")
    show("K Q J", "Ks", "Qh", "Jd")
    show("9 7 5", "9s", "7c", "5h")
    show("A 9 4", "As", "9h", "4d")
    show("Q K T", "Qh", "Ks", "Td")
    show("A K Q J", "Ah", "Ks", "Qc", "Jd")

    # 你也可以随便加更多 case：
    # show("A 5 4", "As", "5s", "4d")
