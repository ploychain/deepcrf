# harry/test_straighty_hint.py
# 不依赖 pokers.Card；用轻量对象模拟 .rank/.suit 即可
from types import SimpleNamespace

# === 你项目里的实现：必须能 from harry.features import compute_straighty_hint ===
from harry.features import compute_straighty_hint

# 映射：'A' -> 'RA', 'T' -> 'RT' 等（与你项目里的 RANK_CHAR_MAP 对齐）
RANK_CHAR = {
    "2":"R2","3":"R3","4":"R4","5":"R5","6":"R6","7":"R7",
    "8":"R8","9":"R9","T":"RT","J":"RJ","Q":"RQ","K":"RK","A":"RA"
}
SUIT_NAME = {"s":"Spades","h":"Hearts","d":"Diamonds","c":"Clubs"}

def make_fake_card(txt):
    """
    构造一个拥有 .rank/.suit 属性的轻量对象，避免依赖 pokers 的构造差异。
    - 输入形如 'As', 'Td', 'Qh' 等两字符
    - .rank 设为 'RA' / 'RT' ...；.suit 设为 'Spades' / 'Hearts' ...
    """
    txt = txt.strip()
    if len(txt) != 2:
        raise ValueError(f"牌面必须两字符（如 'As'），收到: {txt}")
    r, s = txt[0].upper(), txt[1].lower()
    if r not in RANK_CHAR or s not in SUIT_NAME:
        raise ValueError(f"无法解析牌面: {txt}")
    return SimpleNamespace(rank=RANK_CHAR[r], suit=SUIT_NAME[s])

def show(title, *cards_txt):
    board = [make_fake_card(t) for t in cards_txt]
    val = compute_straighty_hint(board)
    # 打印 rank/suit，便于人工核验
    board_dbg = [f"{c.rank}-{c.suit}" for c in board]
    print(f"{title:20s} → straighty_hint = {val:.3f} | board = {board_dbg}")

if __name__ == "__main__":
    # === 典型用例，覆盖 A 低顺、KQJ、带洞顺、无序间距等 ===
    show("A 2 3", "As", "2c", "3h")         # A23 → 低顺窗口
    show("K Q J", "Ks", "Qh", "Jd")         # KQJ
    show("Q K T", "Qh", "Ks", "Td")         # KTQ 无序
    show("9 7 5", "9s", "7c", "5h")         # 大间距（应弱/0）
    show("A K Q J", "Ah", "Ks", "Qc", "Jd") # 四张顺窗口
    show("A 5 4", "As", "5s", "4d")         # A54 间距=4（应提示）
    show("8 T Q", "8c", "Td", "Qh")         # 8TQ（8-10-Q 跨度=4，但缺9/J）
