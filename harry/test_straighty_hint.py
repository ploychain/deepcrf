import pokers as pkrs
from features import compute_straighty_hint  # 改成你文件路径

def show(board_desc, *cards):
    board = [pkrs.Card(*c.split("_")) for c in cards]
    print(f"{board_desc:20s} → straighty_hint = {compute_straighty_hint(board):.3f}")

show("A 2 3", "RA_Spades", "R2_Clubs", "R3_Hearts")
show("K Q J", "RK_Spades", "RQ_Clubs", "RJ_Diamonds")
show("9 7 5", "R9_Spades", "R7_Clubs", "R5_Hearts")
show("A 9 4", "RA_Spades", "R9_Hearts", "R4_Clubs")
show("Q K T", "RQ_Clubs", "RK_Hearts", "RT_Spades")
show("A K Q J", "RA_Hearts", "RK_Spades", "RQ_Clubs", "RJ_Diamonds")
