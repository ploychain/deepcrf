import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pokers as pkrs

# ===================== 全局常量与配置 =====================
VERBOSE = False
EQUITY_PATH = '/home/harry/deepcfr/equity_table.csv'

# 加载 Equity 表（一次性全局加载）
equity_table = pd.read_csv(EQUITY_PATH, dtype={'hand': str, 'equity': float})
equity_table['hand'] = equity_table['hand'].astype(str).str.strip().str.upper()

# rank/suit 映射常量
RANK_MAP = {'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6': 5, 'R7': 6,
            'R8': 7, 'R9': 8, 'RT': 9, 'RJ': 10, 'RQ': 11, 'RK': 12, 'RA': 13}
SUIT_MAP = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}

RANK_CHAR_MAP = {'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6', 'R7': '7',
                 'R8': '8', 'R9': '9', 'RT': 'T', 'RJ': 'J', 'RQ': 'Q', 'RK': 'K', 'RA': 'A'}

# ===================== 函数定义 =====================
def set_verbose(verbose_mode: bool):
    global VERBOSE
    VERBOSE = verbose_mode


def encode_cards(cards):
    """将牌面列表编码为 52 维 one-hot 向量"""
    vec = np.zeros(52)
    for card in cards:
        rank_str = str(card.rank).split('.')[-1]
        suit_str = str(card.suit).split('.')[-1]
        card_idx = SUIT_MAP[suit_str] * 13 + RANK_MAP[rank_str]
        vec[card_idx] = 1
    return vec


def get_preflop_equity(hand_cards):
    """根据两张手牌查表获得 preflop equity"""
    if not hand_cards:
        return 0.0

    ranks = [str(c.rank).split('.')[-1] for c in hand_cards]
    suits = [str(c.suit).split('.')[-1] for c in hand_cards]

    rank_order = list(RANK_MAP.keys())
    sorted_cards = sorted(zip(ranks, suits), key=lambda x: rank_order.index(x[0]), reverse=True)
    rank1, suit1 = sorted_cards[0]
    rank2, suit2 = sorted_cards[1]

    r1 = RANK_CHAR_MAP[rank1]
    r2 = RANK_CHAR_MAP[rank2]
    suited = 's' if suit1 == suit2 else 'o'
    hand_str = f"{r1}{r2}{suited}".upper()

    match = equity_table[equity_table['hand'] == hand_str]
    if not match.empty:
        equity = float(match['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] Found equity for {hand_str}: {equity:.6f}")
        return equity

    if VERBOSE:
        print(f"[WARN] Hand {hand_str} not found in equity table.")
    return 0.0


# ===================== 模型定义 =====================
class PokerNetwork(nn.Module):
    def __init__(self, input_size=500, hidden_size=512, num_actions=3):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, opponent_features=None):
        features = self.base(x)
        action_logits = self.action_head(features)
        bet_size = 0.1 + 2.9 * self.sizing_head(features)
        return action_logits, bet_size


# ===================== 主编码函数 =====================
def encode_state(state, player_id=0):
    """
    Encode a full Pokers state into a fixed-length (500,) vector.
    - 保证维度恒定
    - preflop_equity 永远放在最后
    - 每个模块长度固定，不依赖状态变化
    """
    encoded = []
    num_players = 6  # ✅ 固定6人桌，确保长度恒定

    # ---------- 1. 手牌编码 (52维 one-hot) ----------
    hand_enc = np.zeros(52)
    for card in state.players_state[player_id].hand:
        card_idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    # ---------- 2. 公共牌编码 (52维 one-hot) ----------
    community_enc = np.zeros(52)
    for card in state.public_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    # ---------- 3. 阶段编码 (5维 one-hot) ----------
    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # ---------- 4. 彩池大小 (1维，归一化) ----------
    initial_stake = max(1.0, state.players_state[0].stake)
    pot_enc = np.array([state.pot / initial_stake])
    encoded.append(pot_enc)

    # ---------- 5. 庄位与当前行动者 (6+6维 one-hot) ----------
    button_enc = np.zeros(num_players)
    button_enc[state.button % num_players] = 1
    current_enc = np.zeros(num_players)
    current_enc[state.current_player % num_players] = 1
    encoded.extend([button_enc, current_enc])

    # ---------- 6. 每个玩家状态 (6×4维) ----------
    for p in range(num_players):
        if p < len(state.players_state):
            ps = state.players_state[p]
            active = 1.0 if ps.active else 0.0
            bet = ps.bet_chips / initial_stake
            pot_chips = ps.pot_chips / initial_stake
            stake = ps.stake / initial_stake
        else:
            active = bet = pot_chips = stake = 0.0
        encoded.append(np.array([active, bet, pot_chips, stake]))

    # ---------- 7. 最小下注 (1维) ----------
    encoded.append(np.array([state.min_bet / initial_stake]))

    # ---------- 8. 合法动作 (4维固定 one-hot) ----------
    legal_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        idx = min(int(action_enum), 3)
        legal_enc[idx] = 1
    encoded.append(legal_enc)

    # ---------- 9. 上一步动作 (5维固定: 4种动作 + 金额) ----------
    prev_enc = np.zeros(5)
    if state.from_action is not None:
        idx = min(int(state.from_action.action.action), 3)
        prev_enc[idx] = 1
        prev_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_enc)

    # ---------- 10. preflop_equity (1维固定) ----------
    preflop_equity = 0.0
    if not state.public_cards:
        rank_map = {'R2': '2','R3':'3','R4':'4','R5':'5','R6':'6','R7':'7','R8':'8',
                    'R9':'9','RT':'T','RJ':'J','RQ':'Q','RK':'K','RA':'A'}
        ranks = [str(c.rank).split('.')[-1] for c in state.players_state[player_id].hand]
        suits = [str(c.suit).split('.')[-1] for c in state.players_state[player_id].hand]
        ranks_char = [rank_map[r] for r in ranks]
        suited = 's' if suits[0] == suits[1] else 'o'
        hand_str = f"{ranks_char[0]}{ranks_char[1]}{suited}".upper()
        match = equity_table[equity_table['hand'] == hand_str]
        if not match.empty:
            preflop_equity = float(match['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] Found equity for {hand_str}: {preflop_equity}")

    equity_enc = np.array([preflop_equity])
    encoded.append(equity_enc)

    # ---------- 11. 拼接并固定维度 ----------
    x = np.concatenate(encoded)
    base_len = len(x)

    if base_len < 500:
        x = np.pad(x, (0, 500 - base_len), mode='constant')
    elif base_len > 500:
        raise ValueError(f"[ERROR] Encoded vector too long ({base_len}) — investigate source.")

    if VERBOSE:
        print(f"[DEBUG] Final vector length: {len(x)} | Final equity: {preflop_equity}")

    return x

