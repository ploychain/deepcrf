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
    encoded = []
    num_players = len(state.players_state)

    if VERBOSE:
        print(f"Encoding state: current_player={state.current_player}, stage={state.stage}")
        print(f"Player states: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}")
        print(f"Pot: {state.pot}")

    # Encode cards
    encoded.append(encode_cards(state.players_state[player_id].hand))   # 手牌
    encoded.append(encode_cards(state.public_cards))                    # 公共牌

    # Encode stage (5维 one-hot)
    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # Stake normalization
    initial_stake = max(state.players_state[0].stake, 1.0)

    # Encode pot, button, current player
    encoded.append([state.pot / initial_stake])
    button_enc = np.eye(num_players)[state.button]
    current_player_enc = np.eye(num_players)[state.current_player]
    encoded.append(button_enc)
    encoded.append(current_player_enc)

    # Encode per-player info
    for p in range(num_players):
        ps = state.players_state[p]
        encoded.append([
            1.0 if ps.active else 0.0,
            ps.bet_chips / initial_stake,
            ps.pot_chips / initial_stake,
            ps.stake / initial_stake
        ])

    # Encode min bet and legal actions
    encoded.append([state.min_bet / initial_stake])
    legal_actions_enc = np.zeros(4)
    for a in state.legal_actions:
        legal_actions_enc[int(a)] = 1
    encoded.append(legal_actions_enc)

    # Encode previous action
    prev_action_enc = np.zeros(5)
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_action_enc)

    # Encode preflop equity (only preflop stage)
    preflop_equity = get_preflop_equity(state.players_state[player_id].hand) if not state.public_cards else 0.0
    encoded.append([preflop_equity])

    # Final flatten + fix dimension
    x = np.concatenate(encoded)
    TARGET_DIM = 500
    if len(x) > TARGET_DIM:
        if VERBOSE:
            print(f"[DEBUG] Input too long ({len(x)}), truncating and preserving equity={preflop_equity}")
        x = np.concatenate([x[:TARGET_DIM - 1], [preflop_equity]])
    elif len(x) < TARGET_DIM:
        x = np.pad(x, (0, TARGET_DIM - len(x)), mode='constant')
        x[-1] = preflop_equity

    if VERBOSE:
        print(f"[DEBUG] Final vector length: {len(x)} | Final equity: {preflop_equity}")

    return x
