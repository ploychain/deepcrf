import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pokers as pkrs

VERBOSE = False
equity_table = pd.read_csv('/home/harry/deepcfr/equity_table.csv', dtype={'hand': str, 'equity': float}, skipinitialspace=True)


def set_verbose(verbose_mode):
    global VERBOSE
    VERBOSE = verbose_mode


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


def encode_state(state, player_id=0):
    encoded = []
    num_players = len(state.players_state)

    if VERBOSE:
        print(f"Encoding state: current_player={state.current_player}, stage={state.stage}")
        print(f"Player states: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}")
        print(f"Pot: {state.pot}")

    hand_cards = state.players_state[player_id].hand
    hand_enc = np.zeros(52)
    for card in hand_cards:
        rank_map = {'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6': 5, 'R7': 6, 'R8': 7, 'R9': 8,
                    'RT': 9, 'RJ': 10, 'RQ': 11, 'RK': 12, 'RA': 13}
        suit_map = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}
        rank_str = str(card.rank).split('.')[-1]
        suit_str = str(card.suit).split('.')[-1]
        card_idx = suit_map[suit_str] * 13 + rank_map[rank_str]
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    community_enc = np.zeros(52)
    for card in state.public_cards:
        rank_map = {'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6': 5, 'R7': 6, 'R8': 7, 'R9': 8,
                    'RT': 9, 'RJ': 10, 'RQ': 11, 'RK': 12, 'RA': 13}
        suit_map = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}
        rank_str = str(card.rank).split('.')[-1]
        suit_str = str(card.suit).split('.')[-1]
        card_idx = suit_map[suit_str] * 13 + rank_map[rank_str]
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    initial_stake = state.players_state[0].stake
    if initial_stake <= 0:
        if VERBOSE:
            print(f"WARNING: Initial stake is zero or negative: {initial_stake}")
        initial_stake = 1.0
    pot_enc = [state.pot / initial_stake]
    encoded.append(pot_enc)

    button_enc = np.zeros(num_players)
    button_enc[state.button] = 1
    encoded.append(button_enc)

    current_player_enc = np.zeros(num_players)
    current_player_enc[state.current_player] = 1
    encoded.append(current_player_enc)

    for p in range(num_players):
        player_state = state.players_state[p]
        active_enc = [1.0 if player_state.active else 0.0]
        bet_enc = [player_state.bet_chips / initial_stake]
        pot_chips_enc = [player_state.pot_chips / initial_stake]
        stake_enc = [player_state.stake / initial_stake]
        encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc, stake_enc]))

    min_bet_enc = [state.min_bet / initial_stake]
    encoded.append(min_bet_enc)

    legal_actions_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    prev_action_enc = np.zeros(5)
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_action_enc)

    preflop_equity = 0.0
    if not state.public_cards and hand_cards:
        rank_map = {'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6', 'R7': '7', 'R8': '8',
                    'R9': '9', 'RT': 'T', 'RJ': 'J', 'RQ': 'Q', 'RK': 'K', 'RA': 'A'}
        suit_map = {'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'}
        ranks = [str(card.rank).split('.')[-1] for card in hand_cards]
        suits = [str(card.suit).split('.')[-1] for card in hand_cards]
        rank_values = [list(rank_map.keys()).index(r) for r in ranks]
        sorted_pairs = sorted(zip(rank_values, ranks, suits), reverse=True)
        rank1, rank2 = sorted_pairs[0][1], sorted_pairs[1][1]
        suit1, suit2 = sorted_pairs[0][2], sorted_pairs[1][2]
        hand_str = f"{rank_map[rank1]}{rank_map[rank2]}{'s' if suit1 == suit2 else 'o'}".strip().lower()
        if VERBOSE:
            print(f"Hand cards: {hand_cards}, Ranks: {ranks}, Suits: {suits}, Hand str: {hand_str}")
        try:
            preflop_equity = equity_table.loc[equity_table['hand'].str.strip() == hand_str, 'equity'].values[0]
        except (IndexError, KeyError) as e:
            if VERBOSE:
                print(f"WARNING: Equity not found for hand {hand_str}, error: {e}")
    equity_enc = [preflop_equity]
    encoded.append(equity_enc)

    x = np.concatenate(encoded)
    if len(x) < 500:
        x = np.pad(x, (0, 500 - len(x)), mode='constant')
    elif len(x) > 500:
        x = x[:500]
    return x