import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pokers as pkrs

VERBOSE = False
equity_table = pd.read_csv('../../equity_table.csv')  # From src/core/ to root


def set_verbose(verbose_mode):
    """Set the global verbosity level"""
    global VERBOSE
    VERBOSE = verbose_mode


class PokerNetwork(nn.Module):
    """Poker network with continuous bet sizing capabilities."""

    def __init__(self, input_size=500, hidden_size=512, num_actions=3):  # Keep 500
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
    """Convert a Pokers state to a neural network input tensor."""
    encoded = []
    num_players = len(state.players_state)

    if VERBOSE:
        print(f"Encoding state: current_player={state.current_player}, stage={state.stage}")
        print(f"Player states: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}")
        print(f"Pot: {state.pot}")

    # Encode player's hole cards
    hand_cards = state.players_state[player_id].hand
    hand_enc = np.zeros(52)
    for card in hand_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    # Encode community cards
    community_enc = np.zeros(52)
    for card in state.public_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    # Encode game stage
    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # Encode pot size
    initial_stake = state.players_state[0].stake
    if initial_stake <= 0:
        if VERBOSE:
            print(f"WARNING: Initial stake is zero or negative: {initial_stake}")
        initial_stake = 1.0
    pot_enc = [state.pot / initial_stake]
    encoded.append(pot_enc)

    # Encode button position
    button_enc = np.zeros(num_players)
    button_enc[state.button] = 1
    encoded.append(button_enc)

    # Encode current player
    current_player_enc = np.zeros(num_players)
    current_player_enc[state.current_player] = 1
    encoded.append(current_player_enc)

    # Encode player states
    for p in range(num_players):
        player_state = state.players_state[p]
        active_enc = [1.0 if player_state.active else 0.0]
        bet_enc = [player_state.bet_chips / initial_stake]
        pot_chips_enc = [player_state.pot_chips / initial_stake]
        stake_enc = [player_state.stake / initial_stake]
        encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc, stake_enc]))

    # Encode minimum bet
    min_bet_enc = [state.min_bet / initial_stake]
    encoded.append(min_bet_enc)

    # Encode legal actions
    legal_actions_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    # Encode previous action
    prev_action_enc = np.zeros(5)
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_action_enc)

    # Encode preflop equity
    preflop_equity = 0.0
    if not state.public_cards and hand_cards:
        rank_map = {1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'T', 10: 'J', 11: 'Q', 12: 'K',
                    13: 'A'}
        suit_map = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
        hand_str = ''.join([
            rank_map.get(int(hand_cards[0].rank), 'A'),
            rank_map.get(int(hand_cards[1].rank), 'A'),
            's' if hand_cards[0].suit == hand_cards[1].suit else 'o'
        ])
        try:
            preflop_equity = equity_table.loc[equity_table['hand'] == hand_str, 'equity'].values[0]
        except IndexError:
            if VERBOSE:
                print(f"WARNING: Equity not found for hand {hand_str}")
    equity_enc = [preflop_equity]
    encoded.append(equity_enc)

    # Concatenate and pad/truncate to 500
    x = np.concatenate(encoded)
    if len(x) < 500:
        x = np.pad(x, (0, 500 - len(x)), mode='constant')
    elif len(x) > 500:
        x = x[:500]
    return x