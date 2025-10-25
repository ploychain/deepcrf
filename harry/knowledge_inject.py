import numpy as np
from treys import Card, Deck, Evaluator
import pandas as pd
from typing import List

class PokerKnowledge:
    def __init__(self, sims: int = 10000):
        self.evaluator = Evaluator()
        self.sims = sims

    def preflop_equity(self, hand: List[str], num_opponents: int = 5) -> float:
        """Calculate preflop equity via Monte Carlo for 6-player table."""
        equities = []
        my_hand = [Card.new(card) for card in hand]  # e.g., ['As', 'Ad']
        for _ in range(self.sims):
            deck = Deck()  # Reset deck
            deck_cards = deck.cards[:]
            for card in my_hand:
                if card in deck_cards:
                    deck_cards.remove(card)
                else:
                    continue
            deck.cards = deck_cards
            try:
                opp_hands = [deck.draw(2) for _ in range(num_opponents)]
                board = deck.draw(5)
                # Validate no duplicates
                all_cards = my_hand + [card for opp in opp_hands for card in opp] + board
                if len(set(all_cards)) != len(all_cards):
                    continue
                my_rank = self.evaluator.evaluate(board, my_hand)
                opp_ranks = [self.evaluator.evaluate(board, oh) for oh in opp_hands]
                # Count wins and ties per opponent
                wins = 0.0
                for opp_rank in opp_ranks:
                    if my_rank < opp_rank:  # Win (lower rank is stronger)
                        wins += 1.0
                    elif my_rank == opp_rank:  # Tie
                        wins += 0.5
                wins /= num_opponents  # Average over opponents
                equities.append(wins)
            except (KeyError, ValueError):
                continue
        return np.mean(equities) if equities else 0.0

    def generate_equity_table(self, output_file: str = 'equity_table.csv'):
        """Generate preflop equity table for 169 hand combos."""
        ranks = '23456789TJQKA'
        hands = [f'{r1}{r2}{s}' for r1 in ranks for r2 in ranks for s in ['s', 'o']]
        equities = {}
        for hand in hands:
            cards = [f'{hand[0]}s', f'{hand[1]}{"s" if hand[2] == "s" else "h"}']
            equities[hand] = self.preflop_equity(cards)
        df = pd.DataFrame.from_dict(equities, orient='index', columns=['equity'])
        df.to_csv(output_file)

if __name__ == "__main__":
    pk = PokerKnowledge()
    print(f"AA equity (preflop): {pk.preflop_equity(['As', 'Ad'], 5):.3f}")
    pk.generate_equity_table()