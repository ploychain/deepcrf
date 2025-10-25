import numpy as np
from treys import Card, Deck, Evaluator
import pandas as pd

class PokerKnowledge:
    def __init__(self, sims: int = 10000):
        self.evaluator = Evaluator()
        self.sims = sims

    def preflop_equity(self, hand: list, num_opponents: int = 5) -> float:
        """Calculate preflop equity via Monte Carlo for 6-player table."""
        equities = []
        my_hand = [Card.new(card) for card in hand]  # e.g., ['As', 'Ad']
        for _ in range(self.sims):
            deck = Deck()  # Reset deck each iteration
            deck.shuffle()
            # Validate hand cards are in deck
            if any(Card.new(card) not in deck.cards for card in hand):
                continue
            # Remove hand cards from deck
            for card in my_hand:
                deck.cards.remove(card)
            try:
                opp_hands = [deck.draw(2) for _ in range(num_opponents)]
                board = deck.draw(5)
                # Validate no duplicates
                all_cards = my_hand + [card for opp in opp_hands for card in opp] + board
                if len(set(all_cards)) != len(all_cards):
                    continue
                my_rank = self.evaluator.evaluate(board, my_hand)
                opp_ranks = [self.evaluator.evaluate(board, oh) for oh in opp_hands]
                wins = sum(my_rank < opp_rank for opp_rank in opp_ranks) / num_opponents
                equities.append(wins)
            except KeyError:
                continue  # Skip invalid hand evaluations
        return np.mean(equities) if equities else 0.0

    def generate_equity_table(self, output_file: str = 'equity_table.csv'):
        """Generate preflop equity table for 169 hand combos."""
        ranks = '23456789TJQKA'
        hands = [f'{r1}{r2}{s}' for r1 in ranks for r2 in ranks for s in ['s', 'o']]
        equities = {}
        for hand in hands:
            cards = [f'{hand[0]}s', f'{hand[1]}{"s" if hand[2] == "s" else "h"}']
            equity = self.preflop_equity(cards)
            equities[hand] = equity
        df = pd.DataFrame.from_dict(equities, orient='index', columns=['equity'])
        df.to_csv(output_file)

if __name__ == "__main__":
    pk = PokerKnowledge()
    print(f"AA equity (preflop): {pk.preflop_equity(['As', 'Ad'], 5):.3f}")
    pk.generate_equity_table()