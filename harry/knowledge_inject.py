import numpy as np
from pokerkit import Hand, Deck, Evaluation
import pandas as pd
from typing import List

class PokerKnowledge:
    def __init__(self, sims: int = 10000):
        self.deck = Deck()
        self.sims = sims

    def preflop_equity(self, hand: List[str], num_opponents: int = 5) -> float:
        """Calculate preflop equity via Monte Carlo for 6-player table."""
        equities = []
        my_hand = Hand.parse(hand)
        for _ in range(self.sims):
            self.deck.shuffle()
            opp_hands = [self.deck.deal(2) for _ in range(num_opponents)]
            board = self.deck.deal(5)
            my_eval = Evaluation(my_hand + board)
            opp_evals = [Evaluation(oh + board) for oh in opp_hands]
            wins = sum(my_eval > oe for oe in opp_evals) / num_opponents
            equities.append(wins)
        return np.mean(equities)

    def generate_equity_table(self, output_file: str = 'equity_table.csv'):
        """Generate preflop equity table for 169 hand combos."""
        hands = [f'{r1}{r2}{s}' for r1 in '23456789TJQKA' for r2 in '23456789TJQKA' for s in ['s', 'o']]
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