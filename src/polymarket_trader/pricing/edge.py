"""
Edge detection utilities for Polymarket.

Identifies:
- Spread-adjusted edge (edge must exceed half-spread to be profitable)
- Arbitrage opportunities (YES + NO prices diverge from 1.0)
- Liquidity scoring (can you get filled at your target size?)
"""

from ..models.market import OrderBook


class EdgeDetector:
    def __init__(self, min_edge: float = 0.03) -> None:
        self._min_edge = min_edge

    def compute_edge(
        self,
        market_price: float,
        estimated_probability: float,
        side: str,
    ) -> float:
        """
        Signed edge:
        - Positive → buy side has edge (our prob > market price)
        - Negative → sell side has edge (our prob < market price)
        """
        if side.upper() == "BUY":
            return estimated_probability - market_price
        else:
            return market_price - estimated_probability

    def is_tradeable(self, edge: float, spread: float) -> bool:
        """
        Edge must exceed half-spread + min_edge to be profitable after transaction costs.
        Polymarket charges ~1% taker fee, embedded in the spread.
        """
        required_edge = (spread / 2.0) + self._min_edge
        return abs(edge) >= required_edge

    def score_liquidity(
        self,
        orderbook: OrderBook,
        target_size_usdc: float,
        side: str = "BUY",
    ) -> float:
        """
        Returns 0.0–1.0. Score of 1.0 means ample liquidity for target_size.
        Computes how much of the target size can be filled within 2% price impact.
        """
        if not orderbook.asks and not orderbook.bids:
            return 0.0

        levels = orderbook.asks if side.upper() == "BUY" else orderbook.bids
        if not levels:
            return 0.0

        best_price = levels[0].price
        max_price_impact = 0.02  # 2% slippage tolerance

        available_usdc = 0.0
        for level in levels:
            if abs(level.price - best_price) / best_price > max_price_impact:
                break
            available_usdc += level.price * level.size

        return min(1.0, available_usdc / max(target_size_usdc, 1.0))

    def check_market_efficiency(
        self, yes_price: float, no_price: float
    ) -> dict[str, float | bool]:
        """
        YES + NO prices should sum to ~1.0. Divergence indicates:
        - Sum > 1.0: maker spread is wide (normal)
        - Sum < 0.95: potential arbitrage opportunity
        """
        price_sum = yes_price + no_price
        divergence = abs(1.0 - price_sum)
        has_arb = price_sum < 0.97  # > 3% gap is unusual

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "price_sum": round(price_sum, 4),
            "divergence": round(divergence, 4),
            "has_arbitrage_signal": has_arb,
        }

    def score_market(
        self,
        edge: float,
        spread: float,
        liquidity_score: float,
        time_to_expiry_hours: float = 720.0,
    ) -> float:
        """
        Composite market attractiveness score (0–1).
        Factors: edge magnitude, spread tightness, liquidity, time horizon.
        """
        # Edge score: how many multiples of min_edge is this?
        edge_score = min(1.0, abs(edge) / (self._min_edge * 3))

        # Spread score: tighter is better (0.01 spread → 1.0, 0.20 spread → 0.0)
        spread_score = max(0.0, 1.0 - spread / 0.20)

        # Time score: prefer markets that resolve within 30 days
        time_score = max(0.0, 1.0 - time_to_expiry_hours / (30 * 24))

        return round(
            0.40 * edge_score
            + 0.25 * spread_score
            + 0.25 * liquidity_score
            + 0.10 * time_score,
            4,
        )
