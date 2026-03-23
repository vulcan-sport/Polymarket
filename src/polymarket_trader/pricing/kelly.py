"""
Kelly Criterion pricing engine.

Uses the Kelly formula for binary bets to size positions:
    f* = (p * b - q) / b
    where:
        p = estimated win probability
        b = net odds (payout / stake = 1/price - 1 for a YES bet at `price`)
        q = 1 - p

Incorporates probability calibration to avoid overconfidence:
    - Shrinks estimates toward 0.5 based on confidence
    - Caps Kelly fraction at config.risk.max_kelly_fraction
"""

from loguru import logger

from .base import MarketSnapshot, PricingEngine, PricingSignal, register_engine


@register_engine
class KellyEngine(PricingEngine):
    name = "kelly_criterion"

    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        calibration_alpha: float = 0.8,  # shrinkage factor toward 0.5
        bankroll_usdc: float = 500.0,
        min_bankroll_fraction: float = 0.01,  # never bet less than 1% of bankroll
    ) -> None:
        self._max_kelly = max_kelly_fraction
        self._calibration_alpha = calibration_alpha
        self._bankroll = bankroll_usdc
        self._min_fraction = min_bankroll_fraction

    async def compute_signal(
        self,
        snapshot: MarketSnapshot,
        context: dict | None = None,
    ) -> PricingSignal:
        context = context or {}
        market_price = snapshot.market_price

        # Get probability estimate: prefer Claude's estimate, fall back to spread-based
        raw_prob = context.get("claude_probability")
        raw_confidence = context.get("confidence", 0.5)

        if raw_prob is None:
            raw_prob = self._estimate_from_snapshot(snapshot)
            raw_confidence = 0.4  # lower confidence without Claude

        # Apply calibration (shrink extreme probabilities toward 0.5)
        calibrated_prob = self._calibrate(raw_prob, raw_confidence)

        # Determine side and compute edge
        if calibrated_prob > market_price:
            side = "BUY"
            edge = calibrated_prob - market_price
        elif calibrated_prob < market_price:
            side = "SELL"
            edge = market_price - calibrated_prob
        else:
            return PricingSignal(
                token_id=snapshot.token_id,
                fair_value=calibrated_prob,
                confidence=raw_confidence,
                edge=0.0,
                recommended_side=None,
                recommended_price=market_price,
                recommended_size_usdc=0.0,
                signal_strength=0.0,
            )

        # Kelly formula
        kelly_fraction = self._kelly_fraction(calibrated_prob, market_price, side)
        kelly_fraction = min(kelly_fraction, self._max_kelly)
        kelly_fraction = max(kelly_fraction, 0.0)

        size_usdc = self._bankroll * kelly_fraction
        size_usdc = max(size_usdc, self._bankroll * self._min_fraction)

        # Limit price: buy slightly below mid, sell slightly above
        tick = 0.01
        if side == "BUY":
            limit_price = round(min(snapshot.best_ask, calibrated_prob + tick * 2), 2)
        else:
            limit_price = round(max(snapshot.best_bid, calibrated_prob - tick * 2), 2)

        # Signal strength: combination of edge magnitude and confidence
        signal_strength = min(1.0, (edge / 0.1) * raw_confidence)

        logger.debug(
            f"Kelly signal | token={snapshot.token_id[:8]} "
            f"p_model={calibrated_prob:.3f} p_market={market_price:.3f} "
            f"edge={edge:.3f} side={side} size=${size_usdc:.2f}"
        )

        return PricingSignal(
            token_id=snapshot.token_id,
            fair_value=calibrated_prob,
            confidence=raw_confidence,
            edge=edge if side == "BUY" else -edge,
            recommended_side=side,
            recommended_price=limit_price,
            recommended_size_usdc=size_usdc,
            signal_strength=signal_strength,
            metadata={
                "raw_probability": raw_prob,
                "calibrated_probability": calibrated_prob,
                "kelly_fraction": kelly_fraction,
                "bankroll_usdc": self._bankroll,
            },
        )

    async def calibrate(self, historical_trades: list[dict]) -> None:
        """Adjust bankroll estimate based on historical trade outcomes."""
        if not historical_trades:
            return
        # Simple P&L-based bankroll update
        total_pnl = sum(t.get("pnl", 0.0) for t in historical_trades)
        self._bankroll = max(100.0, self._bankroll + total_pnl)
        logger.info(f"Kelly calibrated: new bankroll=${self._bankroll:.2f}")

    def _kelly_fraction(self, win_prob: float, market_price: float, side: str) -> float:
        """
        Kelly formula for binary prediction markets.

        For a BUY at price p:
            - Win: share resolves YES → payout = 1.0, profit = (1 - p) per share
            - b (net odds) = (1 - p) / p
        """
        if side == "BUY":
            if market_price >= 1.0 or market_price <= 0:
                return 0.0
            b = (1.0 - market_price) / market_price
            q = 1.0 - win_prob
            f = (win_prob * b - q) / b
        else:  # SELL (betting on NO resolution = buying the inverse)
            no_price = 1.0 - market_price
            if no_price >= 1.0 or no_price <= 0:
                return 0.0
            no_win_prob = 1.0 - win_prob
            b = (1.0 - no_price) / no_price
            q = 1.0 - no_win_prob
            f = (no_win_prob * b - q) / b

        return max(0.0, f)

    def _calibrate(self, raw_prob: float, confidence: float) -> float:
        """
        Shrink raw probability toward 0.5 by (1 - confidence * alpha).
        High confidence (1.0) → no shrinkage.
        Low confidence (0.0) → full regression to 0.5.
        """
        shrinkage = 1.0 - confidence * self._calibration_alpha
        return raw_prob * (1.0 - shrinkage) + 0.5 * shrinkage

    @staticmethod
    def _estimate_from_snapshot(snapshot: MarketSnapshot) -> float:
        """
        Fallback probability estimate from orderbook data alone.
        Uses mid-price with a slight mean-reversion adjustment.
        """
        mid = snapshot.market_price
        # Time decay: markets near resolution tend to be more accurate
        if snapshot.time_to_expiry_hours < 24:
            return mid
        # Regress slightly toward 0.5 for distant markets
        return mid * 0.9 + 0.5 * 0.1
