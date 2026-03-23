"""Tests for pricing engines and edge detection."""

import pytest

from polymarket_trader.pricing.base import MarketSnapshot, PricingSignal, build_engine
from polymarket_trader.pricing.edge import EdgeDetector
from polymarket_trader.pricing.kelly import KellyEngine


def make_snapshot(
    token_id: str = "tok1",
    market_price: float = 0.5,
    spread: float = 0.02,
    volume: float = 10000.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        token_id=token_id,
        market_price=market_price,
        best_bid=market_price - spread / 2,
        best_ask=market_price + spread / 2,
        spread=spread,
        volume_24h=volume,
        liquidity=5000.0,
        time_to_expiry_hours=720.0,
    )


class TestKellyEngine:
    @pytest.fixture
    def engine(self) -> KellyEngine:
        return KellyEngine(max_kelly_fraction=0.25, bankroll_usdc=1000.0)

    @pytest.mark.asyncio
    async def test_buy_signal_when_underpriced(self, engine: KellyEngine) -> None:
        snapshot = make_snapshot(market_price=0.40)
        signal = await engine.compute_signal(
            snapshot, context={"claude_probability": 0.60, "confidence": 0.8}
        )
        assert signal.recommended_side == "BUY"
        assert signal.edge > 0
        assert signal.fair_value > snapshot.market_price

    @pytest.mark.asyncio
    async def test_sell_signal_when_overpriced(self, engine: KellyEngine) -> None:
        snapshot = make_snapshot(market_price=0.80)
        signal = await engine.compute_signal(
            snapshot, context={"claude_probability": 0.55, "confidence": 0.8}
        )
        assert signal.recommended_side == "SELL"

    @pytest.mark.asyncio
    async def test_no_signal_at_fair_value(self, engine: KellyEngine) -> None:
        snapshot = make_snapshot(market_price=0.50)
        signal = await engine.compute_signal(
            snapshot, context={"claude_probability": 0.50, "confidence": 1.0}
        )
        assert signal.recommended_side is None

    @pytest.mark.asyncio
    async def test_kelly_respects_max_fraction(self, engine: KellyEngine) -> None:
        snapshot = make_snapshot(market_price=0.10)
        signal = await engine.compute_signal(
            snapshot, context={"claude_probability": 0.90, "confidence": 1.0}
        )
        kelly_fraction = signal.metadata.get("kelly_fraction", 0)
        assert kelly_fraction <= engine._max_kelly

    @pytest.mark.asyncio
    async def test_size_bounded_by_bankroll(self, engine: KellyEngine) -> None:
        snapshot = make_snapshot(market_price=0.30)
        signal = await engine.compute_signal(
            snapshot, context={"claude_probability": 0.70, "confidence": 0.9}
        )
        assert signal.recommended_size_usdc <= engine._bankroll

    @pytest.mark.asyncio
    async def test_calibration_updates_bankroll(self, engine: KellyEngine) -> None:
        initial_bankroll = engine._bankroll
        trades = [{"pnl": 50.0}, {"pnl": -20.0}]
        await engine.calibrate(trades)
        assert engine._bankroll == initial_bankroll + 30.0

    def test_registered_in_factory(self) -> None:
        engine = build_engine("kelly_criterion", bankroll_usdc=500.0)
        assert isinstance(engine, KellyEngine)


class TestEdgeDetector:
    @pytest.fixture
    def detector(self) -> EdgeDetector:
        return EdgeDetector(min_edge=0.03)

    def test_positive_edge_on_buy(self, detector: EdgeDetector) -> None:
        edge = detector.compute_edge(
            market_price=0.40, estimated_probability=0.60, side="BUY"
        )
        assert edge == pytest.approx(0.20)

    def test_positive_edge_on_sell(self, detector: EdgeDetector) -> None:
        edge = detector.compute_edge(
            market_price=0.70, estimated_probability=0.50, side="SELL"
        )
        assert edge == pytest.approx(0.20)

    def test_tradeable_with_sufficient_edge(self, detector: EdgeDetector) -> None:
        assert detector.is_tradeable(edge=0.10, spread=0.02) is True

    def test_not_tradeable_with_small_edge(self, detector: EdgeDetector) -> None:
        assert detector.is_tradeable(edge=0.01, spread=0.02) is False

    def test_arb_detection(self, detector: EdgeDetector) -> None:
        result = detector.check_market_efficiency(yes_price=0.45, no_price=0.45)
        assert result["price_sum"] == pytest.approx(0.90)
        assert result["has_arbitrage_signal"] is True

    def test_no_arb_normal_market(self, detector: EdgeDetector) -> None:
        result = detector.check_market_efficiency(yes_price=0.51, no_price=0.50)
        assert result["has_arbitrage_signal"] is False

    def test_composite_score_range(self, detector: EdgeDetector) -> None:
        score = detector.score_market(edge=0.10, spread=0.02, liquidity_score=0.8)
        assert 0.0 <= score <= 1.0
