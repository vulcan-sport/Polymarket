"""
Abstract pricing engine interface.

This is the HFT extension point. To plug in a custom pricing engine
(ML-based, poker-inspired, news-sentiment, etc.):

1. Subclass PricingEngine
2. Implement compute_signal() and calibrate()
3. Set api.pricing_engine in config.yaml to your engine's name
4. Register in the factory at the bottom of this file

The TradeExecutor calls compute_signal() for every candidate market.
Zero other changes required.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MarketSnapshot:
    """Normalized market state passed as input to every pricing engine."""

    token_id: str
    market_price: float          # current mid price
    best_bid: float
    best_ask: float
    spread: float
    bid_depth: list[tuple[float, float]] = field(default_factory=list)   # [(price, size), ...]
    ask_depth: list[tuple[float, float]] = field(default_factory=list)
    last_trade_price: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    time_to_expiry_hours: float = 720.0  # default 30 days


@dataclass
class PricingSignal:
    """Output of any pricing engine for a single token."""

    token_id: str
    fair_value: float             # engine's estimate of true probability (0–1)
    confidence: float             # how confident the engine is (0–1)
    edge: float                   # fair_value − market_price (positive → buy edge)
    recommended_side: str | None  # "BUY", "SELL", or None
    recommended_price: float      # limit price to post
    recommended_size_usdc: float  # raw size before Kelly scaling
    signal_strength: float        # 0–1 composite signal strength
    metadata: dict = field(default_factory=dict)  # engine-specific data


class PricingEngine(ABC):
    """
    Abstract base class for all pricing engines.

    Built-in: KellyEngine (kelly_criterion)

    Future HFT skill examples:
    - ML probability model with news sentiment
    - Cross-market correlation arbitrage
    - Volatility surface fitting (options-inspired)
    - Poker-GTO expected value models
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this engine. Must match config.yaml api.pricing_engine."""
        ...

    @abstractmethod
    async def compute_signal(
        self,
        snapshot: MarketSnapshot,
        context: dict | None = None,
    ) -> PricingSignal:
        """
        Core method: given a market snapshot, produce a trading signal.

        Args:
            snapshot: Current market state (orderbook, prices, volume, etc.)
            context: Optional extra data — e.g., Claude's probability estimate:
                     {"claude_probability": 0.65, "confidence": 0.8}
        """
        ...

    @abstractmethod
    async def calibrate(self, historical_trades: list[dict]) -> None:
        """
        Update engine parameters based on historical performance.
        Called periodically by the autonomous loop.
        """
        ...

    def is_signal_tradeable(
        self,
        signal: PricingSignal,
        min_edge: float = 0.03,
    ) -> bool:
        """Default tradeability filter — engines can override."""
        return (
            abs(signal.edge) >= min_edge
            and signal.confidence >= 0.5
            and signal.recommended_side is not None
        )


# ------------------------------------------------------------------ #
# Engine Registry & Factory                                            #
# ------------------------------------------------------------------ #

_REGISTRY: dict[str, type[PricingEngine]] = {}


def register_engine(cls: type[PricingEngine]) -> type[PricingEngine]:
    """Decorator to register a pricing engine by name."""
    _REGISTRY[cls.name] = cls  # type: ignore[attr-defined]
    return cls


def build_engine(name: str, **kwargs) -> PricingEngine:
    """Instantiate a pricing engine by its registered name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown pricing engine '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)
