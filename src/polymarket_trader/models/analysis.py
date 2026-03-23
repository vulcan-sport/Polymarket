import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from .market import Market


class MarketScore(BaseModel):
    market: Market
    relevance_score: float = Field(ge=0, le=1)
    edge_score: float = Field(ge=0, le=1)
    liquidity_score: float = Field(ge=0, le=1)
    composite_score: float = Field(ge=0, le=1)
    claude_reasoning: str = ""
    recommended_side: str | None = None  # "YES", "NO", or None
    recommended_price: float | None = None
    estimated_probability: float | None = None  # Claude's probability estimate


class ThesisAnalysis(BaseModel):
    thesis: str
    markets_found: list[MarketScore] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def top_markets(self) -> list[MarketScore]:
        return sorted(self.markets_found, key=lambda m: m.composite_score, reverse=True)[:5]

    @property
    def tradeable_markets(self) -> list[MarketScore]:
        return [m for m in self.markets_found if m.recommended_side is not None]


class TradeDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    market_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size_usdc: float
    kelly_fraction: float = 0.0
    edge: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    approved: bool = False
    approved_by: str | None = None  # "user" or "auto"
    rejected: bool = False
    reject_reason: str = ""
    executed: bool = False
    order_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def size_shares(self) -> float:
        if self.price <= 0:
            return 0.0
        return round(self.size_usdc / self.price, 2)

    def to_summary(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "market_id": self.market_id,
            "side": self.side,
            "price": self.price,
            "size_usdc": self.size_usdc,
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "approved": self.approved,
            "executed": self.executed,
        }
