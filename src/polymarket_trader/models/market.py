import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, computed_field, field_validator


class Token(BaseModel):
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float = 0.0
    winner: bool = False


class OrderLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    token_id: str
    bids: list[OrderLevel] = Field(default_factory=list)
    asks: list[OrderLevel] = Field(default_factory=list)
    timestamp: str = ""
    hash: str = ""

    @computed_field
    @property
    def best_bid(self) -> float | None:
        return max((b.price for b in self.bids), default=None)

    @computed_field
    @property
    def best_ask(self) -> float | None:
        return min((a.price for a in self.asks), default=None)

    @computed_field
    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return round(self.best_ask - self.best_bid, 4)
        return None

    @computed_field
    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return round((self.best_bid + self.best_ask) / 2, 4)
        return None


class Market(BaseModel):
    condition_id: str
    question: str
    description: str = ""
    slug: str = ""
    category: str | None = None
    tokens: list[Token] = Field(default_factory=list)
    end_date: datetime | None = None
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    volume: float = 0.0
    liquidity: float = 0.0
    # Live data populated after orderbook fetch
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    last_trade_price: float | None = None
    # Raw clob token IDs [yes_token_id, no_token_id]
    clob_token_ids: list[str] = Field(default_factory=list)

    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def parse_clob_token_ids(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [s.strip() for s in v.split(",") if s.strip()]
        return []

    @property
    def yes_token_id(self) -> str | None:
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def no_token_id(self) -> str | None:
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None

    @property
    def yes_price(self) -> float | None:
        for t in self.tokens:
            if t.outcome.lower() in ("yes", "true"):
                return t.price
        return None

    @property
    def no_price(self) -> float | None:
        for t in self.tokens:
            if t.outcome.lower() in ("no", "false"):
                return t.price
        return None

    def to_summary(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume_usdc": self.volume,
            "liquidity_usdc": self.liquidity,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "accepting_orders": self.accepting_orders,
        }
