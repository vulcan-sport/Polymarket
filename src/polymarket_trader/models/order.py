from datetime import datetime
from pydantic import BaseModel


class Order(BaseModel):
    order_id: str
    market_id: str = ""
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    size_filled: float = 0.0
    status: str = "LIVE"  # "LIVE", "MATCHED", "CANCELLED", "PARTIAL"
    created_at: datetime = datetime.now()

    @property
    def size_remaining(self) -> float:
        return self.size - self.size_filled


class Position(BaseModel):
    market_id: str
    token_id: str
    outcome: str  # "YES" or "NO"
    size: float
    avg_price: float
    current_price: float = 0.0
    market_question: str = ""

    @property
    def unrealized_pnl(self) -> float:
        return round((self.current_price - self.avg_price) * self.size, 4)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return round((self.current_price - self.avg_price) / self.avg_price * 100, 2)

    @property
    def cost_basis_usdc(self) -> float:
        return round(self.avg_price * self.size, 4)

    @property
    def current_value_usdc(self) -> float:
        return round(self.current_price * self.size, 4)


class Trade(BaseModel):
    trade_id: str
    market_id: str = ""
    token_id: str
    side: str
    price: float
    size: float
    fee: float = 0.0
    created_at: datetime = datetime.now()

    @property
    def notional_usdc(self) -> float:
        return round(self.price * self.size, 4)
