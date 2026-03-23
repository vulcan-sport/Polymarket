"""
Async httpx client for Polymarket's Data API.
Used for fetching user positions, P&L, and trade history.
"""

from typing import Any

import httpx
from loguru import logger

from ..models.order import Position, Trade


class DataConnector:
    def __init__(
        self, host: str = "https://data-api.polymarket.com", wallet_address: str = ""
    ) -> None:
        self._host = host.rstrip("/")
        self._wallet = wallet_address.lower()
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._host, timeout=30.0)
        return self._client

    async def get_positions(self, size_threshold: float = 0.0) -> list[Position]:
        """Fetch open positions for the configured wallet."""
        if not self._wallet:
            return []
        try:
            resp = await self._get_client().get(
                "/positions",
                params={"user": self._wallet, "sizeThreshold": size_threshold},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            return [self._parse_position(p) for p in items]
        except Exception as e:
            logger.error(f"DataConnector get_positions error: {e}")
            return []

    async def get_pnl(self) -> dict[str, float]:
        """Return realized and unrealized P&L summary."""
        positions = await self.get_positions()
        unrealized = sum(p.unrealized_pnl for p in positions)
        trades = await self.get_user_trades(limit=200)
        # Rough realized P&L: sum of (sell - buy) across matched trades
        realized = 0.0
        for t in trades:
            if t.side.upper() == "SELL":
                realized += t.notional_usdc
            else:
                realized -= t.notional_usdc
        return {
            "unrealized_pnl": round(unrealized, 4),
            "realized_pnl": round(realized, 4),
            "total_pnl": round(unrealized + realized, 4),
            "open_positions": len(positions),
        }

    async def get_portfolio_value(self) -> float:
        """Estimate total portfolio value in USDC."""
        positions = await self.get_positions()
        return round(sum(p.current_value_usdc for p in positions), 4)

    async def get_user_trades(self, limit: int = 100) -> list[Trade]:
        if not self._wallet:
            return []
        try:
            resp = await self._get_client().get(
                "/trades",
                params={"user": self._wallet, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            return [self._parse_trade(t) for t in items[:limit]]
        except Exception as e:
            logger.error(f"DataConnector get_user_trades error: {e}")
            return []

    @staticmethod
    def _parse_position(raw: dict[str, Any]) -> Position:
        asset = raw.get("asset", raw)
        return Position(
            market_id=asset.get("market", raw.get("market_id", "")),
            token_id=asset.get("token_id", raw.get("token_id", "")),
            outcome=asset.get("outcome", raw.get("outcome", "YES")),
            size=float(raw.get("size", 0)),
            avg_price=float(raw.get("avgPrice", raw.get("avg_price", 0))),
            current_price=float(
                asset.get("price", raw.get("current_price", 0))
            ),
            market_question=asset.get("question", raw.get("question", "")),
        )

    @staticmethod
    def _parse_trade(raw: dict[str, Any]) -> Trade:
        from datetime import datetime

        return Trade(
            trade_id=raw.get("id", raw.get("trade_id", "")),
            market_id=raw.get("market", raw.get("market_id", "")),
            token_id=raw.get("asset_id", raw.get("token_id", "")),
            side=raw.get("side", "BUY").upper(),
            price=float(raw.get("price", 0)),
            size=float(raw.get("size", 0)),
            fee=float(raw.get("fee", 0)),
            created_at=datetime.now(),
        )
