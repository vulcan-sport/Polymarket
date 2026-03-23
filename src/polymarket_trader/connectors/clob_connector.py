"""
Wrapper around py-clob-client's ClobClient.

Handles:
- L1 EIP-712 auth + L2 HMAC credential derivation
- All market data queries
- All order operations (place, cancel, list)
"""

import asyncio
from functools import partial
from typing import Any

from loguru import logger
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from ..config.schema import AppConfig
from ..models.market import Market, OrderBook, OrderLevel, Token
from ..models.order import Order, Position, Trade


class CLOBConnector:
    """
    Async-friendly wrapper around the synchronous py-clob-client.

    All blocking calls are run in a thread pool via asyncio.to_thread
    so they don't block the MCP server's event loop.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client: ClobClient | None = None

    async def initialize(self) -> None:
        """Create client and derive/set L2 API credentials."""
        cfg = self._config

        self._client = ClobClient(
            host=cfg.api.clob_host,
            key=cfg.auth.private_key,
            chain_id=cfg.auth.chain_id,
            signature_type=cfg.auth.signature_type,
            funder=cfg.auth.wallet_address if cfg.auth.signature_type != 0 else None,
        )

        # Use stored L2 creds if available, otherwise derive new ones
        if cfg.auth.api_key and cfg.auth.api_secret and cfg.auth.api_passphrase:
            creds = ApiCreds(
                api_key=cfg.auth.api_key,
                api_secret=cfg.auth.api_secret,
                api_passphrase=cfg.auth.api_passphrase,
            )
            self._client.set_api_creds(creds)
            logger.info("CLOB: using stored L2 credentials")
        else:
            logger.info("CLOB: deriving L2 credentials from private key...")
            creds = await asyncio.to_thread(self._client.create_or_derive_api_creds)
            self._client.set_api_creds(creds)
            logger.info(f"CLOB: derived L2 creds (api_key={creds.api_key[:8]}...)")

        logger.info("CLOBConnector initialized")

    def _require_client(self) -> ClobClient:
        if self._client is None:
            raise RuntimeError("CLOBConnector not initialized — call initialize() first")
        return self._client

    # ------------------------------------------------------------------ #
    # Market Data                                                          #
    # ------------------------------------------------------------------ #

    async def get_markets(self, cursor: str = "MA==", limit: int = 50) -> tuple[list[Market], str]:
        """Paginated market list. Returns (markets, next_cursor)."""
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_markets, cursor)
        markets = [self._parse_market(m) for m in raw.get("data", [])]
        next_cursor = raw.get("next_cursor", "LTE=")
        return markets, next_cursor

    async def get_market(self, condition_id: str) -> Market:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_market, condition_id)
        return self._parse_market(raw)

    async def get_orderbook(self, token_id: str) -> OrderBook:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_order_book, token_id)
        return self._parse_orderbook(token_id, raw)

    async def get_orderbooks(self, token_ids: list[str]) -> list[OrderBook]:
        tasks = [self.get_orderbook(tid) for tid in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        books = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Failed to fetch orderbook: {r}")
            else:
                books.append(r)
        return books

    async def get_midpoint(self, token_id: str) -> float:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_midpoint, token_id)
        return float(raw.get("mid", 0.5))

    async def get_price(self, token_id: str, side: str = "BUY") -> float:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_price, token_id, side)
        return float(raw.get("price", 0.0))

    async def get_spread(self, token_id: str) -> float:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_spread, token_id)
        return float(raw.get("spread", 0.0))

    async def get_last_trade_price(self, token_id: str) -> float:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_last_trade_price, token_id)
        return float(raw.get("price", 0.0))

    async def get_tick_size(self, token_id: str) -> float:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_tick_size, token_id)
        return float(raw.get("minimum_tick_size", 0.01))

    # ------------------------------------------------------------------ #
    # Account                                                              #
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> float:
        """Returns USDC balance available for trading."""
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_balance_allowance, params={"asset_type": 0})
        return float(raw.get("balance", 0.0)) / 1e6  # USDC has 6 decimals

    async def get_open_orders(self) -> list[Order]:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_orders)
        return [self._parse_order(o) for o in (raw if isinstance(raw, list) else [])]

    async def get_trades(self, limit: int = 100) -> list[Trade]:
        client = self._require_client()
        raw = await asyncio.to_thread(client.get_trades)
        trades = raw if isinstance(raw, list) else []
        return [self._parse_trade(t) for t in trades[:limit]]

    # ------------------------------------------------------------------ #
    # Trading                                                              #
    # ------------------------------------------------------------------ #

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> dict[str, Any]:
        """
        Place a limit order.

        Args:
            token_id: The YES or NO token ID
            side: "BUY" or "SELL"
            price: Limit price (0.0 – 1.0)
            size: Number of shares
            order_type: "GTC" (Good Till Cancel) or "FOK" (Fill or Kill)
        """
        client = self._require_client()
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY if side.upper() == "BUY" else SELL,
        )
        ot = OrderType.GTC if order_type.upper() == "GTC" else OrderType.FOK
        signed = await asyncio.to_thread(client.create_order, order_args)
        result = await asyncio.to_thread(client.post_order, signed, ot)
        logger.info(f"Order placed: {side} {size} shares @ {price} | result={result}")
        return result if isinstance(result, dict) else {"result": result}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        client = self._require_client()
        result = await asyncio.to_thread(client.cancel_order, order_id)
        logger.info(f"Order cancelled: {order_id}")
        return result if isinstance(result, dict) else {"result": result}

    async def cancel_all_orders(self) -> dict[str, Any]:
        client = self._require_client()
        result = await asyncio.to_thread(client.cancel_all)
        logger.info("All orders cancelled")
        return result if isinstance(result, dict) else {"result": result}

    # ------------------------------------------------------------------ #
    # Parsers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_market(raw: dict[str, Any]) -> Market:
        tokens = []
        for t in raw.get("tokens", []):
            tokens.append(
                Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0.0)),
                    winner=t.get("winner", False),
                )
            )

        clob_token_ids = raw.get("clobTokenIds", [])
        if not clob_token_ids:
            clob_token_ids = [t.token_id for t in tokens]

        end_date = None
        if raw.get("endDate") or raw.get("end_date_iso"):
            from datetime import datetime

            try:
                end_str = raw.get("endDate") or raw.get("end_date_iso", "")
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except Exception:
                pass

        return Market(
            condition_id=raw.get("condition_id", raw.get("conditionId", "")),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            slug=raw.get("market_slug", raw.get("slug", "")),
            category=raw.get("category"),
            tokens=tokens,
            end_date=end_date,
            active=raw.get("active", True),
            closed=raw.get("closed", False),
            accepting_orders=raw.get("accepting_orders", True),
            volume=float(raw.get("volume", 0.0)),
            liquidity=float(raw.get("liquidity", 0.0)),
            clob_token_ids=clob_token_ids,
        )

    @staticmethod
    def _parse_orderbook(token_id: str, raw: Any) -> OrderBook:
        bids = []
        asks = []
        if hasattr(raw, "bids"):
            for b in raw.bids or []:
                try:
                    bids.append(OrderLevel(price=float(b.price), size=float(b.size)))
                except Exception:
                    pass
            for a in raw.asks or []:
                try:
                    asks.append(OrderLevel(price=float(a.price), size=float(a.size)))
                except Exception:
                    pass
        elif isinstance(raw, dict):
            for b in raw.get("bids", []):
                try:
                    bids.append(OrderLevel(price=float(b["price"]), size=float(b["size"])))
                except Exception:
                    pass
            for a in raw.get("asks", []):
                try:
                    asks.append(OrderLevel(price=float(a["price"]), size=float(a["size"])))
                except Exception:
                    pass
        return OrderBook(
            token_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
        )

    @staticmethod
    def _parse_order(raw: dict[str, Any]) -> Order:
        from datetime import datetime

        return Order(
            order_id=raw.get("id", raw.get("order_id", "")),
            market_id=raw.get("market", ""),
            token_id=raw.get("asset_id", raw.get("token_id", "")),
            side=raw.get("side", "BUY").upper(),
            price=float(raw.get("price", 0)),
            size=float(raw.get("original_size", raw.get("size", 0))),
            size_filled=float(raw.get("size_matched", 0)),
            status=raw.get("status", "LIVE"),
            created_at=datetime.now(),
        )

    @staticmethod
    def _parse_trade(raw: dict[str, Any]) -> Trade:
        from datetime import datetime

        return Trade(
            trade_id=raw.get("id", raw.get("trade_id", "")),
            market_id=raw.get("market", ""),
            token_id=raw.get("asset_id", raw.get("token_id", "")),
            side=raw.get("side", "BUY").upper(),
            price=float(raw.get("price", 0)),
            size=float(raw.get("size", 0)),
            fee=float(raw.get("fee_rate_bps", 0)) / 10000,
            created_at=datetime.now(),
        )
