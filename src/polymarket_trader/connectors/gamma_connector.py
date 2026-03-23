"""
Async httpx client for Polymarket's Gamma API.
Used for market search and discovery.
"""

from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from ..models.market import Market, Token


class GammaConnector:
    def __init__(self, host: str = "https://gamma-api.polymarket.com") -> None:
        self._host = host.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GammaConnector":
        self._client = httpx.AsyncClient(base_url=self._host, timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._host, timeout=30.0)
        return self._client

    async def search_markets(
        self,
        query: str,
        active: bool = True,
        limit: int = 20,
        order: str = "volume",
        min_liquidity: float = 100.0,
    ) -> list[Market]:
        """Search Polymarket markets by keyword."""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "order": order,
            "ascending": "false",
        }
        if active:
            params["active"] = "true"
            params["closed"] = "false"

        try:
            resp = await self._get_client().get("/markets", params=params)
            resp.raise_for_status()
            raw_markets = resp.json()
            if isinstance(raw_markets, dict):
                raw_markets = raw_markets.get("data", [])
        except Exception as e:
            logger.error(f"Gamma search_markets error: {e}")
            return []

        markets = [self._parse_market(m) for m in raw_markets]
        # Filter by minimum liquidity
        return [m for m in markets if m.liquidity >= min_liquidity]

    async def get_market_by_slug(self, slug: str) -> Market | None:
        try:
            resp = await self._get_client().get(f"/markets/{slug}")
            resp.raise_for_status()
            return self._parse_market(resp.json())
        except Exception as e:
            logger.error(f"Gamma get_market_by_slug error: {e}")
            return None

    async def get_market_by_condition_id(self, condition_id: str) -> Market | None:
        try:
            params = {"condition_id": condition_id}
            resp = await self._get_client().get("/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            if items:
                return self._parse_market(items[0])
        except Exception as e:
            logger.error(f"Gamma get_market_by_condition_id error: {e}")
        return None

    async def get_trending_markets(self, limit: int = 10) -> list[Market]:
        """Return top markets by 24h volume."""
        return await self.search_markets(query="", active=True, limit=limit, order="volume")

    async def get_events(
        self,
        query: str | None = None,
        active: bool = True,
        limit: int = 20,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if active:
            params["active"] = "true"
        try:
            resp = await self._get_client().get("/events", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.error(f"Gamma get_events error: {e}")
            return []

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
        if not clob_token_ids and tokens:
            clob_token_ids = [t.token_id for t in tokens]

        end_date = None
        for field in ("end_date_iso", "endDate", "end_date"):
            if raw.get(field):
                try:
                    end_date = datetime.fromisoformat(raw[field].replace("Z", "+00:00"))
                    break
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
            volume=float(raw.get("volume", raw.get("volumeNum", 0.0))),
            liquidity=float(raw.get("liquidity", raw.get("liquidityNum", 0.0))),
            clob_token_ids=clob_token_ids,
        )
