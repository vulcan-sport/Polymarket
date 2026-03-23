"""Tests for connectors (mocked HTTP via pytest-httpx)."""

import pytest
from pytest_httpx import HTTPXMock

from polymarket_trader.connectors.gamma_connector import GammaConnector
from polymarket_trader.connectors.data_connector import DataConnector
from polymarket_trader.models.market import Market
from polymarket_trader.models.order import Position


MOCK_MARKET = {
    "condition_id": "0xabc123",
    "question": "Will the US pass crypto legislation in 2026?",
    "description": "Resolves YES if a major crypto bill is signed.",
    "market_slug": "us-crypto-legislation-2026",
    "active": True,
    "closed": False,
    "accepting_orders": True,
    "volume": 50000.0,
    "liquidity": 10000.0,
    "tokens": [
        {"token_id": "tok_yes", "outcome": "Yes", "price": 0.45},
        {"token_id": "tok_no", "outcome": "No", "price": 0.55},
    ],
    "clobTokenIds": ["tok_yes", "tok_no"],
}

MOCK_POSITION = {
    "asset": {
        "market": "0xabc123",
        "token_id": "tok_yes",
        "outcome": "YES",
        "question": "Will the US pass crypto legislation in 2026?",
        "price": 0.52,
    },
    "size": 100.0,
    "avgPrice": 0.45,
}


class TestGammaConnector:
    @pytest.mark.asyncio
    async def test_search_markets_returns_markets(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(json=[MOCK_MARKET])
        connector = GammaConnector("https://gamma-api.polymarket.com")
        markets = await connector.search_markets("crypto", min_liquidity=0.0)
        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        assert markets[0].condition_id == "0xabc123"
        assert markets[0].yes_token_id == "tok_yes"

    @pytest.mark.asyncio
    async def test_search_markets_filters_low_liquidity(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(json=[MOCK_MARKET])
        connector = GammaConnector("https://gamma-api.polymarket.com")
        markets = await connector.search_markets("crypto", min_liquidity=50000.0)
        assert len(markets) == 0  # liquidity=10000 < 50000

    @pytest.mark.asyncio
    async def test_search_markets_handles_api_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(status_code=500, json={"error": "server error"})
        connector = GammaConnector("https://gamma-api.polymarket.com")
        markets = await connector.search_markets("crypto")
        assert markets == []


class TestDataConnector:
    @pytest.mark.asyncio
    async def test_get_positions_returns_positions(self, httpx_mock: HTTPXMock) -> None:
        wallet = "0xuser123"
        httpx_mock.add_response(json=[MOCK_POSITION])
        connector = DataConnector("https://data-api.polymarket.com", wallet)
        positions = await connector.get_positions()
        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].avg_price == 0.45
        assert positions[0].size == 100.0

    @pytest.mark.asyncio
    async def test_get_positions_empty_without_wallet(self) -> None:
        connector = DataConnector("https://data-api.polymarket.com", "")
        positions = await connector.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_pnl_aggregates_correctly(self, httpx_mock: HTTPXMock) -> None:
        wallet = "0xuser123"
        httpx_mock.add_response(json=[MOCK_POSITION])  # positions
        httpx_mock.add_response(json=[])               # trades
        connector = DataConnector("https://data-api.polymarket.com", wallet)
        pnl = await connector.get_pnl()
        assert "unrealized_pnl" in pnl
        assert "realized_pnl" in pnl
        assert "total_pnl" in pnl
        # Position: 100 shares, avg 0.45, current 0.52 → unrealized = (0.52-0.45)*100 = 7.0
        assert pnl["unrealized_pnl"] == pytest.approx(7.0, abs=0.01)
