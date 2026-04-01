"""Tests for connectors (mocked HTTP via pytest-httpx)."""

import pytest
from unittest.mock import MagicMock
from pytest_httpx import HTTPXMock

from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
from polymarket_trader.connectors.clob_connector import CLOBConnector
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

# Real Polymarket Data API format: asset is a plain string (token ID)
MOCK_POSITION_REAL_API = {
    "proxyWallet": "0xuser123",
    "asset": "tok_yes",
    "outcome": "Yes",
    "title": "Will the US pass crypto legislation in 2026?",
    "conditionId": "0xabc123",
    "size": 100.0,
    "avgPrice": 0.45,
    "curPrice": 0.52,
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

    @pytest.mark.asyncio
    async def test_get_positions_real_api_format(self, httpx_mock: HTTPXMock) -> None:
        """Bug #2: asset is a string in the real Data API, not a nested dict."""
        wallet = "0xuser123"
        httpx_mock.add_response(json=[MOCK_POSITION_REAL_API])
        connector = DataConnector("https://data-api.polymarket.com", wallet)
        positions = await connector.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert isinstance(pos, Position)
        assert pos.token_id == "tok_yes"
        assert pos.market_id == "0xabc123"
        assert pos.outcome == "Yes"
        assert pos.size == 100.0
        assert pos.avg_price == pytest.approx(0.45)
        assert pos.current_price == pytest.approx(0.52)
        assert pos.market_question == "Will the US pass crypto legislation in 2026?"


class TestMarketModel:
    def test_clob_token_ids_list_passthrough(self) -> None:
        """A proper list is stored as-is."""
        m = Market(condition_id="0x1", question="Q?", clob_token_ids=["tok_yes", "tok_no"])
        assert m.clob_token_ids == ["tok_yes", "tok_no"]
        assert m.yes_token_id == "tok_yes"
        assert m.no_token_id == "tok_no"

    def test_clob_token_ids_json_string(self) -> None:
        """Bug #1: JSON-encoded string from the API is parsed into a list."""
        m = Market(condition_id="0x1", question="Q?", clob_token_ids='["tok_yes","tok_no"]')
        assert m.clob_token_ids == ["tok_yes", "tok_no"]
        assert m.yes_token_id == "tok_yes"
        assert m.no_token_id == "tok_no"

    def test_clob_token_ids_csv_string(self) -> None:
        """Comma-separated string is split into a list."""
        m = Market(condition_id="0x1", question="Q?", clob_token_ids="tok_yes,tok_no")
        assert m.clob_token_ids == ["tok_yes", "tok_no"]
        assert m.yes_token_id == "tok_yes"
        assert m.no_token_id == "tok_no"

    def test_clob_token_ids_empty_string(self) -> None:
        """Empty string yields an empty list."""
        m = Market(condition_id="0x1", question="Q?", clob_token_ids="")
        assert m.clob_token_ids == []
        assert m.yes_token_id is None


class TestGammaConnectorStringTokenIds:
    @pytest.mark.asyncio
    async def test_search_markets_with_string_clob_token_ids(self, httpx_mock: HTTPXMock) -> None:
        """Bug #1: clobTokenIds returned as a JSON string by the API."""
        mock_market_str_ids = {**MOCK_MARKET, "clobTokenIds": '["tok_yes","tok_no"]'}
        httpx_mock.add_response(json=[mock_market_str_ids])
        connector = GammaConnector("https://gamma-api.polymarket.com")
        markets = await connector.search_markets("crypto", min_liquidity=0.0)
        assert len(markets) == 1
        assert markets[0].yes_token_id == "tok_yes"
        assert markets[0].no_token_id == "tok_no"


class TestCLOBConnector:
    @pytest.mark.asyncio
    async def test_get_balance_passes_typed_params(self) -> None:
        """Bug fix: get_balance must pass BalanceAllowanceParams dataclass, not a dict."""
        mock_client = MagicMock()
        mock_client.get_balance_allowance.return_value = {"balance": 50_000_000}  # 50 USDC

        connector = CLOBConnector.__new__(CLOBConnector)
        connector._client = mock_client

        balance = await connector.get_balance()

        mock_client.get_balance_allowance.assert_called_once()
        call_args = mock_client.get_balance_allowance.call_args
        params_arg = call_args.kwargs.get("params") or call_args.args[0]

        assert isinstance(params_arg, BalanceAllowanceParams), (
            f"Expected BalanceAllowanceParams, got {type(params_arg).__name__}"
        )
        assert params_arg.asset_type == AssetType.COLLATERAL
        assert balance == pytest.approx(50.0)
