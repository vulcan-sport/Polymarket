"""
Polymarket Trader MCP Server

Exposes all trading capabilities as MCP tools for Claude Desktop / Claude Code.

To connect to Claude Desktop, add to claude_desktop_config.json:
{
  "mcpServers": {
    "polymarket-trader": {
      "command": "uv",
      "args": [
        "--directory", "/Users/parthgosalia/Desktop/Polymarket",
        "run", "python", "-m", "polymarket_trader.mcp.server"
      ]
    }
  }
}
"""

import json
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP

from ..analyst.claude_analyst import ClaudeAnalyst
from ..config.settings import load_config
from ..connectors.clob_connector import CLOBConnector
from ..connectors.data_connector import DataConnector
from ..connectors.gamma_connector import GammaConnector
from ..pricing.base import build_engine
from ..pricing.kelly import KellyEngine  # ensure registration
from ..risk.manager import RiskManager
from ..trader.executor import TradeExecutor
from ..trader.loop import AutonomousLoop

# ------------------------------------------------------------------ #
# Startup: wire up all components                                     #
# ------------------------------------------------------------------ #

mcp = FastMCP("polymarket-trader")

_config = load_config()
_clob = CLOBConnector(_config)
_gamma = GammaConnector(_config.api.gamma_host)
_data = DataConnector(_config.api.data_host, _config.auth.wallet_address)
_risk = RiskManager(_config.risk)
_analyst = ClaudeAnalyst(_config.api.anthropic_api_key, _config.api.claude_model)

_pricing_engine = build_engine(
    _config.api.pricing_engine,
    max_kelly_fraction=_config.risk.max_kelly_fraction,
    bankroll_usdc=_config.risk.max_total_exposure_usdc,
)

_executor = TradeExecutor(_clob, _gamma, _data, _analyst, _pricing_engine, _risk, _config)
_loop = AutonomousLoop(_executor, _risk, _config.loop)

_initialized = False


async def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        await _clob.initialize()
        _initialized = True


# ------------------------------------------------------------------ #
# Market Discovery Tools                                              #
# ------------------------------------------------------------------ #


@mcp.tool()
async def search_markets(query: str, limit: int = 20, active_only: bool = True) -> str:
    """
    Search Polymarket markets by keyword.
    Returns markets with current prices, volume, and liquidity.
    """
    await _ensure_initialized()
    markets = await _gamma.search_markets(query=query, active=active_only, limit=limit)
    if not markets:
        return f"No markets found for query: '{query}'"

    results = []
    for m in markets:
        results.append(
            {
                "condition_id": m.condition_id,
                "question": m.question,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "volume_usdc": round(m.volume, 2),
                "liquidity_usdc": round(m.liquidity, 2),
                "end_date": m.end_date.strftime("%Y-%m-%d") if m.end_date else None,
                "accepting_orders": m.accepting_orders,
            }
        )
    return json.dumps(results, indent=2)


@mcp.tool()
async def get_market_details(condition_id: str) -> str:
    """
    Get full details for a specific market including live orderbook snapshot.
    """
    await _ensure_initialized()
    try:
        market = await _clob.get_market(condition_id)
    except Exception:
        market = await _gamma.get_market_by_condition_id(condition_id)

    if not market:
        return f"Market not found: {condition_id}"

    # Enrich with orderbook
    book_data = {}
    if market.yes_token_id:
        try:
            book = await _clob.get_orderbook(market.yes_token_id)
            book_data = {
                "best_bid": book.best_bid,
                "best_ask": book.best_ask,
                "spread": book.spread,
                "mid_price": book.mid_price,
            }
        except Exception:
            pass

    return json.dumps(
        {
            **market.to_summary(),
            "orderbook": book_data,
            "yes_token_id": market.yes_token_id,
            "no_token_id": market.no_token_id,
        },
        indent=2,
    )


@mcp.tool()
async def get_trending_markets(limit: int = 10) -> str:
    """Return top Polymarket markets by 24-hour volume."""
    await _ensure_initialized()
    markets = await _gamma.get_trending_markets(limit=limit)
    return json.dumps([m.to_summary() for m in markets], indent=2)


@mcp.tool()
async def get_orderbook(token_id: str) -> str:
    """Get live orderbook (bids and asks) for a specific outcome token."""
    await _ensure_initialized()
    book = await _clob.get_orderbook(token_id)
    return json.dumps(
        {
            "token_id": token_id,
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "spread": book.spread,
            "mid_price": book.mid_price,
            "top_bids": [{"price": b.price, "size": b.size} for b in book.bids[:5]],
            "top_asks": [{"price": a.price, "size": a.size} for a in book.asks[:5]],
        },
        indent=2,
    )


# ------------------------------------------------------------------ #
# Thesis Analysis Tools                                               #
# ------------------------------------------------------------------ #


@mcp.tool()
async def analyze_thesis(thesis: str, max_markets: int = 10) -> str:
    """
    Analyze an investment thesis and find the best Polymarket markets to trade.

    Claude will:
    1. Extract keywords from your thesis
    2. Search for relevant markets
    3. Score each market by relevance, edge, and liquidity
    4. Return ranked trade recommendations

    Example thesis: "I believe Trump will sign a crypto executive order before June 2026"
    """
    await _ensure_initialized()
    analysis = await _executor.thesis_to_trades(thesis, max_markets=max_markets)

    output: dict[str, Any] = {
        "thesis": analysis.thesis,
        "summary": analysis.summary,
        "markets_analyzed": len(analysis.markets_found),
        "tradeable_markets": len(analysis.tradeable_markets),
        "top_recommendations": [],
    }

    for score in analysis.top_markets[:5]:
        output["top_recommendations"].append(
            {
                "question": score.market.question,
                "condition_id": score.market.condition_id,
                "recommended_side": score.recommended_side,
                "recommended_price": score.recommended_price,
                "estimated_probability": score.estimated_probability,
                "composite_score": round(score.composite_score, 3),
                "relevance": round(score.relevance_score, 3),
                "edge": round(score.edge_score, 3),
                "reasoning": score.claude_reasoning,
            }
        )

    # Surface pending approvals
    pending = _risk.get_pending_decisions()
    if pending:
        output["pending_approvals"] = [d.to_summary() for d in pending]

    return json.dumps(output, indent=2)


@mcp.tool()
async def estimate_market_probability(condition_id: str, context: str = "") -> str:
    """
    Ask Claude to estimate the true probability for a market's YES outcome.
    Optionally provide additional context to inform the estimate.
    """
    await _ensure_initialized()
    market = await _gamma.get_market_by_condition_id(condition_id)
    if not market:
        return f"Market not found: {condition_id}"

    estimate = _analyst.estimate_probability(market, context)
    return json.dumps({"market": market.question, **estimate}, indent=2)


@mcp.tool()
async def find_arbitrage(condition_id: str) -> str:
    """
    Check if YES + NO prices sum to something other than 1.0, indicating a potential arbitrage.
    """
    await _ensure_initialized()
    market = await _gamma.get_market_by_condition_id(condition_id)
    if not market:
        return f"Market not found: {condition_id}"

    from ..pricing.edge import EdgeDetector
    detector = EdgeDetector()
    yes_price = market.yes_price or 0.5
    no_price = market.no_price or 0.5
    result = detector.check_market_efficiency(yes_price, no_price)
    return json.dumps({"market": market.question, **result}, indent=2)


# ------------------------------------------------------------------ #
# Portfolio Tools                                                      #
# ------------------------------------------------------------------ #


@mcp.tool()
async def get_portfolio() -> str:
    """Show all current positions with unrealized P&L and current prices."""
    await _ensure_initialized()
    summary = await _executor.get_portfolio_summary()
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_pnl_summary() -> str:
    """Return daily realized P&L, total unrealized P&L, and portfolio value."""
    await _ensure_initialized()
    pnl = await _data.get_pnl()
    portfolio_value = await _data.get_portfolio_value()
    return json.dumps({"portfolio_value_usdc": portfolio_value, **pnl}, indent=2)


@mcp.tool()
async def get_open_orders() -> str:
    """List all live (unfilled) orders."""
    await _ensure_initialized()
    orders = await _clob.get_open_orders()
    return json.dumps(
        [
            {
                "order_id": o.order_id,
                "token_id": o.token_id[:16] + "...",
                "side": o.side,
                "price": o.price,
                "size": o.size,
                "size_filled": o.size_filled,
                "size_remaining": o.size_remaining,
                "status": o.status,
            }
            for o in orders
        ],
        indent=2,
    )


@mcp.tool()
async def get_balance() -> str:
    """Return USDC balance available for trading."""
    await _ensure_initialized()
    balance = await _clob.get_balance()
    return json.dumps({"balance_usdc": balance}, indent=2)


# ------------------------------------------------------------------ #
# Trading Tools                                                        #
# ------------------------------------------------------------------ #


@mcp.tool()
async def place_order(
    condition_id: str,
    side: str,
    price: float,
    size_usdc: float,
    order_type: str = "GTC",
) -> str:
    """
    Place a limit order on a market.

    In confirm mode (default): returns a pending decision for your approval.
    In autonomous mode: executes immediately if within risk limits.

    Args:
        condition_id: Market condition ID
        side: "YES" to buy YES shares, "NO" to buy NO shares
        price: Limit price (0.01 – 0.99)
        size_usdc: Position size in USDC
        order_type: "GTC" (Good Till Cancel) or "FOK" (Fill or Kill)
    """
    await _ensure_initialized()
    from ..models.analysis import TradeDecision

    market = await _gamma.get_market_by_condition_id(condition_id)
    if not market:
        return json.dumps({"error": f"Market not found: {condition_id}"})

    token_id = market.yes_token_id if side.upper() == "YES" else market.no_token_id
    if not token_id:
        return json.dumps({"error": "No token ID found for this market/side"})

    decision = TradeDecision(
        market_id=condition_id,
        token_id=token_id,
        side="BUY",
        price=price,
        size_usdc=size_usdc,
        reasoning=f"Manual order: {side} ${size_usdc} @ {price}",
    )

    positions = await _data.get_positions()
    decision = await _risk.evaluate_trade(decision, positions)

    if decision.rejected:
        return json.dumps(
            {"error": f"Order rejected by risk manager: {decision.reject_reason}"}
        )

    if decision.approved:
        result = await _executor.execute_decision(decision)
        return json.dumps({"status": "executed", **result}, indent=2)

    # Confirm mode: pending approval
    return json.dumps(
        {
            "status": "pending_approval",
            "decision_id": decision.decision_id,
            "message": f"Order pending your approval. "
                       f"Call approve_trade('{decision.decision_id}') to execute "
                       f"or reject_trade('{decision.decision_id}') to cancel.",
            "order": decision.to_summary(),
        },
        indent=2,
    )


@mcp.tool()
async def cancel_order(order_id: str) -> str:
    """Cancel a specific open order."""
    await _ensure_initialized()
    result = await _clob.cancel_order(order_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def cancel_all_orders() -> str:
    """Cancel all open orders across all markets."""
    await _ensure_initialized()
    result = await _clob.cancel_all_orders()
    return json.dumps(result, indent=2)


@mcp.tool()
async def approve_trade(decision_id: str) -> str:
    """
    Approve a pending trade decision and execute it immediately.
    Only needed in confirm mode (the default).
    """
    await _ensure_initialized()
    decision = _risk.approve_pending(decision_id)
    if not decision:
        pending = _risk.get_pending_decisions()
        pending_ids = [d.decision_id for d in pending]
        return json.dumps(
            {
                "error": f"Decision '{decision_id}' not found.",
                "pending_ids": pending_ids,
            }
        )

    result = await _executor.execute_decision(decision)
    return json.dumps({"approved": True, **result}, indent=2)


@mcp.tool()
async def reject_trade(decision_id: str, reason: str = "") -> str:
    """Reject a pending trade decision without executing it."""
    await _ensure_initialized()
    decision = _risk.reject_pending(decision_id, reason)
    if not decision:
        return json.dumps({"error": f"Decision '{decision_id}' not found."})
    return json.dumps(
        {"rejected": True, "decision_id": decision_id, "reason": reason or "User rejected"}
    )


# ------------------------------------------------------------------ #
# Autonomous Loop Tools                                                #
# ------------------------------------------------------------------ #


@mcp.tool()
async def start_autonomous_loop(thesis: str, poll_interval_seconds: int = 60) -> str:
    """
    Start continuous market monitoring for a given investment thesis.

    Claude will scan markets every poll_interval_seconds, identify opportunities,
    and either propose trades (confirm mode) or execute them automatically (autonomous mode).
    """
    await _ensure_initialized()
    _config.loop.poll_interval_seconds = poll_interval_seconds
    await _loop.start(thesis)
    return json.dumps(
        {
            "status": "started",
            "thesis": thesis[:100],
            "poll_interval_seconds": poll_interval_seconds,
            "approval_mode": _config.risk.approval_mode,
            "message": "Loop running. Call get_loop_status() to check progress.",
        },
        indent=2,
    )


@mcp.tool()
async def stop_autonomous_loop() -> str:
    """Gracefully stop the autonomous trading loop."""
    await _loop.stop()
    return json.dumps({"status": "stopped"})


@mcp.tool()
async def get_loop_status() -> str:
    """Return current loop state, active thesis, pending decisions, and iteration count."""
    status = _loop.get_status()
    return json.dumps(status, indent=2)


# ------------------------------------------------------------------ #
# Risk & Configuration Tools                                           #
# ------------------------------------------------------------------ #


@mcp.tool()
async def set_approval_mode(mode: str) -> str:
    """
    Toggle trading approval mode.

    'confirm'    = Claude proposes trades, you must approve each one (safe default)
    'autonomous' = Claude executes trades within risk limits without asking
    """
    if mode not in ("confirm", "autonomous"):
        return json.dumps({"error": "mode must be 'confirm' or 'autonomous'"})
    _risk.set_approval_mode(mode)  # type: ignore[arg-type]
    return json.dumps(
        {
            "approval_mode": mode,
            "message": (
                "Claude will now propose trades for your approval."
                if mode == "confirm"
                else "Claude will now trade autonomously within your risk limits."
            ),
        }
    )


@mcp.tool()
async def get_risk_status() -> str:
    """Return current risk limits, daily loss consumed, exposure, and limit utilization."""
    await _ensure_initialized()
    positions = await _data.get_positions()
    status = _risk.get_status(positions)
    return json.dumps(status, indent=2)


@mcp.tool()
async def update_risk_limits(
    max_position_usdc: float | None = None,
    daily_loss_limit_usdc: float | None = None,
    max_exposure_usdc: float | None = None,
    min_edge: float | None = None,
) -> str:
    """
    Update runtime risk limits without restarting the server.

    Args:
        max_position_usdc: Max size per individual position in USDC
        daily_loss_limit_usdc: Halt trading if daily losses exceed this amount
        max_exposure_usdc: Max total open exposure across all positions
        min_edge: Minimum probability edge (e.g. 0.03 = 3%) required to trade
    """
    _risk.update_limits(max_position_usdc, daily_loss_limit_usdc, max_exposure_usdc, min_edge)
    positions = await _data.get_positions()
    return json.dumps({"updated": True, "new_limits": _risk.get_status(positions)}, indent=2)


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    mcp.run(transport="stdio")
