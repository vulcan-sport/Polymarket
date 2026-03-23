"""
TradeExecutor orchestrates the full pipeline:
thesis → market search → scoring → pricing → risk → execution
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ..analyst.claude_analyst import ClaudeAnalyst
from ..config.schema import AppConfig
from ..connectors.clob_connector import CLOBConnector
from ..connectors.data_connector import DataConnector
from ..connectors.gamma_connector import GammaConnector
from ..models.analysis import MarketScore, ThesisAnalysis, TradeDecision
from ..models.market import Market, OrderBook
from ..models.order import Position
from ..pricing.base import MarketSnapshot, PricingEngine
from ..pricing.edge import EdgeDetector
from ..risk.manager import RiskManager


class TradeExecutor:
    def __init__(
        self,
        clob: CLOBConnector,
        gamma: GammaConnector,
        data: DataConnector,
        analyst: ClaudeAnalyst,
        pricing: PricingEngine,
        risk: RiskManager,
        config: AppConfig,
    ) -> None:
        self._clob = clob
        self._gamma = gamma
        self._data = data
        self._analyst = analyst
        self._pricing = pricing
        self._risk = risk
        self._config = config
        self._edge_detector = EdgeDetector(config.risk.min_edge_threshold)

    # ------------------------------------------------------------------ #
    # Thesis → Trades                                                      #
    # ------------------------------------------------------------------ #

    async def thesis_to_trades(
        self,
        thesis: str,
        max_markets: int = 10,
    ) -> ThesisAnalysis:
        """
        Full pipeline:
        1. Extract keywords from thesis
        2. Search Gamma API (2-3 queries)
        3. Filter by quality
        4. Fetch orderbooks
        5. Claude scores markets
        6. Build TradeDecisions for top markets
        7. Gate through RiskManager
        8. Return ThesisAnalysis
        """
        logger.info(f"thesis_to_trades: '{thesis[:60]}...'")

        # Step 1: keyword extraction
        keywords = self._analyst.extract_keywords(thesis)
        logger.debug(f"Keywords: {keywords}")

        # Step 2: multi-query market search (deduplicated)
        all_markets: dict[str, Market] = {}
        queries = [
            " ".join(keywords[:3]),
            " ".join(keywords[3:6]) if len(keywords) > 3 else keywords[0],
        ]
        for q in queries:
            if not q.strip():
                continue
            found = await self._gamma.search_markets(
                query=q,
                active=True,
                limit=30,
                min_liquidity=200.0,
            )
            for m in found:
                all_markets[m.condition_id] = m

        logger.info(f"Found {len(all_markets)} candidate markets after dedup")

        if not all_markets:
            return ThesisAnalysis(
                thesis=thesis,
                markets_found=[],
                summary="No markets found. Try different keywords.",
            )

        # Step 3: filter by quality
        candidates = [
            m
            for m in all_markets.values()
            if m.accepting_orders and not m.closed and m.liquidity >= 200.0
        ]
        candidates = sorted(candidates, key=lambda m: m.volume, reverse=True)[:max_markets]

        # Step 4: enrich with orderbook data
        candidates = await self._enrich_with_orderbooks(candidates)

        # Step 5: Claude scores
        analysis = await self._analyst.analyze_thesis(thesis, candidates, max_markets=max_markets)

        # Step 6: build TradeDecisions for top tradeable markets
        current_positions = await self._data.get_positions()
        decisions = []
        for score in analysis.tradeable_markets[:5]:
            decision = await self._build_decision(score, current_positions)
            if decision:
                decisions.append(decision)

        # Attach decisions to analysis (embed in summary for now)
        if decisions:
            decision_lines = []
            for d in decisions:
                decision_lines.append(
                    f"  [{d.decision_id}] {d.side} ${d.size_usdc:.2f} "
                    f"@ {d.price:.3f} | edge={d.edge:.1%} | {d.reasoning[:60]}"
                )
            analysis.summary += "\n\nPending trade decisions:\n" + "\n".join(decision_lines)

        return analysis

    async def _enrich_with_orderbooks(self, markets: list[Market]) -> list[Market]:
        """Fetch orderbooks for YES tokens and attach best_bid/ask/spread to markets."""
        token_ids = [m.yes_token_id for m in markets if m.yes_token_id]
        books: dict[str, OrderBook] = {}
        for book in await self._clob.get_orderbooks(token_ids):
            books[book.token_id] = book

        for m in markets:
            if m.yes_token_id and m.yes_token_id in books:
                book = books[m.yes_token_id]
                m.best_bid = book.best_bid
                m.best_ask = book.best_ask
                m.spread = book.spread
        return markets

    async def _build_decision(
        self,
        score: MarketScore,
        current_positions: list[Position],
    ) -> TradeDecision | None:
        market = score.market
        token_id = market.yes_token_id if score.recommended_side == "YES" else market.no_token_id
        if not token_id:
            return None

        mid = market.yes_price or 0.5
        if market.best_bid and market.best_ask:
            mid = (market.best_bid + market.best_ask) / 2

        # Build MarketSnapshot for pricing engine
        book_snapshot = await self._clob.get_orderbook(token_id)
        snapshot = MarketSnapshot(
            token_id=token_id,
            market_price=mid,
            best_bid=book_snapshot.best_bid or mid * 0.99,
            best_ask=book_snapshot.best_ask or mid * 1.01,
            spread=book_snapshot.spread or 0.02,
            bid_depth=[(b.price, b.size) for b in book_snapshot.bids[:5]],
            ask_depth=[(a.price, a.size) for a in book_snapshot.asks[:5]],
            volume_24h=market.volume,
            liquidity=market.liquidity,
            time_to_expiry_hours=(
                (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
                if market.end_date
                else 720.0
            ),
        )

        context = {}
        if score.estimated_probability is not None:
            context["claude_probability"] = score.estimated_probability
            context["confidence"] = score.relevance_score

        signal = await self._pricing.compute_signal(snapshot, context)

        if not self._pricing.is_signal_tradeable(signal, self._config.risk.min_edge_threshold):
            return None

        # clamp size to position limit
        size_usdc = min(signal.recommended_size_usdc, self._config.risk.max_position_size_usdc)

        decision = TradeDecision(
            market_id=market.condition_id,
            token_id=token_id,
            side="BUY" if score.recommended_side == "YES" else "BUY",
            price=signal.recommended_price,
            size_usdc=size_usdc,
            kelly_fraction=signal.metadata.get("kelly_fraction", 0.0),
            edge=signal.edge,
            confidence=signal.confidence,
            reasoning=score.claude_reasoning[:200],
        )

        decision = await self._risk.evaluate_trade(decision, current_positions)
        return decision

    # ------------------------------------------------------------------ #
    # Execute                                                              #
    # ------------------------------------------------------------------ #

    async def execute_decision(self, decision: TradeDecision) -> dict[str, Any]:
        """Execute an approved TradeDecision via the CLOB connector."""
        if not decision.approved:
            return {"error": "Decision not approved", "decision_id": decision.decision_id}
        if decision.executed:
            return {"error": "Already executed", "decision_id": decision.decision_id}

        size_shares = decision.size_shares
        if size_shares <= 0:
            return {"error": "Invalid size", "decision_id": decision.decision_id}

        try:
            result = await self._clob.place_limit_order(
                token_id=decision.token_id,
                side=decision.side,
                price=decision.price,
                size=size_shares,
            )
            decision.executed = True
            decision.order_id = result.get("orderID", result.get("order_id"))
            logger.info(
                f"Executed decision {decision.decision_id}: "
                f"order_id={decision.order_id}"
            )
            return {"success": True, "order_id": decision.order_id, **result}
        except Exception as e:
            logger.error(f"Execute decision failed: {e}")
            return {"error": str(e), "decision_id": decision.decision_id}

    # ------------------------------------------------------------------ #
    # Portfolio                                                            #
    # ------------------------------------------------------------------ #

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Returns positions, P&L, open orders, balance, and risk status."""
        positions, open_orders, pnl, balance = (
            await self._data.get_positions(),
            await self._clob.get_open_orders(),
            await self._data.get_pnl(),
            await self._clob.get_balance(),
        )
        risk_status = self._risk.get_status(positions)

        return {
            "balance_usdc": balance,
            "positions": [
                {
                    "market": p.market_question or p.token_id[:16],
                    "outcome": p.outcome,
                    "size": p.size,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ],
            "open_orders_count": len(open_orders),
            "pnl": pnl,
            "risk": risk_status,
        }
