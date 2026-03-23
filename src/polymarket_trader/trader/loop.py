"""
AutonomousLoop: continuous market monitoring and trading.

State machine:
IDLE → SCANNING → ANALYZING → DECIDING → (AWAITING_APPROVAL | EXECUTING) → IDLE
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger

from ..config.schema import LoopConfig
from ..risk.manager import RiskManager
from .executor import TradeExecutor


class LoopState(str, Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    ANALYZING = "ANALYZING"
    DECIDING = "DECIDING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AutonomousLoop:
    def __init__(
        self,
        executor: TradeExecutor,
        risk: RiskManager,
        config: LoopConfig,
    ) -> None:
        self._executor = executor
        self._risk = risk
        self._config = config

        self._state = LoopState.IDLE
        self._thesis: str = ""
        self._running = False
        self._task: asyncio.Task | None = None
        self._iteration = 0
        self._last_scan: datetime | None = None
        self._last_thesis_check: datetime | None = None
        self._errors: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def start(self, thesis: str) -> None:
        if self._running:
            logger.warning("Loop already running. Stop it first.")
            return
        self._thesis = thesis
        self._running = True
        self._state = LoopState.IDLE
        self._iteration = 0
        self._task = asyncio.create_task(self._run())
        logger.info(f"Autonomous loop started | thesis='{thesis[:60]}'")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._state = LoopState.STOPPED
        logger.info("Autonomous loop stopped")

    def get_status(self) -> dict[str, Any]:
        pending = self._risk.get_pending_decisions()
        return {
            "state": self._state.value,
            "running": self._running,
            "thesis": self._thesis[:100] if self._thesis else None,
            "iteration": self._iteration,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "next_scan_in_seconds": self._seconds_until_next_scan(),
            "pending_approvals": len(pending),
            "pending_decision_ids": [d.decision_id for d in pending],
            "recent_errors": self._errors[-3:],
        }

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        while self._running:
            try:
                await self._scan_iteration()
                self._iteration += 1
                self._last_scan = datetime.now()
                await asyncio.sleep(self._config.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"Loop error (iteration {self._iteration}): {e}"
                logger.error(error_msg)
                self._errors.append(error_msg)
                self._state = LoopState.ERROR
                # Back off and retry
                await asyncio.sleep(min(60, self._config.poll_interval_seconds))

    async def _scan_iteration(self) -> None:
        # ── 1. SCANNING: fetch positions and check limits ──────────────
        self._state = LoopState.SCANNING
        positions = await self._executor._data.get_positions()
        risk_status = self._risk.get_status(positions)

        if not risk_status["limits_ok"]:
            logger.warning("Daily loss limit reached — pausing loop")
            await asyncio.sleep(3600)  # pause for 1h
            return

        # ── 2. Check exit signals on existing positions ─────────────────
        exit_decisions = await self._check_exit_signals(positions)
        for dec in exit_decisions:
            dec = await self._risk.evaluate_trade(dec, positions)
            if dec.approved:
                self._state = LoopState.EXECUTING
                result = await self._executor.execute_decision(dec)
                logger.info(f"Exit executed: {result}")

        # ── 3. ANALYZING: search for new opportunities ──────────────────
        if not self._thesis:
            return

        self._state = LoopState.ANALYZING
        analysis = await self._executor.thesis_to_trades(
            self._thesis,
            max_markets=self._config.max_markets_per_scan,
        )
        logger.info(
            f"Scan #{self._iteration}: found {len(analysis.markets_found)} markets, "
            f"{len(analysis.tradeable_markets)} tradeable"
        )

        # ── 4. DECIDING / EXECUTING ──────────────────────────────────────
        self._state = LoopState.DECIDING

        # If in autonomous mode, decisions from thesis_to_trades are already approved
        # If in confirm mode, they sit in risk.pending_decisions waiting for user
        pending = self._risk.get_pending_decisions()
        if pending:
            self._state = LoopState.AWAITING_APPROVAL
        else:
            self._state = LoopState.IDLE

        # ── 5. Periodic thesis validity check ───────────────────────────
        await self._maybe_recheck_thesis(positions)

    async def _check_exit_signals(self, positions) -> list:
        """
        For each open position, check if we should exit:
        - Near resolution (price > 0.95 or < 0.05)
        - Stop-loss: position down > 30% from avg entry
        """
        from ..models.analysis import TradeDecision

        exits = []
        for pos in positions:
            should_exit = False
            reason = ""

            if pos.current_price > 0.95:
                should_exit = True
                reason = "Market near YES resolution (price > 0.95) — taking profit"
            elif pos.current_price < 0.05:
                should_exit = True
                reason = "Market near NO resolution (price < 0.05) — cutting loss"
            elif pos.unrealized_pnl_pct < -30:
                should_exit = True
                reason = f"Stop-loss triggered (down {pos.unrealized_pnl_pct:.1f}%)"

            if should_exit:
                logger.info(f"Exit signal: {reason} | {pos.market_question[:50]}")
                exits.append(
                    TradeDecision(
                        market_id=pos.market_id,
                        token_id=pos.token_id,
                        side="SELL",
                        price=pos.current_price,
                        size_usdc=pos.current_value_usdc,
                        reasoning=reason,
                    )
                )
        return exits

    async def _maybe_recheck_thesis(self, positions) -> None:
        """Re-evaluate thesis validity every thesis_refresh_hours."""
        if not self._thesis or not positions:
            return
        refresh_delta = timedelta(hours=self._config.thesis_refresh_hours)
        if (
            self._last_thesis_check is None
            or datetime.now() - self._last_thesis_check > refresh_delta
        ):
            validity = self._executor._analyst.check_thesis_validity(self._thesis, positions)
            self._last_thesis_check = datetime.now()
            if not validity.get("still_valid", True):
                logger.warning(
                    f"Thesis may be invalid: {validity.get('reasoning', '')} | "
                    f"Recommended: {validity.get('recommended_action', 'HOLD')}"
                )

    def _seconds_until_next_scan(self) -> int | None:
        if not self._last_scan or not self._running:
            return None
        elapsed = (datetime.now() - self._last_scan).total_seconds()
        remaining = self._config.poll_interval_seconds - elapsed
        return max(0, int(remaining))
