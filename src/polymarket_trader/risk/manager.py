"""
Central risk management gate.

All trade decisions must pass through RiskManager before execution.

In 'confirm' mode:  decisions are returned unapproved; MCP server
                    surfaces them to the user for manual approval.
In 'autonomous' mode: decisions are auto-approved if within all limits.
"""

from datetime import date, datetime
from typing import Literal

from loguru import logger

from ..models.analysis import TradeDecision
from ..models.order import Position
from ..config.schema import RiskConfig


class RiskLimitError(Exception):
    """Raised when a trade exceeds a risk limit."""


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self._cfg = config
        self._daily_realized_pnl: float = 0.0
        self._pnl_date: date = datetime.now().date()
        self._pending_decisions: dict[str, TradeDecision] = {}  # decision_id → decision

    # ------------------------------------------------------------------ #
    # Main gate                                                            #
    # ------------------------------------------------------------------ #

    async def evaluate_trade(
        self,
        decision: TradeDecision,
        current_positions: list[Position] | None = None,
    ) -> TradeDecision:
        """
        Returns the decision with approved=True/False set.

        In confirm mode: always returns approved=False (pending user action).
        In autonomous mode: approves if all limits pass, else sets rejected=True.
        """
        self._reset_daily_pnl_if_needed()
        current_positions = current_positions or []

        # Always check hard limits
        ok, reason = self._check_all_limits(decision, current_positions)
        if not ok:
            decision.rejected = True
            decision.reject_reason = reason
            logger.warning(f"Trade rejected by risk: {reason}")
            return decision

        if self._cfg.approval_mode == "autonomous":
            decision.approved = True
            decision.approved_by = "auto"
            logger.info(
                f"Trade auto-approved: {decision.side} ${decision.size_usdc:.2f} "
                f"market={decision.market_id[:12]}"
            )
        else:
            # Confirm mode: store as pending, surface via MCP
            self._pending_decisions[decision.decision_id] = decision
            logger.info(
                f"Trade pending approval [{decision.decision_id}]: "
                f"{decision.side} ${decision.size_usdc:.2f} market={decision.market_id[:12]}"
            )

        return decision

    def approve_pending(self, decision_id: str) -> TradeDecision | None:
        """User approves a pending decision."""
        decision = self._pending_decisions.get(decision_id)
        if not decision:
            return None
        decision.approved = True
        decision.approved_by = "user"
        self._pending_decisions.pop(decision_id, None)
        logger.info(f"Trade approved by user: {decision_id}")
        return decision

    def reject_pending(self, decision_id: str, reason: str = "") -> TradeDecision | None:
        """User rejects a pending decision."""
        decision = self._pending_decisions.get(decision_id)
        if not decision:
            return None
        decision.rejected = True
        decision.reject_reason = reason or "Rejected by user"
        self._pending_decisions.pop(decision_id, None)
        logger.info(f"Trade rejected by user: {decision_id} | {reason}")
        return decision

    def get_pending_decisions(self) -> list[TradeDecision]:
        return list(self._pending_decisions.values())

    # ------------------------------------------------------------------ #
    # Limit checks                                                         #
    # ------------------------------------------------------------------ #

    def _check_all_limits(
        self,
        decision: TradeDecision,
        current_positions: list[Position],
    ) -> tuple[bool, str]:
        checks = [
            self._check_position_limit(decision.size_usdc),
            self._check_daily_loss_limit(),
            self._check_exposure_limit(current_positions, decision.size_usdc),
        ]
        for ok, reason in checks:
            if not ok:
                return False, reason
        return True, ""

    def _check_position_limit(self, size_usdc: float) -> tuple[bool, str]:
        if size_usdc > self._cfg.max_position_size_usdc:
            return (
                False,
                f"Position ${size_usdc:.2f} exceeds limit ${self._cfg.max_position_size_usdc:.2f}",
            )
        return True, ""

    def _check_daily_loss_limit(self) -> tuple[bool, str]:
        if -self._daily_realized_pnl >= self._cfg.daily_loss_limit_usdc:
            return (
                False,
                f"Daily loss limit ${self._cfg.daily_loss_limit_usdc:.2f} reached",
            )
        return True, ""

    def _check_exposure_limit(
        self,
        current_positions: list[Position],
        new_size_usdc: float,
    ) -> tuple[bool, str]:
        current_exposure = sum(p.cost_basis_usdc for p in current_positions)
        if current_exposure + new_size_usdc > self._cfg.max_total_exposure_usdc:
            return (
                False,
                f"Total exposure ${current_exposure + new_size_usdc:.2f} "
                f"exceeds limit ${self._cfg.max_total_exposure_usdc:.2f}",
            )
        return True, ""

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def record_trade_pnl(self, pnl: float) -> None:
        """Call after a trade is executed to track daily P&L."""
        self._reset_daily_pnl_if_needed()
        self._daily_realized_pnl += pnl

    def _reset_daily_pnl_if_needed(self) -> None:
        today = datetime.now().date()
        if today != self._pnl_date:
            self._daily_realized_pnl = 0.0
            self._pnl_date = today

    def set_approval_mode(self, mode: Literal["confirm", "autonomous"]) -> None:
        self._cfg.approval_mode = mode
        logger.info(f"Approval mode set to: {mode}")

    def update_limits(
        self,
        max_position_usdc: float | None = None,
        daily_loss_limit_usdc: float | None = None,
        max_exposure_usdc: float | None = None,
        min_edge: float | None = None,
    ) -> None:
        if max_position_usdc is not None:
            self._cfg.max_position_size_usdc = max_position_usdc
        if daily_loss_limit_usdc is not None:
            self._cfg.daily_loss_limit_usdc = daily_loss_limit_usdc
        if max_exposure_usdc is not None:
            self._cfg.max_total_exposure_usdc = max_exposure_usdc
        if min_edge is not None:
            self._cfg.min_edge_threshold = min_edge
        logger.info("Risk limits updated")

    def get_status(self, current_positions: list[Position] | None = None) -> dict:
        self._reset_daily_pnl_if_needed()
        current_positions = current_positions or []
        current_exposure = sum(p.cost_basis_usdc for p in current_positions)
        return {
            "approval_mode": self._cfg.approval_mode,
            "daily_pnl": round(self._daily_realized_pnl, 4),
            "daily_loss_limit": self._cfg.daily_loss_limit_usdc,
            "daily_loss_remaining": round(
                self._cfg.daily_loss_limit_usdc + self._daily_realized_pnl, 4
            ),
            "current_exposure_usdc": round(current_exposure, 4),
            "max_exposure_usdc": self._cfg.max_total_exposure_usdc,
            "max_position_size_usdc": self._cfg.max_position_size_usdc,
            "min_edge_threshold": self._cfg.min_edge_threshold,
            "pending_approvals": len(self._pending_decisions),
            "limits_ok": -self._daily_realized_pnl < self._cfg.daily_loss_limit_usdc,
        }
