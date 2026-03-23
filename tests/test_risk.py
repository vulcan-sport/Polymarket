"""Tests for the risk manager."""

import pytest

from polymarket_trader.config.schema import RiskConfig
from polymarket_trader.models.analysis import TradeDecision
from polymarket_trader.models.order import Position
from polymarket_trader.risk.manager import RiskManager


def make_risk_config(**kwargs) -> RiskConfig:
    defaults = dict(
        approval_mode="confirm",
        max_position_size_usdc=100.0,
        max_total_exposure_usdc=300.0,
        daily_loss_limit_usdc=50.0,
        min_edge_threshold=0.03,
        max_kelly_fraction=0.25,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def make_decision(size_usdc: float = 50.0, **kwargs) -> TradeDecision:
    return TradeDecision(
        market_id="mkt1",
        token_id="tok1",
        side="BUY",
        price=0.50,
        size_usdc=size_usdc,
        **kwargs,
    )


def make_position(cost_usdc: float = 100.0) -> Position:
    return Position(
        market_id="mkt2",
        token_id="tok2",
        outcome="YES",
        size=cost_usdc / 0.5,
        avg_price=0.5,
        current_price=0.5,
    )


class TestRiskManager:
    @pytest.fixture
    def risk(self) -> RiskManager:
        return RiskManager(make_risk_config())

    @pytest.mark.asyncio
    async def test_confirm_mode_leaves_decision_pending(self, risk: RiskManager) -> None:
        decision = await risk.evaluate_trade(make_decision())
        assert not decision.approved
        assert not decision.rejected
        assert len(risk.get_pending_decisions()) == 1

    @pytest.mark.asyncio
    async def test_autonomous_mode_auto_approves(self) -> None:
        risk = RiskManager(make_risk_config(approval_mode="autonomous"))
        decision = await risk.evaluate_trade(make_decision())
        assert decision.approved
        assert decision.approved_by == "auto"
        assert len(risk.get_pending_decisions()) == 0

    @pytest.mark.asyncio
    async def test_rejects_oversized_position(self, risk: RiskManager) -> None:
        decision = await risk.evaluate_trade(make_decision(size_usdc=200.0))
        assert decision.rejected
        assert "exceeds limit" in decision.reject_reason

    @pytest.mark.asyncio
    async def test_rejects_when_daily_limit_hit(self, risk: RiskManager) -> None:
        risk.record_trade_pnl(-50.0)  # at limit
        decision = await risk.evaluate_trade(make_decision(size_usdc=10.0))
        assert decision.rejected
        assert "daily loss" in decision.reject_reason.lower()

    @pytest.mark.asyncio
    async def test_rejects_over_exposure(self, risk: RiskManager) -> None:
        positions = [make_position(200.0), make_position(200.0)]
        decision = await risk.evaluate_trade(make_decision(size_usdc=50.0), positions)
        assert decision.rejected

    def test_approve_pending(self, risk: RiskManager) -> None:
        decision = make_decision()
        risk._pending_decisions[decision.decision_id] = decision
        approved = risk.approve_pending(decision.decision_id)
        assert approved is not None
        assert approved.approved
        assert len(risk.get_pending_decisions()) == 0

    def test_reject_pending(self, risk: RiskManager) -> None:
        decision = make_decision()
        risk._pending_decisions[decision.decision_id] = decision
        rejected = risk.reject_pending(decision.decision_id, "changed my mind")
        assert rejected is not None
        assert rejected.rejected
        assert "changed my mind" in rejected.reject_reason
        assert len(risk.get_pending_decisions()) == 0

    def test_approve_nonexistent_returns_none(self, risk: RiskManager) -> None:
        assert risk.approve_pending("nonexistent") is None

    def test_set_approval_mode(self, risk: RiskManager) -> None:
        risk.set_approval_mode("autonomous")
        status = risk.get_status()
        assert status["approval_mode"] == "autonomous"

    def test_update_limits(self, risk: RiskManager) -> None:
        risk.update_limits(max_position_usdc=200.0, daily_loss_limit_usdc=100.0)
        status = risk.get_status()
        assert status["max_position_size_usdc"] == 200.0
        assert status["daily_loss_limit"] == 100.0

    def test_get_status_includes_all_fields(self, risk: RiskManager) -> None:
        status = risk.get_status()
        required = {
            "approval_mode",
            "daily_pnl",
            "daily_loss_limit",
            "current_exposure_usdc",
            "max_exposure_usdc",
            "max_position_size_usdc",
            "pending_approvals",
            "limits_ok",
        }
        assert required.issubset(status.keys())
