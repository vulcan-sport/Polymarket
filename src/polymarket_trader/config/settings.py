import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .schema import AppConfig, AuthConfig, APIConfig, RiskConfig, LoopConfig

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _load_yaml_config() -> dict[str, Any]:
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_config() -> AppConfig:
    yaml_cfg = _load_yaml_config()

    risk_yaml = yaml_cfg.get("risk", {})
    loop_yaml = yaml_cfg.get("loop", {})
    api_yaml = yaml_cfg.get("api", {})

    auth = AuthConfig(
        private_key=os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
        wallet_address=os.environ.get("POLYMARKET_WALLET_ADDRESS", ""),
        api_key=os.environ.get("POLYMARKET_API_KEY", ""),
        api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
        chain_id=api_yaml.get("chain_id", 137),
        signature_type=api_yaml.get("signature_type", 0),
    )

    api = APIConfig(
        clob_host=api_yaml.get("clob_host", "https://clob.polymarket.com"),
        gamma_host=api_yaml.get("gamma_host", "https://gamma-api.polymarket.com"),
        data_host=api_yaml.get("data_host", "https://data-api.polymarket.com"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        pricing_engine=api_yaml.get("pricing_engine", "kelly_criterion"),
        claude_model=api_yaml.get("claude_model", "claude-opus-4-6"),
        chain_id=api_yaml.get("chain_id", 137),
        signature_type=api_yaml.get("signature_type", 0),
    )

    risk = RiskConfig(
        approval_mode=os.environ.get("APPROVAL_MODE", risk_yaml.get("approval_mode", "confirm")),
        max_position_size_usdc=float(
            os.environ.get("MAX_POSITION_SIZE_USDC", risk_yaml.get("max_position_size_usdc", 100))
        ),
        max_total_exposure_usdc=float(
            os.environ.get(
                "MAX_TOTAL_EXPOSURE_USDC", risk_yaml.get("max_total_exposure_usdc", 500)
            )
        ),
        daily_loss_limit_usdc=float(
            os.environ.get("DAILY_LOSS_LIMIT_USDC", risk_yaml.get("daily_loss_limit_usdc", 50))
        ),
        min_edge_threshold=risk_yaml.get("min_edge_threshold", 0.03),
        max_kelly_fraction=risk_yaml.get("max_kelly_fraction", 0.25),
    )

    loop = LoopConfig(
        poll_interval_seconds=loop_yaml.get("poll_interval_seconds", 60),
        max_markets_per_scan=loop_yaml.get("max_markets_per_scan", 50),
        thesis_refresh_hours=loop_yaml.get("thesis_refresh_hours", 6),
    )

    return AppConfig(auth=auth, api=api, risk=risk, loop=loop)
