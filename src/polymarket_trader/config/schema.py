from typing import Literal
from pydantic import BaseModel, Field


class AuthConfig(BaseModel):
    private_key: str
    wallet_address: str
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    chain_id: int = 137
    signature_type: int = 0  # 0=EOA, 1=Magic/email, 2=Proxy


class APIConfig(BaseModel):
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"
    anthropic_api_key: str = ""
    pricing_engine: str = "kelly_criterion"
    claude_model: str = "claude-opus-4-6"


class RiskConfig(BaseModel):
    approval_mode: Literal["confirm", "autonomous"] = "confirm"
    max_position_size_usdc: float = Field(default=100.0, gt=0)
    max_total_exposure_usdc: float = Field(default=500.0, gt=0)
    daily_loss_limit_usdc: float = Field(default=50.0, gt=0)
    min_edge_threshold: float = Field(default=0.03, ge=0, le=1)
    max_kelly_fraction: float = Field(default=0.25, gt=0, le=1)


class LoopConfig(BaseModel):
    poll_interval_seconds: int = Field(default=60, gt=0)
    max_markets_per_scan: int = Field(default=50, gt=0)
    thesis_refresh_hours: int = Field(default=6, gt=0)


class AppConfig(BaseModel):
    auth: AuthConfig
    api: APIConfig
    risk: RiskConfig
    loop: LoopConfig
