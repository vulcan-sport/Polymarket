# Polymarket Autonomous Trader

Claude-powered autonomous trading system for Polymarket prediction markets.

## Features

- **Thesis → Markets**: Describe an investment thesis in natural language; Claude finds, scores, and recommends the best markets to trade
- **Autonomous Trading**: Claude monitors markets continuously and executes within configurable risk limits
- **MCP Integration**: All trading capabilities exposed as MCP tools in Claude Desktop / Claude Code
- **Risk Management**: Configurable position limits, daily loss limits, exposure caps
- **HFT Extension Point**: Abstract pricing engine interface ready for Kelly criterion, ML models, or poker-inspired strategies
- **Confirm-Before-Trade**: Safe default mode — Claude proposes, you approve

## Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv) package manager
- A Polymarket account with USDC on Polygon
- Your wallet's private key
- An Anthropic API key

## Setup

### 1. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/Polymarket
uv sync
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env with your POLYMARKET_PRIVATE_KEY, POLYMARKET_WALLET_ADDRESS, and ANTHROPIC_API_KEY
```

### 3. Generate CLOB API credentials (one-time)

```bash
uv run python scripts/setup_credentials.py
# Copy the printed API_KEY, API_SECRET, API_PASSPHRASE into .env
```

### 4. Configure risk limits

Edit `config.yaml` to set position sizes, loss limits, and approval mode.

### 5. Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "polymarket-trader": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/parthgosalia/Desktop/Polymarket",
        "run",
        "python",
        "-m",
        "polymarket_trader.mcp.server"
      ]
    }
  }
}
```

Restart Claude Desktop. The trading tools will appear automatically.

## Usage

### Search markets
> "Search for markets about the 2026 US midterm elections"

### Thesis-driven trading
> "Analyze my thesis: AI regulation will pass in the EU before end of 2026"

### Check portfolio
> "Show me my current positions and P&L"

### Autonomous loop
> "Start monitoring markets for my thesis that crypto will have a major regulatory event in 2026"

### Toggle autonomous mode
> "Set approval mode to autonomous" (trades within your risk limits without asking)

## Risk Limits (`config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `approval_mode` | `confirm` | `confirm` = approve each trade; `autonomous` = auto-execute |
| `max_position_size_usdc` | 100 | Max size per single position |
| `max_total_exposure_usdc` | 500 | Max total open exposure across all positions |
| `daily_loss_limit_usdc` | 50 | Halt trading if daily losses exceed this |
| `min_edge_threshold` | 0.03 | Min probability edge (3%) required to trade |
| `max_kelly_fraction` | 0.25 | Cap Kelly sizing at 25% of theoretical |

## HFT Extension

To plug in a custom pricing engine (e.g., ML-based or poker-inspired HFT):

1. Subclass `PricingEngine` from `src/polymarket_trader/pricing/base.py`
2. Implement `compute_signal()` and `calibrate()`
3. Register the engine name in `config.yaml` under `api.pricing_engine`

See `src/polymarket_trader/pricing/base.py` for the full interface.

## Architecture

```
MCP Server (Claude ↔ Tools)
    ↓
TradeExecutor (orchestration)
    ├── CLOBConnector   ← py-clob-client (orders, orderbook, auth)
    ├── GammaConnector  ← market search
    ├── DataConnector   ← positions, P&L
    ├── ClaudeAnalyst   ← thesis scoring, probability estimation
    ├── PricingEngine   ← Kelly/HFT signal generation
    └── RiskManager     ← limits, approval gates
```

## Running Tests

```bash
uv run pytest
```
