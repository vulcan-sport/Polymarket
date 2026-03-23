"""
Claude-powered analyst for Polymarket.

Responsibilities:
1. Extract search keywords from an investment thesis
2. Score and rank candidate markets by relevance + edge
3. Estimate true probabilities for binary outcomes
4. Explain trade rationale in plain English
5. Periodically re-validate that a thesis still holds
"""

import json
import re
from datetime import datetime

import anthropic
from loguru import logger

from ..models.analysis import MarketScore, ThesisAnalysis, TradeDecision
from ..models.market import Market

_SYSTEM_PROMPT = """You are an expert prediction market analyst specializing in Polymarket.
You analyze investment theses and identify binary markets worth trading.

For each market you evaluate:
- Relevance to the user's thesis (0–10)
- Your probability estimate for the YES outcome (0.0–1.0)
- The edge vs. current market price (positive = market is underpricing YES)
- A recommended position: "YES", "NO", or "SKIP"

Be calibrated. Acknowledge uncertainty. Never overfit to recent news.
Use base rates and outside-view reasoning where possible.
Return well-formed JSON as instructed — no markdown fences, no commentary."""


class ClaudeAnalyst:
    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    # ------------------------------------------------------------------ #
    # Thesis → Markets                                                     #
    # ------------------------------------------------------------------ #

    def extract_keywords(self, thesis: str) -> list[str]:
        """
        Extract search keywords from a thesis using a fast heuristic.
        For short theses this avoids an extra API call.
        """
        # Strip stop words and short tokens
        stop = {
            "i", "a", "an", "the", "will", "would", "that", "this", "is", "are",
            "be", "my", "in", "on", "to", "of", "and", "or", "for", "by", "with",
            "before", "after", "it", "its", "their", "there", "they", "we", "us",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", thesis.lower())
        keywords = [w for w in words if w not in stop]
        # Deduplicate and take top 6
        seen: set[str] = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:6]

    async def analyze_thesis(
        self,
        thesis: str,
        candidate_markets: list[Market],
        max_markets: int = 10,
    ) -> ThesisAnalysis:
        """
        Score and rank candidate_markets by relevance to the thesis.
        Returns ThesisAnalysis with MarketScore list sorted by composite_score.
        """
        if not candidate_markets:
            return ThesisAnalysis(
                thesis=thesis,
                markets_found=[],
                summary="No candidate markets found for this thesis.",
            )

        # Trim to top-N by volume to keep prompt small
        sorted_candidates = sorted(candidate_markets, key=lambda m: m.volume, reverse=True)[
            :max_markets
        ]

        market_descriptions = []
        for i, m in enumerate(sorted_candidates):
            market_descriptions.append(
                f"{i + 1}. condition_id={m.condition_id}\n"
                f"   Question: {m.question}\n"
                f"   YES price: {m.yes_price or 'N/A'}, NO price: {m.no_price or 'N/A'}\n"
                f"   Volume: ${m.volume:,.0f}, Liquidity: ${m.liquidity:,.0f}\n"
                f"   Ends: {m.end_date.strftime('%Y-%m-%d') if m.end_date else 'unknown'}"
            )

        prompt = f"""Thesis: "{thesis}"

Candidate markets:
{chr(10).join(market_descriptions)}

For each market, return a JSON array where each element has:
{{
  "condition_id": "<id>",
  "relevance_score": <0.0-1.0>,
  "estimated_probability": <0.0-1.0>,
  "edge_score": <0.0-1.0>,
  "liquidity_score": <0.0-1.0>,
  "composite_score": <0.0-1.0>,
  "recommended_side": "<YES|NO|SKIP>",
  "recommended_price": <float or null>,
  "reasoning": "<2-3 sentences>"
}}

Composite score = 0.4*relevance + 0.3*edge + 0.2*liquidity + 0.1*(1-time_until_expiry_weight)
Return only the JSON array, no other text."""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            scores_raw = json.loads(raw_text)
        except Exception as e:
            logger.error(f"ClaudeAnalyst.analyze_thesis error: {e}")
            scores_raw = []

        market_by_id = {m.condition_id: m for m in sorted_candidates}
        market_scores = []
        for s in scores_raw:
            cid = s.get("condition_id", "")
            market = market_by_id.get(cid)
            if not market:
                continue
            recommended_side = s.get("recommended_side", "SKIP")
            market_scores.append(
                MarketScore(
                    market=market,
                    relevance_score=float(s.get("relevance_score", 0)),
                    edge_score=float(s.get("edge_score", 0)),
                    liquidity_score=float(s.get("liquidity_score", 0)),
                    composite_score=float(s.get("composite_score", 0)),
                    claude_reasoning=s.get("reasoning", ""),
                    recommended_side=recommended_side if recommended_side != "SKIP" else None,
                    recommended_price=s.get("recommended_price"),
                    estimated_probability=s.get("estimated_probability"),
                )
            )

        market_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Generate a plain-English summary
        summary = self._build_summary(thesis, market_scores)

        return ThesisAnalysis(
            thesis=thesis,
            markets_found=market_scores,
            summary=summary,
            created_at=datetime.now(),
        )

    # ------------------------------------------------------------------ #
    # Probability estimation                                               #
    # ------------------------------------------------------------------ #

    def estimate_probability(
        self,
        market: Market,
        context: str = "",
    ) -> dict:
        """
        Returns Claude's probability estimate for the YES outcome.
        """
        prompt = f"""Market: "{market.question}"
Current YES price: {market.yes_price or 'unknown'}
Current NO price: {market.no_price or 'unknown'}
Volume: ${market.volume:,.0f}
{f"Additional context: {context}" if context else ""}

Estimate the true probability of the YES outcome resolving.
Return JSON only:
{{
  "yes_probability": <0.0-1.0>,
  "no_probability": <0.0-1.0>,
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
}}"""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"ClaudeAnalyst.estimate_probability error: {e}")
            mid = market.yes_price or 0.5
            return {
                "yes_probability": mid,
                "no_probability": 1.0 - mid,
                "confidence": 0.3,
                "reasoning": "Could not obtain Claude estimate; using market price as proxy.",
                "key_factors": [],
            }

    # ------------------------------------------------------------------ #
    # Trade explanation                                                    #
    # ------------------------------------------------------------------ #

    def explain_trade(self, decision: TradeDecision, market: Market) -> str:
        prompt = f"""Market: "{market.question}"
Proposed trade: {decision.side} {decision.size_shares} shares at ${decision.price:.3f}
Estimated edge: {decision.edge:.1%}
Confidence: {decision.confidence:.0%}
Kelly fraction: {decision.kelly_fraction:.1%}
Reasoning: {decision.reasoning}

Write 2-3 sentences explaining why this trade makes sense, the key risk factors,
and what needs to happen for it to be profitable. Be direct and specific."""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"ClaudeAnalyst.explain_trade error: {e}")
            return decision.reasoning

    # ------------------------------------------------------------------ #
    # Thesis validity                                                      #
    # ------------------------------------------------------------------ #

    def check_thesis_validity(
        self,
        thesis: str,
        current_positions: list,
    ) -> dict:
        """
        Re-evaluates whether the original thesis still holds.
        Called every thesis_refresh_hours by the autonomous loop.
        """
        positions_summary = "\n".join(
            f"- {p.market_question or p.token_id}: {p.size:.2f} shares @ {p.avg_price:.3f}"
            for p in current_positions[:10]
        )

        prompt = f"""Original thesis: "{thesis}"

Current open positions related to this thesis:
{positions_summary or "None"}

Today's date: {datetime.now().strftime("%Y-%m-%d")}

Does the thesis still hold? Return JSON:
{{
  "still_valid": <true|false>,
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences>",
  "recommended_action": "<HOLD|EXIT|REDUCE|INCREASE>",
  "key_changes": ["<change1>", "<change2>"]
}}"""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"ClaudeAnalyst.check_thesis_validity error: {e}")
            return {
                "still_valid": True,
                "confidence": 0.5,
                "reasoning": "Could not re-evaluate thesis.",
                "recommended_action": "HOLD",
                "key_changes": [],
            }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary(thesis: str, scores: list[MarketScore]) -> str:
        tradeable = [s for s in scores if s.recommended_side is not None]
        if not tradeable:
            return f"No tradeable markets found for thesis: '{thesis}'"

        top = tradeable[0]
        lines = [
            f"Found {len(scores)} relevant markets for thesis: '{thesis}'",
            f"Top recommendation: '{top.market.question}' → "
            f"{top.recommended_side} at ~{top.recommended_price or top.market.yes_price:.3f} "
            f"(composite score: {top.composite_score:.2f})",
        ]
        if len(tradeable) > 1:
            lines.append(f"{len(tradeable)} markets have tradeable signals.")
        return " ".join(lines)
