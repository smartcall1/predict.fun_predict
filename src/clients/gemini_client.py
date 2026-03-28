"""
Gemini Flash client for AI-powered trading decisions.
Drop-in replacement for XAIClient — same interface, Gemini backend.

Provides: TradingDecision, DailyUsageTracker, GeminiClient (aliased as XAIClient)
"""

import json
import time
import os
import pickle
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, date, timedelta

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


# ─── Data classes (same interface as original xai_client) ───────────────

@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str          # "buy", "sell", "hold"
    side: str            # "yes", "no"
    confidence: float    # 0.0 to 1.0
    limit_price: Optional[int] = None  # cents (0-99)
    reasoning: Optional[str] = None    # AI 분석 근거


@dataclass
class DailyUsageTracker:
    """Track daily AI usage and costs."""
    date: str
    total_cost: float = 0.0
    request_count: int = 0
    daily_limit: float = 5.0
    is_exhausted: bool = False
    last_exhausted_time: Optional[datetime] = None


# ─── Prompt ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are an expert prediction market trader with deep knowledge across sports, politics, economics, crypto, and current events.

Analyze this prediction market and decide whether to trade.

## Market Information
- **Title**: {title}
- **Category**: {category}
- **YES price**: {yes_price:.2f} (market-implied probability)
- **NO price**: {no_price:.2f}
- **Volume**: {volume}
- **Available balance**: ${balance:.2f}

## News/Context
{news_summary}

## Your Task
1. Identify the domain (sports, politics, economics, crypto, etc.)
2. Estimate the TRUE probability of the YES outcome (0.0 to 1.0)
3. Compare with market price to find edge
4. Only recommend BUY if edge > 5% AND confidence > 60%

## Decision Rules
- BUY YES if your_probability > market_yes_price + 0.05
- BUY NO if your_probability < market_yes_price - 0.05
- SKIP if edge too small or low confidence
- Set limit_price in cents (1-99) — your target entry price

## Response (JSON only, no markdown)
{{"action": "BUY" or "SKIP", "side": "YES" or "NO", "limit_price": 1-99, "confidence": 0.0-1.0, "reasoning": "Brief 2-3 sentence analysis"}}
"""


# ─── GeminiClient ───────────────────────────────────────────────────────

class GeminiClient(TradingLoggerMixin):
    """
    Gemini Flash client for AI-powered trading decisions.
    Same interface as the original XAIClient for drop-in compatibility.
    """

    def __init__(self, api_key: Optional[str] = None, db_manager=None):
        self.api_key = api_key or settings.api.gemini_api_key
        self.db_manager = db_manager
        self.model_name = settings.api.gemini_model

        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0

        # Daily usage
        self.daily_tracker = self._load_daily_tracker()
        self.usage_file = "logs/daily_ai_usage.pkl"

        # Initialize Gemini
        self._genai = None
        self._model = None
        self._init_gemini()

        self.logger.info(
            f"GeminiClient initialized (model={self.model_name}, "
            f"api_key={'present' if self.api_key else 'MISSING'})"
        )

    def _init_gemini(self):
        """Initialize Google Generative AI client."""
        if not self.api_key:
            self.logger.warning("GEMINI_API_KEY not set — dummy mode")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": settings.trading.ai_temperature,
                    "top_p": 0.95,
                    "max_output_tokens": settings.trading.ai_max_tokens,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as e:
            self.logger.error(f"Gemini init failed: {e}")

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create daily usage tracker."""
        today_str = date.today().isoformat()
        usage_file = "logs/daily_ai_usage.pkl"
        try:
            if os.path.exists(usage_file):
                with open(usage_file, "rb") as f:
                    tracker = pickle.load(f)
                    if tracker.date == today_str:
                        return tracker
        except Exception:
            pass
        return DailyUsageTracker(
            date=today_str,
            daily_limit=settings.trading.daily_ai_cost_limit,
        )

    def _save_daily_tracker(self):
        """Persist daily tracker."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, "wb") as f:
                pickle.dump(self.daily_tracker, f)
        except Exception:
            pass

    async def _check_daily_limits(self) -> bool:
        """Check if daily budget allows another request."""
        today_str = date.today().isoformat()
        if self.daily_tracker.date != today_str:
            self.daily_tracker = DailyUsageTracker(
                date=today_str,
                daily_limit=settings.trading.daily_ai_cost_limit,
            )
        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            return False
        return True

    def _estimate_cost(self, input_tokens: int = 500, output_tokens: int = 300) -> float:
        """Gemini Flash cost estimate. Very cheap: $0.10/1M input, $0.40/1M output."""
        return (input_tokens / 1_000_000) * 0.10 + (output_tokens / 1_000_000) * 0.40

    # ─── Main interface (matches XAIClient) ─────────────────────

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision from Gemini Flash.
        Same signature as XAIClient.get_trading_decision().
        """
        if not await self._check_daily_limits():
            self.logger.warning("Daily AI cost limit reached. Skipping.")
            return None

        title = market_data.get("title", "Unknown Market")
        category = market_data.get("category", "unknown")

        # Parse prices — handle both Kalshi-style (cents) and Predict.fun-style (0-1)
        yes_price = float(market_data.get("yes_price", 0.5))
        no_price = float(market_data.get("no_price", 0.5))
        if yes_price > 1:
            yes_price = yes_price / 100.0
            no_price = no_price / 100.0

        volume = market_data.get("volume", 0)
        balance = portfolio_data.get("available_balance", 0)

        prompt = ANALYSIS_PROMPT.format(
            title=title,
            category=category,
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            balance=balance,
            news_summary=news_summary[:500] if news_summary else "No additional context.",
        )

        if not self._model:
            # Dummy mode
            self.logger.info(f"[DUMMY] Analyzing: {title[:50]}...")
            return TradingDecision(action="hold", side="yes", confidence=0.0)

        try:
            import asyncio
            t0 = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(self._model.generate_content, prompt),
                timeout=30.0,
            )
            elapsed = time.time() - t0

            # Cost tracking
            usage = getattr(response, "usage_metadata", None)
            in_tok = usage.prompt_token_count if usage else 500
            out_tok = usage.candidates_token_count if usage else 300
            cost = self._estimate_cost(in_tok, out_tok)

            self.total_cost += cost
            self.request_count += 1
            self.daily_tracker.total_cost += cost
            self.daily_tracker.request_count += 1
            self._save_daily_tracker()

            # Log to DB if available
            if self.db_manager and hasattr(self.db_manager, "log_ai_query"):
                await self.db_manager.log_ai_query(
                    model=self.model_name,
                    query_type="trading_decision",
                    market_id=market_data.get("market_id") or market_data.get("ticker"),
                    cost=cost,
                    response_time=elapsed,
                )

            # Parse response
            text = response.text.strip()
            result = json.loads(text)

            action = result.get("action", "SKIP").upper()
            side = result.get("side", "YES").upper()
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            limit_price = result.get("limit_price")
            reasoning = result.get("reasoning", "")

            # Map to TradingDecision format
            if action == "BUY":
                td_action = "buy"
            elif action == "SELL":
                td_action = "sell"
            else:
                td_action = "hold"

            td_side = side.lower()  # "yes" or "no"

            self.logger.info(
                f"[Gemini] {title[:50]}... → {td_action.upper()} {td_side.upper()} "
                f"(conf={confidence:.2f}, price={limit_price}) "
                f"[{elapsed:.1f}s, ${cost:.4f}] {reasoning[:80]}"
            )

            return TradingDecision(
                action=td_action,
                side=td_side,
                confidence=confidence,
                limit_price=limit_price,
                reasoning=reasoning,
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            return None

    async def search(self, query: str, max_length: int = 200) -> str:
        """
        Stub for XAIClient.search() compatibility.
        Gemini Flash doesn't have a separate search endpoint.
        Returns a simple context string instead.
        """
        return f"Analysis based on market data for: {query[:100]}"

    async def close(self):
        """Cleanup (no persistent connections to close for Gemini)."""
        self.logger.info(
            f"GeminiClient closed. Total: {self.request_count} requests, ${self.total_cost:.4f}"
        )
