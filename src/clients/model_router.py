"""
Simplified model router — all requests go to Gemini Flash.

Replaces the original 5-model ensemble routing layer.
Maintains the same interface (get_completion, get_trading_decision, close)
so all existing code works without modification.
"""

import time
from typing import Any, Dict, Optional

from src.clients.gemini_client import GeminiClient, TradingDecision
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class ModelRouter(TradingLoggerMixin):
    """
    Simplified routing layer — all AI goes through Gemini Flash.
    Maintains the same interface as the original multi-model router.
    """

    def __init__(
        self,
        xai_client: Optional[XAIClient] = None,
        openrouter_client: Optional[OpenRouterClient] = None,
        db_manager: Any = None,
    ):
        self.db_manager = db_manager
        # All AI goes through GeminiClient (aliased as XAIClient)
        self.xai_client = xai_client or GeminiClient(db_manager=db_manager)
        self.openrouter_client = openrouter_client  # unused, kept for interface compat

        self.logger.info(
            "ModelRouter initialized (Gemini Flash single-model mode)",
            model=settings.api.gemini_model,
        )

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        strategy: Optional[str] = None,
        query_type: Optional[str] = None,
        market_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Get a text completion from Gemini Flash.
        model/capability params are accepted but ignored (single model).
        """
        if not self.xai_client._model:
            self.logger.warning("Gemini model not initialized, returning empty")
            return ""

        try:
            import google.generativeai as genai

            # Override generation config if custom params provided
            gen_config = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            if max_tokens is not None:
                gen_config["max_output_tokens"] = max_tokens

            t0 = time.time()
            if gen_config:
                response = self.xai_client._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(**gen_config),
                )
            else:
                response = self.xai_client._model.generate_content(prompt)
            elapsed = time.time() - t0

            # Cost tracking
            usage = getattr(response, "usage_metadata", None)
            in_tok = usage.prompt_token_count if usage else 500
            out_tok = usage.candidates_token_count if usage else 300
            cost = self.xai_client._estimate_cost(in_tok, out_tok)
            self.xai_client.total_cost += cost
            self.xai_client.request_count += 1
            self.xai_client.daily_tracker.total_cost += cost
            self.xai_client.daily_tracker.request_count += 1
            self.xai_client._save_daily_tracker()

            # Log to DB if available
            if self.db_manager and hasattr(self.db_manager, "log_ai_query"):
                await self.db_manager.log_ai_query(
                    model=settings.api.gemini_model,
                    query_type=query_type or "completion",
                    market_id=market_id,
                    cost=cost,
                    response_time=elapsed,
                )

            text = response.text.strip() if response.text else ""
            self.logger.debug(
                f"Completion done [{elapsed:.1f}s, ${cost:.4f}]",
                query_type=query_type,
                market_id=market_id,
            )
            return text

        except Exception as e:
            self.logger.error(f"get_completion failed: {e}")
            return ""

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
    ) -> Optional[TradingDecision]:
        """Delegate to GeminiClient."""
        return await self.xai_client.get_trading_decision(
            market_data, portfolio_data, news_summary
        )

    async def close(self) -> None:
        """Close all clients."""
        if self.xai_client:
            await self.xai_client.close()
        self.logger.info("ModelRouter closed")
