"""
Compatibility shim — redirects XAIClient to GeminiClient.

All files that import XAIClient, TradingDecision, DailyUsageTracker
will transparently use Gemini Flash instead of xAI/Grok.
"""

from src.clients.gemini_client import (
    GeminiClient,
    TradingDecision,
    DailyUsageTracker,
)

# Alias for drop-in compatibility
XAIClient = GeminiClient
