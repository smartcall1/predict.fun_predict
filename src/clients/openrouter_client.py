"""
Compatibility shim — OpenRouterClient is no longer used.

This bot uses Gemini Flash as the sole AI model.
Provides stub classes to prevent ImportError in any remaining references.
"""

# Empty pricing dict for compatibility
MODEL_PRICING = {}


class OpenRouterClient:
    """Stub — all AI calls go through GeminiClient now."""

    def __init__(self, *args, **kwargs):
        pass

    async def get_completion(self, *args, **kwargs):
        return ""

    async def close(self):
        pass
