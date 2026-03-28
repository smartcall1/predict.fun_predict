"""
Compatibility shim — redirects KalshiClient to PredictFunClient.

All 36+ files that import KalshiClient will transparently use Predict.fun API instead.
"""

from src.clients.predictfun_client import PredictFunClient, PredictFunAPIError

# Alias for drop-in compatibility
KalshiClient = PredictFunClient
KalshiAPIError = PredictFunAPIError
