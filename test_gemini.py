"""Quick Gemini SDK test — run from project root: python3 test_gemini.py"""
import asyncio
from src.clients.gemini_client import GeminiClient

async def test():
    client = GeminiClient()
    result = await client.get_trading_decision(
        market_data={'title': 'Will BTC hit 100k?', 'yes_price': 0.65, 'no_price': 0.35},
        portfolio_data={'available_balance': 1000},
        news_summary='Test'
    )
    print(f'Result: {result}')

asyncio.run(test())
