"""
Live Order Executor — Predict.fun

Predict.fun API를 통한 실제 주문 실행.
Paper 모드 시뮬레이션도 포함 (realistic costs).
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any

from src.clients.predictfun_client import PredictFunClient
from src.config.settings import settings

logger = logging.getLogger("live_executor")


class LiveExecutor:
    """Handles live order execution on Predict.fun."""

    def __init__(self, client: PredictFunClient, paper_mode: bool = True):
        self.client = client
        self.paper_mode = paper_mode
        self._pending_orders = {}

    async def execute_buy(
        self,
        market_id: str,
        side: str,
        price: float,
        quantity: int,
        market_title: str = "",
    ) -> Optional[Dict]:
        """
        Execute a BUY order.

        Args:
            market_id: Predict.fun market ID
            side: "YES" or "NO"
            price: Target price (0-1 scale)
            quantity: Number of contracts
            market_title: For logging

        Returns:
            Order result dict or None on failure.
        """
        logger.info(f"{'[PAPER]' if self.paper_mode else '[LIVE]'} "
                     f"BUY {quantity}x {side} @ {price:.4f} | {market_title[:40]}")

        try:
            # Price to cents for API
            yes_price = int(round(price * 100)) if side.upper() == "YES" else None
            no_price = int(round(price * 100)) if side.upper() == "NO" else None

            result = await self.client.place_order(
                ticker=market_id,
                client_order_id=f"ai_{time.time_ns()}_{market_id[:8]}",
                side=side.lower(),
                action="buy",
                count=quantity,
                type_="market",
                yes_price=yes_price,
                no_price=no_price,
            )

            order = result.get("order", {})
            fill_price = order.get("price", price)
            status = order.get("status", "unknown")

            logger.info(f"Order result: status={status}, fill={fill_price:.4f}")

            return {
                "order_id": order.get("order_id", f"ai_{int(time.time())}"),
                "market_id": market_id,
                "side": side,
                "quantity": quantity,
                "entry_price": fill_price,
                "intended_price": price,
                "status": status,
                "paper": self.paper_mode,
                "trade_value": order.get("trade_value", quantity * fill_price),
                "fee": order.get("fee", 0),
                "gas_fee": order.get("gas_fee", 0),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    async def execute_sell(
        self,
        market_id: str,
        side: str,
        price: float,
        quantity: int,
        reason: str = "manual",
    ) -> Optional[Dict]:
        """Execute a SELL order to close position."""
        logger.info(f"{'[PAPER]' if self.paper_mode else '[LIVE]'} "
                     f"SELL {quantity}x {side} @ {price:.4f} ({reason})")

        try:
            yes_price = int(round(price * 100)) if side.upper() == "YES" else None
            no_price = int(round(price * 100)) if side.upper() == "NO" else None

            result = await self.client.place_order(
                ticker=market_id,
                client_order_id=f"exit_{int(time.time())}_{market_id}",
                side=side.lower(),
                action="sell",
                count=quantity,
                type_="market",
                yes_price=yes_price,
                no_price=no_price,
            )

            order = result.get("order", {})
            return {
                "order_id": order.get("order_id"),
                "market_id": market_id,
                "side": side,
                "quantity": quantity,
                "exit_price": order.get("price", price),
                "status": order.get("status", "unknown"),
                "reason": reason,
                "paper": self.paper_mode,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
            return None

    async def get_current_price(self, market_id: str) -> Optional[Dict]:
        """Get current YES/NO prices from orderbook."""
        try:
            prices = await self.client.get_best_prices(market_id)
            if prices:
                return {
                    "yes_price": prices.get("mid") or prices.get("yes_ask"),
                    "no_price": 1.0 - (prices.get("mid") or prices.get("yes_ask") or 0.5),
                    "spread": prices.get("spread"),
                }
        except Exception:
            pass

        # Fallback: market stats
        try:
            stats = await self.client.get_market_stats(market_id)
            if stats:
                lp = float(stats.get("lastPrice", 0.5))
                return {"yes_price": lp, "no_price": 1.0 - lp, "spread": None}
        except Exception:
            pass

        return None

    async def get_balance(self) -> float:
        """Get available balance."""
        try:
            resp = await self.client.get_balance()
            return float(resp.get("balance", 0))
        except Exception:
            return 0.0
