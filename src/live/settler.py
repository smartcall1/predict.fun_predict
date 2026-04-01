"""
Position Settler — Predict.fun

다중 우선순위 정산 로직 (polymarket_trader_bot 패턴 차용):
  P0: 마켓 종료 (tradingStatus=CLOSED + resolution)
  P0-V: VOID 감지 (cancelled/refund)
  P1: 자연 정산 (YES/NO winner 확정)
  P2: 익절 (AI fair value의 60% 도달)
  P2-A: Near certainty (price >= 0.95)
  P3: 손절 (ROI <= -stop_loss_pct)
  P4: 시간 만료 (max_hold_hours 초과)
"""

import time
import logging
from typing import Dict, Optional, Tuple

from src.clients.predictfun_client import PredictFunClient

logger = logging.getLogger("settler")


class PositionSettler:
    """Handles position settlement with multi-priority cascade."""

    def __init__(
        self,
        client: PredictFunClient,
        take_profit_pct: float = 0.30,
        stop_loss_pct: float = 0.15,
        max_hold_hours: int = 240,
    ):
        self.client = client
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_hours = max_hold_hours
        self._void_cycles: Dict[str, int] = {}  # market_id -> consecutive void-like cycles

    async def check_position(self, tid: str, pos: Dict) -> Optional[Dict]:
        """
        Check if a position should be settled.

        Returns:
            Settlement dict with {action, reason, exit_price, pnl} or None.
        """
        market_id = pos.get("market_id")
        side = pos.get("side", "YES").upper()
        entry_price = pos.get("entry_price", 0)
        quantity = pos.get("quantity", 0)
        size_usdc = pos.get("size_usdc", entry_price * quantity)
        opened_at = pos.get("timestamp", time.time())

        # Fetch current market data
        try:
            mkt_resp = await self.client.get_market(market_id=market_id)
            mkt = mkt_resp.get("market", mkt_resp) if isinstance(mkt_resp, dict) else {}
        except Exception as e:
            logger.debug(f"Market fetch failed for {market_id}: {e}")
            return None

        if not mkt:
            return None

        trading_status = str(mkt.get("tradingStatus", "")).upper()
        resolution = mkt.get("resolution")
        status = str(mkt.get("status", "")).upper()

        # Get current price
        current_price = entry_price
        try:
            prices = await self.client.get_best_prices(market_id)
            if prices and prices.get("mid"):
                yes_price = prices["mid"]
                current_price = yes_price if side == "YES" else (1.0 - yes_price)
        except Exception:
            pass

        # Track peak price for trailing stop
        peak_key = f"peak_{tid}"
        peak_price = pos.get("peak_price", current_price)
        if current_price > peak_price:
            peak_price = current_price
            pos["peak_price"] = peak_price

        # ── P0: Market Closed ───────────────────
        if trading_status in ("CLOSED", "HALTED") or status in ("RESOLVED", "SETTLED"):

            # P0-V: VOID / Cancelled detection
            if resolution is None:
                self._void_cycles[market_id] = self._void_cycles.get(market_id, 0) + 1
                if self._void_cycles[market_id] >= 6:  # 6 consecutive cycles
                    logger.info(f"[VOID] {market_id} — closed but no resolution after 6 cycles")
                    return {
                        "action": "VOID",
                        "reason": "market_cancelled",
                        "exit_price": entry_price,  # Refund
                        "pnl": 0.0,
                        "current_price": current_price,
                    }
                return None  # Wait more cycles

            # P1: Natural settlement
            result = str(resolution).upper()
            won = False
            if result in ("YES", "UP", "ABOVE", "TRUE") and side == "YES":
                won = True
            elif result in ("NO", "DOWN", "BELOW", "FALSE") and side == "NO":
                won = True

            if won:
                payout = quantity * 1.0
                pnl = payout - size_usdc  # H5: size_usdc 기반 (실제 투입 비용)
                logger.info(f"[WIN] {market_id} {side} → +${pnl:.2f}")
                return {
                    "action": "WIN",
                    "reason": "market_resolution",
                    "exit_price": 1.0,
                    "pnl": pnl,
                    "current_price": 1.0,
                }
            else:
                pnl = -size_usdc  # H5: size_usdc 기반 (실제 투입 비용)
                logger.info(f"[LOSS] {market_id} {side} → ${pnl:.2f}")
                return {
                    "action": "LOSS",
                    "reason": "market_resolution",
                    "exit_price": 0.0,
                    "pnl": pnl,
                    "current_price": 0.0,
                }

        # Clear void cycle counter if market still open
        self._void_cycles.pop(market_id, None)

        # ── P2: Take Profit (AI fair value의 60% 도달 시 익절) ─────
        # ai_target_price = AI 앙상블이 추정한 YES fair value (항상 저장됨)
        # 예) 진입 10.5¢, AI추정 22% → edge_target = 0.105 + 0.60*(0.22-0.105) = 0.174
        target_price = pos.get("ai_target_price")  # AI가 추정한 YES fair value
        if target_price and entry_price > 0:
            # NO side: YES확률 → NO fair value로 변환
            fair_value = target_price if side == "YES" else (1.0 - target_price)
            if fair_value > entry_price:
                edge_target = entry_price + 0.60 * (fair_value - entry_price)
            else:
                edge_target = None
        else:
            edge_target = None

        if edge_target and current_price >= edge_target:
            pnl = current_price * quantity - size_usdc
            logger.info(f"[AI_TARGET] {market_id} fair={fair_value:.2f} target_60%={edge_target:.2f} current={current_price:.2f} → +${pnl:.2f}")
            return {
                "action": "SELL",
                "reason": f"ai_target_{edge_target:.2f}",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }
        # NOTE: ROI 기반 fallback 제거됨 — AI fair value 익절만 사용
        # ai_target_price는 live_trader.py에서 position.confidence로 항상 저장됨

        # ── P2-A: Near certainty (price >= 0.95) ──
        if current_price >= 0.95:
            pnl = (current_price - entry_price) * quantity
            logger.info(f"[NEAR_CERTAIN] {market_id} price={current_price:.2f}")
            return {
                "action": "SELL",
                "reason": "near_certainty",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }

        # ── P3: Stop Loss ───────────────────────
        if entry_price > 0:
            roi = (current_price - entry_price) / entry_price
            if roi <= -self.stop_loss_pct:
                pnl = (current_price - entry_price) * quantity
                logger.info(f"[STOP_LOSS] {market_id} ROI={roi:.1%} → ${pnl:.2f}")
                return {
                    "action": "SELL",
                    "reason": f"stop_loss_{roi:.0%}",
                    "exit_price": current_price,
                    "pnl": pnl,
                    "current_price": current_price,
                }

        # ── P4: Time-based exit ─────────────────
        hours_held = (time.time() - opened_at) / 3600
        if hours_held >= self.max_hold_hours:
            pnl = (current_price - entry_price) * quantity
            logger.info(f"[TIME_EXIT] {market_id} held {hours_held:.0f}h → ${pnl:.2f}")
            return {
                "action": "SELL",
                "reason": f"time_exit_{hours_held:.0f}h",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }

        return None
