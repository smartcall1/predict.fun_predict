"""
Position Settler — Predict.fun

적응형 SL/TP 정산 로직 (polymarket AI 앙상블 패턴 이식):
  P0: 마켓 종료 (tradingStatus=CLOSED + resolution)
  P0-V: VOID 감지 (cancelled/refund)
  P1: 자연 정산 (YES/NO winner 확정)
  P2: Stop Loss — 적응형 5~10% (confidence 기반, StopLossCalculator)
  P3: Take Profit — 적응형 15~30% (confidence 기반, StopLossCalculator)
  P3-A: Near certainty (price >= 0.98)
  P4: 시간 만료 (max_hold_hours, 기본 72h)
  P5: 비상 SL 10% (SL 미설정 레거시 포지션)
"""

import time
import logging
from typing import Dict, Optional

from src.clients.predictfun_client import PredictFunClient
from src.utils.stop_loss_calculator import StopLossCalculator

logger = logging.getLogger("settler")


class PositionSettler:
    """Handles position settlement with adaptive SL/TP cascade."""

    def __init__(self, client: PredictFunClient):
        self.client = client
        self._void_cycles: Dict[str, int] = {}

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

        # Get current price (exit 기준: YES=yes_bid, NO=1-yes_ask)
        current_price = entry_price
        try:
            prices = await self.client.get_best_prices(market_id)
            if prices:
                if side == "YES":
                    current_price = prices.get("yes_bid") or prices.get("mid") or entry_price
                else:
                    yes_ask = prices.get("yes_ask")
                    current_price = (1.0 - yes_ask) if yes_ask else entry_price
        except Exception:
            pass

        # ── P0: Market Closed ─────────────���─────
        if trading_status in ("CLOSED", "HALTED") or status in ("RESOLVED", "SETTLED"):

            # P0-V: VOID / Cancelled detection
            if resolution is None:
                self._void_cycles[market_id] = self._void_cycles.get(market_id, 0) + 1
                if self._void_cycles[market_id] >= 6:
                    logger.info(f"[VOID] {market_id} — closed but no resolution after 6 cycles")
                    return {
                        "action": "VOID",
                        "reason": "market_cancelled",
                        "exit_price": entry_price,
                        "pnl": 0.0,
                        "current_price": current_price,
                    }
                return None

            # P1: Natural settlement
            result = str(resolution).upper()
            won = False
            if result in ("YES", "UP", "ABOVE", "TRUE") and side == "YES":
                won = True
            elif result in ("NO", "DOWN", "BELOW", "FALSE") and side == "NO":
                won = True

            if won:
                payout = quantity * 1.0
                pnl = payout - size_usdc
                logger.info(f"[WIN] {market_id} {side} → +${pnl:.2f}")
                return {
                    "action": "WIN",
                    "reason": "market_resolution",
                    "exit_price": 1.0,
                    "pnl": pnl,
                    "current_price": 1.0,
                }
            else:
                pnl = -size_usdc
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

        # ── P1-A: Near settlement (price >= 0.98 or <= 0.02) ──
        if current_price >= 0.98:
            pnl = (current_price - entry_price) * quantity
            logger.info(f"[NEAR_WIN] {market_id} price={current_price:.2f}")
            return {
                "action": "SELL",
                "reason": "near_certainty_win",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }
        if current_price <= 0.02:
            pnl = (current_price - entry_price) * quantity
            logger.info(f"[NEAR_LOSS] {market_id} price={current_price:.2f}")
            return {
                "action": "SELL",
                "reason": "near_certainty_loss",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }

        roi = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # ── P2: Adaptive Stop Loss (5~10%, confidence 기반) ──
        sl_price = pos.get("stop_loss_price")
        if sl_price:
            triggered = StopLossCalculator.is_stop_loss_triggered(
                position_side=side,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss_price=sl_price,
            )
            if triggered:
                pnl = (current_price - entry_price) * quantity
                logger.info(f"[STOP_LOSS] {market_id} ROI={roi:+.1%} current={current_price:.3f} SL={sl_price:.3f}")
                return {
                    "action": "SELL",
                    "reason": f"stop_loss_{roi:.0%}",
                    "exit_price": current_price,
                    "pnl": pnl,
                    "current_price": current_price,
                }

        # ── P3: Adaptive Take Profit (15~30%, confidence 기반) ──
        tp_price = pos.get("take_profit_price")
        if tp_price:
            tp_triggered = current_price >= tp_price  # YES/NO 동일 (가격 공간 이미 변환됨)
            if tp_triggered:
                pnl = (current_price - entry_price) * quantity
                logger.info(f"[TAKE_PROFIT] {market_id} ROI={roi:+.1%} current={current_price:.3f} TP={tp_price:.3f}")
                return {
                    "action": "SELL",
                    "reason": f"take_profit_{roi:.0%}",
                    "exit_price": current_price,
                    "pnl": pnl,
                    "current_price": current_price,
                }

        # ── P4: Time-based exit ─────────────────
        max_hold = pos.get("max_hold_hours", 72)
        hours_held = (time.time() - opened_at) / 3600
        if hours_held >= max_hold:
            pnl = (current_price - entry_price) * quantity
            logger.info(f"[TIME_EXIT] {market_id} held {hours_held:.0f}h >= {max_hold}h ROI={roi:+.1%}")
            return {
                "action": "SELL",
                "reason": f"time_exit_{hours_held:.0f}h",
                "exit_price": current_price,
                "pnl": pnl,
                "current_price": current_price,
            }

        # ── P5: Emergency SL 10% (SL 미설정 레거시 포지션) ──
        if not sl_price and entry_price > 0:
            emergency_sl = StopLossCalculator.calculate_simple_stop_loss(
                entry_price=entry_price, side=side, stop_loss_pct=0.10
            )
            emergency_triggered = StopLossCalculator.is_stop_loss_triggered(
                position_side=side,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss_price=emergency_sl,
            )
            if emergency_triggered:
                pnl = (current_price - entry_price) * quantity
                logger.info(f"[EMERGENCY_SL] {market_id} ROI={roi:+.1%} (no SL set)")
                return {
                    "action": "SELL",
                    "reason": f"emergency_sl_{roi:.0%}",
                    "exit_price": current_price,
                    "pnl": pnl,
                    "current_price": current_price,
                }

        return None
