#!/usr/bin/env python3
"""
Live Trader — Predict.fun AI Prediction Bot

AI 분석 기반 자동 매매 봇. Paper trading과 완전 분리.
paper_trader.py와 동일한 ingest→decide 파이프라인을 사용하되,
실제 주문 실행 + JSON 상태 관리 + 텔레그램 인터랙티브 UI.

Usage:
    python live_trader.py                  # Paper 모드 (기본)
    python live_trader.py --live           # 실거래 모드
    python live_trader.py --live --once    # 1회 스캔 후 종료
"""

import asyncio
import argparse
import os
import sys
import time
import logging
from datetime import datetime

from src.config.settings import settings
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.clients.predictfun_client import PredictFunClient
from src.clients.gemini_client import GeminiClient
from src.utils.database import DatabaseManager
from src.jobs.ingest import run_ingestion
from src.jobs.decide import make_decision_for_market

from src.live.state import StateManager
from src.live.executor import LiveExecutor
from src.live.settler import PositionSettler
from src.live.telegram_ui import TelegramUI

logger = get_trading_logger("live_trader")


class LiveTrader:
    """AI-powered live trading bot for Predict.fun."""

    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.mode_str = "LIVE" if live_mode else "PAPER"

        # State (separated from paper_trader)
        self.state = StateManager(mode=self.mode_str)

        # Clients
        self.client = PredictFunClient()
        self.gemini = None
        self.db = None

        # Executor & Settler
        self.executor = LiveExecutor(self.client, paper_mode=not live_mode)
        self.settler = PositionSettler(self.client)

        # Telegram UI
        self.tg = TelegramUI()
        self._setup_telegram_handlers()

        # Control
        self._running = True
        self._scan_interval = settings.trading.scan_interval_seconds

        logger.info(f"LiveTrader initialized [{self.mode_str}]")

    def _setup_telegram_handlers(self):
        """Register Telegram button callbacks."""
        self.tg.on_status(self._cmd_status)
        self.tg.on_trades(self._cmd_trades)
        self.tg.on_positions(self._cmd_positions)
        self.tg.on_stats(self._cmd_stats)
        self.tg.on_stop(self._cmd_stop)

    # ── Telegram Commands ───────────────────────

    def _cmd_status(self):
        s = self.state.stats
        pos_count = self.state.position_count
        self.tg.send(
            f"📊 <b>Bot Status [{self.mode_str}]</b>\n"
            f"{'='*30}\n"
            f"💰 Bankroll: ${self.state.bankroll:.2f}\n"
            f"📌 Positions: {pos_count}/{settings.trading.max_positions}\n"
            f"📈 W/L/V: {s.get('wins',0)}/{s.get('losses',0)}/{s.get('voids',0)}\n"
            f"🎯 Win Rate: {self.state.win_rate}%\n"
            f"💵 PnL: ${s.get('total_pnl',0):+.2f}\n"
            f"🤖 AI Cost: ${s.get('total_ai_cost',0):.4f}\n"
            f"{'='*30}\n"
            f"<i>Updated: {datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )

    def _cmd_trades(self):
        trades = self.state.get_recent_trades(10)
        if not trades:
            self.tg.send("📋 No trades yet.")
            return

        lines = ["📋 <b>Recent Trades</b>\n"]
        for t in trades:
            result = t.get("result", "?")
            emoji = {"WIN": "✅", "LOSS": "❌", "VOID": "🔄", "SELL": "💰"}.get(result, "•")
            pnl = t.get("pnl", 0)
            title = t.get("market_title", "")[:35]
            lines.append(f"{emoji} {result} ${pnl:+.2f} | {title}")

        self.tg.send("\n".join(lines))

    def _cmd_positions(self):
        positions = self.state.positions
        if not positions:
            self.tg.send("📌 No open positions.")
            return

        lines = [f"📌 <b>Open Positions ({len(positions)})</b>\n"]
        for tid, pos in positions.items():
            side = pos.get("side", "?")
            entry = pos.get("entry_price", 0)
            title = pos.get("market_title", "")[:35]
            hours = (time.time() - pos.get("timestamp", time.time())) / 3600
            lines.append(f"{'🟢' if side=='YES' else '🔴'} {side} @ {entry:.2f} ({hours:.0f}h) | {title}")

        self.tg.send("\n".join(lines))

    def _cmd_stats(self):
        s = self.state.stats
        total = s.get("wins", 0) + s.get("losses", 0)
        self.tg.send(
            f"💰 <b>Performance Stats [{self.mode_str}]</b>\n"
            f"{'='*30}\n"
            f"Total Trades: {total}\n"
            f"Wins: {s.get('wins',0)} | Losses: {s.get('losses',0)} | Voids: {s.get('voids',0)}\n"
            f"Win Rate: {self.state.win_rate}%\n"
            f"Total PnL: ${s.get('total_pnl',0):+.2f}\n"
            f"Peak Bankroll: ${self.state.state.get('peak_bankroll',0):.2f}\n"
            f"AI Cost: ${s.get('total_ai_cost',0):.4f}\n"
            f"Net PnL: ${s.get('total_pnl',0) - s.get('total_ai_cost',0):+.2f}\n"
            f"{'='*30}"
        )

    def _cmd_stop(self):
        """Graceful stop via Telegram."""
        logger.info("Stop requested via Telegram")
        self._running = False

    # ── Main Loop ───────────────────────────────

    async def run(self, once: bool = False):
        """Main trading loop."""
        setup_logging()

        # Initialize
        self.db = DatabaseManager()
        await self.db.initialize()
        self.gemini = GeminiClient(db_manager=self.db)

        # Sync bankroll
        balance = await self.executor.get_balance()
        if self.state.bankroll == 0:
            self.state.init_bankroll(balance or settings.trading.initial_bankroll)
        logger.info(f"Bankroll: ${self.state.bankroll:.2f}")

        # Start Telegram polling
        self.tg.start_polling()
        self.tg.notify_startup(self.mode_str, self.state.bankroll, self.state.position_count)

        try:
            while self._running:
                cycle_start = time.time()

                # Step 1: Settle existing positions
                await self._settle_positions()

                # Step 2: Scan for new trades
                await self._scan_and_trade()

                # Step 3: Save state
                self.state.save()

                if once:
                    break

                # Sleep until next scan
                elapsed = time.time() - cycle_start
                sleep_time = max(10, self._scan_interval - elapsed)
                logger.info(f"Cycle done in {elapsed:.0f}s. Sleeping {sleep_time:.0f}s...")
                await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.tg.notify_error(str(e))
        finally:
            self.state.save()
            self.tg.stop_polling()
            await self.client.close()
            await self.gemini.close()
            logger.info("LiveTrader stopped.")

    # ── Scan & Trade ────────────────────────────

    async def _scan_and_trade(self):
        """Ingest markets, run AI decisions, execute trades."""
        logger.info("Scanning markets...")

        # Check position limit
        if self.state.position_count >= settings.trading.max_positions:
            logger.info(f"Position limit ({settings.trading.max_positions}) reached.")
            return

        # Ingest
        try:
            queue = asyncio.Queue()
            await run_ingestion(self.db, queue)

            markets = []
            while not queue.empty():
                markets.append(queue.get_nowait())
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return

        if not markets:
            logger.info("No eligible markets.")
            return

        logger.info(f"{len(markets)} markets to analyze")

        trades_made = 0
        for i, market in enumerate(markets):
            if not self._running:
                break
            if self.state.position_count >= settings.trading.max_positions:
                break

            try:
                # Skip if already have position in this market
                existing_tids = [p.get("market_id") for p in self.state.positions.values()]
                if market.market_id in existing_tids:
                    continue

                position = await make_decision_for_market(
                    market=market,
                    db_manager=self.db,
                    xai_client=self.gemini,
                    kalshi_client=self.client,
                )

                if position is None:
                    continue

                if (position.confidence or 0) < settings.trading.min_confidence_to_trade:
                    continue

                # Execute order
                result = await self.executor.execute_buy(
                    market_id=market.market_id,
                    side=position.side,
                    price=position.entry_price,
                    quantity=position.quantity,
                    market_title=market.title,
                )

                if result and result.get("status") in ("filled", "matched"):
                    tid = result["order_id"]
                    pos_data = {
                        "market_id": market.market_id,
                        "market_title": market.title,
                        "side": position.side,
                        "entry_price": result.get("entry_price", position.entry_price),
                        "quantity": position.quantity,
                        "size_usdc": result.get("trade_value", position.quantity * position.entry_price),
                        "confidence": position.confidence,
                        "rationale": position.rationale or "",
                        "strategy": position.strategy or "directional",
                        "timestamp": time.time(),
                    }
                    self.state.add_position(tid, pos_data)
                    self.state.bankroll -= pos_data["size_usdc"]
                    # H4 fix: Gemini의 마지막 분석 비용만 기록 (누적값 X)
                    last_cost = getattr(self.gemini, '_last_decision_cost', 0.0)
                    if last_cost > 0:
                        self.state.record_ai_cost(last_cost)

                    self.tg.notify_trade(
                        side=position.side,
                        market_title=market.title,
                        price=result.get("entry_price", position.entry_price),
                        confidence=position.confidence,
                        quantity=position.quantity,
                        reasoning=(position.rationale or "")[:150],
                    )
                    trades_made += 1
                    logger.info(f"Trade #{trades_made}: BUY {position.side} {market.title[:40]}")

            except Exception as e:
                logger.error(f"Trade error for {market.market_id}: {e}")

        logger.info(f"Scan done: {trades_made} trades from {len(markets)} markets")

    # ── Settlement ──────────────────────────────

    async def _settle_positions(self):
        """Check all open positions for settlement."""
        positions = dict(self.state.positions)  # snapshot
        if not positions:
            return

        logger.info(f"Checking {len(positions)} positions for settlement...")
        settled = 0

        for tid, pos in positions.items():
            try:
                result = await self.settler.check_position(tid, pos)
                if result is None:
                    continue

                action = result["action"]
                pnl = result.get("pnl", 0)
                reason = result.get("reason", "unknown")
                exit_price = result.get("exit_price", 0)

                # Execute sell if needed
                if action == "SELL":
                    sell_result = await self.executor.execute_sell(
                        market_id=pos["market_id"],
                        side=pos["side"],
                        price=exit_price,
                        quantity=pos["quantity"],
                        reason=reason,
                    )
                    if not sell_result:
                        continue

                # Update state — bankroll 회계 (C4 fix)
                # 매수 시 size_usdc를 차감했으므로, 정산 시 회수액을 더함
                size_usdc = pos.get("size_usdc", 0)
                quantity = pos.get("quantity", 0)

                if action == "WIN":
                    # 마켓 정산 승리: payout = quantity * $1.0
                    payout = quantity * 1.0
                    self.state.record_win(pnl)
                    self.state.bankroll += payout
                elif action == "LOSS":
                    # 마켓 정산 패배: payout = 0 (이미 차감됨)
                    self.state.record_loss(pnl)
                    # bankroll 변동 없음 — 원금은 이미 매수 시 차감됨
                elif action == "VOID":
                    # 취소/환불: 원금 반환
                    self.state.record_void()
                    self.state.bankroll += size_usdc
                elif action == "SELL":
                    # 조기 청산: exit_price * quantity 회수
                    proceeds = exit_price * quantity
                    if pnl >= 0:
                        self.state.record_win(pnl)
                    else:
                        self.state.record_loss(pnl)
                    self.state.bankroll += proceeds

                # Remove position
                self.state.remove_position(tid)

                # Log trade
                self.state.log_trade({
                    "market_id": pos["market_id"],
                    "market_title": pos.get("market_title", ""),
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "quantity": pos["quantity"],
                    "pnl": pnl,
                    "result": action,
                    "reason": reason,
                })

                # Telegram
                self.tg.notify_settlement(
                    result=action,
                    market_title=pos.get("market_title", ""),
                    side=pos["side"],
                    entry=pos["entry_price"],
                    exit_price=exit_price,
                    pnl=pnl,
                    reason=reason,
                )

                settled += 1
                self.state.save()

            except Exception as e:
                logger.error(f"Settlement error for {tid}: {e}")

        if settled:
            logger.info(f"Settled {settled} positions")


# ── CLI ─────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Live Trader — Predict.fun AI Bot")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (real money)")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle then exit")
    args = parser.parse_args()

    if args.live:
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("   Real money will be used!")
        print("   Press Ctrl+C within 5 seconds to cancel...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
        print("🚀 Starting LIVE trading...")

    trader = LiveTrader(live_mode=args.live)
    await trader.run(once=args.once)


if __name__ == "__main__":
    asyncio.run(main())
