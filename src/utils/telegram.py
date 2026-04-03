"""
Telegram notifier for Predict.fun AI Trading Bot.

Sends trade signals, status updates, and daily summaries.
"""

import requests
import logging
from datetime import datetime
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.token = settings.api.telegram_bot_token
        self.chat_id = settings.api.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id
                            and "your_" not in (self.token or "").lower())

        if not self.enabled:
            logger.warning("[Telegram] Token/ChatID missing. Notifications disabled.")

    # ── Core send ────────────────────────────────

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            # 4096자 제한 대응
            chunks = [text[i:i+4090] for i in range(0, len(text), 4090)]
            for chunk in chunks:
                r = requests.post(url, json={
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": parse_mode,
                }, timeout=10)
                r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"[Telegram] Send failed: {e}")
            return False

    # ── Trade notifications ──────────────────────

    def notify_signal(self, market_title: str, side: str, entry_price: float,
                      confidence: float, reasoning: str, edge: float = 0,
                      ai_target: float = 0):
        """Paper trade signal notification — 폴리마켓 동기화 형식."""
        self.send(
            f"\U0001f916 <b>[OPEN] AI 앙상블</b>\n"
            f"\U0001f4c4 마켓: {self._esc(market_title[:100])}\n"
            f"\U0001f48e 방향: {side.upper()}\n"
            f"\U0001f4b0 체결가: ${entry_price:.3f}\n"
            f"\U0001f3af 확신도: {confidence:.0%}\n"
            f"\n"
            f"\U0001f4dd <i>Paper Trade</i>"
        )

    def notify_settlement(self, market_title: str, side: str, entry_price: float,
                          exit_price: float, pnl: float, result: str):
        """Trade settlement notification — 폴리마켓 동기화 형식."""
        if result == "WIN":
            header = "\U0001f4b0 <b>[WIN] 정산 완료</b>"
            pnl_line = f"\U0001f3c6 수익: ${pnl:+.2f}"
        elif result == "LOSS":
            header = "\U0001f6d1 <b>[LOSS] 정산 완료</b>"
            pnl_line = f"\U0001f4c9 손실: ${pnl:+.2f}"
        else:
            header = "\U0001f504 <b>[VOID] 정산 완료</b>"
            pnl_line = f"\U0001f4b2 PnL: ${pnl:+.2f}"
        self.send(
            f"{header}\n"
            f"\U0001f4c4 마켓: {self._esc(market_title[:100])}\n"
            f"{pnl_line}\n"
            f"\n"
            f"\U0001f4dd <i>Paper Trade</i>"
        )

    def notify_skip(self, market_title: str, reason: str):
        """Optional: notify when a market is skipped (for debugging)."""
        self.send(
            f"\U000023ed <b>Skipped</b>\n"
            f"{self._esc(market_title[:80])}\n"
            f"<i>{self._esc(reason[:150])}</i>"
        )

    # ── Status / Summary ─────────────────────────

    def notify_scan_start(self, market_count: int):
        """Scan cycle start."""
        now = datetime.utcnow().strftime("%H:%M UTC")
        self.send(
            f"\U0001f4e1 <b>Scan Started</b> ({now})\n"
            f"Markets to analyze: {market_count}"
        )

    def notify_scan_complete(self, signals: int, skipped: int, ai_cost: float):
        """Scan cycle complete."""
        self.send(
            f"\U00002705 <b>Scan Complete</b>\n"
            f"Signals: {signals} | Skipped: {skipped}\n"
            f"AI Cost: ${ai_cost:.4f}"
        )

    def notify_daily_summary(self, stats: dict):
        """Daily performance summary."""
        wr = stats.get("win_rate", 0)
        pnl = stats.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        net = stats.get("net_pnl", pnl)
        net_sign = "+" if net >= 0 else ""

        self.send(
            f"\U0001f4ca <b>Daily Summary</b>\n"
            f"{'='*25}\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"Total PnL: {pnl_sign}${pnl:.2f}\n"
            f"AI Cost: ${stats.get('total_ai_cost', 0):.4f}\n"
            f"Net PnL: {net_sign}${net:.2f}\n"
            f"Open Positions: {stats.get('open_positions', 0)}\n"
            f"{'='*25}\n"
            f"<i>Predict.fun AI Bot (Paper)</i>"
        )

    def notify_status(self, bankroll: float, open_positions: int,
                      total_signals: int, daily_cost: float):
        """Current bot status."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        self.send(
            f"\U0001f4cc <b>Bot Status</b> ({now})\n"
            f"{'='*25}\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"Open Positions: {open_positions}\n"
            f"Total Signals: {total_signals}\n"
            f"Today AI Cost: ${daily_cost:.4f}\n"
            f"Mode: PAPER\n"
            f"{'='*25}"
        )

    def notify_error(self, error_msg: str):
        """Error notification."""
        self.send(
            f"\U000026a0 <b>Error</b>\n"
            f"<code>{self._esc(str(error_msg)[:500])}</code>"
        )

    # ── Helper ───────────────────────────────────

    @staticmethod
    def _esc(text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
