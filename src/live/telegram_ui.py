"""
Interactive Telegram UI — Predict.fun AI Bot

polymarket_trader_bot의 텔레그램 패턴 이식:
- 영구 버튼 키보드 (Status, Trades, Positions, Stop)
- 폴링 기반 커맨드 처리 (daemon thread)
- 확인 워크플로우 (Stop → "YES" 확인)
"""

import json
import time
import threading
import logging
import requests
from typing import Optional, Callable, Dict

from src.config.settings import settings

logger = logging.getLogger("telegram_ui")


# Button layout
KEYBOARD = {
    "keyboard": [
        ["📊 Status", "📋 Trades"],
        ["📌 Positions", "💰 Stats"],
        ["📄 Logs", "⏹ Stop"],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
}


class TelegramUI:
    """Interactive Telegram bot with persistent buttons and command polling."""

    def __init__(self):
        self.token = settings.api.telegram_bot_token
        self.chat_id = settings.api.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id
                            and "your_" not in (self.token or "").lower())

        self._offset = 0
        self._running = False
        self._poll_thread = None

        # Callbacks (set by live_trader)
        self._on_status: Optional[Callable] = None
        self._on_trades: Optional[Callable] = None
        self._on_positions: Optional[Callable] = None
        self._on_stats: Optional[Callable] = None
        self._on_logs: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None

        # Confirmation state
        self._pending_confirm: Optional[Dict] = None

        if not self.enabled:
            logger.warning("Telegram disabled (token/chat_id missing)")

    # ── Core Send ───────────────────────────────

    def send(self, text: str, with_keyboard: bool = True, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if with_keyboard:
            payload["reply_markup"] = json.dumps(KEYBOARD)

        try:
            chunks = [text[i:i+4090] for i in range(0, len(text), 4090)]
            for chunk in chunks:
                payload["text"] = chunk
                r = requests.post(url, json=payload, timeout=10)
                r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    # ── Notifications ───────────────────────────

    def notify_trade(self, side: str, market_title: str, price: float,
                     confidence: float, quantity: int, reasoning: str = "",
                     ai_target: float = 0, edge: float = 0):
        """거래 알림 — 폴리마켓 동기화 형식."""
        cost = quantity * price
        self.send(
            f"\U0001f916 <b>[OPEN] AI 앙상블</b>\n"
            f"\U0001f4c4 마켓: {self._esc(market_title[:80])}\n"
            f"\U0001f48e 방향: {side.upper()}\n"
            f"\U0001f4b0 체결가: ${price:.3f}\n"
            f"\U0001f4b5 투자금: ${cost:.2f}\n"
            f"\U0001f3af 확신도: {confidence:.0%}"
        )

    def notify_settlement(self, result: str, market_title: str, side: str,
                          entry: float, exit_price: float, pnl: float, reason: str):
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
            f"\U0001f4c4 마켓: {self._esc(market_title[:80])}\n"
            f"{pnl_line}"
        )

    def notify_error(self, msg: str):
        self.send(f"\U000026a0 <b>Error</b>\n<code>{self._esc(str(msg)[:400])}</code>")

    def notify_startup(self, mode: str, bankroll: float, positions: int):
        self.send(
            f"\U0001f680 <b>Bot Started ({mode})</b>\n\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"Open Positions: {positions}"
        )

    # ── Polling Loop ────────────────────────────

    def start_polling(self):
        """Start background polling thread."""
        if not self.enabled:
            return
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Telegram polling started")

    def stop_polling(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._handle_update(upd)
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            time.sleep(2)

    def _get_updates(self):
        if not self.enabled:
            return []
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            r = requests.get(url, params={
                "offset": self._offset, "timeout": 5, "limit": 10
            }, timeout=15)
            data = r.json()
            updates = data.get("result", [])
            if updates:
                self._offset = updates[-1]["update_id"] + 1
            return updates
        except Exception:
            return []

    def _handle_update(self, update: Dict):
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if chat_id != str(self.chat_id):
            return

        # Check pending confirmation
        if self._pending_confirm:
            if time.time() > self._pending_confirm.get("expires", 0):
                self._pending_confirm = None
            elif text.upper() == "YES":
                action = self._pending_confirm.get("action")
                self._pending_confirm = None
                if action == "stop" and self._on_stop:
                    self.send("\U0001f6d1 <b>Stopping bot...</b>")
                    self._on_stop()
                return
            else:
                self._pending_confirm = None
                self.send("Cancelled.")
                return

        # Route commands
        if text in ("📊 Status", "/status"):
            if self._on_status:
                self._on_status()
        elif text in ("📋 Trades", "/trades"):
            if self._on_trades:
                self._on_trades()
        elif text in ("📌 Positions", "/positions"):
            if self._on_positions:
                self._on_positions()
        elif text in ("💰 Stats", "/stats"):
            if self._on_stats:
                self._on_stats()
        elif text in ("📄 Logs", "/logs"):
            if self._on_logs:
                self._on_logs()
        elif text in ("⏹ Stop", "/stop"):
            self._pending_confirm = {"action": "stop", "expires": time.time() + 30}
            self.send(
                "\U000026a0 <b>Stop bot?</b>\n"
                "Type <b>YES</b> within 30 seconds to confirm."
            )

    # ── Callback Registration ───────────────────

    def on_status(self, fn: Callable):
        self._on_status = fn

    def on_trades(self, fn: Callable):
        self._on_trades = fn

    def on_positions(self, fn: Callable):
        self._on_positions = fn

    def on_stats(self, fn: Callable):
        self._on_stats = fn

    def on_logs(self, fn: Callable):
        self._on_logs = fn

    def on_stop(self, fn: Callable):
        self._on_stop = fn

    # ── Helper ──────────────────────────────────

    @staticmethod
    def _esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
