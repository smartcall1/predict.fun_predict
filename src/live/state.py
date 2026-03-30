"""
Live Trading State Manager — JSON persistence.

LIVE와 PAPER 상태를 완전 분리하여 관리.
polymarket_trader_bot의 state_LIVE.json 패턴 차용.
"""

import json
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
logger = logging.getLogger("state_manager")


class StateManager:
    """JSON-based state persistence for live trading. Thread-safe."""

    def __init__(self, mode: str = "LIVE"):
        self.mode = mode.upper()
        self.state_file = os.path.join(STATE_DIR, f"state_{self.mode}.json")
        self.trade_log_file = os.path.join(STATE_DIR, f"trade_history_{self.mode}.jsonl")
        self._lock = threading.Lock()

        os.makedirs(STATE_DIR, exist_ok=True)
        self.state = self._load()

    def _default_state(self) -> Dict:
        return {
            "bankroll": 0.0,
            "initial_bankroll": 0.0,
            "peak_bankroll": 0.0,
            "positions": {},       # tid -> position dict
            "stats": {
                "wins": 0,
                "losses": 0,
                "voids": 0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "total_ai_cost": 0.0,
            },
            "last_updated": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ensure all default keys exist
                    default = self._default_state()
                    for k, v in default.items():
                        if k not in data:
                            data[k] = v
                    if "stats" in data and isinstance(data["stats"], dict):
                        for sk, sv in default["stats"].items():
                            if sk not in data["stats"]:
                                data["stats"][sk] = sv
                    return data
            except Exception as e:
                # L5: 조용한 실패 대신 경고 + 손상 파일 백업
                logger.warning(f"State file corrupted, backing up: {e}")
                try:
                    backup = self.state_file + ".corrupted"
                    os.rename(self.state_file, backup)
                    logger.warning(f"Corrupted state backed up to {backup}")
                except Exception:
                    pass
        return self._default_state()

    def save(self):
        """Atomic save to JSON."""
        with self._lock:
            self.state["last_updated"] = datetime.utcnow().isoformat()
            tmp = self.state_file + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.state_file)
            except Exception as e:
                print(f"[STATE] Save failed: {e}")

    # ── Bankroll ────────────────────────────────

    @property
    def bankroll(self) -> float:
        with self._lock:
            return self.state.get("bankroll", 0.0)

    @bankroll.setter
    def bankroll(self, value: float):
        with self._lock:
            self.state["bankroll"] = round(value, 4)
            if value > self.state.get("peak_bankroll", 0):
                self.state["peak_bankroll"] = round(value, 4)

    def init_bankroll(self, amount: float):
        with self._lock:
            if self.state.get("initial_bankroll", 0) == 0:
                self.state["initial_bankroll"] = amount
            self.state["bankroll"] = round(amount, 4)
            if amount > self.state.get("peak_bankroll", 0):
                self.state["peak_bankroll"] = round(amount, 4)

    # ── Positions ───────────────────────────────

    @property
    def positions(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self.state.get("positions", {}))

    def add_position(self, tid: str, pos: Dict):
        with self._lock:
            self.state["positions"][tid] = pos
            self.state["stats"]["total_trades"] += 1

    def remove_position(self, tid: str) -> Optional[Dict]:
        with self._lock:
            return self.state["positions"].pop(tid, None)

    def get_position(self, tid: str) -> Optional[Dict]:
        with self._lock:
            return self.state["positions"].get(tid)

    @property
    def position_count(self) -> int:
        with self._lock:
            return len(self.state.get("positions", {}))

    # ── Stats ───────────────────────────────────

    @property
    def stats(self) -> Dict:
        with self._lock:
            return dict(self.state.get("stats", {}))

    def record_win(self, pnl: float):
        with self._lock:
            self.state["stats"]["wins"] += 1
            self.state["stats"]["total_pnl"] += pnl

    def record_loss(self, pnl: float):
        with self._lock:
            self.state["stats"]["losses"] += 1
            self.state["stats"]["total_pnl"] += pnl

    def record_void(self):
        with self._lock:
            self.state["stats"]["voids"] += 1

    def record_ai_cost(self, cost: float):
        self.state["stats"]["total_ai_cost"] += cost

    @property
    def win_rate(self) -> float:
        w = self.stats.get("wins", 0)
        l = self.stats.get("losses", 0)
        total = w + l
        return round(w / total * 100, 1) if total > 0 else 0.0

    # ── Trade Log (JSONL) ───────────────────────

    def log_trade(self, trade: Dict):
        """Append a trade to the JSONL log file."""
        trade["logged_at"] = datetime.utcnow().isoformat()
        try:
            with open(self.trade_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[STATE] Trade log failed: {e}")

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Read last N trades from JSONL."""
        if not os.path.exists(self.trade_log_file):
            return []
        try:
            with open(self.trade_log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            trades = []
            for line in lines[-limit:]:
                try:
                    trades.append(json.loads(line.strip()))
                except Exception:
                    continue
            return list(reversed(trades))
        except Exception:
            return []

    # ── Summary ─────────────────────────────────

    def summary(self) -> str:
        s = self.stats
        return (
            f"Bankroll: ${self.bankroll:.2f} | "
            f"Positions: {self.position_count} | "
            f"W/L/V: {s.get('wins',0)}/{s.get('losses',0)}/{s.get('voids',0)} | "
            f"WR: {self.win_rate}% | "
            f"PnL: ${s.get('total_pnl',0):+.2f} | "
            f"AI Cost: ${s.get('total_ai_cost',0):.4f}"
        )
