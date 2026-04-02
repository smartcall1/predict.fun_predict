"""
Paper Trading Signal Tracker

Logs hypothetical trades to SQLite and checks outcomes when markets settle.
No real money is ever risked.
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


DB_PATH = os.environ.get(
    "PAPER_TRADING_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_trades.db"),
)


@dataclass
class Signal:
    """A single paper-trading signal."""
    id: Optional[int]
    timestamp: str          # ISO-8601
    market_id: str
    market_title: str
    side: str               # YES / NO
    entry_price: float      # 0-1 scale (e.g. 0.85 = 85¢)
    confidence: float       # model confidence 0-1
    reasoning: str
    strategy: str           # e.g. directional, market_making
    # Outcome fields (filled after settlement)
    outcome: Optional[str]  # win / loss / pending
    settlement_price: Optional[float]
    pnl: Optional[float]    # per-contract P&L in dollars
    settled_at: Optional[str]


def _ensure_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            market_id       TEXT NOT NULL,
            market_title    TEXT NOT NULL,
            side            TEXT NOT NULL DEFAULT 'NO',
            entry_price     REAL NOT NULL,
            confidence      REAL,
            reasoning       TEXT,
            strategy        TEXT,
            outcome         TEXT DEFAULT 'pending',
            settlement_price REAL,
            pnl             REAL,
            settled_at      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_market
        ON signals(market_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_outcome
        ON signals(outcome)
    """)
    conn.commit()


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_db(conn)
    return conn


def has_pending_signal(market_id: str, side: str) -> bool:
    """같은 market_id + side에 pending 시그널이 이미 있는지 확인."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE market_id=? AND side=? AND outcome='pending'",
            (market_id, side),
        ).fetchone()
        return row[0] > 0
    finally:
        conn.close()


def log_signal(
    market_id: str,
    market_title: str,
    side: str,
    entry_price: float,
    confidence: float,
    reasoning: str,
    strategy: str = "directional",
) -> int:
    """Record a new paper-trading signal. Returns the signal id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO signals
               (timestamp, market_id, market_title, side, entry_price, confidence, reasoning, strategy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                market_id,
                market_title,
                side,
                entry_price,
                confidence,
                reasoning,
                strategy,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def settle_signal(signal_id: int, settlement_price: float):
    """
    Mark a signal as settled.
    PnL = (1 - entry_price) if win, else -entry_price.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        if not row:
            return

        side = row["side"]
        entry = row["entry_price"]

        if side.upper() == "NO":
            if settlement_price < 0.5:  # M2: 경계값 명확화 (< not <=)
                pnl = 1.0 - entry
                outcome = "win"
            else:
                pnl = -entry
                outcome = "loss"
        else:
            if settlement_price > 0.5:  # M2: 경계값 명확화 (> not >=)
                pnl = 1.0 - entry
                outcome = "win"
            else:
                pnl = -entry
                outcome = "loss"

        conn.execute(
            """UPDATE signals
               SET outcome = ?, settlement_price = ?, pnl = ?, settled_at = ?
               WHERE id = ?""",
            (outcome, settlement_price, round(pnl, 4), datetime.now(timezone.utc).isoformat(), signal_id),
        )
        conn.commit()
    finally:
        conn.close()


def take_profit_signal(signal_id: int, exit_price: float):
    """
    익절로 시그널 정산. 마켓 종료 전 AI 목표가 도달 시 사용.
    PnL = exit_price - entry_price (YES) 또는 entry_price - exit_price (NO 방향은 1-exit)
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        if not row:
            return

        side = row["side"]
        entry = row["entry_price"]

        if side.upper() == "YES":
            pnl = exit_price - entry
        else:
            # NO side: entry는 NO 가격, exit도 NO 가격 기준
            pnl = exit_price - entry

        outcome = "win" if pnl > 0 else "loss"

        conn.execute(
            """UPDATE signals SET outcome=?, settlement_price=?, pnl=?, settled_at=? WHERE id=?""",
            (outcome, exit_price, round(pnl, 4),
             datetime.now(timezone.utc).isoformat(), signal_id),
        )
        conn.commit()
    finally:
        conn.close()


def time_exit_signal(signal_id: int, current_price: float):
    """
    72시간 초과 시 현재가 기준으로 강제 청산.
    PnL = current_price - entry_price.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        if not row:
            return

        entry = row["entry_price"]
        side = row["side"]

        if side.upper() == "YES":
            pnl = current_price - entry
        else:
            pnl = current_price - entry

        outcome = "win" if pnl > 0 else "loss"

        conn.execute(
            """UPDATE signals SET outcome=?, settlement_price=?, pnl=?, settled_at=? WHERE id=?""",
            (outcome, current_price, round(pnl, 4),
             datetime.now(timezone.utc).isoformat(), signal_id),
        )
        conn.commit()
    finally:
        conn.close()


def check_time_exits(max_hold_hours: int = 72) -> List[int]:
    """
    72시간 초과 pending 시그널을 찾아 강제 청산.
    반환: 청산된 signal_id 리스트.
    """
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM signals WHERE outcome = 'pending'").fetchall()
        exited = []
        now = datetime.now(timezone.utc)

        for row in rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                hours_held = (now - ts).total_seconds() / 3600
                if hours_held >= max_hold_hours:
                    # 현재가를 모르니 entry_price 기준 본전 청산 (PnL=0)
                    # 실제 현재가가 있으면 외부에서 time_exit_signal()을 직접 호출
                    time_exit_signal(row["id"], row["entry_price"])
                    exited.append(row["id"])
                    print(f"⏰ [PAPER TIME EXIT] id={row['id']} {row['market_title'][:40]} | {hours_held:.1f}h >= {max_hold_hours}h")
            except Exception:
                continue

        return exited
    finally:
        conn.close()


def get_pending_signals() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals WHERE outcome = 'pending' ORDER BY timestamp").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_signals() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals ORDER BY timestamp DESC").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_stats() -> Dict[str, Any]:
    """Compute summary statistics over all settled signals."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals WHERE outcome != 'pending'").fetchall()
    settled = [dict(r) for r in rows]
    pending = conn.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'pending'").fetchone()[0]
    conn.close()

    if not settled:
        return {
            "total_signals": pending,
            "settled": 0,
            "pending": pending,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    wins = sum(1 for s in settled if s["outcome"] == "win")
    losses = sum(1 for s in settled if s["outcome"] == "loss")
    pnls = [s["pnl"] for s in settled if s["pnl"] is not None]
    total_pnl = sum(pnls)

    return {
        "total_signals": len(settled) + pending,
        "settled": len(settled),
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(settled) * 100, 1) if settled else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(total_pnl / len(settled), 4) if settled else 0.0,
        "best_trade": round(max(pnls), 4) if pnls else 0.0,
        "worst_trade": round(min(pnls), 4) if pnls else 0.0,
    }
