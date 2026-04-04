#!/usr/bin/env python3
"""
기존 포지션 일괄 청산 스크립트
수정 전(BUG FIX28 이전) 진입한 포지션을 전부 현재가에 매도.

Usage:
    python close_old_positions.py          # 목록만 표시 (dry run)
    python close_old_positions.py --exec   # 실제 청산 실행
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.config.settings import settings
from src.clients.predictfun_client import PredictFunClient
from src.live.executor import LiveExecutor
from src.live.state import StateManager

KEEP_MARKET_IDS = set()  # 비워두면 timestamp 기반 자동 판별


async def main():
    dry_run = "--exec" not in sys.argv
    if dry_run:
        print("=" * 50)
        print("  DRY RUN - 목록만 표시 (실행: --exec)")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  EXECUTING - 실제 청산 진행!")
        print("=" * 50)

    state = StateManager(mode="LIVE")
    client = PredictFunClient()
    executor = LiveExecutor(client, paper_mode=False)

    await client._authenticate()

    positions = dict(state.positions)
    print(f"\nTotal positions: {len(positions)}\n")

    cutoff = time.time() - 7200  # 2시간 전

    old_positions = {}
    new_positions = {}
    for tid, pos in positions.items():
        ts = pos.get("timestamp", 0)
        mid = pos.get("market_id", "")
        if mid in KEEP_MARKET_IDS or ts > cutoff:
            new_positions[tid] = pos
        else:
            old_positions[tid] = pos

    print(f"KEEP (new): {len(new_positions)}")
    for tid, pos in new_positions.items():
        title = (pos.get("market_title", "") or "")[:35]
        print(f"  [KEEP] {pos.get('market_id')}: {title} [{pos.get('side')}]")

    print(f"\nCLOSE (old): {len(old_positions)}")
    for tid, pos in old_positions.items():
        title = (pos.get("market_title", "") or "")[:35]
        entry = pos.get("entry_price", 0)
        qty = pos.get("quantity", 0)
        side = pos.get("side", "?")
        print(f"  [CLOSE] {pos.get('market_id')}: {title} [{side}] entry={entry:.3f} qty={qty}")

    if dry_run:
        print(f"\nRun with --exec to execute.")
        await client.close()
        return

    print(f"\nClosing {len(old_positions)} positions...")
    closed = 0
    failed = 0

    for tid, pos in old_positions.items():
        market_id = pos.get("market_id", "")
        side = pos.get("side", "YES")
        qty = pos.get("quantity", 0)
        title = (pos.get("market_title", "") or "")[:30]

        try:
            prices = await client.get_best_prices(market_id)
            if prices:
                if side.upper() == "YES":
                    current = prices.get("yes_bid") or prices.get("mid") or pos.get("entry_price", 0.5)
                else:
                    no_bid = prices.get("no_bid")
                    if no_bid is not None and no_bid > 0:
                        current = no_bid
                    else:
                        mid = prices.get("mid")
                        current = max(1.0 - mid, 0.01) if mid else 0.5
            else:
                current = pos.get("entry_price", 0.5)
        except Exception:
            current = pos.get("entry_price", 0.5)

        print(f"\n  SELL {market_id} [{side}] {qty}x @ {current:.4f} | {title}")

        try:
            result = await executor.execute_sell(
                market_id=market_id,
                side=side,
                price=current,
                quantity=qty,
                reason="manual_close_old",
            )
            if result:
                pnl = (current - pos.get("entry_price", current)) * qty
                print(f"    OK | PnL={pnl:+.2f}")
                state.remove_position(tid)
                state.log_trade({
                    "result": "SELL",
                    "reason": "manual_close_old_positions",
                    "exit_price": current,
                    "pnl": pnl,
                    "market_id": market_id,
                    "market_title": pos.get("market_title", ""),
                    "side": side,
                    "entry_price": pos.get("entry_price", 0),
                    "quantity": qty,
                })
                s = state.stats
                s["losses"] = s.get("losses", 0) + 1
                s["total_pnl"] = s.get("total_pnl", 0) + pnl
                closed += 1
            else:
                print(f"    FAIL (result=None)")
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

        await asyncio.sleep(1)

    state.save()
    await client.close()

    print(f"\n{'=' * 50}")
    print(f"  Done: {closed} closed, {failed} failed")
    print(f"  Remaining: {len(new_positions)} positions (new only)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
