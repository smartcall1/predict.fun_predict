#!/usr/bin/env python3
"""
Watchdog — Predict.fun Live Trader

Exit code semantics:
    0   → Graceful stop (watchdog exits)
    99  → Restart signal (3s delay, restart)
    *   → Crash (5s delay, auto-restart)

Usage:
    python run_live.py                  # Paper 모드 (기본)
    python run_live.py --live           # 실거래 모드
"""

import subprocess
import sys
import os
import time
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "live_trader.log")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB


def rotate_log():
    """Rotate log file if > 10MB."""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_SIZE:
        backup = LOG_FILE + ".1"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(LOG_FILE, backup)
        print(f"[WATCHDOG] Log rotated: {LOG_FILE}")


def run():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Pass through CLI args (--live, --once)
    extra_args = sys.argv[1:]
    cmd = [sys.executable, "-u", "live_trader.py"] + extra_args

    mode = "LIVE" if "--live" in extra_args else "PAPER"
    print(f"[WATCHDOG] Starting Predict.fun AI Bot [{mode}]")
    print(f"[WATCHDOG] Command: {' '.join(cmd)}")
    print(f"[WATCHDOG] Log: {LOG_FILE}")
    print()

    restart_count = 0

    while True:
        rotate_log()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WATCHDOG] [{now}] Spawning bot (restart #{restart_count})...")

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n{'='*60}\n")
                log_f.write(f"[WATCHDOG] Bot started at {now} (restart #{restart_count})\n")
                log_f.write(f"{'='*60}\n\n")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()

                proc.wait()
                exit_code = proc.returncode

        except KeyboardInterrupt:
            print("\n[WATCHDOG] KeyboardInterrupt — stopping")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            break

        except Exception as e:
            print(f"[WATCHDOG] Error: {e}")
            exit_code = 1

        # Exit code handling
        now = datetime.now().strftime("%H:%M:%S")
        if exit_code == 0:
            print(f"[WATCHDOG] [{now}] Bot exited normally (code 0). Stopping watchdog.")
            break
        elif exit_code == 99:
            print(f"[WATCHDOG] [{now}] Restart signal (code 99). Restarting in 3s...")
            time.sleep(3)
        else:
            print(f"[WATCHDOG] [{now}] Bot crashed (code {exit_code}). Restarting in 5s...")
            time.sleep(5)

        restart_count += 1


if __name__ == "__main__":
    run()
