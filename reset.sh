#!/bin/bash
# Reset all trading data (market analyses, cost tracking, signals)
echo "Resetting trading data..."
sqlite3 trading_system.db "DELETE FROM market_analyses; DELETE FROM daily_cost_tracking; DELETE FROM positions;"
sqlite3 data/paper_trades.db "DELETE FROM signals;"
rm -f logs/daily_ai_usage.pkl
echo "Done. Run: python3 -u paper_trader.py"
