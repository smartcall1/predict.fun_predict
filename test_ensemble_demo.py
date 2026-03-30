#!/usr/bin/env python3
"""
Ensemble Demo — 각 에이전트의 실제 추론 과정을 시각적으로 보여주는 테스트.

Usage:
    python test_ensemble_demo.py
    python test_ensemble_demo.py "Will Bitcoin hit $100k by end of 2026?"
    python test_ensemble_demo.py "Fed rate hike in 2026?" --yes-price 0.23
"""

import asyncio
import argparse
import json
import sys
import os
import time
from datetime import datetime

# .env 로드
from dotenv import load_dotenv
load_dotenv()

from src.config.settings import settings
from src.agents.ensemble import EnsembleRunner
from src.agents.trader_agent import TraderAgent
from src.clients.model_router import ModelRouter
from src.utils.database import DatabaseManager


# ── 컬러 출력 ───────────────────────────────
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

AGENT_COLORS = {
    "forecaster": C.CYAN,
    "news_analyst": C.BLUE,
    "bull_researcher": C.GREEN,
    "bear_researcher": C.RED,
    "risk_manager": C.YELLOW,
    "trader": C.MAGENTA,
}

AGENT_EMOJI = {
    "forecaster": "🔮",
    "news_analyst": "📰",
    "bull_researcher": "🐂",
    "bear_researcher": "🐻",
    "risk_manager": "🛡️",
    "trader": "👨‍💼",
}


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {C.BOLD}{text}{C.RESET}")
    print(f"{'='*70}")


def print_agent(role, result):
    color = AGENT_COLORS.get(role, "")
    emoji = AGENT_EMOJI.get(role, "•")
    name = role.upper().replace("_", " ")
    model = "gemini-3.1-pro-preview" if role == "trader" else "gemini-2.5-flash"

    print(f"\n{color}{'─'*60}{C.RESET}")
    print(f"{color}{emoji} {C.BOLD}{name}{C.RESET} {C.DIM}({model}){C.RESET}")
    print(f"{color}{'─'*60}{C.RESET}")

    if "error" in result:
        print(f"  {C.RED}ERROR: {result['error']}{C.RESET}")
        return

    # 핵심 수치
    if "probability" in result and result["probability"] is not None:
        prob = result["probability"]
        prob_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  YES 확률: {color}{C.BOLD}{prob:.1%}{C.RESET}  [{prob_bar}]")

    if "confidence" in result:
        print(f"  확신도:   {result['confidence']:.1%}")

    if "base_rate" in result:
        print(f"  Base Rate: {result['base_rate']:.1%}")

    if "probability_floor" in result:
        print(f"  확률 하한: {result['probability_floor']:.1%}")

    if "probability_ceiling" in result:
        print(f"  확률 상한: {result['probability_ceiling']:.1%}")

    if "sentiment" in result:
        s = result["sentiment"]
        label = "긍정" if s > 0.1 else ("부정" if s < -0.1 else "중립")
        print(f"  감성: {s:+.2f} ({label})")

    if "risk_score" in result:
        rs = result["risk_score"]
        risk_label = "낮음" if rs <= 3 else ("보통" if rs <= 6 else "높음")
        print(f"  Risk Score: {rs}/10 ({risk_label})")

    if "should_trade" in result:
        st = result["should_trade"]
        print(f"  거래 권고: {'✅ YES' if st else '❌ NO'}")

    if "ev_estimate" in result:
        print(f"  기대값(EV): {result['ev_estimate']:+.3f}")

    if "key_arguments" in result:
        print(f"  핵심 논거:")
        for i, arg in enumerate(result["key_arguments"][:3], 1):
            print(f"    {i}. {arg}")

    if "key_factors" in result:
        print(f"  핵심 요인:")
        for i, f in enumerate(result["key_factors"][:3], 1):
            print(f"    {i}. {f}")

    # Action (Trader only)
    if "action" in result:
        action = result["action"]
        side = result.get("side", "?")
        emoji_action = "🟢 BUY" if action == "BUY" else ("🔴 SELL" if action == "SELL" else "⏭️ SKIP")
        print(f"\n  {C.BOLD}결정: {emoji_action} {side}{C.RESET}")
        if "limit_price" in result:
            print(f"  지정가: {result['limit_price']}¢")
        if "position_size_pct" in result:
            print(f"  포지션 크기: {result['position_size_pct']:.1f}%")

    # Reasoning
    if "reasoning" in result:
        reasoning = result["reasoning"]
        print(f"\n  {C.DIM}추론:{C.RESET}")
        # Wrap at 70 chars
        words = reasoning.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 68:
                print(line)
                line = "    "
            line += w + " "
        if line.strip():
            print(line)


async def run_demo(market_title: str, yes_price: float, volume: float, category: str):
    """Run full ensemble + trader demo with verbose output."""

    no_price = round(1.0 - yes_price, 2)
    market_data = {
        "ticker": "demo_market",
        "title": market_title,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": volume,
        "expiration_ts": int(time.time()) + 86400 * 14,
        "category": category,
        "news_summary": f"Analysis of: {market_title}",
    }

    print_header(f"MARKET: {market_title}")
    print(f"  YES Price: {yes_price:.2f} ({yes_price*100:.0f}%)")
    print(f"  NO Price:  {no_price:.2f} ({no_price*100:.0f}%)")
    print(f"  Volume:    ${volume:,.0f}")
    print(f"  Category:  {category}")

    # ── Phase 1: Ensemble (5 agents, Flash) ───────────
    print_header("PHASE 1: ENSEMBLE VOTING (Gemini 2.5 Flash × 5)")

    db = DatabaseManager()
    await db.initialize()

    model_router = ModelRouter(db_manager=db)
    agent_models = settings.ensemble.agent_models

    runner = EnsembleRunner(
        min_models=3,
        disagreement_threshold=settings.ensemble.disagreement_threshold,
    )

    # Build completion callables
    completions = {}
    for role in runner.agents:
        role_model = agent_models.get(role, settings.api.gemini_model)

        async def _make_fn(prompt, _model=role_model, _role=role):
            return await model_router.get_completion(
                prompt=prompt, model=_model,
                strategy="ensemble", query_type=f"ensemble_{_role}",
                market_id="demo_market",
            )
        completions[role] = _make_fn

    enriched = {**market_data}
    t0 = time.time()
    ensemble_result = await runner.run_ensemble(enriched, completions, context={})
    ensemble_time = time.time() - t0

    # Print each agent's result
    for mr in ensemble_result.get("model_results", []):
        role = mr.get("_agent", "unknown")
        print_agent(role, mr)

    # ── Ensemble Summary ──────────────────────────
    print_header("ENSEMBLE AGGREGATE")
    prob = ensemble_result.get("probability")
    conf = ensemble_result.get("confidence")
    disagree = ensemble_result.get("disagreement")

    if prob is not None:
        prob_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  가중 확률:  {C.BOLD}{prob:.1%}{C.RESET}  [{prob_bar}]")
        print(f"  종합 확신:  {conf:.1%}")
        print(f"  의견 불일치: {disagree:.3f} (threshold: {settings.ensemble.disagreement_threshold})")
        print(f"  참여 모델:  {ensemble_result.get('num_models_used', 0)}개")
        print(f"  소요 시간:  {ensemble_time:.1f}초")

        # Edge calculation
        edge_yes = prob - yes_price
        edge_no = (1 - prob) - no_price
        print(f"\n  Edge (YES): {edge_yes:+.3f} {'✅ 유리' if edge_yes >= 0.05 else '❌ 부족'}")
        print(f"  Edge (NO):  {edge_no:+.3f} {'✅ 유리' if edge_no >= 0.05 else '❌ 부족'}")
    else:
        print(f"  {C.RED}Ensemble 실패: {ensemble_result.get('error')}{C.RESET}")
        return

    # ── Phase 2: Trader (Pro model) ──────────────────
    min_edge = settings.trading.min_edge
    if max(edge_yes, edge_no) < min_edge:
        print(f"\n  {C.YELLOW}Edge < {min_edge:.0%} → Trader 미호출 (SKIP){C.RESET}")
        print_header("FINAL DECISION: SKIP (edge 부족)")
        return

    suggested_side = "YES" if edge_yes >= edge_no else "NO"
    edge = max(edge_yes, edge_no)

    print_header(f"PHASE 2: TRADER VERIFICATION (Gemini 3.1 Pro)")
    print(f"  Ensemble 제안: BUY {suggested_side} (edge={edge:+.3f})")
    print(f"  Trader가 독립적으로 검증 중...\n")

    trader = TraderAgent()
    trader_model = agent_models.get("trader", "gemini-3.1-pro-preview")

    async def trader_completion(prompt):
        return await model_router.get_completion(
            prompt=prompt, model=trader_model,
            strategy="ensemble_trader", query_type="ensemble_trader",
            market_id="demo_market",
        )

    model_results = ensemble_result.get("model_results", [])
    trader_context = {
        "forecaster_result": next((r for r in model_results if r.get("_agent") == "forecaster"), None),
        "news_result": next((r for r in model_results if r.get("_agent") == "news_analyst"), None),
        "bull_result": next((r for r in model_results if r.get("_agent") == "bull_researcher"), None),
        "bear_result": next((r for r in model_results if r.get("_agent") == "bear_researcher"), None),
        "risk_result": next((r for r in model_results if r.get("_agent") == "risk_manager"), None),
        "ensemble_meta": {
            "probability": prob, "confidence": conf,
            "disagreement": disagree, "suggested_side": suggested_side,
            "edge": edge, "num_models": ensemble_result.get("num_models_used", 0),
        },
    }

    t1 = time.time()
    trader_result = await trader.analyze(enriched, trader_context, trader_completion)
    trader_time = time.time() - t1

    print_agent("trader", trader_result)
    print(f"\n  {C.DIM}소요 시간: {trader_time:.1f}초{C.RESET}")

    # ── Final Summary ─────────────────────────────
    action = trader_result.get("action", "SKIP")
    side = trader_result.get("side", "?")
    final_conf = trader_result.get("confidence", 0)

    if action == "BUY":
        emoji = "🟢"
        color = C.GREEN
    elif action == "SELL":
        emoji = "🔴"
        color = C.RED
    else:
        emoji = "⏭️"
        color = C.YELLOW

    print_header(f"FINAL DECISION: {emoji} {action} {side}")
    print(f"  {color}{C.BOLD}Action: {action} {side}{C.RESET}")
    print(f"  Confidence: {final_conf:.0%}")
    print(f"  Total time: {ensemble_time + trader_time:.1f}초")
    print(f"  AI cost:    ~${(ensemble_time + trader_time) * 0.0001:.4f}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Ensemble Agent Demo")
    parser.add_argument("market", nargs="?",
                        default="Fed rate hike in 2026?",
                        help="Market title/question")
    parser.add_argument("--yes-price", type=float, default=0.23,
                        help="Current YES price (0-1)")
    parser.add_argument("--volume", type=float, default=50000,
                        help="Market volume in USD")
    parser.add_argument("--category", default="economics",
                        help="Market category")
    args = parser.parse_args()

    if not settings.api.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    await run_demo(args.market, args.yes_price, args.volume, args.category)


if __name__ == "__main__":
    asyncio.run(main())
