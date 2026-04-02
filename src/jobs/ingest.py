"""
Market Ingestion Job — Predict.fun

Fetches active markets from Predict.fun API, transforms them into the Market schema,
and upserts them into the database. Compatible with the existing pipeline.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional, List

from src.clients.kalshi_client import KalshiClient  # → PredictFunClient via shim
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


def _parse_predict_market(market_data: dict, existing_position_market_ids: set, logger) -> Optional[Market]:
    """
    Parse a single Predict.fun market response into the Market dataclass.

    실제 API 응답 필드 (2026-03-28 확인):
      - id: 95071 (int)
      - title: "11-15" (짧음, question/description과 조합 필요)
      - question: "11-15" (title과 동일할 수 있음)
      - description: 마켓 상세 설명
      - categorySlug: "number-of-cz-tweets-mar-23rd-mar-30th-2026"
      - status: "REGISTERED" (등록 상태)
      - tradingStatus: "OPEN" (거래 상태 — 이것으로 필터링!)
      - outcomes: [{"indexSet": 1, "name": "Yes", "onChainId": "..."}, ...]
      - stats: None or {"volume": ..., "lastPrice": ...}
      - feeRateBps: 200
      - createdAt: "2026-03-23T12:02:14.665Z"
      - resolution: None or resolved value
    """
    try:
        # Market ID
        market_id = str(market_data.get("id") or market_data.get("marketId") or "")
        if not market_id:
            return None

        # Title — 짧은 title에 categorySlug 정보를 보강
        raw_title = market_data.get("title") or market_data.get("question") or ""
        category_slug = market_data.get("categorySlug") or ""
        description = market_data.get("description") or ""

        # title이 너무 짧으면 (팀 이름만 있는 경우 등) categorySlug에서 컨텍스트 추가
        if len(raw_title) < 30 and category_slug:
            # "number-of-cz-tweets-mar-23rd-mar-30th-2026" → "Number Of Cz Tweets Mar 23Rd Mar 30Th 2026"
            slug_readable = category_slug.replace("-", " ").title()
            title = f"{raw_title} ({slug_readable})"
        else:
            title = raw_title or "Unknown Market"

        # Status — tradingStatus가 핵심!
        trading_status = str(market_data.get("tradingStatus", "")).upper()
        reg_status = str(market_data.get("status", "")).upper()

        if trading_status == "OPEN":
            status = "active"
        elif trading_status in ("CLOSED", "HALTED"):
            status = "closed"
        elif reg_status in ("RESOLVED", "SETTLED"):
            status = "closed"
        else:
            status = "active" if reg_status == "REGISTERED" else reg_status.lower()

        # Prices (0~1 범위)
        yes_price = 0.5
        no_price = 0.5

        # Method 1: stats 객체에서 추출
        stats = market_data.get("stats")
        if stats and isinstance(stats, dict):
            lp = stats.get("lastPrice") or stats.get("last_price")
            if lp is not None:
                yes_price = float(lp)
                no_price = 1.0 - yes_price

        # Method 2: outcomes 배열에서 추출
        if yes_price == 0.5:
            outcomes = market_data.get("outcomes", [])
            if outcomes and isinstance(outcomes, list):
                for outcome in outcomes:
                    oname = str(outcome.get("name", "")).upper()
                    oprice = outcome.get("price")
                    if oprice is not None:
                        oprice = float(oprice)
                        if oname in ("YES", "UP", "ABOVE", "OVER"):
                            yes_price = oprice
                            no_price = 1.0 - oprice
                        elif oname in ("NO", "DOWN", "BELOW", "UNDER"):
                            no_price = oprice
                            yes_price = 1.0 - oprice

        # Method 3: 직접 가격 필드
        if yes_price == 0.5:
            lp = market_data.get("lastPrice") or market_data.get("last_price")
            if lp is not None:
                lp = float(lp)
                if lp > 1:
                    lp = lp / 100.0
                yes_price = lp
                no_price = 1.0 - lp

        # Volume
        vol_raw = 0
        if stats and isinstance(stats, dict):
            vol_raw = stats.get("volume") or stats.get("totalVolume") or 0
        if not vol_raw:
            vol_raw = market_data.get("volume") or market_data.get("totalVolume") or 0
        volume = int(float(vol_raw))

        # Expiration timestamp
        exp_ts = 0
        exp_raw = (
            market_data.get("endDate")
            or market_data.get("expirationDate")
            or market_data.get("close_time")
            or market_data.get("closesAt")
        )
        if exp_raw:
            if isinstance(exp_raw, (int, float)):
                exp_ts = int(exp_raw)
                if exp_ts > 1_000_000_000_000:
                    exp_ts = exp_ts // 1000
            elif isinstance(exp_raw, str):
                try:
                    exp_ts = int(datetime.fromisoformat(exp_raw.replace("Z", "+00:00")).timestamp())
                except (ValueError, TypeError):
                    exp_ts = int(time.time()) + 86400 * 30

        if exp_ts == 0:
            exp_ts = int(time.time()) + 86400 * 30  # default: 30일 뒤

        # Category — categorySlug 사용
        category = category_slug.split("-")[0] if category_slug else "unknown"
        category = category.lower()

        has_position = market_id in existing_position_market_ids

        return Market(
            market_id=market_id,
            title=title,
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            expiration_ts=exp_ts,
            category=category,
            status=status,
            last_updated=datetime.now(),
            has_position=has_position,
        )

    except Exception as e:
        logger.error(f"Failed to parse market: {e}", market_data=str(market_data)[:200])
        return None


async def process_and_queue_markets(
    markets_data: List[dict],
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    """
    Transforms Predict.fun market data, upserts to DB, and queues eligible markets.
    """
    markets_to_upsert = []
    for md in markets_data:
        market = _parse_predict_market(md, existing_position_market_ids, logger)
        if market and market.status == "active":
            markets_to_upsert.append(market)

    if markets_to_upsert:
        await db_manager.upsert_markets(markets_to_upsert)
        logger.info(f"Upserted {len(markets_to_upsert)} markets from Predict.fun.")

        # Category filter first (cheap, no API calls) — 대소문자 무관
        import re
        _SPORTS_TITLE_RE = re.compile(
            r'\b(Nba|Nfl|Nhl|Mlb|Mls|Lol|Csgo|Cs2|Dota|Valorant|Atp|Wta|Ufc|'
            r'Premier League|La Liga|Serie A|Bundesliga|Ncaa[bf]?|Epl|Match Winner)\b',
            re.IGNORECASE,
        )
        excluded_lower = {c.lower() for c in settings.trading.excluded_categories}
        preferred_lower = {c.lower() for c in settings.trading.preferred_categories}
        category_filtered = [
            m for m in markets_to_upsert
            if (not preferred_lower or m.category.lower() in preferred_lower)
            and m.category.lower() not in excluded_lower
            and not _SPORTS_TITLE_RE.search(m.title or "")
        ]

        # Fetch stats for category-filtered markets to get real volume
        # Use the same client from run_ingestion (passed via closure or re-create)
        from src.clients.kalshi_client import KalshiClient
        stats_client = KalshiClient()
        eligible_markets = []
        checked = 0

        for m in category_filtered:
            checked += 1
            if checked % 50 == 0:
                print(f"[STATS] Checking volume... {checked}/{len(category_filtered)} ({len(eligible_markets)} eligible)")
            try:
                stats = await stats_client.get_market_stats(m.market_id)
                if stats:
                    vol_total = float(stats.get("volumeTotalUsd", 0))
                    vol_24h = float(stats.get("volume24hUsd", 0))
                    liquidity = float(stats.get("totalLiquidityUsd", 0))
                    m.volume = int(vol_total)

                    if vol_total >= settings.trading.min_volume:
                        eligible_markets.append(m)
                    else:
                        continue  # Skip low-volume markets
                else:
                    # stats 조회 실패 시 일단 포함 (AI가 판단)
                    eligible_markets.append(m)
            except Exception as e:
                logger.debug(f"Stats fetch failed for {m.market_id}: {e}")
                eligible_markets.append(m)

            await asyncio.sleep(0.15)  # Rate limiting for stats calls

        await stats_client.close()

        print(f"[STATS] Done. {len(eligible_markets)} eligible (vol>=${settings.trading.min_volume}) from {len(category_filtered)} checked")

        # 그룹 마켓 필터: 제목 괄호 안 그룹명이 같은 마켓 → 볼륨 상위 3개만 허용
        import re
        _group_re = re.compile(r'\(([^)]+)\)\s*$')
        group_counts = {}
        # 볼륨 높은 순으로 정렬하여 상위 마켓 우선 통과
        eligible_markets.sort(key=lambda m: m.volume, reverse=True)
        filtered_markets = []
        for m in eligible_markets:
            match = _group_re.search(m.title or "")
            if match:
                group_name = match.group(1).strip().lower()
                group_counts[group_name] = group_counts.get(group_name, 0) + 1
                if group_counts[group_name] > 3:
                    continue  # 같은 그룹 4번째부터 스킵
            filtered_markets.append(m)

        if len(filtered_markets) < len(eligible_markets):
            print(f"[GROUP] 그룹 필터 적용: {len(eligible_markets)} → {len(filtered_markets)}개")

        for market in filtered_markets:
            await queue.put(market)
    else:
        logger.info("No active markets to upsert in this batch.")


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_ticker: Optional[str] = None,
):
    """
    Main ingestion job — fetches markets from Predict.fun API.
    """
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting Predict.fun market ingestion.", market_ticker=market_ticker)

    client = KalshiClient()  # → PredictFunClient via shim

    try:
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_ticker:
            logger.info(f"Fetching single market: {market_ticker}")
            market_response = await client.get_market(ticker=market_ticker)
            market_data = market_response.get("market")
            if market_data:
                await process_and_queue_markets(
                    [market_data], db_manager, queue, existing_position_market_ids, logger,
                )
            else:
                logger.warning(f"Market not found: {market_ticker}")
        else:
            print("[FETCH] Fetching markets from Predict.fun...")
            cursor = None
            page_size = 20  # Predict.fun max per page
            total_fetched = 0
            max_pages = 30  # Safety limit

            for _ in range(max_pages):
                response = await client.get_markets(limit=page_size, cursor=cursor)
                markets_page = response.get("markets", [])
                cursor = response.get("cursor")

                if not markets_page:
                    break

                total_fetched += len(markets_page)
                if total_fetched % 100 == 0 or total_fetched <= 20:
                    print(f"[FETCH] {total_fetched} markets collected...")

                await process_and_queue_markets(
                    markets_page, db_manager, queue, existing_position_market_ids, logger,
                )

                if not cursor or len(markets_page) < page_size:
                    break

                await asyncio.sleep(0.3)

    except Exception as e:
        logger.error("Error during market ingestion.", error=str(e), exc_info=True)
    finally:
        await client.close()
        logger.info("Market ingestion job finished.")
