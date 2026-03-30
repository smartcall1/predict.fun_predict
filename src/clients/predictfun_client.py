"""
Predict.fun API client for trading operations.
Replaces kalshi_client.py — adapted for Predict.fun (BNB Chain) prediction markets.

API docs: https://dev.predict.fun/
Swagger: https://api.predict.fun/docs
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

import httpx

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class PredictFunAPIError(Exception):
    """Custom exception for Predict.fun API errors."""
    pass


class PredictFunClient(TradingLoggerMixin):
    """
    Predict.fun API client for automated trading.
    Handles market data retrieval and (future) trade execution.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, backoff_factor: float = 1.0):
        self.api_key = api_key or settings.api.predict_api_key
        self.base_url = settings.api.predict_base_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self.client = httpx.AsyncClient(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                "User-Agent": "PredictAIBot/1.0",
                "Accept": "application/json",
                "x-api-key": self.api_key or "",
            },
        )

        self.logger.info("Predict.fun client initialized", api_key_present=bool(self.api_key))

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make API request with retry logic.
        If raw_response=True, returns the full JSON without unwrapping 'data'.
        """
        url = f"{self.base_url}{endpoint}"

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(0.3)  # rate limit: ~3 req/s

                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                )
                response.raise_for_status()
                data = response.json()

                if raw_response:
                    return data

                # Predict.fun wraps some responses in {"data": ...} or {"success": true, "data": ...}
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(
                        f"API {e.response.status_code}, retrying in {sleep_time:.1f}s",
                        endpoint=endpoint, attempt=attempt + 1,
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise PredictFunAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                last_exception = e
                sleep_time = self.backoff_factor * (2 ** attempt)
                self.logger.warning(f"Request failed, retrying", error=str(e), attempt=attempt + 1)
                await asyncio.sleep(sleep_time)

        raise PredictFunAPIError(f"Failed after {self.max_retries} retries: {last_exception}")

    # ── Markets ──────────────────────────────────

    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str = "OPEN",
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get markets list. Returns dict with 'markets' key for compatibility
        with the existing ingestion pipeline.
        """
        params = {"limit": limit, "status": status}
        if cursor:
            params["after"] = cursor  # Predict.fun uses 'after' for cursor pagination

        raw = await self._request("GET", "/v1/markets", params=params, raw_response=True)

        # Response: {"success": true, "data": [...], "cursor": "base64..."}
        if isinstance(raw, list):
            markets = raw
            next_cursor = None
        elif isinstance(raw, dict):
            markets = raw.get("data", raw.get("markets", []))
            next_cursor = raw.get("cursor")
        else:
            markets = []
            next_cursor = None

        return {"markets": markets, "cursor": next_cursor}

    async def get_all_open_markets(self, max_pages: int = 30) -> List[Dict]:
        """Paginate through all open markets using cursor-based pagination."""
        PAGE_SIZE = 20  # Predict.fun returns max 20 per page
        all_markets = []
        cursor = None

        for _ in range(max_pages):
            result = await self.get_markets(limit=PAGE_SIZE, cursor=cursor)
            batch = result.get("markets", [])
            cursor = result.get("cursor")

            if not batch:
                break
            all_markets.extend(batch)

            if not cursor or len(batch) < PAGE_SIZE:
                break

        self.logger.info(f"Fetched {len(all_markets)} total open markets")
        return all_markets

    async def get_market(self, ticker: Optional[str] = None, market_id: Optional[str] = None) -> Dict[str, Any]:
        """Get specific market data."""
        mid = ticker or market_id
        if not mid:
            raise PredictFunAPIError("ticker or market_id required")
        data = await self._request("GET", f"/v1/markets/{mid}")
        if isinstance(data, dict) and "market" in data:
            return data
        return {"market": data}

    async def get_orderbook(self, ticker: str, depth: int = 20) -> Dict[str, Any]:
        """Get market orderbook."""
        return await self._request("GET", f"/v1/markets/{ticker}/orderbook", params={"depth": depth})

    async def get_market_stats(self, ticker: str) -> Optional[Dict]:
        """Get market stats (volume, liquidity). Separate endpoint from market detail."""
        try:
            data = await self._request("GET", f"/v1/markets/{ticker}/stats")
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            self.logger.debug(f"get_market_stats({ticker}) failed: {e}")
            return None

    async def get_market_history(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        """Get market price history / activity."""
        data = await self._request("GET", f"/v1/markets/{ticker}/activity", params={"limit": limit})
        if isinstance(data, list):
            return {"history": data}
        return data

    # ── Events ───────────────────────────────────

    async def get_events(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get events (market groups)."""
        data = await self._request("GET", "/v1/events", params={"limit": limit, "offset": offset, "status": "OPEN"})
        if isinstance(data, list):
            return data
        return data.get("events", [])

    # ── Orderbook helpers ────────────────────────

    async def get_best_prices(self, ticker: str) -> Optional[Dict]:
        """Extract best ask/bid from orderbook."""
        try:
            ob = await self.get_orderbook(ticker)
        except Exception:
            return None

        def first_price(levels):
            if not levels:
                return None
            lv = levels[0]
            if isinstance(lv, (list, tuple)):
                return float(lv[0])
            if isinstance(lv, dict):
                return float(lv.get("price") or lv.get("p") or 0) or None
            return None

        asks = ob.get("asks", []) if isinstance(ob, dict) else []
        bids = ob.get("bids", []) if isinstance(ob, dict) else []
        best_ask = first_price(asks)
        best_bid = first_price(bids)

        if best_ask is None and best_bid is None:
            return None

        return {
            "yes_ask": best_ask,
            "yes_bid": best_bid,
            "spread": round((best_ask or 0) - (best_bid or 0), 4) if best_ask and best_bid else None,
            "mid": round(((best_ask or 0) + (best_bid or 0)) / 2, 4) if best_ask and best_bid else (best_ask or best_bid),
        }

    # ── Portfolio (paper mode stubs) ─────────────

    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance. Paper mode returns simulated balance."""
        if settings.trading.paper_trading_mode:
            return {"balance": settings.trading.initial_bankroll}
        # TODO: implement real balance check via SDK/API
        return {"balance": 0}

    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio positions. Paper mode returns empty."""
        if settings.trading.paper_trading_mode:
            return {"positions": []}
        # TODO: implement real position check
        return {"positions": []}

    async def get_fills(self, ticker: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Get order fills."""
        return {"fills": []}

    async def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        """Get orders."""
        return {"orders": []}

    async def get_trades(self, ticker: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Get trade history."""
        return {"trades": []}

    # ── Order execution (paper only for now) ─────

    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: int,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Place order with realistic paper simulation.

        현실 비용 시뮬레이션 (90%+ 유사도 목표):
        1. 실제 오더북에서 체결가 추정 (best ask/bid)
        2. 슬리피지 적용 (오더북 depth 기반, 최소 1%)
        3. 거래 수수료 (마켓별 feeRateBps, 기본 2%)
        4. 가스비 (BNB Chain ~$0.10/tx)
        5. 스프레드 반영 (bid-ask 차이)
        """
        if settings.trading.paper_trading_mode:
            # ── 1. 실제 오더북에서 체결가 추정 ──
            intended_price = (yes_price or 50) / 100.0 if yes_price else 0.50
            actual_price = intended_price
            spread = 0.0

            try:
                ob = await self.get_orderbook(ticker)
                if ob and isinstance(ob, dict):
                    asks = ob.get("asks", [])
                    bids = ob.get("bids", [])

                    def _first(levels):
                        if not levels:
                            return None
                        lv = levels[0]
                        if isinstance(lv, (list, tuple)):
                            return float(lv[0])
                        if isinstance(lv, dict):
                            return float(lv.get("price") or lv.get("p") or 0) or None
                        return None

                    best_ask = _first(asks)
                    best_bid = _first(bids)

                    is_buy = action.lower() == "buy"
                    is_yes = side.lower() in ("yes", "up", "above")

                    if is_buy:
                        # BUY YES → pay ask, BUY NO → pay 1-bid
                        if is_yes and best_ask:
                            actual_price = best_ask
                        elif not is_yes and best_bid:
                            actual_price = max(1.0 - best_bid, 0.01)
                        elif best_ask:
                            actual_price = best_ask
                    else:
                        # SELL YES → receive bid, SELL NO → receive 1-ask
                        if is_yes and best_bid:
                            actual_price = best_bid
                        elif not is_yes and best_ask:
                            actual_price = max(1.0 - best_ask, 0.01)
                        elif best_bid:
                            actual_price = best_bid

                    # Spread 계산
                    if best_ask and best_bid:
                        spread = round(best_ask - best_bid, 4)

            except Exception as e:
                self.logger.debug(f"Orderbook fetch failed for paper order, using intended price: {e}")

            # ── 2. 슬리피지 적용 (최소 1%, 스프레드 비례) ──
            # 실제 환경: 소량은 1~2%, 대량은 3~5%
            base_slippage = max(0.01, spread * 0.5)  # 스프레드의 50% or 최소 1%
            # 수량 기반 추가 슬리피지 (10 contracts 이상이면 추가)
            size_slippage = min(0.03, count * 0.001)  # 최대 3% 추가
            total_slippage = base_slippage + size_slippage

            if action.lower() == "buy":
                actual_price = min(actual_price * (1 + total_slippage), 0.99)
            else:
                actual_price = max(actual_price * (1 - total_slippage), 0.01)

            actual_price = round(actual_price, 4)

            # ── 3. 거래 수수료 (마켓별 feeRateBps 조회) ──
            fee_bps = 200  # 기본 2%
            try:
                mkt = await self.get_market(ticker=ticker)
                mkt_data = mkt.get("market", mkt) if isinstance(mkt, dict) else {}
                fee_bps = int(mkt_data.get("feeRateBps", 200))
            except Exception:
                pass

            trade_value = count * actual_price
            fee = trade_value * fee_bps / 10_000
            fee = round(fee, 4)

            # ── 4. 가스비 (BNB Chain, ~$0.05~0.15) ──
            gas_fee = 0.10  # 고정 $0.10 (평균치)

            # ── 5. 총 비용 계산 ──
            total_cost = fee + gas_fee
            net_value = trade_value - total_cost if action.lower() == "buy" else trade_value + total_cost

            self.logger.info(
                f"[PAPER] {action.upper()} {count}x {side.upper()} @ {actual_price:.4f} "
                f"(intended={intended_price:.4f}, slip={total_slippage:.1%}, "
                f"fee=${fee:.4f}, gas=${gas_fee:.2f}, spread={spread:.4f})"
            )

            return {
                "order": {
                    "order_id": f"paper_{int(time.time())}_{ticker}",
                    "ticker": ticker,
                    "side": side,
                    "action": action,
                    "count": count,
                    "price": actual_price,
                    "intended_price": intended_price,
                    "status": "filled",
                    "paper": True,
                    # 비용 상세
                    "trade_value": round(trade_value, 4),
                    "fee": fee,
                    "fee_bps": fee_bps,
                    "gas_fee": gas_fee,
                    "total_cost": round(total_cost, 4),
                    "net_value": round(net_value, 4),
                    "slippage_pct": round(total_slippage * 100, 2),
                    "spread": spread,
                }
            }

        raise PredictFunAPIError(
            "Live trading not yet implemented. "
            "Use paper mode or implement Predict.fun order API here."
        )

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order."""
        return {"status": "cancelled", "order_id": order_id}

    # ── Lifecycle ────────────────────────────────

    async def close(self) -> None:
        await self.client.aclose()
        self.logger.info("Predict.fun client closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
