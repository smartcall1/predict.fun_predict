"""
Predict.fun API client for trading operations.
Supports both Paper simulation and Live trading via predict-sdk.

Paper mode: realistic cost simulation (orderbook, slippage, fees, gas)
Live mode:  predict-sdk OrderBuilder → EIP-712 signing → POST /v1/orders

API docs: https://dev.predict.fun/
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

import httpx

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin

# ── Live Trading SDK imports (lazy, fail gracefully) ──
_SDK_AVAILABLE = False
try:
    from predict_sdk import (
        OrderBuilder, BuildOrderInput,
        ChainId, Side as SdkSide,
        ADDRESSES_BY_CHAIN_ID, RPC_URLS_BY_CHAIN_ID,
        generate_order_salt,
        LimitHelperInput, MarketHelperInput, MarketHelperValueInput,
        Book, DepthLevel, SignedOrder,
    )
    from predict_sdk.order_builder import make_contracts
    from eth_account import Account
    from web3 import Web3
    _SDK_AVAILABLE = True
except ImportError:
    pass

CHAIN_ID_BNB = ChainId.BNB_MAINNET if _SDK_AVAILABLE else None
WEI = 10 ** 18
DEFAULT_PRECISION = 18  # BNB Chain USDT = 18 decimals


def _usdt_to_wei(amount: float) -> int:
    return int(amount * WEI)


def _wei_to_usdt(amount_wei: int) -> float:
    return amount_wei / WEI


class PredictFunAPIError(Exception):
    """Custom exception for Predict.fun API errors."""
    pass


class PredictFunClient(TradingLoggerMixin):
    """
    Predict.fun API client for automated trading.
    Paper mode: orderbook-based simulation with realistic costs.
    Live mode:  predict-sdk EIP-712 signed orders on BNB Chain.
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

        # ── Live Trading: OrderBuilder (predict-sdk) ──
        self._builder: Optional[Any] = None
        self._w3: Optional[Any] = None
        self._live_ready = False

        self._jwt_token: Optional[str] = None

        if not settings.trading.paper_trading_mode and settings.api.private_key and _SDK_AVAILABLE:
            self._init_order_builder()

        mode = "LIVE" if self._live_ready else "PAPER"
        sdk_status = "OK" if self._live_ready else ("NO_SDK" if not _SDK_AVAILABLE else "SKIP(paper/no-key)")
        self.logger.info(f"Predict.fun client initialized [{mode}] SDK={sdk_status}")

    def _init_order_builder(self):
        """Initialize predict-sdk OrderBuilder for live trading."""
        try:
            rpc_url = RPC_URLS_BY_CHAIN_ID.get(CHAIN_ID_BNB, "https://bsc-dataseed.bnbchain.org/")
            self._w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))

            if not self._w3.is_connected():
                for url in [
                    "https://bsc-dataseed1.binance.org",
                    "https://bsc-dataseed2.binance.org",
                    "https://bsc.publicnode.com",
                ]:
                    w3 = Web3(Web3.HTTPProvider(url))
                    if w3.is_connected():
                        self._w3 = w3
                        break

            signer = Account.from_key(settings.api.private_key)
            addresses = ADDRESSES_BY_CHAIN_ID[CHAIN_ID_BNB]
            contracts = make_contracts(self._w3, addresses, signer)

            self._builder = OrderBuilder(
                chain_id=CHAIN_ID_BNB,
                precision=DEFAULT_PRECISION,
                addresses=addresses,
                generate_salt_fn=generate_order_salt,
                logger=logging.getLogger("predict_sdk"),
                signer=signer,
                predict_account=settings.api.deposit_address or settings.api.wallet_address,
                contracts=contracts,
                web3=self._w3,
            )
            self._live_ready = True
            self.logger.info(f"OrderBuilder initialized (RPC: {rpc_url})")
        except Exception as e:
            self.logger.warning(f"OrderBuilder init failed: {e}")
            self._builder = None
            self._live_ready = False

    # ── JWT Authentication ──────────────────────

    async def _authenticate(self) -> bool:
        """3-step JWT authentication: get message → sign → exchange for token."""
        if not self._builder or not self._live_ready:
            return False
        try:
            # Step 1: Get message to sign
            resp = await self.client.get(
                f"{self.base_url}/v1/auth/message",
                headers={"x-api-key": self.api_key or ""},
            )
            resp.raise_for_status()
            message = resp.json().get("data", {}).get("message", "")
            if not message:
                self.logger.warning("JWT auth: empty message from /v1/auth/message")
                return False

            # Step 2: Sign message with predict account
            signature = self._builder.sign_predict_account_message(message)

            # Step 3: Exchange signature for JWT
            signer = settings.api.deposit_address or settings.api.wallet_address
            auth_resp = await self.client.post(
                f"{self.base_url}/v1/auth",
                headers={"x-api-key": self.api_key or "", "Content-Type": "application/json"},
                json={"signer": signer, "message": message, "signature": signature},
            )
            auth_resp.raise_for_status()
            token = auth_resp.json().get("data", {}).get("token", "")
            if token:
                self._jwt_token = token
                self.client.headers["Authorization"] = f"Bearer {token}"
                self.logger.info("JWT authentication successful")
                return True
            self.logger.warning("JWT auth: no token in response")
            return False
        except Exception as e:
            self.logger.warning(f"JWT authentication failed: {e}")
            return False

    # ── HTTP Request Helper ─────────────────────

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make API request with retry logic."""
        url = f"{self.base_url}{endpoint}"

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(0.3)

                response = await self.client.request(
                    method=method, url=url, params=params, json=json_data,
                )
                response.raise_for_status()
                data = response.json()

                if raw_response:
                    return data
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data

            except httpx.HTTPStatusError as e:
                last_exception = e
                # 401 → JWT 인증 후 재시도 (1회만)
                if e.response.status_code == 401 and attempt == 0 and self._live_ready:
                    self.logger.info("401 → JWT 인증 시도...")
                    if await self._authenticate():
                        continue  # 인증 성공 → 재시도
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(f"API {e.response.status_code}, retrying in {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                else:
                    raise PredictFunAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                last_exception = e
                sleep_time = self.backoff_factor * (2 ** attempt)
                self.logger.warning(f"Request failed, retrying: {e}")
                await asyncio.sleep(sleep_time)

        raise PredictFunAPIError(f"Failed after {self.max_retries} retries: {last_exception}")

    # ── Markets ──────────────────────────────────

    async def get_markets(
        self, limit: int = 100, offset: int = 0, status: str = "OPEN", cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {"limit": limit, "status": status}
        if cursor:
            params["after"] = cursor

        raw = await self._request("GET", "/v1/markets", params=params, raw_response=True)

        if isinstance(raw, list):
            return {"markets": raw, "cursor": None}
        elif isinstance(raw, dict):
            markets = raw.get("data", raw.get("markets", []))
            return {"markets": markets, "cursor": raw.get("cursor")}
        return {"markets": [], "cursor": None}

    async def get_all_open_markets(self, max_pages: int = 30) -> List[Dict]:
        PAGE_SIZE = 20
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
        mid = ticker or market_id
        if not mid:
            raise PredictFunAPIError("ticker or market_id required")
        data = await self._request("GET", f"/v1/markets/{mid}")
        if isinstance(data, dict) and "market" in data:
            return data
        return {"market": data}

    async def get_orderbook(self, ticker: str, depth: int = 20) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/markets/{ticker}/orderbook", params={"depth": depth})

    async def get_market_stats(self, ticker: str) -> Optional[Dict]:
        try:
            data = await self._request("GET", f"/v1/markets/{ticker}/stats")
            return data if isinstance(data, dict) else None
        except Exception as e:
            self.logger.debug(f"get_market_stats({ticker}) failed: {e}")
            return None

    async def get_market_history(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        data = await self._request("GET", f"/v1/markets/{ticker}/activity", params={"limit": limit})
        if isinstance(data, list):
            return {"history": data}
        return data

    # ── Events ───────────────────────────────────

    async def get_events(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        data = await self._request("GET", "/v1/events", params={"limit": limit, "offset": offset, "status": "OPEN"})
        if isinstance(data, list):
            return data
        return data.get("events", [])

    # ── Orderbook helpers ────────────────────────

    async def get_best_prices(self, ticker: str) -> Optional[Dict]:
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

        # NO 가격 = YES 오더북의 보수 (공식 문서: dev.predict.fun/doc-685654)
        # NO 매수가(no_ask) = 1 - YES best_bid  (YES bids → NO asks)
        # NO 매도가(no_bid) = 1 - YES best_ask  (YES asks → NO bids)
        no_ask = round(1.0 - best_bid, 4) if best_bid is not None else None
        no_bid = round(1.0 - best_ask, 4) if best_ask is not None else None

        return {
            "yes_ask": best_ask,
            "yes_bid": best_bid,
            "no_ask": no_ask,     # NO 매수가 (= 1 - YES bid)
            "no_bid": no_bid,     # NO 매도가 (= 1 - YES ask)
            "spread": round((best_ask or 0) - (best_bid or 0), 4) if best_ask and best_bid else None,
            "no_spread": round(no_ask - no_bid, 4) if no_ask is not None and no_bid is not None else None,
            "mid": round(((best_ask or 0) + (best_bid or 0)) / 2, 4) if best_ask and best_bid else (best_ask or best_bid),
        }

    def _build_sdk_book(self, market_id: str, ob: dict) -> Optional[Any]:
        """Convert API orderbook response → predict_sdk.Book for SDK order calculation."""
        if not _SDK_AVAILABLE:
            return None
        try:
            def parse_levels(levels: list) -> list:
                result = []
                for lv in levels:
                    if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                        price = float(lv[0])
                        size = float(lv[1])
                    elif isinstance(lv, dict):
                        p = lv.get("price") or lv.get("p") or 0
                        s = lv.get("size") or lv.get("quantity") or lv.get("amount") or 0
                        price = float(p)
                        size = float(s)
                    else:
                        continue
                    if price > 0 and size > 0:
                        result.append(DepthLevel((price, size)))
                return result

            asks = parse_levels(ob.get("asks", []))
            bids = parse_levels(ob.get("bids", []))
            mid_raw = ob.get("marketId") or ob.get("market_id") or market_id
            try:
                mid_int = int(mid_raw)
            except (ValueError, TypeError):
                mid_int = 0
            ts = ob.get("updateTimestampMs") or ob.get("update_timestamp_ms") or int(time.time() * 1000)
            return Book(market_id=mid_int, update_timestamp_ms=int(ts), asks=asks, bids=bids)
        except Exception as e:
            self.logger.warning(f"Book conversion failed: {e}")
            return None

    # ── Portfolio ─────────────────────────────────

    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance. Live mode: on-chain USDT balance at deposit address."""
        if settings.trading.paper_trading_mode:
            return {"balance": settings.trading.initial_bankroll}

        # deposit_address (smart account) → 실제 USDT 보유 주소
        balance_address = settings.api.deposit_address or settings.api.wallet_address

        # SDK balance_of_async
        if self._builder and _SDK_AVAILABLE:
            try:
                bal_wei = await self._builder.balance_of_async(
                    token="USDT",
                    address=balance_address,
                )
                return {"balance": _wei_to_usdt(bal_wei)}
            except Exception as e:
                self.logger.warning(f"balance_of_async failed: {e}")

        # web3 fallback
        if self._w3:
            try:
                abi = [{"inputs": [{"name": "account", "type": "address"}],
                        "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "view", "type": "function"}]
                contract = self._w3.eth.contract(
                    address=Web3.to_checksum_address(ADDRESSES_BY_CHAIN_ID[CHAIN_ID_BNB].USDT),
                    abi=abi,
                )
                bal_wei = contract.functions.balanceOf(
                    Web3.to_checksum_address(balance_address)
                ).call()
                return {"balance": _wei_to_usdt(bal_wei)}
            except Exception as e:
                self.logger.error(f"get_balance web3 fallback failed: {e}")

        return {"balance": 0}

    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        if settings.trading.paper_trading_mode:
            return {"positions": []}
        try:
            data = await self._request(
                "GET", "/v1/positions",
                params={"wallet": settings.api.wallet_address},
            )
            if isinstance(data, list):
                return {"positions": data}
            return {"positions": data.get("positions", [])}
        except Exception as e:
            self.logger.error(f"get_positions failed: {e}")
            return {"positions": []}

    async def get_fills(self, ticker: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        return {"fills": []}

    async def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        return {"orders": []}

    async def get_trades(self, ticker: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> Dict[str, Any]:
        return {"trades": []}

    # ── Redeem (auto_claim) ────────────────────────
    async def redeem_position(self, condition_id: str, index_set: int,
                              amount: Optional[int] = None,
                              is_neg_risk: bool = False,
                              is_yield_bearing: bool = False) -> bool:
        """Redeem resolved position → USDT 회수."""
        if not self._builder or settings.trading.paper_trading_mode:
            return False
        try:
            kwargs = {
                "condition_id": condition_id,
                "index_set": index_set,
                "is_neg_risk": is_neg_risk,
                "is_yield_bearing": is_yield_bearing,
            }
            if is_neg_risk:
                kwargs["amount"] = amount or 0
            result = await self._builder.redeem_positions_async(**kwargs)
            self.logger.info(f"Redeem result: success={result.success} conditionId={condition_id[:16]}...")
            return result.success
        except Exception as e:
            self.logger.warning(f"Redeem failed: {e}")
            return False

    async def get_market_redeem_info(self, market_id: str) -> Optional[Dict]:
        """마켓에서 redeem에 필요한 conditionId, outcomes, negRisk 정보 조회."""
        try:
            data = await self._request("GET", f"/v1/markets/{market_id}")
            if not data:
                return None
            return {
                "condition_id": data.get("conditionId"),
                "is_neg_risk": bool(data.get("isNegRisk")),
                "is_yield_bearing": bool(data.get("isYieldBearing")),
                "outcomes": data.get("outcomes", []),
            }
        except Exception as e:
            self.logger.warning(f"get_market_redeem_info failed: {e}")
            return None

    # ── Approvals ────────────────────────────────

    async def ensure_approvals(self):
        """Set ERC20/ERC1155 approvals for live trading."""
        if settings.trading.paper_trading_mode or not self._builder:
            return
        try:
            result = await self._builder.set_approvals_async()
            self.logger.info(f"Approvals set: {result}")
        except Exception as e:
            self.logger.warning(f"ensure_approvals failed: {e}")

    # ── Order execution ──────────────────────────

    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: float,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Place order — Paper simulation or Live via predict-sdk.

        Args:
            ticker: market ID
            client_order_id: unique order ID
            side: "yes" or "no"
            action: "buy" or "sell"
            count: number of contracts (shares)
            type_: "market" or "limit"
            yes_price: price in cents (0-100) for YES side
            no_price: price in cents (0-100) for NO side
        """

        if settings.trading.paper_trading_mode:
            return await self._paper_order(
                ticker, client_order_id, side, action, count, type_, yes_price, no_price
            )

        # ── LIVE ORDER via predict-sdk ──
        return await self._live_order(
            ticker, client_order_id, side, action, count, type_, yes_price, no_price
        )

    async def _paper_order(
        self, ticker, client_order_id, side, action, count, type_, yes_price, no_price
    ) -> Dict[str, Any]:
        """Paper mode: realistic simulation with orderbook, slippage, fees."""
        intended_price = (yes_price or no_price or 50) / 100.0
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
                    if is_yes and best_ask:
                        actual_price = best_ask
                    elif not is_yes and best_bid:
                        actual_price = max(1.0 - best_bid, 0.01)
                    elif best_ask:
                        actual_price = best_ask
                else:
                    if is_yes and best_bid:
                        actual_price = best_bid
                    elif not is_yes and best_ask:
                        actual_price = max(1.0 - best_ask, 0.01)
                    elif best_bid:
                        actual_price = best_bid

                if best_ask and best_bid:
                    spread = round(best_ask - best_bid, 4)
        except Exception:
            pass

        # Slippage
        base_slippage = max(0.01, spread * 0.5)
        size_slippage = min(0.03, count * 0.001)
        total_slippage = base_slippage + size_slippage

        if action.lower() == "buy":
            actual_price = min(actual_price * (1 + total_slippage), 0.99)
        else:
            actual_price = max(actual_price * (1 - total_slippage), 0.01)
        actual_price = round(actual_price, 4)

        # Fees
        fee_bps = 200
        try:
            mkt = await self.get_market(ticker=ticker)
            mkt_data = mkt.get("market", mkt) if isinstance(mkt, dict) else {}
            fee_bps = int(mkt_data.get("feeRateBps", 200))
        except Exception:
            pass

        trade_value = count * actual_price
        fee = round(trade_value * fee_bps / 10_000, 4)
        gas_fee = 0.10

        self.logger.info(
            f"[PAPER] {action.upper()} {count}x {side.upper()} @ {actual_price:.4f} "
            f"(slip={total_slippage:.1%}, fee=${fee:.4f}, gas=${gas_fee:.2f})"
        )

        return {
            "order": {
                "order_id": client_order_id,
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price": actual_price,
                "intended_price": intended_price,
                "status": "filled",
                "paper": True,
                "trade_value": round(trade_value, 4),
                "fee": fee,
                "fee_bps": fee_bps,
                "gas_fee": gas_fee,
                "total_cost": round(fee + gas_fee, 4),
                "slippage_pct": round(total_slippage * 100, 2),
                "spread": spread,
            }
        }

    async def _live_order(
        self, ticker, client_order_id, side, action, count, type_, yes_price, no_price
    ) -> Dict[str, Any]:
        """Live order via predict-sdk: sign EIP-712 typed data → POST /v1/orders."""
        if not self._live_ready or not self._builder:
            raise PredictFunAPIError(
                "Live trading not ready. Check PRIVATE_KEY, WALLET_ADDRESS, and predict-sdk installation."
            )

        is_buy = action.lower() == "buy"
        sdk_side = SdkSide.BUY if is_buy else SdkSide.SELL
        price = (yes_price or no_price or 50) / 100.0
        size_usdt = count * price

        # 1. Market info (token_id, fee_rate_bps)
        try:
            mkt_resp = await self.get_market(ticker=ticker)
            market = mkt_resp.get("market", mkt_resp) if isinstance(mkt_resp, dict) else {}
        except Exception as e:
            raise PredictFunAPIError(f"Failed to fetch market {ticker}: {e}")

        outcomes = market.get("outcomes", [])
        fee_rate_bps = int(market.get("feeRateBps") or 200)
        # decimalPrecision은 가격 표시용 소수점 자릿수 (2~3)이며, 토큰 decimals가 아님
        # BNB Chain USDT는 항상 18 decimals → DEFAULT_PRECISION 사용
        precision = DEFAULT_PRECISION

        # Resolve token_id from side
        is_yes = side.lower() in ("yes", "up", "above")
        token_id = None
        if is_yes:
            yes_outcome = next(
                (o for o in outcomes if o.get("name", "").upper() in ("YES", "UP", "ABOVE")),
                outcomes[0] if outcomes else None,
            )
            token_id = yes_outcome.get("onChainId") if yes_outcome else None
        else:
            no_outcome = next(
                (o for o in outcomes if o.get("name", "").upper() in ("NO", "DOWN", "BELOW")),
                outcomes[1] if len(outcomes) > 1 else None,
            )
            token_id = no_outcome.get("onChainId") if no_outcome else None

        if not token_id:
            # Fallback: first outcome
            if outcomes:
                token_id = outcomes[0].get("onChainId")
        if not token_id:
            raise PredictFunAPIError(f"No onChainId found for market {ticker}, outcomes: {outcomes}")

        # Sync precision (10**N 형태로 세팅 — OrderBuilder.__init__과 동일)
        try:
            self._builder._precision = 10 ** precision
        except Exception:
            pass

        # 2. Orderbook → SDK Book
        try:
            ob = await self.get_orderbook(ticker)
        except Exception as e:
            raise PredictFunAPIError(f"Orderbook fetch failed: {e}")

        book = self._build_sdk_book(ticker, ob)
        if not book:
            raise PredictFunAPIError("Failed to build SDK Book from orderbook")

        # 3. Calculate order amounts
        slippage_bps = 300  # 3% default
        try:
            if type_.lower() == "limit":
                amounts = self._builder.get_limit_order_amounts(
                    LimitHelperInput(
                        side=sdk_side,
                        price_per_share_wei=int(price * WEI),
                        quantity_wei=_usdt_to_wei(count),
                    )
                )
            elif is_buy:
                amounts = self._builder.get_market_order_amounts(
                    MarketHelperValueInput(
                        side=SdkSide.BUY,
                        value_wei=_usdt_to_wei(size_usdt),
                        slippage_bps=slippage_bps,
                    ),
                    book,
                )
            else:
                shares_wei = _usdt_to_wei(count)
                amounts = self._builder.get_market_order_amounts(
                    MarketHelperInput(
                        side=SdkSide.SELL,
                        quantity_wei=shares_wei,
                        slippage_bps=slippage_bps,
                    ),
                    book,
                )
        except Exception as e:
            raise PredictFunAPIError(f"Order amount calculation failed: {e}")

        # 4. Sign EIP-712 typed data
        is_neg_risk = bool(market.get("isNegRisk"))
        is_yield_bearing = bool(market.get("isYieldBearing"))
        try:
            order_input = BuildOrderInput(
                side=sdk_side,
                token_id=token_id,
                maker_amount=amounts.maker_amount,
                taker_amount=amounts.taker_amount,
                fee_rate_bps=fee_rate_bps,
            )
            strategy = "LIMIT" if type_.lower() == "limit" else "MARKET"
            order = self._builder.build_order(strategy=strategy, data=order_input)
            typed_data = self._builder.build_typed_data(
                order,
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
            )
            signed: SignedOrder = self._builder.sign_typed_data_order(typed_data)
        except Exception as e:
            raise PredictFunAPIError(f"Order signing failed: {e}")

        # 5. POST /v1/orders (camelCase + data wrapper)
        try:
            order_hash = self._builder.build_typed_data_hash(typed_data)
            price_per_share = str(amounts.price_per_share)
            order_payload = {
                "salt": str(signed.salt),
                "maker": signed.maker,
                "signer": signed.signer,
                "taker": signed.taker,
                "tokenId": str(signed.token_id),
                "makerAmount": str(signed.maker_amount),
                "takerAmount": str(signed.taker_amount),
                "expiration": str(signed.expiration),
                "nonce": str(signed.nonce),
                "feeRateBps": str(signed.fee_rate_bps),
                "side": signed.side.value if hasattr(signed.side, "value") else int(signed.side),
                "signatureType": signed.signature_type.value if hasattr(signed.signature_type, "value") else int(signed.signature_type),
                "signature": signed.signature,
            }
            strategy = "LIMIT" if type_.lower() == "limit" else "MARKET"
            payload = {
                "data": {
                    "order": order_payload,
                    "hash": order_hash,
                    "pricePerShare": price_per_share,
                    "strategy": strategy,
                }
            }
            if strategy == "MARKET":
                payload["data"]["slippageBps"] = str(slippage_bps)

            # 401 → JWT 재인증 후 재시도 (1회)
            for _attempt in range(2):
                response = await self.client.post(
                    f"{self.base_url}/v1/orders",
                    json=payload,
                    timeout=15.0,
                )
                if response.status_code == 401 and _attempt == 0:
                    self.logger.info("POST /v1/orders 401 → JWT 재인증...")
                    if await self._authenticate():
                        continue
                if response.status_code >= 400:
                    self.logger.error(f"[LIVE] Order error {response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                break
            result = response.json()
            self.logger.info(f"[LIVE] Order response: {result}")
        except Exception as e:
            raise PredictFunAPIError(f"POST /v1/orders failed: {e}")

        # 6. Normalize response to expected format
        order_id = (
            result.get("orderId") or result.get("order_id") or
            result.get("id") or client_order_id
        )
        status = result.get("status", "matched")

        return {
            "order": {
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price": price,
                "status": status,
                "paper": False,
                "trade_value": round(size_usdt, 4),
                "fee": round(size_usdt * fee_rate_bps / 10_000, 4),
                "gas_fee": 0.10,
            }
        }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if settings.trading.paper_trading_mode:
            return {"status": "cancelled", "order_id": order_id}
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/orders/remove",
                json={"orderId": order_id},
                timeout=10.0,
            )
            return {"status": "cancelled" if response.status_code == 200 else "failed", "order_id": order_id}
        except Exception as e:
            self.logger.error(f"cancel_order failed: {e}")
            return {"status": "failed", "order_id": order_id}

    # ── Lifecycle ────────────────────────────────

    async def close(self) -> None:
        await self.client.aclose()
        self.logger.info("Predict.fun client closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
