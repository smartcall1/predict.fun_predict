"""
auto_redeem.py — 독립 리딤 프로세스 (Predict.fun)

polymarket auto_redeem.py 패턴 이식.
predict-sdk의 redeem_positions_async 사용.

동작:
  2시간마다 Predict.fun API에서 resolved 포지션 조회
  conditionId 기반 온체인 redeem → USDT 회수
  사이클 종료 후 텔레그램 통합 요약 1회 전송

실행:
  python auto_redeem.py          # 2시간 주기 루프
  python auto_redeem.py --once   # 1회 실행 후 종료
"""

import sys
import os
import time
import asyncio
import logging
import requests

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import settings

try:
    from predict_sdk import (
        OrderBuilder, ChainId,
        ADDRESSES_BY_CHAIN_ID, RPC_URLS_BY_CHAIN_ID,
        generate_order_salt,
    )
    from predict_sdk.order_builder import make_contracts
    from eth_account import Account
    from web3 import Web3
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REDEEM] %(levelname)s %(message)s",
)
logger = logging.getLogger("auto_redeem")

# ── 상수 ────────────────────────────────────────────────────────────────────
CHAIN_ID_BNB = ChainId.BNB_MAINNET if SDK_AVAILABLE else None
POLL_INTERVAL = 7200  # 2시간
DEFAULT_PRECISION = 18  # BNB Chain USDT = 18 decimals


# ── 텔레그램 ────────────────────────────────────────────────────────────────
def tg_send(msg: str):
    token = settings.api.telegram_bot_token
    chat = settings.api.telegram_chat_id
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ── Redeemer ────────────────────────────────────────────────────────────────
class PredictRedeemer:
    def __init__(self):
        if not SDK_AVAILABLE:
            raise RuntimeError("predict-sdk not installed")
        if not settings.api.private_key:
            raise RuntimeError("PRIVATE_KEY not set")

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

        self._signer = Account.from_key(settings.api.private_key)
        addresses = ADDRESSES_BY_CHAIN_ID[CHAIN_ID_BNB]
        contracts = make_contracts(self._w3, addresses, self._signer)

        self._builder = OrderBuilder(
            chain_id=CHAIN_ID_BNB,
            precision=DEFAULT_PRECISION,
            addresses=addresses,
            generate_salt_fn=generate_order_salt,
            logger=logging.getLogger("predict_sdk"),
            signer=self._signer,
            predict_account=settings.api.wallet_address,
            contracts=contracts,
            web3=self._w3,
        )

        self._base_url = settings.api.predict_base_url
        self._api_key = settings.api.predict_api_key
        self._deposit_addr = settings.api.deposit_address or settings.api.wallet_address

        logger.info(
            "PredictRedeemer initialized | wallet=%s | deposit=%s",
            settings.api.wallet_address[:10], self._deposit_addr[:10],
        )

    def _api_get(self, endpoint: str, params: dict = None) -> dict:
        headers = {"x-api-key": self._api_key} if self._api_key else {}
        r = requests.get(f"{self._base_url}{endpoint}", params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()

    async def get_usdt_balance(self) -> float:
        """온체인 USDT 잔고 조회 (deposit address)."""
        try:
            bal_wei = await self._builder.balance_of_async(token="USDT", address=self._deposit_addr)
            return bal_wei / 10**18
        except Exception:
            return 0.0

    def get_resolved_positions(self) -> list:
        """Predict.fun API에서 resolved/settled 마켓의 포지션 조회."""
        try:
            # 내 포지션 조회
            data = self._api_get("/v1/positions", {"wallet": settings.api.wallet_address})
            positions = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(positions, list):
                return []

            resolved = []
            for pos in positions:
                market = pos.get("market", {})
                outcome = pos.get("outcome", {})
                status = str(market.get("status", "")).upper()
                trading_status = str(market.get("tradingStatus", "")).upper()

                # resolved/settled 마켓만
                if status in ("RESOLVED", "SETTLED") or trading_status in ("CLOSED", "HALTED"):
                    resolved.append({
                        "market_id": str(market.get("id", "")),
                        "title": market.get("title", "?")[:60],
                        "condition_id": market.get("conditionId"),
                        "is_neg_risk": bool(market.get("isNegRisk")),
                        "is_yield_bearing": bool(market.get("isYieldBearing")),
                        "outcome_name": outcome.get("name", ""),
                        "outcome_status": str(outcome.get("status", "")).upper(),
                        "index_set": outcome.get("indexSet", 1),
                        "amount_raw": pos.get("amount", "0"),
                    })
            return resolved
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    async def redeem_one(self, pos: dict) -> str:
        """단일 포지션 리딤. Returns: 'success' | 'fail' | 'skip'."""
        title = pos["title"]
        condition_id = pos.get("condition_id")
        if not condition_id:
            return "skip"

        # WON 포지션만 redeem (LOST는 잔액 0이라 스킵)
        if pos.get("outcome_status") == "LOST":
            logger.info("SKIP %s (LOST position)", title)
            return "skip"

        index_set = pos.get("index_set", 1)
        is_neg_risk = pos.get("is_neg_risk", False)
        is_yield_bearing = pos.get("is_yield_bearing", False)

        # amount for neg_risk
        amount = None
        if is_neg_risk:
            try:
                amount = int(pos.get("amount_raw", "0"))
            except (ValueError, TypeError):
                amount = 0

        try:
            result = await self._builder.redeem_positions_async(
                condition_id=condition_id,
                index_set=index_set,
                amount=amount,
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
            )
            if result.success:
                logger.info("SUCCESS %s (conditionId=%s...)", title, condition_id[:16])
                return "success"
            else:
                logger.warning("FAIL %s (TX reverted)", title)
                return "fail"
        except Exception as e:
            err = str(e).lower()
            if "payout" in err or "not resolved" in err or "revert" in err:
                logger.info("SKIP %s (not yet resolved: %s)", title, str(e)[:80])
                return "skip"
            logger.warning("FAIL %s: %s", title, str(e)[:100])
            return "fail"

    async def redeem_all(self) -> dict:
        """전체 리딤 사이클."""
        usdt_before = await self.get_usdt_balance()
        positions = self.get_resolved_positions()
        logger.info("Resolved positions: %d | USDT: $%.2f", len(positions), usdt_before)

        if not positions:
            return {"success": 0, "fail": 0, "skip": 0, "usdt_before": usdt_before, "usdt_after": usdt_before}

        success_n = 0
        fail_n = 0
        skip_n = 0
        win_details = []
        fail_details = []

        for pos in positions:
            status = await self.redeem_one(pos)
            if status == "success":
                success_n += 1
                win_details.append(pos["title"])
            elif status == "fail":
                fail_n += 1
                fail_details.append(pos["title"])
            else:
                skip_n += 1
            await asyncio.sleep(2)

        await asyncio.sleep(5)
        usdt_after = await self.get_usdt_balance()
        gained = usdt_after - usdt_before

        logger.info(
            "Redeem done: %d success, %d fail, %d skip | USDT +$%.2f",
            success_n, fail_n, skip_n, gained,
        )

        return {
            "success": success_n,
            "fail": fail_n,
            "skip": skip_n,
            "usdt_before": usdt_before,
            "usdt_after": usdt_after,
            "gained": gained,
            "win_details": win_details,
            "fail_details": fail_details,
        }


# ── 메인 루프 ───────────────────────────────────────────────────────────────
async def run_async():
    once = "--once" in sys.argv
    redeemer = PredictRedeemer()

    while True:
        logger.info("── 사이클 시작 ──")
        try:
            result = await redeemer.redeem_all()
            claimable = result["success"] + result["fail"]

            if claimable > 0:
                lines = [f"💰 <b>[REDEEM 결과]</b> 성공 {result['success']}/{claimable}건"]

                if result["win_details"]:
                    lines.append(f"\n✅ <b>리딤 성공 {len(result['win_details'])}건</b>")
                    for t in result["win_details"]:
                        lines.append(f"  • {t}")

                if result["fail_details"]:
                    lines.append(f"\n❌ <b>리딤 실패 {len(result['fail_details'])}건</b>")
                    for t in result["fail_details"]:
                        lines.append(f"  • {t}")

                lines.append(
                    f"\n💵 USDT: ${result['usdt_before']:.2f} → ${result['usdt_after']:.2f} "
                    f"({result['gained']:+.2f})"
                )
                tg_send("\n".join(lines))

        except Exception as e:
            logger.error("사이클 에러: %s", e)
            tg_send(f"❌ <b>[REDEEM 에러]</b> {str(e)[:200]}")

        if once:
            break
        logger.info("다음 실행까지 %d분 대기...", POLL_INTERVAL // 60)
        await asyncio.sleep(POLL_INTERVAL)


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
