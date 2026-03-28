"""
Configuration settings for the Predict.fun AI trading system.
Forked from kalshi-ai-trading-bot → Predict.fun + Gemini Flash.

Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """API configuration — Predict.fun + Gemini Flash."""
    # Predict.fun
    predict_api_key: str = field(default_factory=lambda: os.getenv("PREDICT_API_KEY", ""))
    predict_base_url: str = "https://api.predict.fun"
    predict_ws_url: str = "wss://ws.predict.fun/ws"

    # BNB Chain (LIVE 전용)
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("WALLET_ADDRESS", ""))
    bsc_rpc_url: str = field(default_factory=lambda: os.getenv("BSC_RPC_URL", "https://bsc-dataseed1.binance.org"))

    # Gemini (단일 모델)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))


@dataclass
class TradingConfig:
    """Trading strategy configuration — disciplined defaults."""
    # Position sizing and risk management
    max_position_size_pct: float = 3.0      # 포트폴리오 대비 최대 포지션 크기
    max_daily_loss_pct: float = 10.0        # 일일 최대 손실률
    max_positions: int = 15                 # 동시 최대 포지션
    min_balance: float = 50.0               # 최소 잔고

    # Market filtering
    min_volume: float = 0.0                 # Predict.fun API는 목록에서 volume 미제공 → 0으로 설정
    max_time_to_expiry_days: int = 30       # 최대 만기일

    # AI decision thresholds
    min_confidence_to_trade: float = 0.60   # 최소 AI 확신도
    min_edge: float = 0.05                  # 최소 엣지 (AI확률 - 마켓가격)
    scan_interval_seconds: int = 300        # 스캔 간격 (5분)

    # Gemini model config
    ai_temperature: float = 0.1            # 낮은 temperature → 일관된 출력
    ai_max_tokens: int = 1024

    # Position sizing — Kelly Criterion
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25            # quarter-Kelly (보수적)
    max_single_position: float = 0.05       # 5% 최대 포지션
    default_position_size: float = 3.0      # 기본 포지션 크기 (%)
    position_size_multiplier: float = 1.0

    # Live/Paper mode
    live_trading_enabled: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true")
    paper_trading_mode: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true")
    initial_bankroll: float = field(default_factory=lambda: float(os.getenv("INITIAL_BANKROLL", "1000.0")))

    # Trading frequency
    market_scan_interval: int = 300         # 5분마다 스캔
    position_check_interval: int = 60       # 1분마다 포지션 체크
    max_trades_per_hour: int = 10
    run_interval_minutes: int = 5

    # Category preferences (빈 리스트 = 모든 카테고리)
    preferred_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])

    # AI cost control
    daily_ai_budget: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "5.0")))
    daily_ai_cost_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "5.0")))
    enable_daily_cost_limiting: bool = True
    max_ai_cost_per_decision: float = 0.01  # Gemini Flash는 매우 저렴
    analysis_cooldown_hours: int = 3
    max_analyses_per_market_per_day: int = 4
    sleep_when_limit_reached: bool = True

    # Enhanced filtering
    min_volume_for_ai_analysis: float = 0.0
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [])

    # News/sentiment (Gemini handles inline — no separate search needed)
    skip_news_for_low_volume: bool = True
    news_search_volume_threshold: float = 500.0

    # High-confidence near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.90
    high_confidence_market_odds: float = 0.85
    high_confidence_expiry_hours: int = 24

    # Category confidence adjustments
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,
        "politics": 1.05,
        "economics": 1.15,
        "crypto": 1.00,
        "default": 1.0,
    })

    # Processor workers
    num_processor_workers: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# === RISK MANAGEMENT ===
max_volatility: float = 0.40
max_correlation: float = 0.70
max_drawdown: float = 0.15              # 15% 최대 드로다운
max_sector_exposure: float = 0.50       # 50% 섹터 집중 한도

# === PERFORMANCE TARGETS ===
target_sharpe: float = 0.3
target_return: float = 0.15
min_trade_edge: float = 0.05            # 5% 최소 엣지
min_confidence_for_large_size: float = 0.70

# === EXIT STRATEGIES ===
use_dynamic_exits: bool = True
profit_threshold: float = 0.20          # 20% 익절
loss_threshold: float = 0.15            # 15% 손절
confidence_decay_threshold: float = 0.25
max_hold_time_hours: int = 240          # 10일
volatility_adjustment: bool = True

# === MARKET SELECTION ===
min_volume_for_analysis: float = 100.0
min_price_movement: float = 0.02
max_bid_ask_spread: float = 0.20

# === SYSTEM ===
beast_mode_enabled: bool = False        # 기본 비활성화 (보수적 모드)
fallback_to_legacy: bool = True
log_level: str = "INFO"
performance_monitoring: bool = True


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.predict_api_key:
            raise ValueError("PREDICT_API_KEY environment variable is required")

        if not self.api.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")

        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")

        return True


# Global settings instance
settings = Settings()

# Validate on import (warn, don't crash)
try:
    settings.validate()
except ValueError as e:
    print(f"[WARN] Configuration: {e}")
    print("Please check your .env file. Copy env.template -> .env and fill in keys.")
