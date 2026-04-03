# Predict.fun AI Trading Bot

[Predict.fun](https://predict.fun) 예측 마켓에서 **Gemini 2.5 Flash** AI를 활용한 자동 트레이딩 봇.

BNB Chain 기반 예측 마켓에서 AI가 독자적으로 확률을 추정하고, 엣지가 있는 마켓에 자동 진입/청산하는 시스템.

> **면책 조항** — 실험적 소프트웨어이며, 투자 조언이 아닙니다. 잃어도 되는 자본으로만 트레이딩하세요.

---

## 핵심 기능

- **AI 자율 분석** — Gemini 2.5 Flash가 600+ 마켓을 스캔, 확률 추정 및 엣지 계산
- **LIVE 거래** — predict-sdk EIP-712 서명 → BNB Chain 온체인 주문 실행
- **적응형 SL/TP** — AI confidence 기반 손절(5~10%) / 익절(15~30%) 자동 조절
- **Auto Redeem** — 정산된 포지션 USDT 자동 회수 (2시간 주기)
- **텔레그램 UI** — 실시간 알림 + 인터랙티브 버튼 (Status/Trades/Positions/Stats/Logs/Stop)
- **Watchdog** — 크래시 시 자동 재시작

---

## 작동 방식

```
  INGEST              DECIDE (Gemini 2.5 Flash)     EXECUTE          SETTLE
 --------            ─────────────────────────     ---------       --------

  Predict.fun  ────►  Gemini 2.5 Flash              predict-sdk      적응형 SL/TP
  REST API            - Bull/Bear 분석               EIP-712 서명     (5~10% SL)
                      - 확률 추정                    온체인 주문       (15~30% TP)
  600+ 마켓   ────►   - 엣지 계산 (EV≥5%)           슬리피지 반영     시간 만료 72h
  (cursor             - BUY YES / BUY NO / SKIP     수수료 반영       Auto Redeem
   pagination)                                                        텔레그램 알림
```

### 파이프라인

1. **수집 (Ingest)** — Predict.fun API에서 활성 마켓 수집 (cursor 기반 페이지네이션, 600+ 마켓, 볼륨 $500+ 필터)
2. **분석 (Decide)** — Gemini 2.5 Flash AI가 마켓별 확률 추정, EV ≥ 0.05 + confidence ≥ 0.60 시 BUY
3. **실행 (Execute)** — predict-sdk로 EIP-712 서명 → 온체인 주문. Kelly Criterion 포지션 사이징
4. **정산 (Settle)** — 다중 우선순위 정산: 마켓 종료 → 적응형 SL/TP → 시간 만료 → 비상 SL
5. **회수 (Redeem)** — `auto_redeem.py`가 2시간마다 resolved 포지션 USDT 자동 회수

---

## 정산 로직 (적응형 SL/TP)

| 우선순위 | 조건 | 동작 |
|---------|------|------|
| P0 | 마켓 종료 (CLOSED/RESOLVED) | WIN/LOSS 자연 정산 |
| P0-V | 종료 후 resolution 없음 6사이클 | VOID (원금 환불) |
| P1-A | 현재가 ≥ 0.98 또는 ≤ 0.02 | 조기 청산 |
| P2 | **Stop Loss 5~10%** (confidence 기반) | 손절 |
| P3 | **Take Profit 15~30%** (confidence 기반) | 익절 |
| P4 | 보유 시간 ≥ **72시간** | 시간 만료 청산 |
| P5 | 비상 SL 10% (레거시 포지션) | 비상 손절 |

**Confidence별 SL/TP:**
| AI Confidence | Stop Loss | Take Profit |
|--------------|-----------|-------------|
| ≥ 0.80 (고) | 5% | 30% |
| ≥ 0.60 (중) | 7% | 20% |
| < 0.60 (저) | 10% | 15% |

---

## 빠른 시작

### 1. 클론 및 설치

```bash
git clone https://github.com/smartcall1/predict.fun_predict.git
cd predict.fun_predict
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp env.template .env
```

`.env` 필수 키:

| 변수 | 설명 | 발급처 |
|------|------|--------|
| `PREDICT_API_KEY` | Predict.fun API 키 | [Discord](https://discord.gg/predictdotfun) |
| `GEMINI_API_KEY` | Gemini API 키 | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `PRIVATE_KEY` | BNB Chain 지갑 Private Key | MetaMask 등 |
| `WALLET_ADDRESS` | EOA 지갑 주소 | MetaMask 등 |
| `DEPOSIT_ADDRESS` | Predict.fun Deposit 주소 (Smart Account) | Predict.fun 웹사이트 |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | 텔레그램 채팅 ID | [@userinfobot](https://t.me/userinfobot) |

### 3. 실행

```bash
# 페이퍼 트레이딩 (1회 스캔)
python paper_trader.py

# 페이퍼 연속 스캔 (15분 간격)
python paper_trader.py --loop

# LIVE 트레이딩 (실제 자금)
python run_live.py --live

# LIVE 1회 스캔
python run_live.py --live --once

# Auto Redeem (2시간 주기, 별도 프로세스)
python auto_redeem.py

# Auto Redeem 1회 실행
python auto_redeem.py --once
```

---

## Termux (Android VPS) 실행

```bash
pkg install python git
git clone https://github.com/smartcall1/predict.fun_predict.git
cd predict.fun_predict
pip install -r requirements.txt
# .env 파일 설정 후:

# Screen 1: 메인 봇
python run_live.py --live

# Screen 2: Auto Redeem
python auto_redeem.py
```

---

## 텔레그램 버튼

| 버튼 | 기능 |
|------|------|
| 📊 Status | 포트폴리오, 잔고, 승률, ROI |
| 📋 Trades | 최근 거래 내역 (WIN/LOSS/SELL) |
| 📌 Positions | 활성 포지션 + 미실현 손익 |
| 💰 Stats | 성과 통계 (승/패/승률/PnL) |
| 📄 Logs | 최근 봇 로그 (tail 30줄) |
| ⏹ Stop | 봇 안전 종료 (확인 필요) |

### 텔레그램 알림

| 이벤트 | 알림 내용 |
|--------|----------|
| BUY 시그널 | 마켓명, YES/NO, 진입가, 확신도, 엣지, AI 분석 근거 |
| 정산 | WIN/LOSS/SELL, 진입가 → 정산가, PnL, 사유 |
| Redeem | 성공/실패 건수, USDT 변동액 |

---

## 설정

### .env 주요 파라미터

```bash
# 스캔 간격 (초, 기본 300 = 5분)
SCAN_INTERVAL=300

# 최대 동시 포지션
MAX_POSITIONS=15

# Kelly Criterion 분수 (0.25 = quarter-Kelly)
KELLY_FRACTION=0.25

# 최소 AI 확신도
MIN_CONFIDENCE=0.60

# 최소 엣지 (AI확률 - 마켓가격)
MIN_EDGE=0.05

# 일일 AI 비용 한도 (USD)
DAILY_AI_COST_LIMIT=5.0

# 분석 쿨다운 (시간, 같은 마켓 재분석 방지)
ANALYSIS_COOLDOWN_HOURS=3

# Gemini 모델
GEMINI_MODEL=gemini-2.5-flash
```

---

## AI 비용

Gemini 2.5 Flash 기준:

| 시나리오 | 마켓 수 | 일일 분석 | 예상 비용 |
|---------|--------|----------|----------|
| 보수적 | 100개 | ~400회 | ~$0.07 |
| 표준 | 600개 | ~1,200회 | ~$0.20 |
| 공격적 | 600개 | ~5,000회 | ~$0.85 |

`DAILY_AI_COST_LIMIT`에 도달하면 당일 AI 분석 자동 중단.

---

## 프로젝트 구조

```
predict.fun_predict/
├── run_live.py                  # LIVE 트레이딩 엔트리포인트 (Watchdog)
├── live_trader.py               # LIVE 메인 트레이더 (스캔→분석→실행→정산)
├── auto_redeem.py               # 독립 리딤 프로세스 (2시간 주기)
├── paper_trader.py              # 페이퍼 트레이딩
├── env.template                 # 환경변수 템플릿
├── requirements.txt             # 의존성
│
├── src/
│   ├── clients/
│   │   ├── predictfun_client.py # Predict.fun REST API + predict-sdk 주문 실행
│   │   └── gemini_client.py     # Gemini 2.5 Flash AI 클라이언트
│   │
│   ├── config/
│   │   └── settings.py          # 전체 설정 관리 (@dataclass)
│   │
│   ├── jobs/
│   │   ├── ingest.py            # 마켓 수집 (cursor pagination, 600+)
│   │   └── decide.py            # AI 분석 및 결정 (EV/confidence 기반)
│   │
│   ├── live/
│   │   ├── executor.py          # 주문 실행기 (BUY/SELL)
│   │   ├── settler.py           # 적응형 SL/TP 정산 엔진
│   │   ├── state.py             # 포지션/잔고 상태 관리
│   │   └── telegram_ui.py       # 텔레그램 인터랙티브 UI
│   │
│   ├── agents/                  # AI 에이전트 (Bull/Bear Researcher, Ensemble)
│   ├── strategies/              # 트레이딩 전략 (포트폴리오, 마켓메이킹)
│   ├── utils/
│   │   ├── stop_loss_calculator.py  # 적응형 SL/TP 계산기
│   │   ├── database.py          # SQLite 데이터베이스
│   │   └── telegram.py          # 텔레그램 알림
│   └── paper/                   # 페이퍼 트레이딩 DB + 대시보드
│
├── data/                        # 분석 결과 JSONL
├── logs/                        # 봇 로그
└── tests/                       # 테스트
```

---

## 원본 프로젝트

[ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) (MIT License)를 포크하여 재구성.

| 항목 | 원본 (Kalshi) | 본 프로젝트 (Predict.fun) |
|------|-------------|------------------------|
| 마켓 | Kalshi (미국 전용) | Predict.fun (BNB Chain, 글로벌) |
| AI 모델 | 5모델 앙상블 | Gemini 2.5 Flash 단일 |
| AI 비용 | ~$10-15/일 | ~$0.10-0.50/일 |
| 거래 실행 | Kalshi REST API | predict-sdk (EIP-712 온체인) |
| 통화 | USD (cents) | USDT (BNB Chain, 18 decimals) |
| 손절/익절 | 고정값 | 적응형 SL/TP (confidence 기반) |
| 정산 회수 | 자동 | Auto Redeem (독립 프로세스) |
| 알림 | 없음 | 텔레그램 인터랙티브 UI |

---

## 라이선스

MIT License. [LICENSE](LICENSE) 참조.
