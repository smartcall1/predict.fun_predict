# Predict.fun AI Trading Bot

[Predict.fun](https://predict.fun) 예측 마켓에서 **Gemini 2.5 Flash**를 활용한 AI 자동 트레이딩 봇.

[ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot)을 포크하여 Predict.fun (BNB Chain) + Gemini Flash 단일 모델로 재구성.

---

> **면책 조항** — 실험적 소프트웨어이며, 투자 조언이 아닙니다. 잃어도 되는 자본으로만 트레이딩하세요.

---

## 작동 방식

```
  INGEST              DECIDE (Gemini 2.5 Flash)     EXECUTE          TRACK
 --------            ─────────────────────────     ---------       --------

  Predict.fun  ────►  Gemini 2.5 Flash              Paper/Live       PnL
  REST API            - 마켓 분석                    주문 실행        Win Rate
                      - 확률 추정                    Kelly 사이징     AI Cost
  600+ 마켓   ────►   - 엣지 계산                    수수료 반영      Telegram
  (cursor             - BUY YES / BUY NO / SKIP     슬리피지 반영    알림
   pagination)                                       가스비 반영
```

### 4단계 파이프라인

1. **수집 (Ingest)** — Predict.fun API에서 활성 마켓 수집 (cursor 기반 페이지네이션, 600+ 마켓)
2. **분석 (Decide)** — Gemini 2.5 Flash가 각 마켓의 진짜 확률을 추정, 마켓 가격과 비교하여 엣지 계산
3. **실행 (Execute)** — 엣지가 5% 이상이고 확신도 60% 이상이면 진입. Kelly Criterion으로 포지션 사이징
4. **추적 (Track)** — 모든 시그널/정산을 SQLite에 기록. 텔레그램으로 실시간 알림

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

`.env` 파일에 아래 키 입력:

| 변수 | 설명 | 발급처 |
|------|------|--------|
| `PREDICT_API_KEY` | Predict.fun API 키 | [Discord](https://discord.gg/predictdotfun) |
| `GEMINI_API_KEY` | Gemini API 키 | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | 텔레그램 채팅 ID | [@userinfobot](https://t.me/userinfobot) |

### 3. 실행

```bash
# 페이퍼 트레이딩 (1회 스캔)
python paper_trader.py

# 연속 스캔 (15분 간격)
python paper_trader.py --loop

# 정산 확인
python paper_trader.py --settle

# 통계 조회
python paper_trader.py --stats

# 대시보드 생성
python paper_trader.py --dashboard
```

---

## Termux (Android) 실행

```bash
pkg install python git
git clone https://github.com/smartcall1/predict.fun_predict.git
cd predict.fun_predict
pip install -r requirements.txt
cp env.template .env
# nano .env  <- API 키 입력
python paper_trader.py --loop
```

> numpy/scipy/pandas는 선택 사항. 핵심 파이프라인은 이 라이브러리 없이 동작.

---

## 텔레그램 알림

| 이벤트 | 알림 내용 |
|--------|----------|
| BUY 시그널 | 마켓명, YES/NO, 진입가, 확신도, 엣지, AI 분석 근거 |
| 정산 | WIN/LOSS, 진입가 -> 정산가, PnL |
| 스캔 완료 | 시그널 수, 스킵 수, AI 비용 |
| 일일 요약 | 승률, 총 PnL, AI 비용, 순수익 |
| 에러 | 에러 메시지 |

---

## 페이퍼 트레이딩 현실성

실제 거래와 90%+ 유사도를 목표로 다음 비용을 시뮬레이션:

| 비용 항목 | 적용 방식 | 비율 |
|----------|----------|------|
| 체결가 | 실제 오더북 best ask/bid 기반 | 실시간 |
| 슬리피지 | 스프레드 50% + 수량 비례 (min 1%) | 1~4% |
| 거래 수수료 | 마켓별 `feeRateBps` API 조회 | 기본 2% |
| 가스비 | BNB Chain 고정 | $0.10/건 |
| 스프레드 | bid-ask 차이 자동 반영 | 실시간 |

---

## 설정

### .env 주요 파라미터

```bash
# 스캔 간격 (초, 기본 300 = 5분)
SCAN_INTERVAL=300

# 최대 동시 포지션
MAX_POSITIONS=15

# Kelly Criterion 분수 (0.25 = quarter-Kelly, 보수적)
KELLY_FRACTION=0.25

# 최소 AI 확신도 (이 이하면 스킵)
MIN_CONFIDENCE=0.60

# 최소 엣지 (AI확률 - 마켓가격 차이)
MIN_EDGE=0.05

# 일일 AI 비용 한도 (USD, 안전장치)
DAILY_AI_COST_LIMIT=5.0

# Gemini 모델 (기본값: 2.5 Flash)
GEMINI_MODEL=gemini-2.5-flash-preview-05-20
```

---

## AI 비용

Gemini 2.5 Flash는 매우 저렴:

| 시나리오 | 마켓 수 | 일일 분석 | 예상 비용 |
|---------|--------|----------|----------|
| 보수적 | 100개 | ~400회 | ~$0.07 |
| 표준 | 600개 | ~1,200회 | ~$0.20 |
| 공격적 | 600개 | ~5,000회 | ~$0.85 |

`DAILY_AI_COST_LIMIT`에 도달하면 그날 남은 시간 동안 AI 분석을 자동 중단. 다음날 0시에 리셋.

---

## 프로젝트 구조

```
predict.fun_predict/
├── paper_trader.py              # 메인 엔트리포인트 (페이퍼 트레이딩)
├── beast_mode_bot.py            # 고급 멀티전략 봇
├── cli.py                       # 통합 CLI
├── env.template                 # 환경변수 템플릿
├── requirements.txt             # 의존성 (Termux 호환)
│
├── src/
│   ├── clients/
│   │   ├── predictfun_client.py # Predict.fun REST API 클라이언트
│   │   ├── gemini_client.py     # Gemini 2.5 Flash AI 클라이언트
│   │   ├── kalshi_client.py     # 호환 심 → PredictFunClient
│   │   ├── xai_client.py        # 호환 심 → GeminiClient
│   │   └── model_router.py      # 단일 모델 라우터
│   │
│   ├── config/
│   │   └── settings.py          # 전체 설정 관리
│   │
│   ├── jobs/
│   │   ├── ingest.py            # 마켓 수집 (Predict.fun)
│   │   ├── decide.py            # AI 분석 및 결정
│   │   ├── execute.py           # 주문 실행
│   │   └── track.py             # 포지션 추적 및 정산
│   │
│   ├── strategies/              # 트레이딩 전략 (포트폴리오, 마켓메이킹 등)
│   ├── paper/                   # 페이퍼 트레이딩 DB + 대시보드
│   └── utils/
│       ├── telegram.py          # 텔레그램 알림
│       ├── database.py          # SQLite 데이터베이스
│       └── ...
│
├── docs/                        # 대시보드 HTML
└── tests/                       # 테스트
```

---

## 원본 프로젝트

[ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) (MIT License)

### 원본 대비 변경 사항

| 항목 | 원본 (Kalshi) | 본 프로젝트 (Predict.fun) |
|------|-------------|------------------------|
| 마켓 | Kalshi (미국 전용) | Predict.fun (BNB Chain, 글로벌) |
| AI 모델 | 5모델 앙상블 (Grok, Claude, GPT-4o, Gemini, DeepSeek) | Gemini 2.5 Flash 단일 |
| AI 비용 | ~$10-15/일 | ~$0.10-0.50/일 |
| 인증 | RSA PSS 서명 | API Key |
| 통화 | USD (cents) | USDT (BNB Chain) |
| 알림 | 없음 | 텔레그램 실시간 알림 |

---

## 라이선스

MIT License. [LICENSE](LICENSE) 참조.

