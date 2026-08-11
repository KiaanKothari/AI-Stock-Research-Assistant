# [RELEASE V 1.3] 📈 AI Stock Research Assistant 

A single-file Streamlit application for researching stocks: technical and fundamental analysis, rule-based AI scoring, portfolio and watchlist tracking, backtesting, and a built-in Education Mode for teaching financial literacy — all in one interactive dashboard.


---

## Overview

AI Stock Research Assistant pulls live market data from Yahoo Finance and turns it into a single research dashboard: candlestick charts with technical overlays, financial statement breakdowns, rule-based investment scoring, a portfolio/watchlist tracker, price alerts, peer benchmarking, a simple backtester, and a dedicated Education Mode for students and teachers.

It was built to answer a practical question: *what would it take to combine live financial data, quantitative analysis, and interactive visualization into a single tool that's also useful for teaching people how to read that data?* Rather than treating "analysis" and "education" as separate products, the app layers a classroom mode directly on top of the same live data the research dashboard uses.

**How the pieces fit together:**
- **Data** — live prices, company fundamentals, financial statements, and news are retrieved from Yahoo Finance via `yfinance`.
- **Calculations** — technical indicators (SMA, RSI, MACD, Bollinger Bands, support/resistance), valuation models (Graham Number, DCF), and a backtesting engine are computed locally with `pandas`/`numpy`.
- **Rule-based scoring** — four composite scores (AI Investment, Buffett, Graham, Risk) are generated from deterministic, weighted checklists over the fetched fundamentals — see [AI-Powered Research](#ai-powered-research) for exactly what that means.
- **Visualization** — every chart is rendered with Plotly, styled consistently across the app.

---

## Why I Built This

I'm a student interested in the overlap between artificial intelligence, finance, and software engineering, and I wanted a project that would force me to get hands-on with all three at once rather than treating them separately.

Most beginner finance tools either show raw numbers with no context, or wrap everything in a black-box "AI verdict" with no way to see the reasoning. I wanted to build something in between: a dashboard where every score is traceable back to the specific metrics that produced it, and where someone without a finance background — a classmate, a younger student — could actually learn what P/E ratio, ROE, or a Bollinger Band means while looking at real, live data instead of a textbook example.

That's also why Education Mode exists as a first-class part of the app rather than a bolt-on: I think the most interesting version of a "learning tool" is one that uses production data and production-quality analysis, not a simplified toy dataset.

---

## Features

### Market Data
- Live price, daily change, and % change for any valid ticker
- Company profile, sector/industry classification, employee count, exchange, and currency
- Historical OHLCV price data across configurable periods (1mo–max) and intervals (daily/weekly/monthly)

### Financial Analysis
- Income statement, balance sheet, and cash flow statement browsing
- Revenue and EPS (estimate vs. reported) trend charts
- Key ratios: P/E, forward P/E, PEG, price/book, price/sales, EV/EBITDA, ROE, ROA, margins, debt/equity, current ratio
- **Graham Number** and a simplified **Discounted Cash Flow (DCF)** intrinsic value estimate, each compared against the current price for a margin-of-safety read
- Analyst recommendation history and price targets

### Technical Analysis
- Interactive candlestick chart with SMA 20/50/200 and Bollinger Band overlays
- RSI (14) and MACD (12, 26, 9) indicator charts
- Automatic support & resistance level detection
- Volume chart, colored by up/down day

### AI Analysis
- **AI Investment Score** — a weighted composite (0–100) across valuation, quality, momentum, and financial-health pillars, with a plain-English breakdown of what drove the score
- **Buffett Score** and **Graham Score** — checklist-style scores modeled on each investor's publicly known criteria
- **Risk Score** — volatility- and leverage-aware risk read
- All four scores are shown as gauges with their contributing factors listed out, not just a single number

### Backtesting
- Configurable SMA-crossover strategy backtest (adjustable short/long windows and starting capital)
- Strategy equity curve plotted against buy-and-hold
- Summary stats: total return, win rate, number of trades, max drawdown
- Full trade log (entry/exit price and date, return per trade)

### Portfolio & Alerts
- Manual portfolio tracker (shares + cost basis) with live P/L
- Watchlist with live price/market-cap snapshots
- **Price alerts** — set an above/below target price for any ticker; alerts are evaluated against live prices each session

### Peer & Sector Comparison
- Curated peer groups for major tickers (auto-suggested), with manual peer entry for any other ticker
- Side-by-side benchmarking table: market cap, revenue growth, profit margin, P/E, ROE
- Selected stock vs. peer-average deltas

### News & Sentiment
- Latest company news with headline, publisher, and thumbnail
- Lightweight keyword-based sentiment scoring per headline (Positive/Neutral/Negative), with an aggregate breakdown chart
- *(See [AI-Powered Research](#ai-powered-research) — this is a deterministic lexicon scan, not a language model.)*

### Dividend Tracking
- Full historical dividend payment series with a payment-history chart
- Annual dividend, dividend yield, payout ratio, and computed dividend growth rate (CAGR)

### Education Mode
See the [dedicated section below](#education-mode) — this is the app's other major differentiator.

### Visualization
- Every chart (candlestick, RSI, MACD, Bollinger, revenue, earnings, comparison, gauges, peer bars, sentiment, backtest equity curve, dividend history) is built with Plotly and shares a consistent dark/light-aware color system

### User Experience
- **Dark/Light theme toggle**, applied live via CSS
- **Multi-currency display** — convert Market Cap, Revenue, and other dollar figures into EUR/GBP/JPY/INR/CAD/AUD using a live FX rate (native-currency data is preserved alongside the converted figure)
- Compare-stocks view with normalized (% return) performance charts across multiple tickers
- PDF research report export and CSV export (price history and key metrics)

---

## AI-Powered Research

It's worth being precise about what "AI" means in this project, since that term gets overloaded:

| Layer | What it is | Example in this app |
|---|---|---|
| **Data retrieval** | Raw facts pulled from Yahoo Finance, unmodified | Current price, P/E ratio, revenue, dividend history |
| **Calculations** | Deterministic math performed on that data | RSI, MACD, SMA crossover backtest, Graham Number, DCF |
| **Rule-based scoring** | Fixed, human-written weighting logic applied to the data — not a trained model | AI Investment Score, Buffett Score, Graham Score, Risk Score |
| **Keyword sentiment** | A fixed positive/negative word lexicon scanning headline text | News Sentiment breakdown |

**None of the "AI" scores in this app are a machine-learning model, and none of them predict future stock prices.** The "AI Investment Score" is a descriptive label for a transparent, weighted checklist (valuation + quality + momentum + financial health), the same way the Buffett and Graham scores encode each investor's publicly known criteria as a checklist. Every score comes with a breakdown listing exactly which factors contributed to it, so the reasoning is inspectable rather than opaque. The sentiment analysis in the News tab is a simple keyword scan, not a language model — it's explicitly labeled as such in the app.

The goal is realistic, explainable analysis — not a black box that claims predictive certainty it doesn't have.

---

## Education Mode

Education Mode turns the same live dashboard into a classroom tool, without needing a second app or a simplified dataset.

**For students:**
- A dedicated **Learn** tab with a plain-language company profile, business model, products, competitors, industry, competitive advantages, and risks — auto-generated from live data for any ticker, with curated write-ups for several major companies
- Expandable, plain-English explanations for financial metrics wherever they appear (definition, why investors care, a worked example, a real-world analogy, and a common misconception), covering all major metrics from Market Cap to EBITDA
- A **"Did You Know?"** fact card and a **"Think Like an Investor"** section with reflection questions generated from the selected company's actual metrics
- **Student Mode**, which hides more advanced metrics and swaps in simplified labels

**For teachers:**
- A **Classroom** tab with one-click generators for discussion questions, homework, a quiz, an exit ticket, a vocabulary assignment, and a case study — all templated and customized with the live company's name and data (no external AI/LLM call)
- A **Classroom Activities** panel (Think–Pair–Share prompt, a small-group comparison activity, and whole-class discussion questions)
- A built-in **Investing Vocabulary** glossary and a **Quick Quiz** (5 multiple-choice questions with instant feedback and a running score)
- A **Teacher Notes** panel — grade level, course, estimated lesson time, learning objectives, and a homework prompt — plus free-form notes saved via Streamlit session state

Because Education Mode is additive rather than a separate build, a teacher and a student can be looking at the exact same live ticker data — just with different levels of scaffolding around it.

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| App framework | [Streamlit](https://streamlit.io/) |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) |
| Data processing | pandas, numpy |
| Visualization | Plotly |
| PDF export | fpdf2 |
| HTTP | requests |

No external AI/LLM API is used — all "AI" scoring and sentiment analysis is implemented as deterministic Python logic, as described above.

---

## How It Works

```
User
  ↓
Streamlit Interface (sidebar controls + tabs)
  ↓
Yahoo Finance (yfinance) — live prices, fundamentals, statements, news, dividends, FX rates
  ↓
Data Processing & Analytics (pandas/numpy) — indicators, scoring, backtesting, sentiment
  ↓
Plotly Charts + Streamlit Metrics/Tables
  ↓
Research Dashboard  +  Education Mode Interface
```

---

## Example Workflow

1. Enter a ticker symbol in the sidebar (e.g. `AAPL`)
2. Review live price, company profile, and key metrics in **Overview**
3. Explore **Technical** and **Fundamentals** for indicators, ratios, and valuation
4. Check the **AI Scores** tab for the composite AI/Buffett/Graham/Risk breakdown
5. Set a **Price Alert**, compare against **Peers**, or run a **Backtest**
6. Read the latest **News** with sentiment tags, or check the **Dividends** tab
7. Toggle **Education Mode** to explore the **Learn** and **Classroom** tabs
8. **Export** a PDF report or CSV of the current ticker's data

---


---

## Installation

```bash
git clone https://github.com/KiaanKothari/AI-Stock-Research-Assistant.git
cd AI-Stock-Research-Assistant

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Configuration

No API keys are required. Market data, news, and FX rates are retrieved through the public `yfinance` interface to Yahoo Finance.

If you deploy this app (e.g. Streamlit Community Cloud) and later add any credentials, use [Streamlit secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management) (`.streamlit/secrets.toml`, which should be gitignored) or environment variables — never commit credentials to the repository.

---

## Project Structure

```
AI-Stock-Research-Assistant/
│
├── app.py              # Full application: data layer, indicators, charts,
│                        # scoring, Education Mode, and the Streamlit UI
├── requirements.txt     # Python dependencies
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## Version History

| Version | Highlights |
|---|---|
| **v1.3.0** | Price Alerts, Peer/Sector Comparison, News Sentiment Analysis, SMA-Crossover Backtesting, Dark/Light theme toggle, Dividend History Tracker, Multi-Currency display support |
| **v1.2.0** | Classroom Activities panel, Investing Vocabulary glossary, Quick Quiz, and an expanded Teacher Notes (lesson-plan) panel added to Education Mode |
| **v1.1.0** | Education Mode: Student Mode, Learn tab, Classroom tab (discussion questions, homework, quiz, exit tickets, vocabulary, reflection questions), inline metric explanations, teacher notes |
| **v1.0.0** | Initial release: AI-powered scoring, interactive price charts, fundamentals, financial statements, technical indicators, portfolio tracker, news, analyst data |

Full details in [CHANGELOG.md](CHANGELOG.md).

---

## Future Improvements

These are ideas, not implemented features:

- Options chain / derivatives data viewer
- Earnings calendar with countdown to next report
- Correlation matrix across multiple tracked stocks
- Monte Carlo–based risk simulation
- Insider/institutional ownership tracking
- Custom stock screener (filter by metric thresholds)
- Excel export with embedded charts
- Broader per-share price conversion under Multi-Currency (currently large-figure metrics and the header price are converted; a few per-share fields elsewhere remain USD-native)

---

## Disclaimer

This project is intended for educational and informational purposes only and does not constitute financial advice.

---

## Author

**Kiaan Kothari** — an independent, student-developed project exploring the intersection of AI, finance, and software engineering.

GitHub: [github.com/KiaanKothari](https://github.com/KiaanKothari)

---

## License

This project is licensed under the MIT License.

---

⭐ If you found this project interesting, consider giving it a star!
