# 📈 AI Stock Research Assistant [RELEASE v1.4.0-beta.1 - DCF Valuation BETA]

An AI-powered stock research platform built with **Python, Streamlit, Plotly, and yfinance**, combining financial data, technical analysis, fundamental analysis, transparent rule-based scoring, portfolio tools, backtesting, and an interactive **DCF valuation model**.

The project also includes an **Education Mode** designed to help students and teachers learn financial concepts using real market data.

> **Current Release: v1.4.0-beta.1 — DCF Valuation Beta**

---

## 🚀 Overview

AI Stock Research Assistant turns live market data into an interactive research dashboard.

Enter a stock ticker and explore:

* Live market data
* Company fundamentals
* Financial statements
* Technical indicators
* Transparent AI-style investment scores
* DCF valuation
* Bear/Base/Bull scenarios
* DCF sensitivity analysis
* Portfolio and watchlist tracking
* Price alerts
* Peer and sector comparisons
* Backtesting
* News and sentiment analysis
* Dividend history
* Education Mode
* PDF and CSV exports

The goal is to combine **finance, quantitative analysis, software engineering, and financial education** into one practical application.

---

# 🆕 What's New — v1.4.0-beta.1

## 💰 DCF Valuation Model

The latest release introduces a new **Discounted Cash Flow (DCF) valuation model**.

Users can enter a company ticker and generate an estimated intrinsic value based on projected free cash flow and configurable valuation assumptions.

### DCF Features

* 📊 **5-year free cash flow projections**
* 💰 Estimated intrinsic value
* ⚙️ Configurable valuation assumptions
* 📈 Current stock price vs. estimated intrinsic value
* 🐻 Bear scenario
* 📊 Base scenario
* 🐂 Bull scenario
* 📉 DCF sensitivity analysis
* WACC sensitivity
* Terminal-growth sensitivity
* Margin-of-safety comparison

The model is designed not only to produce a valuation estimate, but also to make the underlying assumptions visible to the user.

---

## 📚 DCF Education Mode

The DCF feature is integrated into the application's Education Mode.

Users can learn concepts such as:

* Free Cash Flow
* Discounted Cash Flow
* WACC
* Terminal Growth
* Terminal Value
* Enterprise Value
* Equity Value
* Intrinsic Value
* Margin of Safety

The goal is to help users understand **how a DCF works**, rather than simply displaying a final valuation number.

---

# ⭐ Core Features

## 📊 Market Data

* Live stock price
* Daily price change
* Percentage change
* Company profile
* Sector and industry
* Employee count
* Exchange and currency
* Historical OHLCV data
* Configurable historical periods
* Daily, weekly, and monthly intervals

---

## 💰 Fundamental Analysis

* Income statement
* Balance sheet
* Cash flow statement
* Revenue trends
* EPS trends
* Analyst estimates
* P/E
* Forward P/E
* PEG
* Price/Book
* Price/Sales
* EV/EBITDA
* ROE
* ROA
* Profit margins
* Debt/Equity
* Current ratio
* Graham Number
* DCF intrinsic value

---

## 📈 DCF Valuation

The DCF model provides:

1. Historical free cash flow data
2. Five-year FCF projections
3. Configurable assumptions
4. Terminal value calculation
5. Discounted cash flow valuation
6. Estimated intrinsic value
7. Current-price comparison
8. Bear/Base/Bull scenarios
9. WACC and terminal-growth sensitivity analysis

The model is intended as an **educational valuation framework**, not a prediction of future stock prices.

---

## 📉 Technical Analysis

Interactive technical analysis includes:

* Candlestick charts
* SMA 20
* SMA 50
* SMA 200
* Bollinger Bands
* RSI
* MACD
* Volume
* Support and resistance detection

---

## 🤖 AI-Powered Research

The application uses transparent, rule-based scoring rather than a black-box machine-learning model.

### AI Investment Score

A weighted score based on:

* Valuation
* Business quality
* Momentum
* Financial health

### Additional Scores

* Buffett Score
* Graham Score
* Risk Score

Each score includes a breakdown of the factors contributing to the result.

### Important

The application's "AI" scores are **deterministic rule-based calculations**.

They are not trained machine-learning models and do not claim to predict future stock prices.

---

## 📰 News & Sentiment

The application provides:

* Latest company news
* Publisher information
* News thumbnails
* Headline sentiment
* Positive / Neutral / Negative classification
* Aggregate sentiment visualization

Sentiment analysis uses a transparent keyword-based approach rather than an external language model.

---

## 🔄 Backtesting

The application includes an SMA crossover backtesting engine.

Users can configure:

* Short moving-average period
* Long moving-average period
* Starting capital

Results include:

* Total return
* Win rate
* Number of trades
* Maximum drawdown
* Strategy equity curve
* Buy-and-hold comparison
* Individual trade log

---

## 💼 Portfolio & Watchlist

### Portfolio Tracker

* Add shares
* Track cost basis
* Calculate live profit/loss

### Watchlist

* Track multiple stocks
* Monitor price
* Monitor market capitalization

### Price Alerts

Set an above/below price target for a stock and monitor the current market price.

---

## 🏢 Peer & Sector Comparison

Compare companies using:

* Market capitalization
* Revenue growth
* Profit margin
* P/E
* ROE

The application can suggest peer companies for major tickers and also allows manual peer selection.

---

## 💵 Dividend Tracking

Track:

* Historical dividend payments
* Annual dividend
* Dividend yield
* Payout ratio
* Dividend growth
* Dividend CAGR

---

# 🎓 Education Mode

Education Mode transforms the research dashboard into a financial-learning environment.

## For Students

Students can access:

* Plain-English financial explanations
* Company business-model explanations
* Competitor information
* Industry information
* Competitive advantages
* Business risks
* Real-world examples
* Financial vocabulary
* Investor reflection questions
* Quick quizzes
* Student Mode with simplified terminology

---

## For Teachers

Teachers can use:

* Discussion-question generators
* Homework generators
* Quiz generators
* Exit-ticket generators
* Vocabulary activities
* Case studies
* Think-Pair-Share activities
* Group comparison activities
* Teacher notes
* Lesson objectives
* Suggested lesson timing

Education Mode uses the same live market data as the research dashboard, allowing students to learn from real companies rather than simplified example datasets.

---

# 📊 Visualization

Charts and financial visualizations are built with Plotly.

The application includes:

* Candlestick charts
* Technical indicators
* Revenue charts
* EPS charts
* Financial metrics
* AI score gauges
* Peer comparisons
* Sentiment charts
* Backtesting equity curves
* Dividend history
* DCF valuation outputs
* DCF sensitivity analysis

---

# 🌎 Multi-Currency Support

Financial figures can be displayed in multiple currencies, including:

* USD
* EUR
* GBP
* JPY
* INR
* CAD
* AUD

Native-currency information is preserved alongside converted values.

---

# 📄 Export Tools

The application supports:

* PDF research reports
* CSV exports
* Price-history exports
* Key-metric exports

---

# 🧠 How the Project Works

```text
User
  ↓
Streamlit Interface
  ↓
Yahoo Finance / yfinance
  ↓
Market Data & Financial Statements
  ↓
Python Data Processing
  ↓
Technical Indicators
Fundamental Analysis
DCF Valuation
Backtesting
Rule-Based Scoring
Sentiment Analysis
  ↓
Plotly Visualizations
  ↓
Research Dashboard
+
Education Mode
```

---

# 🛠️ Technology Stack

| Category        | Technology               |
| --------------- | ------------------------ |
| Language        | Python 3.10+             |
| App Framework   | Streamlit                |
| Market Data     | yfinance / Yahoo Finance |
| Data Processing | pandas, NumPy            |
| Visualization   | Plotly                   |
| PDF Export      | fpdf2                    |
| HTTP            | requests                 |

No external AI/LLM API is required for the application's scoring or sentiment features.

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/KiaanKothari/AI-Stock-Research-Assistant.git
cd AI-Stock-Research-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# ⚙️ Configuration

No API keys are required for the core application.

Market data, financial information, news, and FX data are retrieved through the `yfinance` interface to Yahoo Finance.

If credentials are added in the future, they should be stored using environment variables or Streamlit secrets and should never be committed to the repository.

---

# 📁 Project Structure

```text
AI-Stock-Research-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

# 📜 Version History

| Version           | Highlights                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **v1.4.0-beta.1** | DCF Valuation Beta, 5-year FCF projections, configurable assumptions, Bear/Base/Bull scenarios, DCF sensitivity analysis, intrinsic-value comparison, DCF Education Mode |
| **v1.3.0**        | Price Alerts, Peer/Sector Comparison, News Sentiment Analysis, SMA-Crossover Backtesting, Dark/Light Theme, Dividend History, Multi-Currency Support                     |
| **v1.2.0**        | Classroom Activities, Investing Vocabulary, Quick Quiz, expanded Teacher Notes                                                                                           |
| **v1.1.0**        | Education Mode, Student Mode, Learn tab, Classroom tab, metric explanations, teacher notes                                                                               |
| **v1.0.0**        | Initial release with AI scoring, price charts, fundamentals, financial statements, technical indicators, portfolio tracking, news, and analyst data                      |

See `CHANGELOG.md` for additional details.

---

# 🔭 Future Improvements

Potential future features include:

* Options and derivatives data
* Earnings calendar
* Multi-stock correlation analysis
* Monte Carlo risk simulation
* Insider and institutional ownership tracking
* Custom stock screener
* Excel export with embedded charts
* Additional valuation models
* Expanded portfolio analytics

---

# ⚠️ Disclaimer

This project is intended for **educational and informational purposes only**.

It does not constitute financial advice, investment advice, or a recommendation to buy or sell any security.

DCF outputs, scores, technical indicators, and other calculations are estimates based on assumptions and available market data. They should not be interpreted as guaranteed predictions of future performance.

---

# 👨‍💻 Author

**Kiaan Kothari**

Independent student-developed project exploring the intersection of:

* Artificial intelligence
* Finance
* Quantitative analysis
* Software engineering
* Financial education

---

# 📄 License

This project is licensed under the **MIT License**.

---

⭐ **If you find the project interesting, consider giving the repository a star!**
