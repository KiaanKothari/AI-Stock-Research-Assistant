# Changelog

All notable changes to this project will be documented here.

---

## [1.3.0] - Alerts, Peers, Backtesting & Multi-Currency Update

### 🔔 New Features

- Added **Price Alerts**: set an above/below target price for any ticker, tracked in a dedicated Alerts tab and evaluated against live prices.
- Added **Peer & Sector Comparison**: curated peer groups for major tickers (with manual entry for any other ticker), a benchmarking table (market cap, revenue growth, profit margin, P/E, ROE), and peer-average deltas.
- Added **News Sentiment Analysis**: lightweight keyword-based sentiment scoring for each headline, plus an aggregate Positive/Neutral/Negative breakdown chart in the News tab.
- Added **Backtesting**: a configurable SMA-crossover strategy backtester with an equity curve vs. buy-and-hold, summary stats (return, win rate, trade count, max drawdown), and a full trade log.
- Added a **Dark/Light theme toggle** in the sidebar, applied live via custom CSS.
- Added a **Dividend History Tracker**: full historical dividend payment chart and table, annual dividend, dividend yield, payout ratio, and computed dividend growth rate (CAGR).
- Added **Multi-Currency display support**: convert Market Cap, Revenue, and other large financial figures (plus the header live price) into EUR, GBP, JPY, INR, CAD, or AUD using a live FX rate, alongside the native-currency figures.

### 🔧 Technical Improvements

- Centralized currency formatting so large-number metrics across the app respect the selected display currency by default (USD behavior is unchanged).
- All new tabs are additive to the existing tab set; Education Mode's tab visibility logic was extended rather than restructured.

---

## [1.2.0] - Classroom Tools Expansion

### 🏫 New Features

- Added a **Classroom Activities** panel to the Learn tab: a Think–Pair–Share prompt, a small-group peer-comparison activity, and whole-class discussion prompts.
- Added an **Investing Vocabulary** glossary covering core investing terms (Market Cap, Revenue, Net Income, EPS, P/E Ratio, Dividend, Bull/Bear Market, Volatility, Risk, Diversification, ROE).
- Added a **Quick Quiz**: 5 multiple-choice questions with instant correct/incorrect feedback, a one-sentence explanation per question, and a running score.
- Expanded **Teacher Notes** into a lesson-plan panel: recommended grade level, course name, estimated lesson time, learning objectives, and a homework prompt, in addition to the existing free-form session notes.

---

## [1.1.0] - Education Mode Update

**Release Date:** July 7, 2026

### 🎓 New Features

- Added **Education Mode** toggle in the sidebar.
- Introduced **Student Mode** for classroom-friendly learning.
- Added a dedicated **Learn** tab with beginner-friendly explanations.
- Added a **Classroom** tab with:
  - Discussion questions
  - Homework generator
  - Quiz generator
  - Exit tickets
  - Vocabulary builder
  - Reflection questions
- Added financial term explanations throughout the application.
- Added teacher notes using Streamlit session state.
- Improved navigation for educational use.
- Optimized the interface for classroom demonstrations.

### 📚 Educational Improvements

- Simplified financial concepts for high school students.
- Added plain-English explanations for investing terminology.
- Improved readability of company information.
- Made the app suitable for finance and economics classes.

### 🔧 Technical Improvements

- Improved application stability.
- Refined sidebar organization.
- Enhanced overall user experience.

---

## Version 1.0.0 - Initial Release

### Features

- AI-powered stock research dashboard
- Interactive price charts
- Company fundamentals
- Financial statements
- Technical indicators
- Portfolio tracker
- Company news
- Analyst information
- AI research tools

## v1.4.0-beta.1 — Post-Release Testing

- Verified DCF valuation workflow with multiple stock tickers.
- Verified Bear, Base, and Bull valuation scenarios.
- Verified DCF sensitivity analysis.
- Verified current price vs. intrinsic value comparison.
- Verified the updated user interface and navigation.
- Verified Education Mode DCF explanations.
- Performed general UI smoke testing after deployment.

