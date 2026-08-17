"""
AI Stock Research Assistant — single-file build
--------------------------------------------------
Everything (data access, indicators, charts, AI scoring, valuation,
news formatting, portfolio/watchlist state, and the Streamlit UI) lives
in this one file for convenience.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import io
import random  # ADDED FOR EDUCATION MODE: used for "Did You Know?" random facts
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================================== #
# SECTION 1: DATA LAYER (originally data.py)
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Ticker object
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_ticker_object(symbol: str) -> yf.Ticker:
    """Return a cached yfinance Ticker object for the given symbol."""
    return yf.Ticker(symbol.strip().upper())


# --------------------------------------------------------------------------- #
# Company info / profile
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_company_info(symbol: str) -> dict[str, Any]:
    """
    Fetch company profile & key statistics.

    Returns an empty dict if the ticker is invalid or data is unavailable.
    """
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        info = ticker.info or {}
        # yfinance sometimes returns a near-empty dict for invalid tickers
        if not info or info.get("regularMarketPrice") is None and info.get(
            "currentPrice"
        ) is None and info.get("previousClose") is None:
            return info if info else {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_logo_url(symbol: str, website: Optional[str] = None) -> Optional[str]:
    """
    Attempt to resolve a company logo using the Clearbit logo API, which
    derives a logo from a company's domain name. Returns None if no website
    is available or the logo cannot be resolved.
    """
    try:
        if not website:
            info = get_company_info(symbol)
            website = info.get("website")
        if not website:
            return None
        domain = (
            website.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
        )
        logo_url = f"https://logo.clearbit.com/{domain}"
        # Quick existence check (best-effort, short timeout)
        resp = requests.head(logo_url, timeout=2)
        if resp.status_code == 200:
            return logo_url
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Price history
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 5, show_spinner=False)
def get_price_history(
    symbol: str, period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch OHLCV price history.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    period : str
        One of yfinance's accepted periods (e.g. '1mo', '6mo', '1y', '5y', 'max').
    interval : str
        Bar interval (e.g. '1d', '1wk', '1h').

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date.
    Empty DataFrame on failure.
    """
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 1, show_spinner=False)
def get_live_price(symbol: str) -> dict[str, Any]:
    """
    Return a small dict describing the latest available price and change.
    Falls back to the most recent daily close if intraday data is missing.
    """
    try:
        info = get_company_info(symbol)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get(
            "regularMarketPreviousClose"
        )
        if price is None:
            hist = get_price_history(symbol, period="5d", interval="1d")
            if hist.empty:
                return {"price": None, "change": None, "pct_change": None}
            price = float(hist["Close"].iloc[-1])
            prev_close = (
                float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            )
        change = None
        pct_change = None
        if price is not None and prev_close:
            change = price - prev_close
            pct_change = (change / prev_close) * 100 if prev_close else None
        return {
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "pct_change": pct_change,
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {"price": None, "change": None, "pct_change": None}


# --------------------------------------------------------------------------- #
# Financial statements
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_income_statement(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """Return the income statement (annual by default)."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        stmt = (
            ticker.quarterly_income_stmt if quarterly else ticker.income_stmt
        )
        return stmt if stmt is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_balance_sheet(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """Return the balance sheet (annual by default)."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        stmt = (
            ticker.quarterly_balance_sheet if quarterly else ticker.balance_sheet
        )
        return stmt if stmt is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_cash_flow(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """Return the cash flow statement (annual by default)."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        stmt = ticker.quarterly_cashflow if quarterly else ticker.cashflow
        return stmt if stmt is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_earnings(symbol: str) -> pd.DataFrame:
    """Return historical & estimated earnings figures."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        df = ticker.earnings_dates
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Analyst data
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_recommendations(symbol: str) -> pd.DataFrame:
    """Return analyst recommendation history."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        rec = ticker.recommendations
        return rec if rec is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_price_targets(symbol: str) -> dict[str, Any]:
    """Return analyst price target summary statistics."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        targets = ticker.analyst_price_targets
        if targets is None:
            return {}
        if isinstance(targets, dict):
            return targets
        return targets.to_dict()
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_company_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return the latest news items for a given ticker."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        news = ticker.news or []
        return news[:limit]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Multi-ticker helper (used for comparisons)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60 * 5, show_spinner=False)
def get_multi_price_history(
    symbols: tuple[str, ...], period: str = "1y", interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers at once (for comparison view)."""
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        result[sym] = get_price_history(sym, period=period, interval=interval)
    return result


def is_valid_ticker(symbol: str) -> bool:
    """Lightweight validity check used before rendering a ticker's dashboard."""
    if not symbol:
        return False
    info = get_company_info(symbol)
    hist = get_price_history(symbol, period="5d")
    return bool(info) or not hist.empty


# =========================================================================== #
# SECTION 2: TECHNICAL INDICATORS (originally indicators.py)
# =========================================================================== #
def sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """Simple Moving Average."""
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].rolling(window=window, min_periods=window).mean()


def ema(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """Exponential Moving Average."""
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].ewm(span=window, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing method).

    Returns a series bounded between 0 and 100.
    """
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)

    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.fillna(50)  # neutral until enough data exists
    return rsi_series


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Returns a DataFrame with columns: MACD, Signal, Histogram.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=["MACD", "Signal", "Histogram"])

    ema_fast = ema(df, fast, column)
    ema_slow = ema(df, slow, column)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
    )


def bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close"
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns a DataFrame with columns: Middle, Upper, Lower.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=["Middle", "Upper", "Lower"])

    middle = sma(df, window, column)
    std = df[column].rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std

    return pd.DataFrame({"Middle": middle, "Upper": upper, "Lower": lower})


def support_resistance(
    df: pd.DataFrame, window: int = 10, num_levels: int = 3
) -> dict[str, list[float]]:
    """
    Identify approximate support and resistance levels using local
    minima/maxima over a rolling window on High/Low prices.

    Returns a dict: {"support": [...], "resistance": [...]}
    sorted nearest-to-price first, capped at `num_levels` each.
    """
    if df.empty or "High" not in df.columns or "Low" not in df.columns:
        return {"support": [], "resistance": []}

    highs = df["High"]
    lows = df["Low"]

    local_max = highs[
        (highs.shift(window) < highs) & (highs.shift(-window) < highs)
    ]
    local_min = lows[(lows.shift(window) > lows) & (lows.shift(-window) > lows)]

    current_price = float(df["Close"].iloc[-1])

    resistance_levels = sorted(
        {round(float(v), 2) for v in local_max.dropna().tolist() if v > current_price}
    )
    support_levels = sorted(
        {round(float(v), 2) for v in local_min.dropna().tolist() if v < current_price},
        reverse=True,
    )

    return {
        "support": support_levels[:num_levels],
        "resistance": resistance_levels[:num_levels],
    }


def volatility(df: pd.DataFrame, window: int = 30, column: str = "Close") -> float:
    """Annualized historical volatility (standard deviation of returns)."""
    if df.empty or column not in df.columns or len(df) < 2:
        return 0.0
    returns = df[column].pct_change().dropna()
    if returns.empty:
        return 0.0
    recent = returns.tail(window)
    return float(recent.std() * np.sqrt(252) * 100)


def beta_proxy(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Rough beta calculation given aligned daily return series for the stock
    and a benchmark (e.g., S&P 500). Returns 0.0 if insufficient data.
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return 0.0
    return float(cov / var)


# =========================================================================== #
# SECTION 3: PLOTLY CHART BUILDERS (originally charts.py)
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

COLOR_UP = "#00C805"
COLOR_DOWN = "#FF3B30"
COLOR_BG = "rgba(0,0,0,0)"
COLOR_GRID = "rgba(150,150,150,0.15)"
COLOR_TEXT = "#D1D5DB"
COLOR_ACCENT = "#F5A623"
COLOR_SMA20 = "#4FC3F7"
COLOR_SMA50 = "#FFB74D"
COLOR_SMA200 = "#CE93D8"
COLOR_BAND = "rgba(120,120,255,0.15)"


def _base_layout(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Apply consistent, professional dashboard styling to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLOR_TEXT)),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, size=12),
        margin=dict(l=40, r=20, t=50 if title else 20, b=30),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=COLOR_GRID, showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=COLOR_GRID, showgrid=True),
        hovermode="x unified",
    )
    return fig


def candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    show_sma20: bool = True,
    show_sma50: bool = True,
    show_sma200: bool = False,
    show_bollinger: bool = False,
) -> go.Figure:
    """Build the primary candlestick price chart with optional overlays."""
    fig = go.Figure()

    if df.empty:
        return _base_layout(fig, f"{symbol} — No price data available")

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
        )
    )

    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=sma(df, 20), name="SMA 20",
                line=dict(color=COLOR_SMA20, width=1.3),
            )
        )
    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=sma(df, 50), name="SMA 50",
                line=dict(color=COLOR_SMA50, width=1.3),
            )
        )
    if show_sma200:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=sma(df, 200), name="SMA 200",
                line=dict(color=COLOR_SMA200, width=1.3),
            )
        )
    if show_bollinger:
        bands = bollinger_bands(df)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bands["Upper"], name="Bollinger Upper",
                line=dict(color="rgba(150,150,255,0.6)", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bands["Lower"], name="Bollinger Lower",
                line=dict(color="rgba(150,150,255,0.6)", width=1, dash="dot"),
                fill="tonexty", fillcolor=COLOR_BAND,
            )
        )

    return _base_layout(fig, f"{symbol} — Price Chart", height=520)


def volume_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Volume bar chart colored by up/down day."""
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, "Volume — No data", height=200)

    colors = [
        COLOR_UP if c >= o else COLOR_DOWN
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"))
    return _base_layout(fig, f"{symbol} — Volume", height=220)


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI oscillator with overbought/oversold reference lines."""
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, "RSI — No data", height=220)

    rsi_series = rsi(df)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI (14)", line=dict(color=COLOR_ACCENT)))
    fig.add_hline(y=70, line_dash="dash", line_color=COLOR_DOWN, opacity=0.6)
    fig.add_hline(y=30, line_dash="dash", line_color=COLOR_UP, opacity=0.6)
    fig.update_yaxes(range=[0, 100])
    return _base_layout(fig, "RSI (14)", height=220)


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal line, and histogram."""
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, "MACD — No data", height=220)

    macd_df = macd(df)
    colors = [
        COLOR_UP if v >= 0 else COLOR_DOWN for v in macd_df["Histogram"].fillna(0)
    ]
    fig.add_trace(go.Bar(x=df.index, y=macd_df["Histogram"], name="Histogram", marker_color=colors))
    fig.add_trace(go.Scatter(x=df.index, y=macd_df["MACD"], name="MACD", line=dict(color=COLOR_SMA20, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=macd_df["Signal"], name="Signal", line=dict(color=COLOR_ACCENT, width=1.5)))
    return _base_layout(fig, "MACD (12, 26, 9)", height=220)


def support_resistance_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Price line chart annotated with detected support/resistance levels."""
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, "Support & Resistance — No data")

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=COLOR_SMA20))
    )
    levels = support_resistance(df)
    for level in levels["resistance"]:
        fig.add_hline(y=level, line_dash="dot", line_color=COLOR_DOWN, opacity=0.7,
                       annotation_text=f"R {level}", annotation_position="right")
    for level in levels["support"]:
        fig.add_hline(y=level, line_dash="dot", line_color=COLOR_UP, opacity=0.7,
                       annotation_text=f"S {level}", annotation_position="right")

    return _base_layout(fig, f"{symbol} — Support & Resistance", height=420)


def revenue_chart(income_stmt: pd.DataFrame) -> go.Figure:
    """Bar chart of total revenue across the reported fiscal periods."""
    fig = go.Figure()
    if income_stmt.empty:
        return _base_layout(fig, "Revenue — No data", height=350)

    row_name = next(
        (r for r in income_stmt.index if "Total Revenue" in str(r) or r == "Total Revenue"),
        None,
    )
    if row_name is None:
        row_name = next((r for r in income_stmt.index if "Revenue" in str(r)), None)
    if row_name is None:
        return _base_layout(fig, "Revenue — No data", height=350)

    series = income_stmt.loc[row_name].dropna()
    periods = [str(c.date()) if hasattr(c, "date") else str(c) for c in series.index]

    fig.add_trace(go.Bar(x=periods, y=series.values, marker_color=COLOR_SMA20, name="Revenue"))
    return _base_layout(fig, "Total Revenue by Period", height=350)


def earnings_chart(earnings_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of EPS estimate vs. reported EPS."""
    fig = go.Figure()
    if earnings_df.empty:
        return _base_layout(fig, "Earnings — No data", height=350)

    df = earnings_df.dropna(how="all").tail(8).sort_index()
    x_labels = [str(i.date()) if hasattr(i, "date") else str(i) for i in df.index]

    if "EPS Estimate" in df.columns:
        fig.add_trace(go.Bar(x=x_labels, y=df["EPS Estimate"], name="EPS Estimate", marker_color=COLOR_ACCENT))
    if "Reported EPS" in df.columns:
        fig.add_trace(go.Bar(x=x_labels, y=df["Reported EPS"], name="Reported EPS", marker_color=COLOR_SMA20))

    fig.update_layout(barmode="group")
    return _base_layout(fig, "EPS: Estimate vs. Reported", height=350)


def comparison_chart(price_histories: dict[str, pd.DataFrame]) -> go.Figure:
    """Normalized (% return) comparison chart across multiple tickers."""
    fig = go.Figure()
    palette = [COLOR_SMA20, COLOR_ACCENT, COLOR_UP, COLOR_DOWN, COLOR_SMA200, "#90CAF9"]

    for i, (symbol, df) in enumerate(price_histories.items()):
        if df.empty:
            continue
        normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=df.index, y=normalized, name=symbol,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )

    fig.update_yaxes(ticksuffix="%")
    return _base_layout(fig, "Normalized Performance Comparison (%)", height=480)


def gauge_chart(score: float, title: str, max_value: int = 100) -> go.Figure:
    """Circular gauge used for AI / Buffett / Graham / Risk scores."""
    if score >= 70:
        bar_color = COLOR_UP
    elif score >= 40:
        bar_color = COLOR_ACCENT
    else:
        bar_color = COLOR_DOWN

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 14, "color": COLOR_TEXT}},
            number={"font": {"color": COLOR_TEXT, "size": 28}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": COLOR_TEXT},
                "bar": {"color": bar_color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,59,48,0.12)"},
                    {"range": [40, 70], "color": "rgba(245,166,35,0.12)"},
                    {"range": [70, 100], "color": "rgba(0,200,5,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=COLOR_BG, height=220, margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


# =========================================================================== #
# SECTION 4: AI / BUFFETT / GRAHAM / RISK SCORING (originally ai_engine.py)
# =========================================================================== #
@dataclass
class ScoreResult:
    """Container for a score plus the human-readable factors behind it."""
    score: float
    breakdown: list[str] = field(default_factory=list)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            if self.score >= 75:
                self.label = "Strong"
            elif self.score >= 50:
                self.label = "Moderate"
            elif self.score >= 25:
                self.label = "Weak"
            else:
                self.label = "Poor"


def _safe(value: Any) -> Optional[float]:
    """Coerce a value to float, returning None on failure or None input."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# AI Investment Score
# --------------------------------------------------------------------------- #


def ai_investment_score(info: dict[str, Any], hist: pd.DataFrame) -> ScoreResult:
    """
    Composite score blending four pillars, each worth up to 25 points:
      1. Valuation  (P/E, PEG, forward P/E vs. trailing)
      2. Quality    (ROE, ROA, profit margins)
      3. Momentum   (price vs. moving averages, RSI)
      4. Financial health (debt-to-equity, current ratio)
    """
    breakdown: list[str] = []
    total = 0.0

    # --- Valuation (0-25) ---
    valuation_score = 12.5  # neutral baseline
    pe = _safe(info.get("trailingPE"))
    peg = _safe(info.get("pegRatio"))
    if pe is not None:
        if 0 < pe <= 15:
            valuation_score = 22
            breakdown.append(f"Attractive P/E of {pe:.1f}")
        elif 15 < pe <= 25:
            valuation_score = 16
            breakdown.append(f"Reasonable P/E of {pe:.1f}")
        elif pe > 25:
            valuation_score = 8
            breakdown.append(f"Elevated P/E of {pe:.1f}")
        else:
            valuation_score = 6
            breakdown.append("Negative earnings (P/E not meaningful)")
    if peg is not None and 0 < peg < 1.5:
        valuation_score = min(25, valuation_score + 3)
        breakdown.append(f"Favorable PEG ratio of {peg:.2f}")
    total += _clamp(valuation_score, 0, 25)

    # --- Quality (0-25) ---
    quality_score = 12.5
    roe = _safe(info.get("returnOnEquity"))
    roa = _safe(info.get("returnOnAssets"))
    margin = _safe(info.get("profitMargins"))
    q_points = 0
    q_count = 0
    if roe is not None:
        q_count += 1
        if roe > 0.20:
            q_points += 9
            breakdown.append(f"Strong ROE of {roe*100:.1f}%")
        elif roe > 0.10:
            q_points += 6
        else:
            q_points += 2
    if roa is not None:
        q_count += 1
        if roa > 0.10:
            q_points += 9
            breakdown.append(f"Strong ROA of {roa*100:.1f}%")
        elif roa > 0.05:
            q_points += 6
        else:
            q_points += 2
    if margin is not None:
        q_count += 1
        if margin > 0.15:
            q_points += 7
            breakdown.append(f"Healthy profit margin of {margin*100:.1f}%")
        elif margin > 0.05:
            q_points += 4
        else:
            q_points += 1
    if q_count > 0:
        quality_score = q_points
    total += _clamp(quality_score, 0, 25)

    # --- Momentum (0-25) ---
    momentum_score = 12.5
    if not hist.empty and len(hist) > 50:
        close = hist["Close"]
        current = float(close.iloc[-1])
        sma50 = sma(hist, 50).iloc[-1]
        sma200 = sma(hist, 200).iloc[-1] if len(hist) >= 200 else None
        rsi_val = rsi(hist).iloc[-1]

        m_points = 0
        if pd.notna(sma50):
            if current > sma50:
                m_points += 8
                breakdown.append("Trading above 50-day moving average")
            else:
                m_points += 2
        if sma200 is not None and pd.notna(sma200):
            if current > sma200:
                m_points += 8
                breakdown.append("Trading above 200-day moving average")
            else:
                m_points += 2
        else:
            m_points += 4
        if pd.notna(rsi_val):
            if 40 <= rsi_val <= 65:
                m_points += 9
                breakdown.append(f"RSI in healthy range ({rsi_val:.0f})")
            elif rsi_val > 70:
                m_points += 3
                breakdown.append(f"RSI overbought ({rsi_val:.0f})")
            elif rsi_val < 30:
                m_points += 4
                breakdown.append(f"RSI oversold ({rsi_val:.0f}) — potential rebound")
            else:
                m_points += 6
        momentum_score = m_points
    total += _clamp(momentum_score, 0, 25)

    # --- Financial health (0-25) ---
    health_score = 12.5
    de = _safe(info.get("debtToEquity"))
    current_ratio = _safe(info.get("currentRatio"))
    h_points = 0
    h_count = 0
    if de is not None:
        h_count += 1
        if de < 50:
            h_points += 13
            breakdown.append(f"Low debt-to-equity of {de:.1f}")
        elif de < 100:
            h_points += 8
        else:
            h_points += 3
            breakdown.append(f"High debt-to-equity of {de:.1f}")
    if current_ratio is not None:
        h_count += 1
        if current_ratio > 1.5:
            h_points += 12
            breakdown.append(f"Strong liquidity, current ratio {current_ratio:.2f}")
        elif current_ratio > 1.0:
            h_points += 8
        else:
            h_points += 3
            breakdown.append(f"Weak liquidity, current ratio {current_ratio:.2f}")
    if h_count > 0:
        health_score = h_points
    total += _clamp(health_score, 0, 25)

    return ScoreResult(score=round(_clamp(total), 1), breakdown=breakdown)


# --------------------------------------------------------------------------- #
# Buffett Score
# --------------------------------------------------------------------------- #


def buffett_score(info: dict[str, Any]) -> ScoreResult:
    """
    Checklist-style score (0-100) inspired by Warren Buffett's preference for
    high-quality, moat-protected, conservatively financed businesses.
    Each criterion contributes up to ~14-17 points across 6 checks.
    """
    breakdown: list[str] = []
    points = 0.0
    checks = 6
    per_check = 100 / checks

    roe = _safe(info.get("returnOnEquity"))
    if roe is not None and roe > 0.15:
        points += per_check
        breakdown.append(f"Consistent high ROE ({roe*100:.1f}%) suggests a durable moat")
    elif roe is not None:
        breakdown.append(f"ROE of {roe*100:.1f}% below Buffett's 15% preference")

    margin = _safe(info.get("profitMargins"))
    if margin is not None and margin > 0.15:
        points += per_check
        breakdown.append(f"High profit margin ({margin*100:.1f}%) indicates pricing power")
    elif margin is not None:
        breakdown.append(f"Profit margin of {margin*100:.1f}% is unremarkable")

    de = _safe(info.get("debtToEquity"))
    if de is not None and de < 80:
        points += per_check
        breakdown.append("Conservative balance sheet (low debt-to-equity)")
    elif de is not None:
        breakdown.append("Higher leverage than Buffett typically favors")

    fcf = _safe(info.get("freeCashflow"))
    if fcf is not None and fcf > 0:
        points += per_check
        breakdown.append("Positive free cash flow generation")

    earnings_growth = _safe(info.get("earningsGrowth"))
    if earnings_growth is not None and earnings_growth > 0.05:
        points += per_check
        breakdown.append(f"Earnings growing at {earnings_growth*100:.1f}%")

    pe = _safe(info.get("trailingPE"))
    if pe is not None and 0 < pe < 25:
        points += per_check
        breakdown.append("Valuation within a reasonable range")

    return ScoreResult(score=round(_clamp(points), 1), breakdown=breakdown)


# --------------------------------------------------------------------------- #
# Graham Score
# --------------------------------------------------------------------------- #


def graham_score(info: dict[str, Any]) -> ScoreResult:
    """
    Checklist-style score (0-100) inspired by Benjamin Graham's criteria for
    the "defensive investor": reasonable size, strong financial condition,
    earnings stability, and a sensible valuation.
    """
    breakdown: list[str] = []
    points = 0.0
    checks = 5
    per_check = 100 / checks

    market_cap = _safe(info.get("marketCap"))
    if market_cap is not None and market_cap > 2_000_000_000:
        points += per_check
        breakdown.append("Adequate company size (market cap > $2B)")

    current_ratio = _safe(info.get("currentRatio"))
    if current_ratio is not None and current_ratio >= 2.0:
        points += per_check
        breakdown.append(f"Strong current ratio of {current_ratio:.2f} (>= 2.0)")
    elif current_ratio is not None:
        breakdown.append(f"Current ratio of {current_ratio:.2f} below Graham's 2.0 threshold")

    eps = _safe(info.get("trailingEps"))
    if eps is not None and eps > 0:
        points += per_check
        breakdown.append("Positive trailing earnings per share")

    dividend_yield = _safe(info.get("dividendYield"))
    if dividend_yield is not None and dividend_yield > 0:
        points += per_check
        breakdown.append("Pays a dividend, consistent with defensive criteria")

    pe = _safe(info.get("trailingPE"))
    pb = _safe(info.get("priceToBook"))
    if pe is not None and pb is not None and (pe * pb) < 22.5:
        points += per_check
        breakdown.append(f"P/E x P/B of {pe*pb:.1f} is within Graham's 22.5 guideline")
    elif pe is not None and pb is not None:
        breakdown.append(f"P/E x P/B of {pe*pb:.1f} exceeds Graham's 22.5 guideline")

    return ScoreResult(score=round(_clamp(points), 1), breakdown=breakdown)


# --------------------------------------------------------------------------- #
# Risk Score  (higher score = LOWER risk, for consistent gauge display)
# --------------------------------------------------------------------------- #


def risk_score(info: dict[str, Any], hist: pd.DataFrame) -> ScoreResult:
    """
    Risk assessment score where 100 = lowest risk and 0 = highest risk,
    based on historical volatility, beta, leverage, and market cap size.
    """
    breakdown: list[str] = []
    points = 0.0
    checks = 4
    per_check = 100 / checks

    vol = volatility(hist) if not hist.empty else None
    if vol is not None:
        if vol < 25:
            points += per_check
            breakdown.append(f"Low annualized volatility ({vol:.1f}%)")
        elif vol < 45:
            points += per_check * 0.6
            breakdown.append(f"Moderate annualized volatility ({vol:.1f}%)")
        else:
            points += per_check * 0.2
            breakdown.append(f"High annualized volatility ({vol:.1f}%)")

    beta = _safe(info.get("beta"))
    if beta is not None:
        if beta < 1.0:
            points += per_check
            breakdown.append(f"Beta of {beta:.2f} — less volatile than the market")
        elif beta < 1.5:
            points += per_check * 0.5
            breakdown.append(f"Beta of {beta:.2f} — roughly in line with the market")
        else:
            points += per_check * 0.15
            breakdown.append(f"Beta of {beta:.2f} — more volatile than the market")

    de = _safe(info.get("debtToEquity"))
    if de is not None:
        if de < 50:
            points += per_check
            breakdown.append("Low balance-sheet leverage")
        elif de < 150:
            points += per_check * 0.5
        else:
            points += per_check * 0.15
            breakdown.append("High balance-sheet leverage increases risk")

    market_cap = _safe(info.get("marketCap"))
    if market_cap is not None:
        if market_cap > 10_000_000_000:
            points += per_check
            breakdown.append("Large-cap size reduces idiosyncratic risk")
        elif market_cap > 2_000_000_000:
            points += per_check * 0.6
        else:
            points += per_check * 0.25
            breakdown.append("Small-cap size increases volatility risk")

    return ScoreResult(score=round(_clamp(points), 1), breakdown=breakdown)


# =========================================================================== #
# SECTION 5: VALUATION MODELS (originally valuation.py)
# =========================================================================== #
# NOTE: the original simple single-line "grow last year's FCF by X%" DCF
# (DCFResult / run_dcf) that used to live here has been REPLACED by the
# forecast-driven DCF engine in SECTION 5B below, which builds a proper
# Revenue -> EBIT -> NOPAT -> FCFF forecast instead of growing a single
# number. Graham Number / margin_of_safety below are unchanged.


def graham_number(eps: Optional[float], book_value_per_share: Optional[float]) -> Optional[float]:
    """
    Benjamin Graham's classic intrinsic value formula:
        sqrt(22.5 * EPS * Book Value per Share)

    Returns None if inputs are missing, non-positive, or the product is negative.
    """
    if not eps or not book_value_per_share:
        return None
    if eps <= 0 or book_value_per_share <= 0:
        return None
    product = 22.5 * eps * book_value_per_share
    if product < 0:
        return None
    return round(product ** 0.5, 2)


def graham_growth_formula(
    eps: Optional[float],
    growth_rate_pct: Optional[float],
    aaa_bond_yield_pct: float = 4.5,
) -> Optional[float]:
    """
    Graham's revised growth-adjusted formula:
        V = EPS * (8.5 + 2g) * 4.4 / Y

    where g is the expected annual growth rate (%) and Y is the current
    AAA corporate bond yield (%). Useful as a secondary sanity check
    alongside the classic Graham Number.
    """
    if eps is None or growth_rate_pct is None or eps <= 0:
        return None
    if aaa_bond_yield_pct <= 0:
        return None
    value = eps * (8.5 + 2 * growth_rate_pct) * 4.4 / aaa_bond_yield_pct
    return round(value, 2) if value > 0 else None


def margin_of_safety(intrinsic_value: Optional[float], current_price: Optional[float]) -> Optional[float]:
    """
    Percentage margin of safety between an estimated intrinsic value and
    the current market price. Positive = undervalued, negative = overvalued.
    """
    if not intrinsic_value or not current_price or current_price <= 0:
        return None
    return round(((intrinsic_value - current_price) / current_price) * 100, 2)


# =========================================================================== #
# SECTION 5B: FORECAST-DRIVEN DCF ENGINE
# ---------------------------------------------------------------------
# Everything below is deterministic Python: retrieving the historical
# financial figures needed for a forecast, deriving forecast assumptions
# from that history, building a year-by-year Free Cash Flow to the Firm
# (FCFF) forecast, discounting it, and turning the result into a plain-
# English narrative from the actual computed numbers.
#
# Architecture (matches the "AI explains, doesn't invent" pattern already
# used by the Buffett/Graham/Risk scores above):
#
#     yfinance data -> deterministic Python math (this section) ->
#     DCFScenarioResult (plain dataclass, just numbers) -> UI displays the
#     numbers + generate_dcf_narrative() turns them into a sentence.
#
# No step in this pipeline invents a number. generate_dcf_narrative() only
# describes results that were already computed by run_dcf_advanced().
# =========================================================================== #

# =========================================================================== #
# 1. Historical financials retrieval
# =========================================================================== #

@dataclass
class HistoricalFinancials:
    """Up to N years of the historical figures needed to build a forecast.

    Columns from yfinance are most-recent-year-first; this class preserves
    that order (index 0 = most recent year, index -1 = oldest year).
    """

    years: list[str]
    revenue: list[float]
    ebit: list[float]
    da: list[float]
    capex: list[float]
    change_in_nwc: list[float]
    effective_tax_rate: Optional[float]
    data_warnings: list[str] = field(default_factory=list)


def _first_available_row(df: Any, row_names: list[str]) -> Any:
    """Return the first matching row (a pandas Series) from a yfinance
    statement DataFrame, trying several possible row-label spellings,
    since yfinance's exact labels can vary slightly by ticker/version."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            return df.loc[name]
    return None


def get_historical_financials(symbol: str, years: int = 3) -> Optional[HistoricalFinancials]:
    """
    Pull up to `years` years of Revenue, EBIT, D&A, CapEx, and change in
    net working capital from yfinance's income statement and cash flow
    statement, plus an effective tax rate.

    Returns
    -------
    None
        If no usable income statement data can be retrieved at all
        (e.g. invalid ticker, or yfinance/network failure). The caller
        (the Streamlit UI) is responsible for showing a clear message
        rather than crashing.
    HistoricalFinancials
        Otherwise — with `data_warnings` populated whenever a specific
        line item was missing and had to be estimated or defaulted, so
        the UI can tell the user exactly what was approximated.
    """
    warnings: list[str] = []
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        income_stmt = ticker.income_stmt
        cashflow = ticker.cashflow
    except Exception:
        return None

    if income_stmt is None or income_stmt.empty:
        return None

    cols = list(income_stmt.columns)[:years]
    if not cols:
        return None
    year_labels = [str(getattr(c, "year", c)) for c in cols]

    revenue_row = _first_available_row(income_stmt, ["Total Revenue", "TotalRevenue"])
    ebit_row = _first_available_row(income_stmt, ["EBIT", "Operating Income"])
    pretax_row = _first_available_row(income_stmt, ["Pretax Income"])
    tax_row = _first_available_row(income_stmt, ["Tax Provision"])

    da_row = _first_available_row(
        cashflow, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]
    )
    capex_row = _first_available_row(cashflow, ["Capital Expenditure", "CapitalExpenditure"])
    nwc_row = _first_available_row(cashflow, ["Change In Working Capital"])

    if revenue_row is None:
        # Without revenue there is nothing to build a forecast from.
        return None

    if ebit_row is None:
        warnings.append(
            "EBIT was not reported directly by the data source; Pretax "
            "Income was used as a rough substitute."
        )
        ebit_row = pretax_row

    def _val(row: Any, col: Any) -> float:
        if row is None:
            return float("nan")
        try:
            return float(row.get(col, float("nan")))
        except Exception:
            return float("nan")

    revenue = [_val(revenue_row, c) for c in cols]
    ebit = [_val(ebit_row, c) for c in cols]
    da = [_val(da_row, c) if da_row is not None else 0.0 for c in cols]
    capex_raw = [_val(capex_row, c) if capex_row is not None else 0.0 for c in cols]
    # yfinance reports CapEx as a cash outflow (negative); we store the
    # positive "amount spent" and subtract it explicitly in the formula.
    capex = [abs(v) if v == v else 0.0 for v in capex_raw]
    nwc_raw = [_val(nwc_row, c) if nwc_row is not None else 0.0 for c in cols]
    # yfinance's "Change In Working Capital" is already a CASH FLOW STATEMENT
    # figure: positive means NWC released cash (NWC decreased), negative
    # means NWC absorbed cash (NWC increased). The textbook FCFF formula
    # instead uses "increase in NWC" as a positive number that gets
    # SUBTRACTED, so we flip the sign here to match that convention.
    change_in_nwc = [-v if v == v else 0.0 for v in nwc_raw]

    if da_row is None:
        warnings.append("D&A was not available and was assumed to be $0 (this understates FCFF).")
    if capex_row is None:
        warnings.append("CapEx was not available and was assumed to be $0 (this overstates FCFF).")
    if nwc_row is None:
        warnings.append("Change in Net Working Capital was not available and was assumed to be $0.")

    effective_tax_rate = None
    if pretax_row is not None and tax_row is not None:
        try:
            pretax = float(pretax_row.iloc[0])
            tax = float(tax_row.iloc[0])
            if pretax > 0:
                effective_tax_rate = max(0.0, min(0.40, tax / pretax))
        except Exception:
            pass
    if effective_tax_rate is None:
        warnings.append(
            "Effective tax rate could not be computed from the data source; "
            "defaulted to 21% (the current US federal statutory rate)."
        )
        effective_tax_rate = 0.21

    return HistoricalFinancials(
        years=year_labels,
        revenue=revenue,
        ebit=ebit,
        da=da,
        capex=capex,
        change_in_nwc=change_in_nwc,
        effective_tax_rate=effective_tax_rate,
        data_warnings=warnings,
    )


def derive_base_assumptions(hist: HistoricalFinancials) -> dict[str, Optional[float]]:
    """
    Turn historical financials into suggested Base-case forecast
    assumptions: average revenue growth, average EBIT margin, and average
    D&A / CapEx / change-in-NWC as a % of revenue.

    Returns None for any assumption that couldn't be computed (e.g. only
    one year of history is available, so no growth rate can be derived).
    """
    # hist.revenue is most-recent-first; reverse to chronological order
    # so growth = later / earlier - 1 reads naturally.
    chron_rev = list(reversed(hist.revenue))
    growth_rates = []
    for i in range(1, len(chron_rev)):
        prev, curr = chron_rev[i - 1], chron_rev[i]
        if prev == prev and curr == curr and prev > 0:
            growth_rates.append(curr / prev - 1)
    avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else None

    ebit_margins, da_pct, capex_pct, nwc_pct = [], [], [], []
    for rev, ebit, da, capex, dnwc in zip(
        hist.revenue, hist.ebit, hist.da, hist.capex, hist.change_in_nwc
    ):
        if rev == rev and rev > 0:
            if ebit == ebit:
                ebit_margins.append(ebit / rev)
            da_pct.append(da / rev)
            capex_pct.append(capex / rev)
            nwc_pct.append(dnwc / rev)

    def _avg(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    return {
        "revenue_growth": avg_growth,
        "ebit_margin": _avg(ebit_margins),
        "da_pct_revenue": _avg(da_pct),
        "capex_pct_revenue": _avg(capex_pct),
        "nwc_pct_revenue": _avg(nwc_pct),
    }


# =========================================================================== #
# 2. WACC estimation
# =========================================================================== #

@dataclass
class WACCResult:
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_weight: float
    debt_weight: float
    wacc: float


def compute_wacc(
    risk_free_rate: float,
    equity_risk_premium: float,
    beta: float,
    credit_spread: float,
    tax_rate: float,
    market_cap: float,
    total_debt: float,
) -> WACCResult:
    """
    Estimate WACC using CAPM for cost of equity and a credit-spread proxy
    for pre-tax cost of debt, weighted by market cap (equity) and total
    debt at current market values.

        Cost of Equity = risk_free_rate + beta * equity_risk_premium
        Cost of Debt (after-tax) = (risk_free_rate + credit_spread) * (1 - tax_rate)
        WACC = (E / (E+D)) * CoE + (D / (E+D)) * CoD_after_tax
    """
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    cost_of_debt_pretax = risk_free_rate + credit_spread
    cost_of_debt_aftertax = cost_of_debt_pretax * (1 - tax_rate)

    total_value = max(market_cap + total_debt, 1e-9)
    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt_aftertax
    # Floor the estimate: if market cap and total debt are both missing/zero
    # (some tickers report neither reliably), equity_weight and debt_weight
    # both come out to 0 and wacc would compute to exactly 0.0 — not a
    # meaningful cost of capital, and dangerous as a downstream default
    # (it's used to seed the WACC slider elsewhere).
    wacc = max(wacc, 0.01)
    return WACCResult(
        cost_of_equity=cost_of_equity,
        cost_of_debt_pretax=cost_of_debt_pretax,
        cost_of_debt_aftertax=cost_of_debt_aftertax,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        wacc=wacc,
    )


# =========================================================================== #
# 3. Forecast build-up and the core DCF engine
# =========================================================================== #

@dataclass
class ForecastYear:
    year_label: str
    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    change_in_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float


@dataclass
class DCFScenarioResult:
    scenario: str  # "Bear" / "Base" / "Bull" / "Sensitivity"
    assumptions: dict[str, float]
    forecast: list[ForecastYear]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: Optional[float]
    intrinsic_value_per_share: Optional[float]


def build_fcff_forecast(
    base_revenue: float,
    revenue_growth: float,
    ebit_margin: float,
    tax_rate: float,
    da_pct_revenue: float,
    capex_pct_revenue: float,
    nwc_pct_revenue: float,
    wacc: float,
    projection_years: int,
) -> list[ForecastYear]:
    """
    Build a year-by-year FCFF forecast from top-line assumptions:

        Revenue(t)  = Revenue(t-1) * (1 + revenue_growth)
        EBIT(t)     = Revenue(t) * ebit_margin
        NOPAT(t)    = EBIT(t) * (1 - tax_rate)
        FCFF(t)     = NOPAT(t) + D&A(t) - CapEx(t) - ChangeInNWC(t)
        PV[FCFF(t)] = FCFF(t) / (1 + wacc) ** t
    """
    forecast: list[ForecastYear] = []
    revenue = base_revenue
    for year in range(1, projection_years + 1):
        revenue = revenue * (1 + revenue_growth)
        ebit = revenue * ebit_margin
        nopat = ebit * (1 - tax_rate)
        da = revenue * da_pct_revenue
        capex = revenue * capex_pct_revenue
        change_in_nwc = revenue * nwc_pct_revenue
        fcff = nopat + da - capex - change_in_nwc
        discount_factor = (1 + wacc) ** year
        pv_fcff = fcff / discount_factor
        forecast.append(
            ForecastYear(
                year_label=f"Year {year}",
                revenue=revenue,
                ebit=ebit,
                nopat=nopat,
                da=da,
                capex=capex,
                change_in_nwc=change_in_nwc,
                fcff=fcff,
                discount_factor=discount_factor,
                pv_fcff=pv_fcff,
            )
        )
    return forecast


def run_dcf_advanced(
    scenario: str,
    base_revenue: float,
    revenue_growth: float,
    ebit_margin: float,
    tax_rate: float,
    da_pct_revenue: float,
    capex_pct_revenue: float,
    nwc_pct_revenue: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int,
    net_debt: float,
    shares_outstanding: Optional[float],
) -> DCFScenarioResult:
    """Run one full scenario: forecast -> terminal value -> enterprise value
    -> equity value -> intrinsic value per share."""
    # Floor WACC to a sane minimum first. Without this, a pathological
    # input (e.g. a negative-beta stock producing a zero or negative
    # CAPM-estimated WACC) could make wacc <= 0, and the guard below only
    # adjusts terminal_growth relative to wacc — it never protects against
    # wacc itself being non-positive, which would still divide by zero.
    wacc = max(wacc, 0.01)
    if wacc <= terminal_growth:
        # A terminal value is mathematically invalid (negative or infinite)
        # if the discount rate doesn't exceed the perpetual growth rate.
        # Guard against it instead of crashing or returning garbage. With
        # wacc floored above, this always leaves wacc - terminal_growth
        # equal to exactly 0.01, never 0.
        terminal_growth = max(0.0, wacc - 0.01)

    forecast = build_fcff_forecast(
        base_revenue,
        revenue_growth,
        ebit_margin,
        tax_rate,
        da_pct_revenue,
        capex_pct_revenue,
        nwc_pct_revenue,
        wacc,
        projection_years,
    )

    last_fcff = forecast[-1].fcff
    terminal_value = last_fcff * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / (1 + wacc) ** projection_years

    enterprise_value = sum(f.pv_fcff for f in forecast) + pv_terminal_value
    equity_value = enterprise_value - net_debt

    intrinsic_value_per_share = None
    if shares_outstanding and shares_outstanding > 0:
        intrinsic_value_per_share = equity_value / shares_outstanding

    return DCFScenarioResult(
        scenario=scenario,
        assumptions={
            "revenue_growth": revenue_growth,
            "ebit_margin": ebit_margin,
            "tax_rate": tax_rate,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "projection_years": projection_years,
        },
        forecast=forecast,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic_value_per_share,
    )


# =========================================================================== #
# 4. Bear / Base / Bull scenarios
# =========================================================================== #

DEFAULT_SCENARIO_DELTAS: dict[str, float] = {
    "revenue_growth": 0.02,    # Bear/Bull shift Base by +/- 2 percentage points
    "ebit_margin": 0.015,      # +/- 1.5 percentage points
    "wacc": 0.01,              # Bear = +1pp WACC (higher discount rate = more conservative)
    "terminal_growth": 0.005,  # +/- 0.5 percentage points
}


def run_bear_base_bull(
    base_revenue: float,
    base_assumptions: dict[str, float],
    net_debt: float,
    shares_outstanding: Optional[float],
    projection_years: int,
    deltas: Optional[dict[str, float]] = None,
) -> dict[str, DCFScenarioResult]:
    """
    Run Base with the assumptions given, then derive Bear and Bull by
    shifting revenue growth, EBIT margin, WACC, and terminal growth by
    fixed, labeled deltas in the unfavorable / favorable direction.
    """
    deltas = deltas or DEFAULT_SCENARIO_DELTAS
    scenarios: dict[str, DCFScenarioResult] = {}

    for label, sign in [("Bear", -1), ("Base", 0), ("Bull", 1)]:
        wacc_sign = -sign  # Bear = HIGHER wacc (more conservative), so flip
        scenarios[label] = run_dcf_advanced(
            scenario=label,
            base_revenue=base_revenue,
            revenue_growth=base_assumptions["revenue_growth"] + sign * deltas["revenue_growth"],
            ebit_margin=base_assumptions["ebit_margin"] + sign * deltas["ebit_margin"],
            tax_rate=base_assumptions["tax_rate"],
            da_pct_revenue=base_assumptions["da_pct_revenue"],
            capex_pct_revenue=base_assumptions["capex_pct_revenue"],
            nwc_pct_revenue=base_assumptions["nwc_pct_revenue"],
            wacc=base_assumptions["wacc"] + wacc_sign * deltas["wacc"],
            terminal_growth=base_assumptions["terminal_growth"] + sign * deltas["terminal_growth"],
            projection_years=projection_years,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
        )
    return scenarios


# =========================================================================== #
# 5. Sensitivity analysis (WACC x Terminal Growth)
# =========================================================================== #

def sensitivity_table(
    base_revenue: float,
    base_assumptions: dict[str, float],
    net_debt: float,
    shares_outstanding: Optional[float],
    projection_years: int,
    wacc_range: list[float],
    terminal_growth_range: list[float],
) -> list[list[Optional[float]]]:
    """
    Build a WACC x Terminal-Growth grid of intrinsic value per share,
    holding all other Base assumptions fixed. Rows = wacc_range,
    columns = terminal_growth_range.
    """
    grid: list[list[Optional[float]]] = []
    for wacc in wacc_range:
        row: list[Optional[float]] = []
        for tg in terminal_growth_range:
            result = run_dcf_advanced(
                scenario="Sensitivity",
                base_revenue=base_revenue,
                revenue_growth=base_assumptions["revenue_growth"],
                ebit_margin=base_assumptions["ebit_margin"],
                tax_rate=base_assumptions["tax_rate"],
                da_pct_revenue=base_assumptions["da_pct_revenue"],
                capex_pct_revenue=base_assumptions["capex_pct_revenue"],
                nwc_pct_revenue=base_assumptions["nwc_pct_revenue"],
                wacc=wacc,
                terminal_growth=tg,
                projection_years=projection_years,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
            )
            row.append(result.intrinsic_value_per_share)
        grid.append(row)
    return grid


# =========================================================================== #
# 6. Suitability warnings
# =========================================================================== #

_BANK_INSURANCE_KEYWORDS = ["bank", "insurance", "reinsurance", "thrifts", "credit services"]
_CYCLICAL_INDUSTRY_KEYWORDS = [
    "oil", "gas", "steel", "mining", "airline", "auto", "shipping",
    "semiconductor", "metals", "coal", "drilling",
]


def check_dcf_suitability(
    sector: Optional[str],
    industry: Optional[str],
    base_fcff: Optional[float],
    hist: Optional[HistoricalFinancials],
    revenue_growth_assumption: float,
) -> list[str]:
    """
    Return plain-English warnings about why a basic single-stage FCFF DCF
    may be less reliable for this company. An EMPTY list does not mean the
    DCF is guaranteed reliable — only that these specific heuristic checks
    didn't flag anything.
    """
    warnings: list[str] = []
    industry_l = (industry or "").lower()
    sector_l = (sector or "").lower()

    if any(kw in industry_l or kw in sector_l for kw in _BANK_INSURANCE_KEYWORDS):
        warnings.append(
            "This company is in banking, insurance, or financial services. "
            "A basic FCFF DCF is generally NOT appropriate here — these "
            "businesses' capital structure and cash flow drivers (deposits, "
            "loan loss provisions, float) don't map cleanly onto this model. "
            "A dividend discount model or excess-return model is typically "
            "used instead."
        )

    if any(kw in industry_l for kw in _CYCLICAL_INDUSTRY_KEYWORDS):
        warnings.append(
            "This industry is commonly cyclical (revenue/margins swing with "
            "the broader economic or commodity cycle). A single Base growth "
            "and margin assumption applied evenly across 5 years may not "
            "reflect boom/bust swings — treat the Bear-to-Bull range as more "
            "informative than any single number."
        )

    if base_fcff is not None and base_fcff <= 0:
        warnings.append(
            "This company's most recent free cash flow is negative or zero. "
            "Since a DCF discounts future cash flow, a negative starting "
            "point makes the forecast far more assumption-dependent and "
            "less reliable."
        )

    if hist is None or len(hist.years) < 2:
        warnings.append(
            "Limited multi-year financial history is available for this "
            "ticker, so growth/margin assumptions rest on very little data "
            "(or defaults) — treat the output with extra caution. This is "
            "common for newly public or very young companies."
        )

    if revenue_growth_assumption > 0.25:
        warnings.append(
            "The revenue growth assumption is very high (over 25% per year). "
            "Sustaining that for 5 straight years is uncommon in practice, "
            "and small changes to this single number will swing the "
            "valuation a lot."
        )

    return warnings


# =========================================================================== #
# 7. Plain-English narrative (the "AI explains, doesn't invent" step)
# =========================================================================== #

def generate_dcf_narrative(
    company_name: str,
    symbol: str,
    scenarios: dict[str, DCFScenarioResult],
    current_price: Optional[float],
) -> str:
    """
    Turn ALREADY-COMPUTED scenario results into a plain-English summary.

    This function performs no valuation math of its own and invents no
    numbers — it only describes results produced upstream by
    run_dcf_advanced() / run_bear_base_bull(), the same "numbers first,
    narrative second" pattern the app already uses for its Buffett/Graham/
    Risk scores.
    """
    base = scenarios.get("Base")
    if base is None or base.intrinsic_value_per_share is None:
        return (
            f"A DCF could not produce a per-share estimate for {company_name} "
            f"({symbol}) — shares outstanding data was unavailable."
        )

    iv = base.intrinsic_value_per_share
    wacc = base.assumptions["wacc"]
    tg = base.assumptions["terminal_growth"]

    lines = [
        f"Using a WACC of {wacc:.1%} and a terminal growth rate of {tg:.1%}, "
        f"the Base-case model estimates {company_name} ({symbol})'s intrinsic "
        f"value at ${iv:,.2f} per share."
    ]

    if current_price:
        diff_pct = (iv - current_price) / current_price * 100
        if diff_pct > 0:
            lines.append(
                f"That is {diff_pct:.1f}% above the current price of "
                f"${current_price:,.2f} — under these assumptions, the model "
                f"considers the stock potentially undervalued."
            )
        else:
            lines.append(
                f"That is {abs(diff_pct):.1f}% below the current price of "
                f"${current_price:,.2f} — under these assumptions, the model "
                f"considers the stock potentially overvalued."
            )

    bear, bull = scenarios.get("Bear"), scenarios.get("Bull")
    if bear and bull and bear.intrinsic_value_per_share and bull.intrinsic_value_per_share:
        lines.append(
            f"Across the Bear-to-Bull range, the estimate spans "
            f"${bear.intrinsic_value_per_share:,.2f} to "
            f"${bull.intrinsic_value_per_share:,.2f} per share — that spread "
            f"reflects how sensitive DCF outputs are to growth, margin, and "
            f"discount-rate assumptions. It is not a prediction of where the "
            f"stock will actually trade."
        )

    return " ".join(lines)


# =========================================================================== #
# SECTION 6: NEWS FORMATTING (originally news.py)
# =========================================================================== #


def normalize_news_item(raw: dict[str, Any]) -> dict[str, Any]:
    """
    yfinance's `Ticker.news` payload has shifted shape across versions
    (sometimes flat, sometimes nested under a "content" key). This function
    normalizes both shapes into a consistent dict:
        {title, publisher, link, published, thumbnail}
    """
    content = raw.get("content", raw)

    title = content.get("title") or raw.get("title") or "Untitled"

    publisher = (
        (content.get("provider") or {}).get("displayName")
        if isinstance(content.get("provider"), dict)
        else raw.get("publisher")
    ) or "Unknown source"

    link = (
        (content.get("canonicalUrl") or {}).get("url")
        if isinstance(content.get("canonicalUrl"), dict)
        else raw.get("link")
    ) or "#"

    published_raw = content.get("pubDate") or raw.get("providerPublishTime")
    published_str = "Unknown date"
    try:
        if isinstance(published_raw, (int, float)):
            published_str = dt.datetime.fromtimestamp(published_raw).strftime(
                "%b %d, %Y %H:%M"
            )
        elif isinstance(published_raw, str):
            # ISO 8601 format e.g. 2024-05-01T12:00:00Z
            parsed = dt.datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            published_str = parsed.strftime("%b %d, %Y %H:%M")
    except Exception:
        pass

    thumbnail = None
    thumb_data = content.get("thumbnail") or raw.get("thumbnail")
    if isinstance(thumb_data, dict):
        resolutions = thumb_data.get("resolutions") or []
        if resolutions:
            thumbnail = resolutions[0].get("url")
        elif "originalUrl" in thumb_data:
            thumbnail = thumb_data.get("originalUrl")

    summary = content.get("summary") or content.get("description") or ""

    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published": published_str,
        "thumbnail": thumbnail,
        "summary": summary,
    }


def normalize_news_list(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a full list of raw news items."""
    normalized = []
    for item in raw_items:
        try:
            normalized.append(normalize_news_item(item))
        except Exception:
            continue
    return normalized


# =========================================================================== #
# SECTION 7: PORTFOLIO & WATCHLIST STATE (originally portfolio.py)
# =========================================================================== #
WATCHLIST_KEY = "watchlist"
PORTFOLIO_KEY = "portfolio_holdings"


@dataclass
class Holding:
    """A single portfolio position."""
    symbol: str
    shares: float
    cost_basis: float  # average cost per share


def init_state() -> None:
    """Ensure the required session_state containers exist."""
    if WATCHLIST_KEY not in st.session_state:
        st.session_state[WATCHLIST_KEY] = []
    if PORTFOLIO_KEY not in st.session_state:
        st.session_state[PORTFOLIO_KEY] = []
    if ALERTS_KEY not in st.session_state:  # NEW: price alerts state
        st.session_state[ALERTS_KEY] = []


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #


def add_to_watchlist(symbol: str) -> None:
    """Add a ticker to the watchlist if not already present."""
    init_state()
    symbol = symbol.strip().upper()
    if symbol and symbol not in st.session_state[WATCHLIST_KEY]:
        st.session_state[WATCHLIST_KEY].append(symbol)


def remove_from_watchlist(symbol: str) -> None:
    """Remove a ticker from the watchlist."""
    init_state()
    symbol = symbol.strip().upper()
    if symbol in st.session_state[WATCHLIST_KEY]:
        st.session_state[WATCHLIST_KEY].remove(symbol)


def get_watchlist_snapshot() -> pd.DataFrame:
    """
    Build a summary table (price, change %, market cap) for every ticker
    currently on the watchlist.
    """
    init_state()
    rows = []
    for symbol in st.session_state[WATCHLIST_KEY]:
        live = get_live_price(symbol)
        info = get_company_info(symbol)
        rows.append(
            {
                "Symbol": symbol,
                "Name": info.get("shortName", "—"),
                "Price": live.get("price"),
                "Change": live.get("change"),
                "% Change": live.get("pct_change"),
                "Market Cap": info.get("marketCap"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Portfolio calculator
# --------------------------------------------------------------------------- #


def add_holding(symbol: str, shares: float, cost_basis: float) -> None:
    """Add or update a portfolio holding."""
    init_state()
    symbol = symbol.strip().upper()
    if not symbol or shares <= 0 or cost_basis < 0:
        return

    holdings = st.session_state[PORTFOLIO_KEY]
    for h in holdings:
        if h["symbol"] == symbol:
            h["shares"] = shares
            h["cost_basis"] = cost_basis
            return
    holdings.append({"symbol": symbol, "shares": shares, "cost_basis": cost_basis})


def remove_holding(symbol: str) -> None:
    """Remove a holding from the portfolio."""
    init_state()
    symbol = symbol.strip().upper()
    st.session_state[PORTFOLIO_KEY] = [
        h for h in st.session_state[PORTFOLIO_KEY] if h["symbol"] != symbol
    ]


def get_portfolio_summary() -> pd.DataFrame:
    """
    Compute a full valuation table for the current portfolio: market value,
    cost basis, unrealized P/L in dollars and percent, and portfolio weight.
    """
    init_state()
    holdings = st.session_state[PORTFOLIO_KEY]
    if not holdings:
        return pd.DataFrame()

    rows = []
    for h in holdings:
        symbol = h["symbol"]
        shares = h["shares"]
        cost_basis = h["cost_basis"]
        live = get_live_price(symbol)
        price = live.get("price") or 0.0

        market_value = price * shares
        total_cost = cost_basis * shares
        pl_dollars = market_value - total_cost
        pl_pct = (pl_dollars / total_cost * 100) if total_cost > 0 else 0.0

        rows.append(
            {
                "Symbol": symbol,
                "Shares": shares,
                "Avg Cost": cost_basis,
                "Current Price": price,
                "Market Value": market_value,
                "Total Cost": total_cost,
                "Unrealized P/L ($)": pl_dollars,
                "Unrealized P/L (%)": pl_pct,
            }
        )

    df = pd.DataFrame(rows)
    total_value = df["Market Value"].sum()
    df["Weight (%)"] = (
        (df["Market Value"] / total_value * 100) if total_value > 0 else 0.0
    )
    return df


def get_portfolio_totals(summary_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate totals (value, cost, P/L) for the whole portfolio."""
    if summary_df.empty:
        return {"total_value": 0.0, "total_cost": 0.0, "total_pl": 0.0, "total_pl_pct": 0.0}

    total_value = float(summary_df["Market Value"].sum())
    total_cost = float(summary_df["Total Cost"].sum())
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0.0

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pl": total_pl,
        "total_pl_pct": total_pl_pct,
    }


# =========================================================================== #
# SECTION 7B: EDUCATION MODE CONTENT  ***NEW — Education Mode feature***
# ---------------------------------------------------------------------
# Everything in this section is purely additive. None of it is imported,
# executed, or displayed unless the "🎓 Education Mode" toggle in the
# sidebar is switched on, so default (non-education) app behavior is
# completely unaffected by this section existing.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 7B.1 — Metric explanations (Requirement 2: expandable metric education,
# and Requirement 5: vocabulary cards reuse this same data).
# --------------------------------------------------------------------------- #

METRIC_EDUCATION: dict[str, dict[str, str]] = {
    "Market Capitalization": {
        "definition": "Market capitalization ('market cap') is the total dollar value of all a company's outstanding shares — its share price multiplied by the number of shares.",
        "why_it_matters": "It tells investors how big a company is and is used to classify stocks as small-cap, mid-cap, or large-cap, which affects risk and growth potential.",
        "example": "If a company has 1 billion shares priced at $50 each, its market cap is $50 billion.",
        "analogy": "It's the total price tag on a company if you wanted to buy the whole business at today's stock price.",
        "fun_fact": "Apple was the first U.S. company to reach a $1 trillion market cap in 2018, and later became the first to reach $3 trillion.",
        "common_mistake": "Students often confuse market cap with revenue or profit — market cap only reflects what investors will pay for the whole company, not its sales or earnings.",
    },
    "P/E Ratio": {
        "definition": "The Price-to-Earnings (P/E) ratio compares a company's share price to its earnings per share (EPS), showing how much investors pay for each dollar of profit.",
        "why_it_matters": "It helps investors judge whether a stock looks expensive or cheap relative to its profits, and compare valuations across companies.",
        "example": "A stock trading at $100 with an EPS of $5 has a P/E of 20 — investors are paying $20 for every $1 of annual profit.",
        "analogy": "It's like figuring out how many years of current profit it would take to 'pay back' the price you paid for a business.",
        "fun_fact": "A high P/E doesn't always mean overpriced — fast-growing companies often trade at high P/E ratios because investors expect much bigger future profits.",
        "common_mistake": "Students often assume a low P/E always means a 'bargain,' but it can also mean investors expect the company's earnings to decline.",
    },
    "Forward P/E": {
        "definition": "Forward P/E uses analysts' estimated future earnings, instead of past earnings, to calculate the price-to-earnings ratio.",
        "why_it_matters": "It gives a forward-looking view of valuation, useful for companies expected to grow (or shrink) quickly.",
        "example": "If a stock's trailing P/E is 30 but its forward P/E is 20, analysts expect earnings to grow significantly next year.",
        "analogy": "It's like judging a restaurant not just by last year's reviews, but by how good people expect the food to be next year.",
        "fun_fact": "Forward P/E can be wrong — it relies on analyst predictions, which don't always come true.",
        "common_mistake": "Students sometimes treat forward P/E as a guaranteed number rather than an estimate that can change.",
    },
    "PEG Ratio": {
        "definition": "The PEG ratio divides the P/E ratio by a company's expected earnings growth rate, adjusting valuation for growth.",
        "why_it_matters": "It helps investors compare 'expensive-looking' fast-growing companies to slower-growing ones on a more level playing field.",
        "example": "A stock with a P/E of 30 and 30% expected growth has a PEG of 1.0, generally considered fairly valued.",
        "analogy": "It's like adjusting a tree's price not just by how tall it is now, but by how fast it's expected to keep growing.",
        "fun_fact": "A PEG ratio below 1.0 is often seen as a sign of undervaluation relative to growth, an idea popularized by investor Peter Lynch.",
        "common_mistake": "Students often forget PEG relies on *estimated* growth rates, which can be inaccurate or overly optimistic.",
    },
    "Beta": {
        "definition": "Beta measures how much a stock's price moves compared to the overall market. A beta of 1.0 means it moves in line with the market.",
        "why_it_matters": "It helps investors understand a stock's volatility and risk relative to the broader market.",
        "example": "A stock with a beta of 1.5 tends to move 50% more than the market — if the market rises 10%, the stock might rise 15%.",
        "analogy": "Think of a small boat versus a cruise ship in rough water — the small boat (high beta) rocks around far more.",
        "fun_fact": "Utility companies often have low betas (under 1.0) because their business is stable, while tech startups often have high betas.",
        "common_mistake": "Students often think beta measures a stock's overall risk, but it only measures volatility *relative to the market*, not company-specific risks like debt.",
    },
    "Dividend Yield": {
        "definition": "Dividend yield shows how much a company pays shareholders in dividends each year, as a percentage of its share price.",
        "why_it_matters": "It helps income-focused investors evaluate how much cash return they get just from holding the stock, aside from price gains.",
        "example": "If a stock trades at $100 and pays $3 per share in annual dividends, its dividend yield is 3%.",
        "analogy": "It's similar to the interest rate on a savings account, but paid out by a company instead of a bank.",
        "fun_fact": "Not all companies pay dividends — many growth companies, like Amazon, reinvest profits back into the business instead.",
        "common_mistake": "Students often assume a high dividend yield is always good, but sometimes it means the stock price has fallen sharply, artificially inflating the yield.",
    },
    "EPS": {
        "definition": "Earnings Per Share (EPS) is a company's total profit divided by its number of outstanding shares.",
        "why_it_matters": "It's a key measure of profitability per share, used to calculate the P/E ratio and compare profitability over time.",
        "example": "A company with $1 billion in profit and 500 million shares has an EPS of $2.",
        "analogy": "If a company's profit were a pizza, EPS tells you how big each shareholder's slice is.",
        "fun_fact": "Companies can boost EPS by buying back their own shares, even if total profit doesn't grow, since there are fewer shares to divide profit among.",
        "common_mistake": "Students sometimes confuse EPS with dividends — EPS is total profit per share, not the cash actually paid out to shareholders.",
    },
    "Revenue": {
        "definition": "Revenue (or 'sales' — the 'top line') is the total money a company brings in from selling products or services, before any expenses are subtracted.",
        "why_it_matters": "It shows how much business a company is actually doing, the starting point for measuring growth and profitability.",
        "example": "If a company sells 1 million phones at $500 each, it generates $500 million in revenue.",
        "analogy": "Revenue is like your total paycheck before taxes and expenses are taken out — it's not what you keep, just what came in.",
        "fun_fact": "A company can have huge revenue and still lose money if its expenses are even bigger — revenue alone doesn't guarantee profit.",
        "common_mistake": "Students often assume high revenue means a company is profitable, but revenue says nothing about costs or actual profit.",
    },
    "Gross Margin": {
        "definition": "Gross margin is the percentage of revenue left after subtracting the direct cost of producing goods or services (cost of goods sold).",
        "why_it_matters": "It shows how efficiently a company produces its products before considering other expenses like marketing or R&D.",
        "example": "If a company earns $100 in revenue and it costs $40 to make the product, gross margin is 60%.",
        "analogy": "It's the profit a lemonade stand makes after paying for lemons and sugar, before paying for the sign or the table.",
        "fun_fact": "Software companies often have gross margins above 70-80% because digital products cost very little to reproduce once built.",
        "common_mistake": "Students often confuse gross margin with net margin — gross margin ignores overhead costs like salaries, rent, and taxes.",
    },
    "Operating Margin": {
        "definition": "Operating margin measures the percentage of revenue left after subtracting cost of goods sold AND operating expenses (salaries, rent, marketing).",
        "why_it_matters": "It shows how efficiently a company runs its core business operations, separate from taxes and interest expenses.",
        "example": "If a company has $100 in revenue and $20 in operating profit, its operating margin is 20%.",
        "analogy": "It's like your take-home pay after covering rent and groceries, but before paying taxes or loan interest.",
        "fun_fact": "Operating margin is often used to compare efficiency between companies in the same industry, since it excludes financing decisions.",
        "common_mistake": "Students sometimes mix this up with net margin, which also subtracts taxes and interest — operating margin stops one step earlier.",
    },
    "Net Margin": {
        "definition": "Net margin (net profit margin) is the percentage of revenue that remains as actual profit after ALL expenses, including taxes and interest.",
        "why_it_matters": "It's the ultimate measure of how much profit a company keeps from every dollar of sales.",
        "example": "If a company earns $100 in revenue and keeps $10 in profit after everything, its net margin is 10%.",
        "analogy": "It's like your final take-home savings after rent, groceries, taxes, and loan payments — what's truly left in your pocket.",
        "fun_fact": "Grocery stores often operate on razor-thin net margins (1-3%), while software companies can have net margins above 20-30%.",
        "common_mistake": "Students often assume all industries should have similar margins — 'normal' margin ranges vary hugely by industry.",
    },
    "ROE": {
        "definition": "Return on Equity (ROE) measures how much profit a company generates for every dollar of shareholders' equity (money invested by owners).",
        "why_it_matters": "It shows how efficiently management uses shareholders' money to generate profits.",
        "example": "If a company has $200 million in profit and $1 billion in shareholder equity, its ROE is 20%.",
        "analogy": "It's like measuring how much interest you'd earn if your invested savings were the company's equity.",
        "fun_fact": "Warren Buffett often looks for companies with consistently high ROE (above 15%) as a sign of a strong competitive advantage.",
        "common_mistake": "Students often forget ROE can be artificially boosted by taking on more debt, which increases risk even as ROE looks better.",
    },
    "ROA": {
        "definition": "Return on Assets (ROA) measures how much profit a company generates for every dollar of total assets it owns (not just shareholder equity).",
        "why_it_matters": "It shows how efficiently a company uses ALL its resources — including borrowed money — to generate profit.",
        "example": "If a company has $100 million in profit and $2 billion in total assets, its ROA is 5%.",
        "analogy": "If ROE measures how well you use your own savings, ROA measures how well you use everything you own, including things bought with a loan.",
        "fun_fact": "Banks typically have low ROA (often under 2%) because they hold enormous assets (deposits and loans) relative to their profit.",
        "common_mistake": "Students often confuse ROA with ROE — ROA includes debt-financed assets, while ROE only reflects owners' equity.",
    },
    "Debt to Equity": {
        "definition": "The debt-to-equity ratio compares a company's total debt to its shareholders' equity, showing how much of the business is financed by borrowing versus ownership.",
        "why_it_matters": "It helps investors assess financial risk — companies with high debt loads can be riskier, especially during economic downturns.",
        "example": "A debt-to-equity ratio of 100 means a company has as much debt as it has shareholder equity.",
        "analogy": "It's like comparing how much of your house is paid off with a mortgage versus how much you actually own outright.",
        "fun_fact": "Capital-intensive industries like utilities and airlines often carry much higher debt-to-equity ratios than software companies.",
        "common_mistake": "Students often assume all debt is bad, but companies can use debt strategically to grow faster — the key is whether they can comfortably repay it.",
    },
    "Current Ratio": {
        "definition": "The current ratio compares a company's current assets (cash and things convertible to cash within a year) to its current liabilities (bills due within a year).",
        "why_it_matters": "It measures a company's short-term ability to pay its bills without borrowing more money or selling long-term assets.",
        "example": "A current ratio of 2.0 means a company has $2 in short-term assets for every $1 of short-term debt.",
        "analogy": "It's like checking whether you have enough money in your checking account to cover this month's bills.",
        "fun_fact": "A current ratio far above 2 or 3 isn't always great — it can mean a company is hoarding cash instead of investing it productively.",
        "common_mistake": "Students often think a higher current ratio is always better, but an excessively high ratio can signal inefficient use of assets.",
    },
    "Free Cash Flow": {
        "definition": "Free Cash Flow (FCF) is the cash a company generates from operations after subtracting money spent on equipment, property, and other capital investments.",
        "why_it_matters": "It represents the actual cash a company has left to pay dividends, buy back stock, pay down debt, or reinvest in growth.",
        "example": "If a company generates $500 million from operations and spends $100 million on new equipment, its FCF is $400 million.",
        "analogy": "It's like your paycheck after covering all your monthly bills and necessary big purchases — what's truly free to save or spend elsewhere.",
        "fun_fact": "Some fast-growing companies report accounting losses but still generate positive free cash flow, since non-cash expenses reduce reported profit without using actual cash.",
        "common_mistake": "Students often confuse free cash flow with net income — a company can be profitable on paper but have negative free cash flow if it's investing heavily.",
    },
    "Enterprise Value": {
        "definition": "Enterprise Value (EV) measures a company's total value: market cap plus total debt, minus cash and cash equivalents.",
        "why_it_matters": "It represents the theoretical cost to acquire the entire company, including paying off its debt, useful for comparing companies with different debt levels.",
        "example": "A company with a $50 billion market cap, $10 billion in debt, and $5 billion in cash has an enterprise value of $55 billion.",
        "analogy": "It's like the true cost of buying a house — the sticker price (market cap) plus any mortgage you'd pay off (debt), minus cash left in the deal.",
        "fun_fact": "Enterprise value can be higher OR lower than market cap, depending on whether a company has more debt or more cash.",
        "common_mistake": "Students often assume EV and market cap are the same thing — but EV accounts for debt and cash, giving a fuller picture of acquisition cost.",
    },
    "EBITDA": {
        "definition": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization — a measure of core operating profitability.",
        "why_it_matters": "It lets investors compare profitability across companies without the effects of financing decisions, tax rates, or accounting choices.",
        "example": "A company with $50 million in operating profit plus $10 million in depreciation/amortization has an EBITDA of roughly $60 million.",
        "analogy": "It's like judging how well a lemonade stand runs day-to-day, ignoring how it was financed or how much the stand itself has worn out.",
        "fun_fact": "EBITDA is often used in valuation ratios like EV/EBITDA, especially for capital-intensive businesses like telecoms and airlines.",
        "common_mistake": "Students often treat EBITDA as equivalent to cash flow, but it ignores capital expenditures and real cash costs like interest and taxes.",
    },
    # ----- NEW: DCF MODULE EDUCATION CONTENT ----- #
    "WACC": {
        "definition": "WACC (Weighted Average Cost of Capital) is the average rate of return a company must pay to all of its investors — stockholders and lenders — blended by how much of the company is financed by each.",
        "why_it_matters": "It's the discount rate used in a DCF to convert future cash into today's dollars — a higher WACC makes future cash worth less today, and vice versa.",
        "example": "If a company is 80% financed by stock (costing 10%/year) and 20% by debt (costing 5%/year after tax), WACC = 0.8×10% + 0.2×5% = 9%.",
        "analogy": "It's like the blended interest rate on a household that has both a low-interest mortgage and a higher-interest credit card — the overall rate depends on how much is owed on each.",
        "fun_fact": "Small changes in WACC can swing a DCF valuation by 20-30% or more — it's one of the most sensitive inputs in the whole model.",
        "common_mistake": "Students often assume a lower WACC is always 'better,' but WACC should reflect real investor risk, not be lowered just to produce a higher valuation.",
    },
    "NOPAT": {
        "definition": "NOPAT (Net Operating Profit After Tax) is a company's operating profit after removing taxes, but before subtracting interest payments.",
        "why_it_matters": "It measures core-business profitability independent of how the company is financed with debt — exactly what a DCF needs for unbiased cash flow.",
        "example": "If EBIT is $100 and the tax rate is 25%, NOPAT = $100 × (1 − 0.25) = $75.",
        "analogy": "It's like your take-home pay after taxes, but before rent or a car loan — it isolates what your job earned you from how you chose to finance your life.",
        "fun_fact": "NOPAT deliberately ignores interest expense, because a DCF discounts cash available to ALL investors (stock AND debt holders), not just shareholders.",
        "common_mistake": "Students often confuse NOPAT with net income — net income subtracts interest expense, NOPAT does not.",
    },
    "EBIT": {
        "definition": "EBIT (Earnings Before Interest and Taxes) is a company's profit from core operations, before interest payments on debt or income taxes.",
        "why_it_matters": "It shows how profitable the underlying business is, independent of financing or tax rate — a fair starting point for valuation.",
        "example": "Revenue $500 − Cost of Goods Sold $300 − Operating Expenses $100 = EBIT of $100.",
        "analogy": "It's like judging how good a lemonade stand's business idea is before considering whether it was funded by a loan (interest) or how much tax the town charges.",
        "fun_fact": "EBIT is sometimes called 'operating income' — they're often (though not always) the exact same line on a company's income statement.",
        "common_mistake": "Students often mix up EBIT with EBITDA — EBIT already includes depreciation and amortization as expenses; EBITDA adds them back.",
    },
    "Terminal Value": {
        "definition": "Terminal Value is the estimated value of all of a company's cash flows beyond the explicit forecast period, assumed to grow at a constant rate forever.",
        "why_it_matters": "A company doesn't stop existing after 5 years — Terminal Value captures 'everything after that' in one number, and often makes up the majority of a DCF's total value.",
        "example": "If Year 5 FCFF is $200, terminal growth is 3%, and WACC is 9%, Terminal Value ≈ $200 × 1.03 / (0.09 − 0.03) ≈ $3,433.",
        "analogy": "It's like valuing a fruit tree by estimating every future harvest for as long as it keeps producing, discounted back to today — not just counting this year's apples.",
        "fun_fact": "In many real-world DCFs, Terminal Value accounts for 60-80% of the total estimated value.",
        "common_mistake": "Students often forget Terminal Value must itself be discounted back to today's dollars, and that the formula breaks if growth is set equal to or above WACC.",
    },
    "Net Working Capital (NWC)": {
        "definition": "Net Working Capital is the cash tied up in day-to-day operations — mainly inventory and money owed by customers, minus money owed to suppliers.",
        "why_it_matters": "As a company grows, it usually ties up MORE cash in things like inventory before that cash comes back — an increase in NWC is a real cost that reduces free cash flow.",
        "example": "If inventory and unpaid customer bills grow $50 more than what's owed to suppliers, that $50 is cash tied up and unavailable this year.",
        "analogy": "It's like a lemonade stand needing to buy a bigger stock of lemons and sugar before it can sell more lemonade — that upfront cash is tied up even though the stand is growing.",
        "fun_fact": "Subscription software companies often have NEGATIVE change in NWC as they grow, since customers pay upfront — growth actually generates extra cash instead of using it.",
        "common_mistake": "Students often assume growth is always cash-positive, but for many businesses fast growth actually consumes cash in the short term via rising NWC.",
    },
    "FCFF": {
        "definition": "Free Cash Flow to the Firm (FCFF) is the cash a company generates that's available to ALL investors — stockholders and lenders — after operating costs and reinvestment.",
        "why_it_matters": "FCFF is the actual number a DCF discounts — it's the cash the company could hand to investors each year, which is exactly what a DCF is trying to value.",
        "example": "FCFF = NOPAT + D&A − CapEx − Change in NWC. If NOPAT is $75, D&A is $20, CapEx is $30, and Change in NWC is $5: FCFF = 75+20−30−5 = $60.",
        "analogy": "It's like your true money left over at month's end — take-home pay, plus adding back non-cash costs, minus what you spent on a car, minus extra cash tied up in a bigger grocery stockpile.",
        "fun_fact": "FCFF is calculated BEFORE debt payments — that's what makes it usable to value the whole company (Enterprise Value) before separately subtracting debt to reach Equity Value.",
        "common_mistake": "Students often try to build FCFF from Net Income instead of EBIT — but Net Income already subtracts interest expense, double-counting financing costs WACC already accounts for.",
    },
    # ----- END NEW: DCF MODULE EDUCATION CONTENT ----- #
}


def render_metric_education(term: str, key_suffix: str = "") -> None:
    """
    Render a classroom-friendly, expandable explanation for a financial term
    (Requirement 2: metric explanations, and reused for Requirement 5:
    vocabulary cards). Safe no-op if the term isn't in METRIC_EDUCATION.
    """
    content = METRIC_EDUCATION.get(term)
    if not content:
        return
    with st.expander(f"📚 What is {term}?", expanded=False):
        st.markdown(f"**Plain-English definition:** {content['definition']}")
        st.markdown(f"**Why investors care:** {content['why_it_matters']}")
        st.markdown(f"**Simple example:** {content['example']}")
        st.markdown(f"**Real-world analogy:** {content['analogy']}")
        st.markdown(f"**Interesting fact:** {content['fun_fact']}")
        st.caption(f"⚠️ Common mistake students make: {content['common_mistake']}")


# --------------------------------------------------------------------------- #
# 7B.2 — Student Mode helpers (Requirement 7: simplified labels).
# --------------------------------------------------------------------------- #

SIMPLE_LABELS: dict[str, str] = {
    "Market Cap": "Company's Total Value",
    "P/E (Trailing)": "Price vs. Profit Ratio",
    "Forward P/E": "Expected Price vs. Profit Ratio",
    "PEG Ratio": "Growth-Adjusted Price Ratio",
    "ROE": "Return on Owners' Money",
    "ROA": "Return on Everything Owned",
    "Revenue (TTM)": "Total Sales (Past Year)",
    "Debt / Equity": "Borrowed Money vs. Owned Money",
    "Current Ratio": "Ability to Pay Short-Term Bills",
    "Beta": "How Wild the Stock Price Swings",
    "Price / Book": "Price vs. Net Worth",
    "Price / Sales": "Price vs. Total Sales",
    "EV / EBITDA": "Company Price vs. Core Profit",
}


def label_for(base_label: str, education_mode: bool, student_mode: bool) -> str:
    """Return a simplified metric label when Student Mode is active, else the original label."""
    if education_mode and student_mode and base_label in SIMPLE_LABELS:
        return SIMPLE_LABELS[base_label]
    return base_label


# --------------------------------------------------------------------------- #
# 7B.3 — Learn tab content (Requirement 3).
# --------------------------------------------------------------------------- #

COMPANY_LEARN_CONTENT: dict[str, dict[str, Any]] = {
    "AAPL": {
        "overview": "Apple Inc. is one of the world's largest technology companies, known for designing and selling consumer electronics like the iPhone, iPad, and Mac computers, along with software and digital services.",
        "business_model": "Apple designs its own hardware and software, has products manufactured through contracted partners overseas, and sells them directly to consumers through its own stores, website, and retail partners.",
        "products": "iPhone, iPad, Mac computers, Apple Watch, AirPods, and services like the App Store, Apple Music, iCloud, and Apple TV+.",
        "how_it_makes_money": "Most of Apple's revenue comes from hardware sales, especially the iPhone, but its services division (App Store fees, subscriptions, licensing) has grown into a major, high-margin revenue source.",
        "competitors": "Samsung and Google (smartphones), Microsoft (computers/software), and Sony and Amazon in various hardware and services categories.",
        "industry": "Consumer electronics and technology.",
        "advantages": "A tightly integrated ecosystem of hardware, software, and services, an extremely strong brand, and a large base of loyal, repeat customers.",
        "risks": "Heavy reliance on iPhone sales, exposure to global supply chain disruptions, regulatory scrutiny of the App Store, and intense competition.",
        "facts": [
            "Apple was founded in 1976 in a garage by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
            "Apple was the first U.S. company to reach a $1 trillion market valuation, in 2018.",
            "Apple spends billions of dollars every year on research and development to design new products.",
        ],
    },
    "MSFT": {
        "overview": "Microsoft Corporation is a major technology company best known for its Windows operating system, Office productivity software, and its Azure cloud computing platform.",
        "business_model": "Microsoft sells software licenses and cloud subscriptions to businesses and consumers, increasingly earning recurring revenue through subscriptions rather than one-time software purchases.",
        "products": "Windows, Microsoft 365 (Word, Excel, Teams), Azure cloud services, Xbox gaming, and LinkedIn.",
        "how_it_makes_money": "A large and growing share of revenue comes from cloud computing (Azure) and subscription software (Microsoft 365), alongside gaming and professional networking (LinkedIn).",
        "competitors": "Amazon and Google (cloud computing), Apple and Google (operating systems), Sony and Nintendo (gaming).",
        "industry": "Software, cloud computing, and technology services.",
        "advantages": "A dominant position in workplace software, a fast-growing and profitable cloud business, and deep relationships with large businesses worldwide.",
        "risks": "Intense cloud competition from Amazon and Google, cybersecurity threats, and regulatory attention on its size and acquisitions.",
        "facts": [
            "Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
            "Microsoft Azure is one of the largest cloud computing platforms in the world, alongside Amazon's AWS.",
            "Microsoft owns LinkedIn, GitHub, and the video game franchise Minecraft.",
        ],
    },
    "GOOGL": {
        "overview": "Alphabet Inc. is the parent company of Google, the world's most widely used internet search engine, along with YouTube, Android, and various other technology ventures.",
        "business_model": "Alphabet earns most of its money by selling digital advertising space across Google Search, YouTube, and partner websites, using data about user behavior to target ads effectively.",
        "products": "Google Search, YouTube, Android, Google Cloud, Google Maps, and Chrome.",
        "how_it_makes_money": "The vast majority of revenue comes from digital advertising, with a smaller but fast-growing portion from Google Cloud computing services.",
        "competitors": "Meta and Amazon (digital advertising), Microsoft and Amazon (cloud computing), Apple (mobile operating systems).",
        "industry": "Internet services, digital advertising, and technology.",
        "advantages": "Dominance in internet search, an enormous amount of user data, and a huge ecosystem of free products (like Gmail and Maps) that keep users engaged.",
        "risks": "Heavy reliance on advertising revenue, antitrust and regulatory scrutiny worldwide, and competition from AI-powered search alternatives.",
        "facts": [
            "Google was founded in 1998 by Larry Page and Sergey Brin while they were PhD students at Stanford.",
            "YouTube, owned by Google, is one of the most-visited websites in the world.",
            "Alphabet was created in 2015 as a parent company to separate Google's core business from other ventures like self-driving car company Waymo.",
        ],
    },
    "AMZN": {
        "overview": "Amazon.com Inc. began as an online bookstore and has grown into one of the world's largest e-commerce and cloud computing companies.",
        "business_model": "Amazon earns money by selling and shipping products directly, taking a cut from third-party sellers using its platform, and renting out cloud computing infrastructure to other businesses.",
        "products": "Amazon.com marketplace, Amazon Prime, Amazon Web Services (AWS), Kindle, and Alexa devices.",
        "how_it_makes_money": "While retail sales generate the most revenue, Amazon Web Services (AWS) generates a disproportionately large share of the company's total profit due to its high margins.",
        "competitors": "Walmart and Target (retail), Microsoft and Google (cloud computing), Netflix and Disney (streaming).",
        "industry": "E-commerce, cloud computing, and logistics.",
        "advantages": "A massive logistics and delivery network, the leading cloud computing platform (AWS), and a huge, loyal Prime subscriber base.",
        "risks": "Thin profit margins in retail, labor and regulatory scrutiny, and rising competition in both e-commerce and cloud computing.",
        "facts": [
            "Amazon was founded by Jeff Bezos in 1994, originally selling only books.",
            "Amazon Web Services (AWS) generates a large share of Amazon's total operating profit despite being a smaller share of revenue.",
            "Amazon Prime has hundreds of millions of subscribers worldwide.",
        ],
    },
    "TSLA": {
        "overview": "Tesla, Inc. designs and manufactures electric vehicles, battery technology, and solar energy products, aiming to accelerate the world's shift to sustainable energy.",
        "business_model": "Tesla makes money primarily by manufacturing and selling electric vehicles directly to consumers, bypassing traditional car dealerships, along with energy storage and solar products.",
        "products": "Electric vehicles (Model S, 3, X, Y, Cybertruck), battery energy storage (Powerwall, Megapack), and solar panels.",
        "how_it_makes_money": "The vast majority of revenue comes from vehicle sales, with smaller contributions from energy generation/storage and regulatory credits sold to other automakers.",
        "competitors": "Traditional automakers like Ford, GM, and Toyota, and other EV makers like BYD, Rivian, and Chinese manufacturers.",
        "industry": "Automotive and clean energy.",
        "advantages": "A strong brand, an early lead in electric vehicle technology and battery manufacturing, and a proprietary charging network.",
        "risks": "Increasing competition from traditional and new EV makers, sensitivity to interest rates and vehicle affordability, and reliance on CEO Elon Musk's public image.",
        "facts": [
            "Tesla was founded in 2003 and named after inventor Nikola Tesla.",
            "Tesla built the world's largest electric vehicle charging network, called the Supercharger network.",
            "Tesla became one of the first pure electric vehicle companies to be consistently profitable on an annual basis.",
        ],
    },
    "META": {
        "overview": "Meta Platforms, Inc. (formerly Facebook) operates some of the world's largest social media platforms, including Facebook, Instagram, and WhatsApp.",
        "business_model": "Meta earns nearly all of its revenue from selling targeted digital advertising across its family of apps, using data about user interests and behavior.",
        "products": "Facebook, Instagram, WhatsApp, Messenger, and virtual/augmented reality products under its Reality Labs division.",
        "how_it_makes_money": "The overwhelming majority of revenue comes from advertising sold to businesses wanting to reach Meta's billions of users.",
        "competitors": "Google/YouTube and TikTok (advertising and attention), Snapchat (social media), and various VR companies (Reality Labs).",
        "industry": "Social media and digital advertising.",
        "advantages": "Billions of active users across its apps, powerful ad-targeting technology, and network effects that make its platforms more valuable as more people join.",
        "risks": "Heavy reliance on advertising revenue, competition from TikTok and other platforms, regulatory scrutiny over privacy and content moderation, and heavy investment in its Reality Labs (metaverse) division.",
        "facts": [
            "Facebook was founded by Mark Zuckerberg in 2004 while he was a student at Harvard University.",
            "Meta owns four of the world's most-used social media and messaging platforms: Facebook, Instagram, WhatsApp, and Messenger.",
            "Meta has invested tens of billions of dollars into virtual and augmented reality technology through its Reality Labs division.",
        ],
    },
    "NVDA": {
        "overview": "NVIDIA Corporation designs advanced computer chips, especially graphics processing units (GPUs), which have become essential hardware for gaming, artificial intelligence, and data centers.",
        "business_model": "NVIDIA designs chips and licenses/sells them to computer makers, cloud providers, and businesses, without owning the factories that manufacture the physical chips.",
        "products": "GeForce graphics cards (gaming), data center GPUs used for AI training, and specialized chips for autonomous vehicles.",
        "how_it_makes_money": "While NVIDIA started primarily in gaming graphics cards, most of its revenue and profit now comes from data center GPUs used to power artificial intelligence.",
        "competitors": "AMD and Intel (chip design), and increasingly, cloud companies designing their own AI chips.",
        "industry": "Semiconductors and artificial intelligence hardware.",
        "advantages": "A dominant position in AI training hardware, a mature software ecosystem (CUDA) that developers rely on, and strong brand recognition among gamers and AI researchers alike.",
        "risks": "Heavy reliance on a few large customers for AI chips, potential new competition from custom AI chips built by big tech companies, and geopolitical risks around chip manufacturing and export restrictions.",
        "facts": [
            "NVIDIA was founded in 1993 and originally focused on graphics cards for video games.",
            "NVIDIA's GPUs have become critical hardware for training large artificial intelligence models.",
            "NVIDIA became one of the most valuable companies in the world during the AI boom of the 2020s.",
        ],
    },
}


def get_learn_content(symbol: str, info: dict[str, Any]) -> dict[str, Any]:
    """
    Return Learn-tab content for a symbol. Falls back to a generic profile
    auto-built from live company info when no curated profile exists,
    so the Learn tab works for ANY valid ticker, not just the curated ones.
    """
    if symbol in COMPANY_LEARN_CONTENT:
        return COMPANY_LEARN_CONTENT[symbol]
    name = info.get("shortName") or info.get("longName") or symbol
    sector = info.get("sector") or "its industry"
    industry = info.get("industry") or "this market"
    summary = info.get("longBusinessSummary") or "No detailed description is available for this company."
    return {
        "overview": summary,
        "business_model": f"{name} operates in the {industry} industry, part of the broader {sector} sector.",
        "products": "See the Company Profile in the Overview tab for specific products and services.",
        "how_it_makes_money": f"{name} generates revenue primarily through its core operations in {industry}.",
        "competitors": "Competitor details aren't pre-loaded for this ticker — try researching other companies in the same industry for comparison.",
        "industry": f"{sector} — {industry}",
        "advantages": "Consider researching what makes this company different from others in its industry (brand, technology, cost, scale).",
        "risks": "Consider general business risks: competition, regulation, economic downturns, and changing consumer preferences.",
        "facts": [f"{name} is classified under the {sector} sector by Yahoo Finance."],
    }


def get_did_you_know(symbol: str, info: dict[str, Any]) -> str:
    """Return a random 'Did You Know?' fact for the given company (Requirement 8)."""
    content = get_learn_content(symbol, info)
    facts = content.get("facts") or []
    if facts:
        return random.choice(facts)
    name = info.get("shortName") or info.get("longName") or symbol
    return f"{name} has a market capitalization of {fmt_large_number(info.get('marketCap'))}."


def generate_reflection_questions(company_name: str, info: dict[str, Any]) -> list[str]:
    """Return 5 'Think Like an Investor' reflection questions tailored to the company (Requirement 9)."""
    margin = info.get("profitMargins")
    pe = info.get("trailingPE")
    sector = info.get("sector") or "its industry"
    return [
        f"Why do you think {company_name}'s profit margin is "
        f"{fmt_pct(margin) if margin is not None else 'what it is'}? What could cause it to rise or fall?",
        f"Would you personally invest in {company_name}? Why or why not?",
        f"How might competition within {sector} affect {company_name}'s future growth?",
        f"{company_name} has a P/E ratio of {fmt_ratio(pe) if pe is not None else 'N/A'}. "
        f"What does this tell you about what investors expect from the company?",
        f"What risks could hurt {company_name}'s stock price over the next five years?",
    ]


# --------------------------------------------------------------------------- #
# 7B.4 — Classroom tab template generators (Requirement 4).
# No LLM is used — these are predefined templates customized with the
# selected company's name, ticker, and live metrics.
# --------------------------------------------------------------------------- #

def generate_discussion_questions(company_name: str, symbol: str) -> list[str]:
    """Predefined classroom discussion questions for the selected company."""
    return [
        f"What does {company_name} ({symbol}) actually do, and who are its customers?",
        f"How does {company_name} make most of its money?",
        f"What are the biggest risks facing {company_name} today?",
        f"How might {company_name}'s industry look different in ten years?",
        f"If you were a financial advisor, would you recommend {company_name} stock to a "
        f"client saving for retirement? Why or why not?",
    ]


def generate_homework(company_name: str, symbol: str) -> str:
    """Predefined homework assignment template for the selected company."""
    return (
        f"**Homework Assignment: Researching {company_name} ({symbol})**\n\n"
        f"1. Look up {company_name}'s three most recent quarterly earnings reports. "
        f"Did revenue go up or down?\n"
        f"2. Identify two competitors of {company_name} and compare their market capitalization.\n"
        f"3. In 2-3 sentences, explain {company_name}'s business model in your own words.\n"
        f"4. Research what a $1,000 investment in {company_name} five years ago would be worth today.\n"
        f"5. Write a one-paragraph argument for OR against investing in {company_name}."
    )


def generate_quiz(company_name: str, symbol: str, info: dict[str, Any]) -> list[dict[str, str]]:
    """Predefined quiz questions for the selected company."""
    sector = info.get("sector", "N/A")
    return [
        {"question": f"What sector does {company_name} operate in?", "answer": sector},
        {
            "question": "True or False: a higher P/E ratio than the industry average always "
            "means a stock is a bad investment.",
            "answer": "False — a high P/E can also reflect strong expected future growth.",
        },
        {"question": "What does EPS stand for?", "answer": "Earnings Per Share"},
        {
            "question": f"Based on the Overview tab, is {company_name} a large-cap, mid-cap, "
            f"or small-cap company?",
            "answer": "Answers will vary based on current market cap data.",
        },
        {
            "question": "What is the difference between revenue and profit?",
            "answer": "Revenue is total sales; profit is what's left after subtracting expenses.",
        },
    ]


def generate_exit_ticket(company_name: str) -> list[str]:
    """Predefined exit-ticket prompts for the selected company."""
    return [
        f"In one sentence, summarize what {company_name} does.",
        "Name one financial metric you learned about today.",
        "What is one question you still have about how stocks are valued?",
    ]


def generate_vocabulary_assignment(terms: list[str]) -> str:
    """Predefined vocabulary assignment listing key financial terms."""
    term_list = ", ".join(terms)
    return (
        f"Define each of the following terms in your own words, and explain why it matters "
        f"to investors: {term_list}."
    )


def generate_case_study(company_name: str, symbol: str, info: dict[str, Any]) -> str:
    """Predefined case-study scenario for the selected company."""
    return (
        f"**Case Study: Should You Invest in {company_name}?**\n\n"
        f"You have been given a hypothetical $5,000 to invest for a class project. Using the "
        f"data available in this dashboard for {company_name} ({symbol}):\n\n"
        f"1. Summarize the company's business model and competitive position.\n"
        f"2. Analyze at least three financial metrics (e.g., P/E ratio, profit margin, "
        f"debt-to-equity) and explain what they suggest about the company's health.\n"
        f"3. Research one major risk facing the company and explain how it could affect the "
        f"stock price.\n"
        f"4. Make a final recommendation: would you invest the $5,000 in {company_name}? "
        f"Justify your answer using at least two pieces of evidence from the dashboard."
    )


# =========================================================================== #
# SECTION 7C: NEW FEATURE MODULES  ***NEW — Aug 2026 update***
# ---------------------------------------------------------------------
# Price Alerts, Peer/Sector Comparison, News Sentiment Analysis, and
# SMA-Crossover Backtesting. Purely additive — none of the existing tabs,
# charts, or analytics are modified by this section.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 7C.1 — Price Alerts / Watchlist Notifications
# --------------------------------------------------------------------------- #

ALERTS_KEY = "price_alerts"


def init_alerts_state() -> None:
    """Ensure the price-alerts session_state container exists."""
    if ALERTS_KEY not in st.session_state:
        st.session_state[ALERTS_KEY] = []


def add_price_alert(symbol: str, target_price: float, direction: str, note: str = "") -> None:
    """Add a new price alert for a ticker (direction is 'above' or 'below')."""
    init_alerts_state()
    symbol = symbol.strip().upper()
    if not symbol or target_price <= 0:
        return
    st.session_state[ALERTS_KEY].append(
        {
            "symbol": symbol,
            "target_price": float(target_price),
            "direction": direction,
            "note": note.strip(),
            "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )


def remove_price_alert(index: int) -> None:
    """Remove a price alert by its position in the list."""
    init_alerts_state()
    if 0 <= index < len(st.session_state[ALERTS_KEY]):
        st.session_state[ALERTS_KEY].pop(index)


def evaluate_price_alerts() -> pd.DataFrame:
    """
    Check every saved alert against the latest live price and return a
    DataFrame with a "Triggered" column indicating whether the alert
    condition has been met.
    """
    init_alerts_state()
    rows = []
    for i, alert in enumerate(st.session_state[ALERTS_KEY]):
        live = get_live_price(alert["symbol"])
        current_price = live.get("price")
        triggered = False
        if current_price is not None:
            if alert["direction"] == "above" and current_price >= alert["target_price"]:
                triggered = True
            elif alert["direction"] == "below" and current_price <= alert["target_price"]:
                triggered = True
        rows.append(
            {
                "Index": i,
                "Symbol": alert["symbol"],
                "Target": alert["target_price"],
                "Direction": alert["direction"],
                "Current Price": current_price,
                "Note": alert["note"],
                "Created": alert["created_at"],
                "Triggered": triggered,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7C.2 — Peer / Sector Comparison Benchmarking
# --------------------------------------------------------------------------- #

PEER_MAP: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ"],
    "MSFT": ["GOOGL", "AAPL", "ORCL", "IBM"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "AMZN": ["WMT", "TGT", "BABA"],
    "TSLA": ["GM", "F", "RIVN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "NVDA": ["AMD", "INTC", "QCOM"],
}


def get_peer_symbols(symbol: str) -> list[str]:
    """Return a curated peer group for a ticker, or an empty list if unknown."""
    return PEER_MAP.get(symbol.strip().upper(), [])


def build_peer_comparison_table(symbol: str, peers: list[str]) -> pd.DataFrame:
    """Build a side-by-side benchmarking table for a symbol and its peers."""
    rows = []
    for sym in [symbol] + peers:
        peer_info = get_company_info(sym)
        rows.append(
            {
                "Symbol": sym,
                "Market Cap": peer_info.get("marketCap"),
                "Revenue Growth": peer_info.get("revenueGrowth"),
                "Profit Margin": peer_info.get("profitMargins"),
                "P/E Ratio": peer_info.get("trailingPE"),
                "ROE": peer_info.get("returnOnEquity"),
            }
        )
    return pd.DataFrame(rows)


def peer_comparison_chart(comparison_df: pd.DataFrame, metric: str, symbol: str) -> go.Figure:
    """Bar chart comparing a single metric across a symbol and its peers."""
    fig = go.Figure()
    colors = [COLOR_ACCENT if s == symbol else COLOR_SMA20 for s in comparison_df["Symbol"]]
    fig.add_trace(
        go.Bar(
            x=comparison_df["Symbol"],
            y=comparison_df[metric],
            marker_color=colors,
        )
    )
    return _base_layout(fig, title=f"{metric} — Peer Comparison", height=380)


# --------------------------------------------------------------------------- #
# 7C.3 — News Sentiment Analysis (lexicon-based, no external API/LLM)
# --------------------------------------------------------------------------- #

POSITIVE_SENTIMENT_WORDS = {
    "beat", "beats", "growth", "surge", "surges", "soar", "soars", "record",
    "profit", "profits", "gain", "gains", "rally", "rallies", "upgrade",
    "upgraded", "strong", "outperform", "bullish", "rise", "rises", "rising",
    "boost", "boosts", "positive", "win", "wins", "expand", "expands",
    "expansion", "success", "successful", "breakthrough", "innovation",
    "buy", "top", "high", "higher", "optimistic",
}

NEGATIVE_SENTIMENT_WORDS = {
    "miss", "misses", "missed", "decline", "declines", "plunge", "plunges",
    "fall", "falls", "falling", "drop", "drops", "loss", "losses", "downgrade",
    "downgraded", "weak", "underperform", "bearish", "sell-off", "selloff",
    "lawsuit", "investigation", "recall", "layoff", "layoffs", "cut", "cuts",
    "warning", "warns", "risk", "risks", "concern", "concerns", "negative",
    "low", "lower", "crash", "slump", "struggl", "fraud", "scandal",
}


def analyze_headline_sentiment(text: str) -> tuple[str, int]:
    """
    Score a headline using a simple positive/negative keyword lexicon.
    Returns a (label, score) tuple where label is Positive/Negative/Neutral.
    """
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    score = 0
    for word in words:
        if word in POSITIVE_SENTIMENT_WORDS:
            score += 1
        elif word in NEGATIVE_SENTIMENT_WORDS:
            score -= 1
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score


def analyze_headlines_sentiment(news_items: list[dict[str, Any]]) -> pd.DataFrame:
    """Run sentiment analysis across a normalized list of news items."""
    rows = []
    for item in news_items:
        label, score = analyze_headline_sentiment(item.get("title", ""))
        rows.append({"title": item.get("title", ""), "sentiment": label, "score": score})
    return pd.DataFrame(rows)


def sentiment_summary_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """Bar chart of Positive / Neutral / Negative headline counts."""
    counts = sentiment_df["sentiment"].value_counts()
    order = ["Positive", "Neutral", "Negative"]
    values = [int(counts.get(label, 0)) for label in order]
    colors = [COLOR_UP, COLOR_ACCENT, COLOR_DOWN]
    fig = go.Figure(
        data=[go.Bar(x=order, y=values, marker_color=colors)]
    )
    return _base_layout(fig, title="News Sentiment Breakdown", height=320)


# --------------------------------------------------------------------------- #
# 7C.4 — Backtesting: Simple SMA Crossover Strategy
# --------------------------------------------------------------------------- #

def run_sma_crossover_backtest(
    hist: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    """
    Simulate a simple SMA-crossover strategy: go long whenever the short SMA
    is above the long SMA, and stay in cash otherwise. Returns the resulting
    equity curves (strategy vs. buy-and-hold), summary stats, and trade log.
    """
    if hist.empty or len(hist) < long_window + 2:
        return {"equity": pd.DataFrame(), "stats": {}, "trades": []}

    df = hist.copy()
    df["SMA_Short"] = df["Close"].rolling(short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(long_window).mean()
    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Position"] * df["Daily_Return"]
    df["Strategy_Equity"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["BuyHold_Equity"] = initial_capital * (1 + df["Daily_Return"]).cumprod()

    trades: list[dict[str, Any]] = []
    in_position = False
    entry_price = None
    entry_date = None
    for idx, row in df.iterrows():
        if row["Position"] == 1 and not in_position:
            in_position = True
            entry_price = row["Close"]
            entry_date = idx
        elif row["Position"] == 0 and in_position:
            in_position = False
            exit_price = row["Close"]
            ret_pct = (exit_price / entry_price - 1) * 100 if entry_price else 0.0
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                }
            )
    if in_position and entry_price:
        exit_price = df["Close"].iloc[-1]
        ret_pct = (exit_price / entry_price - 1) * 100
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": df.index[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": ret_pct,
            }
        )

    final_strategy = df["Strategy_Equity"].iloc[-1]
    final_buyhold = df["BuyHold_Equity"].iloc[-1]
    total_return_pct = (final_strategy / initial_capital - 1) * 100
    buyhold_return_pct = (final_buyhold / initial_capital - 1) * 100

    running_max = df["Strategy_Equity"].cummax()
    drawdown = (df["Strategy_Equity"] - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100) if not drawdown.empty else 0.0

    winning_trades = [t for t in trades if t["return_pct"] > 0]
    win_rate_pct = (len(winning_trades) / len(trades) * 100) if trades else 0.0

    stats = {
        "total_return_pct": total_return_pct,
        "buyhold_return_pct": buyhold_return_pct,
        "num_trades": len(trades),
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }
    return {"equity": df[["Strategy_Equity", "BuyHold_Equity"]], "stats": stats, "trades": trades}


def backtest_equity_chart(equity_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Line chart comparing strategy equity vs. buy-and-hold equity."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["Strategy_Equity"],
            name="SMA Crossover Strategy",
            line=dict(color=COLOR_ACCENT, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["BuyHold_Equity"],
            name="Buy & Hold",
            line=dict(color=COLOR_SMA20, width=2, dash="dash"),
        )
    )
    return _base_layout(fig, title=f"{symbol} — Strategy vs. Buy & Hold", height=420)


# --------------------------------------------------------------------------- #
# 7C.5 — Dark / Light Theme Toggle
# --------------------------------------------------------------------------- #

def build_custom_css(theme_mode: str) -> str:
    """
    Return the app's custom CSS, adapted for the selected theme mode.

    Typography: Inter (a highly legible, professional UI/display face used
    by Stripe, Linear, and most modern fintech products) for all headings
    and body text, paired with JetBrains Mono — a monospace face — for
    numeric data specifically (metric values, dataframes). Real trading
    terminals set figures in monospace so digits align in a fixed-width
    grid; that convention is deliberately carried through here rather than
    leaving every number in the same proportional font as the prose.
    """
    if theme_mode == "Light":
        bg_color = "#FFFFFF"
        surface_color = "#F7F8FA"
        text_color = "#1A1A1A"
        muted_text_color = "#5B6472"
        metric_bg = "rgba(0, 0, 0, 0.03)"
        metric_border = "rgba(0, 0, 0, 0.10)"
        tab_active_bg = "rgba(0, 0, 0, 0.05)"
        divider_color = "rgba(0, 0, 0, 0.08)"
    else:
        bg_color = "#0E1117"
        surface_color = "#161B22"
        text_color = "#E6E8EB"
        muted_text_color = "#9AA4B2"
        metric_bg = "rgba(127, 127, 127, 0.08)"
        metric_border = "rgba(127, 127, 127, 0.16)"
        tab_active_bg = "rgba(245, 166, 35, 0.10)"
        divider_color = "rgba(255, 255, 255, 0.08)"

    accent = "#F5A623"

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}

    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-size: 0.95rem;
        line-height: 1.55;
    }}

    /* Headings: tighter tracking, heavier weight, clear hierarchy */
    h1, h2, h3, h4 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: {text_color};
    }}
    h1 {{ font-size: 1.65rem; }}
    h2 {{ font-size: 1.3rem; }}
    h3, h4 {{ font-size: 1.05rem; margin-top: 0.4rem; }}
    p, span, div, label {{ font-family: 'Inter', sans-serif; }}

    /* Numeric data gets the monospace utility face — metrics, dataframes */
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    div[data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stDataFrame, .stDataFrame [class*="glide"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {metric_bg};
        border: 1px solid {metric_border};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.75rem;
        font-weight: 500;
        color: {muted_text_color};
        text-transform: uppercase;
        letter-spacing: 0.4px;
        opacity: 1;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {surface_color};
        border-right: 1px solid {divider_color};
    }}

    /* Tabs — wrap onto multiple rows instead of forcing a horizontal
       scrollbar, and give the active tab a clear, quiet indicator rather
       than a loud full-background highlight. */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        flex-wrap: wrap;
        row-gap: 4px;
        border-bottom: 1px solid {divider_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
        font-weight: 600;
        font-size: 0.88rem;
        color: {muted_text_color};
        transition: background-color 0.15s ease, color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {tab_active_bg};
        color: {text_color};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_active_bg};
        color: {accent} !important;
        border-bottom: 2px solid {accent};
    }}

    /* Buttons */
    .stButton button {{
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
    }}

    /* Dividers: quieter than Streamlit's default */
    hr {{ border-color: {divider_color}; }}

    /* Ticker badge */
    .ticker-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #F5A623, #F76B1C);
        color: #111;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }}

    .price-up {{ color: #00C805; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .price-down {{ color: #FF3B30; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}

    footer {{ visibility: hidden; }}
</style>
"""


# =========================================================================== #
# SECTION 7D: DIVIDEND HISTORY TRACKER & MULTI-CURRENCY SUPPORT  ***v1.3***
# ---------------------------------------------------------------------
# Purely additive — none of the existing tabs, charts, or analytics above
# are modified, aside from the small currency-awareness change to
# fmt_large_number() noted in Section 8's Helpers block.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 7D.1 — Dividend History Tracker
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_dividend_history(symbol: str) -> pd.Series:
    """Fetch the full historical dividend-payment series for a ticker."""
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        dividends = ticker.dividends
        return dividends if dividends is not None else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def compute_dividend_growth_rate(dividends: pd.Series) -> Optional[float]:
    """
    Compute the annualized dividend growth rate (CAGR) between the first
    and most recent full year of dividend payments. Returns None if there
    isn't enough history to compute a meaningful rate.
    """
    if dividends.empty:
        return None
    annual = dividends.groupby(dividends.index.year).sum()
    if len(annual) < 2:
        return None
    years = annual.index.tolist()
    n_years = years[-1] - years[0]
    first_value, last_value = annual.iloc[0], annual.iloc[-1]
    if n_years <= 0 or first_value <= 0:
        return None
    return (last_value / first_value) ** (1 / n_years) - 1


def dividend_history_chart(dividends: pd.Series, symbol: str) -> go.Figure:
    """Bar chart of a company's historical dividend payments."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=dividends.index, y=dividends.values, marker_color=COLOR_UP)
    )
    return _base_layout(fig, title=f"{symbol} — Dividend Payment History", height=380)


# --------------------------------------------------------------------------- #
# 7D.2 — Multi-Currency Display Support
# --------------------------------------------------------------------------- #

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "INR", "CAD", "AUD"]

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "INR": "₹",
    "CAD": "C$",
    "AUD": "A$",
}


@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_fx_rate(target_currency: str) -> float:
    """
    Fetch the current USD -> target_currency exchange rate via Yahoo
    Finance's FX tickers. Returns 1.0 (no conversion) for USD, or if the
    live rate can't be fetched, so the app never breaks on an FX outage.
    """
    target_currency = target_currency.strip().upper()
    if target_currency == "USD":
        return 1.0
    try:
        pair = yf.Ticker(f"USD{target_currency}=X")
        rate_hist = pair.history(period="5d")
        if not rate_hist.empty:
            return float(rate_hist["Close"].iloc[-1])
    except Exception:
        pass
    return 1.0


def fmt_currency_price(value: Any, fx_rate: float = 1.0, symbol: str = "$") -> str:
    """Format a per-share (non-abbreviated) price value in the display currency."""
    try:
        value = float(value) * fx_rate
    except (TypeError, ValueError):
        return "N/A"
    return f"{symbol}{value:,.2f}"




# =========================================================================== #
# SECTION 7E: PAGE RENDER FUNCTIONS  ***UI overhaul — sidebar navigation***
# ---------------------------------------------------------------------
# Each function below renders exactly what used to live inside a
# `with tab_X:` block. Nothing here is new business logic — every
# function is the same content that already existed, just wrapped so it
# can be registered as an st.Page() and reached via the grouped sidebar
# navigation built in SECTION 8 below (replacing the old flat/nested tab
# bar). Functions reference module-level globals (symbol_input, info,
# hist, live, company_name, education_mode, student_mode, display_currency,
# _DISPLAY_FX_RATE, _DISPLAY_CURRENCY_SYMBOL) that are set in SECTION 8
# before any page function is actually called via pg.run().
# =========================================================================== #

def render_dashboard_page() -> None:
    """Dashboard — company snapshot, price chart, key metrics."""
    # ----- NEW: EDUCATION MODE — student callout (Requirement 7) ----- #
    if education_mode and student_mode:
        st.success(
            "🌟 Key Concept: A stock's price reflects what investors are willing to pay "
            "today for a share of the company's future profits."
        )
    # ----- END NEW ----- #

    with st.expander("📄 Company Profile", expanded=True):
        summary = info.get("longBusinessSummary")
        if summary:
            st.write(summary)
        else:
            st.info("No company description available.")

        cols = st.columns(4)
        cols[0].metric("Employees", f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get("fullTimeEmployees") else "N/A")
        cols[1].metric("Website", info.get("website", "N/A"))
        cols[2].metric("Exchange", info.get("exchange", "N/A"))
        cols[3].metric("Currency", info.get("currency", "N/A"))

    st.markdown("#### Price Chart")
    fig_price = candlestick_chart(
        hist, symbol_input, show_sma20, show_sma50, show_sma200, show_bollinger
    )
    st.plotly_chart(
        fig_price,
        use_container_width=True,
        key="price_chart",
    )
    fig_vol = volume_chart(hist, symbol_input)
    st.plotly_chart(
        fig_vol,
        use_container_width=True,
        key="volume_chart",
    )
    st.markdown("#### Key Financial Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(label_for("Market Cap", education_mode, student_mode), fmt_large_number(info.get("marketCap")))
    m2.metric(label_for("P/E (Trailing)", education_mode, student_mode), fmt_ratio(info.get("trailingPE")))
    m3.metric("Forward P/E", fmt_ratio(info.get("forwardPE")))
    if not (education_mode and student_mode):
        m4.metric("PEG Ratio", fmt_ratio(info.get("pegRatio")))
    m5.metric("Dividend Yield", fmt_pct(info.get("dividendYield")))

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric(label_for("ROE", education_mode, student_mode), fmt_pct(info.get("returnOnEquity")))
    m7.metric(label_for("ROA", education_mode, student_mode), fmt_pct(info.get("returnOnAssets")))
    m8.metric(label_for("Revenue (TTM)", education_mode, student_mode), fmt_large_number(info.get("totalRevenue")))
    m9.metric("Total Cash", fmt_large_number(info.get("totalCash")))
    m10.metric("Total Debt", fmt_large_number(info.get("totalDebt")))

    # ----- NEW: EDUCATION MODE — expandable metric explanations (Requirement 2) ----- #
    if education_mode:
        st.markdown("##### 📚 Learn About These Metrics")
        for _metric_term in [
            "Market Capitalization",
            "P/E Ratio",
            "Forward P/E",
            "PEG Ratio",
            "Dividend Yield",
            "ROE",
            "ROA",
            "Revenue",
        ]:
            render_metric_education(_metric_term)


def render_technical_page() -> None:
    """Technical analysis: candlesticks, RSI, MACD, support/resistance, fundamentals snapshot."""
    st.markdown("#### Candlestick Chart with Overlays")

    st.plotly_chart(
        candlestick_chart(
            hist,
            symbol_input,
            show_sma20,
            show_sma50,
            show_sma200,
            show_bollinger,
        ),
        use_container_width=True,
        key="chart1",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            rsi_chart(hist),
            use_container_width=True,
            key="chart2",
        )

    with col2:
        st.plotly_chart(
            macd_chart(hist),
            use_container_width=True,
            key="chart3",
        )

    st.plotly_chart(
        support_resistance_chart(hist, symbol_input),
        use_container_width=True,
        key="chart4",
    )

    with st.expander("ℹ️ Support & Resistance Levels"):
        levels = support_resistance(hist)

        c1, c2 = st.columns(2)

        with c1:
            st.write(levels["support"])

        with c2:
            st.write(levels["resistance"])
    st.markdown("#### Valuation & Profitability")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label_for("Market Cap", education_mode, student_mode), fmt_large_number(info.get("marketCap")))
    col2.metric("Trailing P/E", fmt_ratio(info.get("trailingPE")))
    col3.metric("Forward P/E", fmt_ratio(info.get("forwardPE")))
    col4.metric("PEG Ratio", fmt_ratio(info.get("pegRatio")))

    # ----- NEW: EDUCATION MODE — Student Mode hides advanced metrics (Requirement 7) ----- #
    if not (education_mode and student_mode):
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Price / Book", fmt_ratio(info.get("priceToBook")))
        col6.metric("Price / Sales", fmt_ratio(info.get("priceToSalesTrailing12Months")))
        col7.metric("EV / EBITDA", fmt_ratio(info.get("enterpriseToEbitda")))
        col8.metric("Beta", fmt_ratio(info.get("beta")))
    # ----- END NEW: EDUCATION MODE ----- #

    st.markdown("#### Profitability & Returns")
    col9, col10, col11, col12 = st.columns(4)
    col9.metric(label_for("ROE", education_mode, student_mode), fmt_pct(info.get("returnOnEquity")))
    col10.metric(label_for("ROA", education_mode, student_mode), fmt_pct(info.get("returnOnAssets")))
    col11.metric("Profit Margin", fmt_pct(info.get("profitMargins")))
    col12.metric("Operating Margin", fmt_pct(info.get("operatingMargins")))

    st.markdown("#### Balance Sheet Snapshot")
    col13, col14, col15, col16 = st.columns(4)
    col13.metric("Total Cash", fmt_large_number(info.get("totalCash")))
    col14.metric("Total Debt", fmt_large_number(info.get("totalDebt")))
    # ----- NEW: EDUCATION MODE — Student Mode hides advanced metrics (Requirement 7) ----- #
    if not (education_mode and student_mode):
        col15.metric("Debt / Equity", fmt_ratio(info.get("debtToEquity")))
        col16.metric("Current Ratio", fmt_ratio(info.get("currentRatio")))
    # ----- END NEW: EDUCATION MODE ----- #

    # ----- NEW: EDUCATION MODE — expandable metric explanations (Requirement 2) ----- #
    if education_mode:
        st.markdown("##### 📚 Learn About These Metrics")
        for _metric_term in [
            "Beta",
            "Gross Margin",
            "Operating Margin",
            "Net Margin",
            "Debt to Equity",
            "Current Ratio",
            "Enterprise Value",
            "EBITDA",
        ]:
            render_metric_education(_metric_term)
    # ----- END NEW: EDUCATION MODE ----- #

    st.markdown("#### Revenue & Earnings Trend")
    income_stmt = get_income_statement(symbol_input)
    earnings_df = get_earnings(symbol_input)

    col_rev, col_earn = st.columns(2)
    with col_rev:
        st.plotly_chart(
            revenue_chart(income_stmt),
            use_container_width=True,
            key="chart5",
        )
    with col_earn:
        st.plotly_chart(
            earnings_chart(earnings_df),
            use_container_width=True,
            key="chart6",
        )


def render_financials_page() -> None:
    """Income statement, balance sheet, and cash flow statement."""
    quarterly = st.toggle("Show Quarterly Data", value=False)

    with st.expander("💵 Income Statement", expanded=True):
        income = get_income_statement(symbol_input, quarterly=quarterly)
        if income.empty:
            st.info("Income statement data not available.")
        else:
            st.dataframe(income, use_container_width=True)

    with st.expander("🏦 Balance Sheet"):
        balance = get_balance_sheet(symbol_input, quarterly=quarterly)
        if balance.empty:
            st.info("Balance sheet data not available.")
        else:
            st.dataframe(balance, use_container_width=True)

    with st.expander("💸 Cash Flow Statement"):
        cashflow = get_cash_flow(symbol_input, quarterly=quarterly)
        if cashflow.empty:
            st.info("Cash flow data not available.")
        else:
            st.dataframe(cashflow, use_container_width=True)


def render_analyst_page() -> None:
    """Analyst recommendations and price targets."""
    st.markdown("#### Analyst Recommendations")
    recs = get_recommendations(symbol_input)
    if recs.empty:
        st.info("No analyst recommendation data available.")
    else:
        st.dataframe(recs, use_container_width=True)

    st.markdown("#### Price Targets")
    targets = get_price_targets(symbol_input)
    if not targets:
        st.info("No analyst price target data available.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Low", fmt_large_number(targets.get("low")) if targets.get("low") else "N/A")
        col2.metric("Mean", fmt_large_number(targets.get("mean")) if targets.get("mean") else "N/A")
        col3.metric("Median", fmt_large_number(targets.get("median")) if targets.get("median") else "N/A")
        col4.metric("High", fmt_large_number(targets.get("high")) if targets.get("high") else "N/A")

        current_price = live.get("price")
        mean_target = targets.get("mean")
        if current_price and mean_target:
            upside = (mean_target - current_price) / current_price * 100
            st.metric("Implied Upside vs. Mean Target", f"{upside:+.2f}%")


def render_scores_page() -> None:
    """AI / Buffett / Graham / Risk composite scores with breakdowns."""
    st.markdown("#### AI-Generated Investment Scores")
    st.caption("Transparent, rule-based scoring — not machine-learning black boxes.")

    ai_result = ai_investment_score(info, hist)
    buffett_result = buffett_score(info)
    graham_result = graham_score(info)
    risk_result = risk_score(info, hist)

    gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)
    with gauge_col1:
        st.plotly_chart(
            gauge_chart(ai_result.score, "AI Score"),
            use_container_width=True,
            key="chart7",
        )
    with gauge_col2:
        st.plotly_chart(
            gauge_chart(buffett_result.score, "Buffett Score"),
            use_container_width=True,
            key="chart8",
        )
    with gauge_col3:
        st.plotly_chart(
            gauge_chart(graham_result.score, "Graham Score"),
            use_container_width=True,
            key="chart9",
        )
    with gauge_col4:
        st.plotly_chart(
            gauge_chart(risk_result.score, "Risk Score"),
            use_container_width=True,
            key="chart10",
        )
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        with st.expander(f"AI Investment Score Breakdown — {ai_result.label}", expanded=True):
            for line in ai_result.breakdown:
                st.write(f"• {line}")
        with st.expander(f"Buffett Score Breakdown — {buffett_result.label}"):
            for line in buffett_result.breakdown:
                st.write(f"• {line}")
    with score_col2:
        with st.expander(f"Graham Score Breakdown — {graham_result.label}"):
            for line in graham_result.breakdown:
                st.write(f"• {line}")
        with st.expander(f"Risk Score Breakdown — {risk_result.label}"):
            for line in risk_result.breakdown:
                st.write(f"• {line}")


def render_earnings_page() -> None:
    """Earnings — revenue trend, EPS estimate vs. reported, and upcoming/past earnings dates."""
    st.markdown(f"#### Earnings — {symbol_input}")

    eps_col1, eps_col2, eps_col3 = st.columns(3)
    eps_col1.metric("Trailing EPS", fmt_ratio(info.get("trailingEps")))
    eps_col2.metric("Forward EPS", fmt_ratio(info.get("forwardEps")))
    eps_col3.metric("Earnings Growth", fmt_pct(info.get("earningsGrowth")))

    income_stmt_earn = get_income_statement(symbol_input)
    earnings_df_earn = get_earnings(symbol_input)

    earn_col1, earn_col2 = st.columns(2)
    with earn_col1:
        st.plotly_chart(
            revenue_chart(income_stmt_earn),
            use_container_width=True,
            key="earnings_page_revenue_chart",
        )
    with earn_col2:
        st.plotly_chart(
            earnings_chart(earnings_df_earn),
            use_container_width=True,
            key="earnings_page_eps_chart",
        )

    with st.expander("📅 Earnings Dates (Estimate vs. Reported)", expanded=True):
        if earnings_df_earn.empty:
            st.info("No earnings date data available for this ticker.")
        else:
            st.dataframe(earnings_df_earn, use_container_width=True)


def render_stock_research_page() -> None:
    """Stock Research — Technical, Fundamentals, Financials, Analyst, and AI Scores
    grouped under one page as inner tabs, so the sidebar nav doesn't need a
    separate top-level entry for each."""
    st.markdown(f"#### Stock Research — {symbol_input}")
    research_inner_tabs = st.tabs(["Technical", "Fundamentals", "Financials", "Analyst", "AI Scores"])
    with research_inner_tabs[0]:
        render_technical_page()
    with research_inner_tabs[1]:
        st.info(
            "Fundamentals are integrated into the Technical and Financials tabs above — "
            "see Technical for valuation/profitability metrics and Financials for the "
            "full statements."
        )
    with research_inner_tabs[2]:
        render_financials_page()
    with research_inner_tabs[3]:
        render_analyst_page()
    with research_inner_tabs[4]:
        render_scores_page()


def _dcf_kpi_card(icon, icon_bg, label, value, value_color, subtext, subtext_color, card_bg, card_border, muted):
    """Small HTML KPI card used in the DCF Valuation page header row."""
    return f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px;
                padding:16px 18px; height:132px; display:flex; flex-direction:column; justify-content:space-between;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <span style="color:{muted}; font-size:0.76rem; font-weight:600;">{label}</span>
            <span style="background:{icon_bg}; min-width:28px; height:28px; border-radius:50%;
                         display:flex; align-items:center; justify-content:center; font-size:0.85rem;">{icon}</span>
        </div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:{value_color};">{value}</div>
            <div style="color:{subtext_color}; font-size:0.78rem; margin-top:2px;">{subtext}</div>
        </div>
    </div>
    """


def _dcf_scenario_card(emoji, label, label_color, header_bg, tg_pct, wacc_pct, value, upside_text, upside_color, card_bg, card_border, muted):
    """Bear / Base / Bull scenario card used in the DCF Valuation page."""
    return f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px;
                padding:18px; text-align:center;">
        <div style="width:44px; height:44px; border-radius:50%; background:{header_bg};
                    display:flex; align-items:center; justify-content:center; font-size:1.3rem; margin:0 auto 10px auto;">{emoji}</div>
        <div style="color:{label_color}; font-weight:700; font-size:0.98rem;">{label}</div>
        <div style="color:{muted}; font-size:0.76rem; margin-top:8px;">Terminal Growth: {tg_pct}</div>
        <div style="color:{muted}; font-size:0.76rem;">WACC: {wacc_pct}</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:800; color:{label_color}; margin-top:10px;">{value}</div>
        <div style="color:{upside_color}; font-size:0.8rem; font-weight:600; margin-top:4px;">{upside_text}</div>
    </div>
    """


def _dcf_sensitivity_cell_color(value, lo, hi):
    """Interpolate a red -> yellow -> green background color for a sensitivity table cell."""
    if value is None or hi == lo:
        return "rgba(127,127,127,0.15)"
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    if t < 0.5:
        # red -> yellow
        u = t / 0.5
        r, g, b = 214, int(70 + u * (196 - 70)), int(66 + u * (77 - 66))
    else:
        # yellow -> green
        u = (t - 0.5) / 0.5
        r, g, b = int(214 - u * (214 - 26)), int(196 - u * (196 - 166)), int(77 - u * (77 - 69))
    return f"rgba({r},{g},{b},0.55)"


def _dcf_price_range_bar(bear_val, current_val, base_val, bull_val, muted):
    """Gradient horizontal bar showing Bear / Current / Base / Bull positioned by value."""
    values = [v for v in (bear_val, current_val, base_val, bull_val) if v is not None]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0

    def pos(v):
        return max(3, min(97, (v - lo) / span * 100))

    markers = [
        (bear_val, "Bear Case", "#FF5A4E"),
        (current_val, "Current Price", "#E8EAED"),
        (base_val, "Intrinsic Value (Base)", "#5B9BD9"),
        (bull_val, "Bull Case", "#3DDC63"),
    ]
    labels_html = ""
    ticks_html = ""
    for val, label, color in markers:
        if val is None:
            continue
        p = pos(val)
        labels_html += (
            f'<div style="position:absolute; left:{p}%; transform:translateX(-50%); text-align:center; top:-48px; white-space:nowrap;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace; font-weight:700; color:{color}; font-size:0.85rem;">${val:,.2f}</div>'
            f'<div style="color:{muted}; font-size:0.68rem;">{label}</div>'
            f"</div>"
        )
        ticks_html += (
            f'<div style="position:absolute; left:{p}%; top:0; width:2px; height:100%; '
            f'background:rgba(255,255,255,0.7); transform:translateX(-1px);"></div>'
        )

    return f"""
    <div style="position:relative; margin-top:56px; margin-bottom:10px; height:10px; border-radius:6px;
                background:linear-gradient(to right, #FF5A4E, #A25FE8, #5B9BD9, #3DDC63);">
        {ticks_html}
        {labels_html}
    </div>
    """


def render_dcf_valuation_page() -> None:
    """
    DCF Valuation — redesigned to match the v2 dashboard mockup: a KPI card
    row, a 5-year forecast table, a valuation-summary donut chart,
    Bear/Base/Bull scenario cards, a color-coded WACC x Terminal-Growth
    sensitivity table, and a price-range gradient bar.

    All underlying numbers still come from the exact same functions as
    before (get_historical_financials, derive_base_assumptions,
    compute_wacc, run_bear_base_bull, sensitivity_table,
    check_dcf_suitability, generate_dcf_narrative — see SECTION 5B above).
    Only the presentation layer changed here.
    """
    _dcf_theme = st.session_state.get("theme_mode_toggle", "Dark")
    if _dcf_theme == "Light":
        card_bg, card_border, muted = "#F7F8FA", "rgba(0,0,0,0.10)", "#5B6472"
    else:
        card_bg, card_border, muted = "#161B22", "rgba(255,255,255,0.10)", "#9AA4B2"

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:1.5rem; font-weight:800;">DCF Valuation</span>
                <span style="background:rgba(0,200,5,0.15); color:#00C805; padding:2px 10px;
                             border-radius:999px; font-size:0.68rem; font-weight:700;">BETA</span>
            </div>
            <p style="color:{muted}; margin-top:2px; margin-bottom:0;">
                Intrinsic value estimation using Discounted Cash Flow analysis
            </p>
            """,
            unsafe_allow_html=True,
        )
    with header_col2:
        st.caption(f"🕐 Updated {dt.datetime.now().strftime('%b %d, %Y %I:%M %p')}")
        if st.button("🔄 Run DCF Analysis", use_container_width=True, key="dcf2_run_button"):
            st.rerun()

    st.divider()

    with st.expander("📐 Graham Intrinsic Value (classic formula, separate from the DCF below)"):
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        g_number = graham_number(eps, book_value)
        gcol1, gcol2, gcol3 = st.columns(3)
        gcol1.metric("Trailing EPS", f"{eps:.2f}" if eps else "N/A")
        gcol2.metric("Book Value / Share", f"{book_value:.2f}" if book_value else "N/A")
        gcol3.metric("Graham Number", f"${g_number:.2f}" if g_number else "N/A")
        if g_number and live.get("price"):
            g_mos = margin_of_safety(g_number, live["price"])
            if g_mos is not None:
                st.metric("Margin of Safety vs. Current Price", f"{g_mos:+.2f}%")

    hist_financials = get_historical_financials(symbol_input)

    if hist_financials is None:
        st.warning(
            f"Historical financial statement data isn't available for "
            f"{symbol_input}, so a forecast-driven DCF can't be built for "
            f"this ticker. Try a different company."
        )
        return

    if hist_financials.data_warnings:
        with st.expander("⚠️ Data Notes", expanded=False):
            for note in hist_financials.data_warnings:
                st.caption(f"• {note}")

    derived = derive_base_assumptions(hist_financials)
    default_growth = derived["revenue_growth"] if derived["revenue_growth"] is not None else 0.08
    default_margin = derived["ebit_margin"] if derived["ebit_margin"] is not None else 0.15
    default_da_pct = derived["da_pct_revenue"] if derived["da_pct_revenue"] is not None else 0.04
    default_capex_pct = derived["capex_pct_revenue"] if derived["capex_pct_revenue"] is not None else 0.05
    default_nwc_pct = derived["nwc_pct_revenue"] if derived["nwc_pct_revenue"] is not None else 0.01
    default_tax_rate = hist_financials.effective_tax_rate or 0.21

    beta_val = info.get("beta") or 1.0
    market_cap_val = info.get("marketCap") or 0.0
    total_debt_val = info.get("totalDebt") or 0.0
    risk_free_rate = 0.04
    equity_risk_premium = 0.05
    credit_spread = 0.015

    wacc_estimate = compute_wacc(
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        beta=beta_val,
        credit_spread=credit_spread,
        tax_rate=default_tax_rate,
        market_cap=market_cap_val,
        total_debt=total_debt_val,
    )

    with st.expander("⚙️ Base-Case Assumptions", expanded=False):
        st.caption(
            "Defaults are derived from up to 3 years of financial history "
            "and an estimated WACC — adjust anything below."
        )
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            rev_growth_input = st.slider(
                "Revenue Growth", -10.0, 40.0,
                min(40.0, max(-10.0, round(default_growth * 100, 1))), 0.5, key="dcf2_rev_growth"
            ) / 100
            ebit_margin_input = st.slider(
                "EBIT Margin", -50.0, 60.0,
                min(60.0, max(-50.0, round(default_margin * 100, 1))), 0.5, key="dcf2_ebit_margin"
            ) / 100
        with a_col2:
            tax_rate_input = st.slider(
                "Tax Rate", 0.0, 40.0,
                min(40.0, max(0.0, round(default_tax_rate * 100, 1))), 0.5, key="dcf2_tax_rate"
            ) / 100
            wacc_input = st.slider(
                "WACC", 3.0, 20.0, min(20.0, max(3.0, round(wacc_estimate.wacc * 100, 1))), 0.1, key="dcf2_wacc"
            ) / 100
        with a_col3:
            terminal_growth_input = st.slider(
                "Terminal Growth", 0.0, 5.0, 2.5, 0.1, key="dcf2_terminal_growth"
            ) / 100
            projection_years_input = st.slider("Forecast Years", 3, 10, 5, 1, key="dcf2_years")

        st.markdown(
            f"**How the suggested WACC was built:** Cost of Equity (CAPM) = "
            f"{risk_free_rate:.1%} + {beta_val:.2f}β × {equity_risk_premium:.1%} = "
            f"{wacc_estimate.cost_of_equity:.2%}; after-tax Cost of Debt = "
            f"{wacc_estimate.cost_of_debt_aftertax:.2%}; weighted "
            f"{wacc_estimate.equity_weight:.0%} equity / {wacc_estimate.debt_weight:.0%} debt "
            f"→ **{wacc_estimate.wacc:.2%}** suggested."
        )

    base_revenue = hist_financials.revenue[0]
    net_debt_val = total_debt_val - (info.get("totalCash") or 0.0)
    shares_out_val = info.get("sharesOutstanding")

    base_assumptions_dict = {
        "revenue_growth": rev_growth_input,
        "ebit_margin": ebit_margin_input,
        "tax_rate": tax_rate_input,
        "da_pct_revenue": default_da_pct,
        "capex_pct_revenue": default_capex_pct,
        "nwc_pct_revenue": default_nwc_pct,
        "wacc": wacc_input,
        "terminal_growth": terminal_growth_input,
    }

    scenarios = run_bear_base_bull(
        base_revenue=base_revenue,
        base_assumptions=base_assumptions_dict,
        net_debt=net_debt_val,
        shares_outstanding=shares_out_val,
        projection_years=projection_years_input,
    )
    base_result = scenarios["Base"]
    bear_result = scenarios["Bear"]
    bull_result = scenarios["Bull"]
    current_price = live.get("price")

    suitability_warnings = check_dcf_suitability(
        sector=info.get("sector"),
        industry=info.get("industry"),
        base_fcff=base_result.forecast[0].fcff if base_result.forecast else None,
        hist=hist_financials,
        revenue_growth_assumption=rev_growth_input,
    )
    if suitability_warnings:
        with st.expander("⚠️ Is a basic DCF appropriate for this company?", expanded=True):
            for warning_text in suitability_warnings:
                st.warning(warning_text)

    # ----------------------------------------------------------------- #
    # KPI card row
    # ----------------------------------------------------------------- #
    base_iv = base_result.intrinsic_value_per_share
    bear_iv = bear_result.intrinsic_value_per_share
    bull_iv = bull_result.intrinsic_value_per_share
    base_upside = margin_of_safety(base_iv, current_price) if base_iv and current_price else None

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.markdown(
            _dcf_kpi_card(
                "💲", "rgba(0,200,5,0.15)", "Intrinsic Value (Base)",
                f"${base_iv:,.2f}" if base_iv else "N/A", "#3DDC63",
                f"{base_upside:+.2f}% Upside" if base_upside is not None else "vs. current price unavailable",
                "#3DDC63" if (base_upside or 0) >= 0 else "#FF5A4E",
                card_bg, card_border, muted,
            ),
            unsafe_allow_html=True,
        )
    with kcol2:
        st.markdown(
            _dcf_kpi_card(
                "🎯", "rgba(162,95,232,0.18)", "Fair Value Range",
                f"${bear_iv:,.2f} – ${bull_iv:,.2f}" if bear_iv and bull_iv else "N/A", "#E6E8EB",
                f"Bear ${bear_iv:,.0f}  ·  Bull ${bull_iv:,.0f}" if bear_iv and bull_iv else "",
                muted, card_bg, card_border, muted,
            ),
            unsafe_allow_html=True,
        )
    with kcol3:
        st.markdown(
            _dcf_kpi_card(
                "📈", "rgba(91,155,217,0.18)", "WACC",
                f"{wacc_input:.2%}", "#5B9BD9", "Cost of Capital",
                muted, card_bg, card_border, muted,
            ),
            unsafe_allow_html=True,
        )
    with kcol4:
        st.markdown(
            _dcf_kpi_card(
                "📊", "rgba(162,95,232,0.18)", "Terminal Growth",
                f"{terminal_growth_input:.2%}", "#A25FE8", "Perpetual Growth Rate",
                muted, card_bg, card_border, muted,
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    # ----------------------------------------------------------------- #
    # 5-Year Cash Flow Projections + DCF Valuation Summary (donut)
    # ----------------------------------------------------------------- #
    proj_col, summary_col = st.columns([3, 2])

    with proj_col:
        st.markdown(
            f'<div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px; padding:18px;">'
            f'<div style="font-weight:700; margin-bottom:10px;">5-Year Cash Flow Projections</div>',
            unsafe_allow_html=True,
        )
        forecast_df = pd.DataFrame(
            [
                {
                    "Year": f.year_label,
                    "Revenue": f.revenue,
                    "% Growth": (f.revenue / base_result.forecast[i - 1].revenue - 1) if i > 0 else (f.revenue / base_revenue - 1),
                    "FCFF": f.fcff,
                    "FCFF Margin": f.fcff / f.revenue if f.revenue else None,
                }
                for i, f in enumerate(base_result.forecast)
            ]
        )
        st.dataframe(
            forecast_df.style.format(
                {"Revenue": "${:,.0f}", "% Growth": "{:+.2%}", "FCFF": "${:,.0f}", "FCFF Margin": "{:.1%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("All figures in the security's native reporting currency.")
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_col:
        st.markdown(
            f'<div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px; padding:18px;">'
            f'<div style="font-weight:700; margin-bottom:6px;">DCF Valuation Summary</div>',
            unsafe_allow_html=True,
        )
        sum_pv_fcf = sum(f.pv_fcff for f in base_result.forecast)
        sum_pv_terminal = base_result.pv_terminal_value
        ev_total = sum_pv_fcf + sum_pv_terminal
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["PV of FCF (Years 1-N)", "PV of Terminal Value"],
                    values=[max(sum_pv_fcf, 0), max(sum_pv_terminal, 0)],
                    hole=0.62,
                    marker=dict(colors=["#5B9BD9", "#A25FE8"]),
                    textinfo="none",
                )
            ]
        )
        fig_donut = _base_layout(fig_donut, height=230)
        fig_donut.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True, key="dcf_summary_donut")

        pv_fcf_pct = (sum_pv_fcf / ev_total * 100) if ev_total else 0
        pv_term_pct = (sum_pv_terminal / ev_total * 100) if ev_total else 0
        st.markdown(
            f"""
            <div style="font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span>🔵 PV of FCF (Years 1-{projection_years_input})</span>
                    <span style="font-family:'JetBrains Mono',monospace;">{fmt_large_number(sum_pv_fcf)} · {pv_fcf_pct:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span>🟣 PV of Terminal Value</span>
                    <span style="font-family:'JetBrains Mono',monospace;">{fmt_large_number(sum_pv_terminal)} · {pv_term_pct:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-top:1px solid {card_border}; padding-top:6px; margin-top:6px;">
                    <span>Enterprise Value</span>
                    <span style="font-family:'JetBrains Mono',monospace;">{fmt_large_number(ev_total)}</span>
                </div>
            </div>
            <div style="margin-top:12px; border:1px solid {card_border}; border-radius:10px; padding:10px 14px;">
                <div style="color:{muted}; font-size:0.76rem;">Equity Value</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:800; color:#3DDC63;">{fmt_large_number(base_result.equity_value)}</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ----------------------------------------------------------------- #
    # Scenario Analysis + Sensitivity Analysis
    # ----------------------------------------------------------------- #
    scen_col, sens_col = st.columns([2, 3])

    with scen_col:
        st.markdown("**Scenario Analysis**")
        sc1, sc2, sc3 = st.columns(3)
        bear_upside = margin_of_safety(bear_iv, current_price) if bear_iv and current_price else None
        bull_upside = margin_of_safety(bull_iv, current_price) if bull_iv and current_price else None
        with sc1:
            st.markdown(
                _dcf_scenario_card(
                    "🐻", "Bear Case", "#FF5A4E", "rgba(255,90,78,0.18)",
                    f"{bear_result.assumptions['terminal_growth']:.1%}", f"{bear_result.assumptions['wacc']:.1%}",
                    f"${bear_iv:,.2f}" if bear_iv else "N/A",
                    f"{bear_upside:+.1f}% vs. price" if bear_upside is not None else "",
                    "#FF5A4E", card_bg, card_border, muted,
                ),
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                _dcf_scenario_card(
                    "⚖️", "Base Case", "#5B9BD9", "rgba(91,155,217,0.18)",
                    f"{base_result.assumptions['terminal_growth']:.1%}", f"{base_result.assumptions['wacc']:.1%}",
                    f"${base_iv:,.2f}" if base_iv else "N/A",
                    f"{base_upside:+.1f}% vs. price" if base_upside is not None else "",
                    "#5B9BD9", card_bg, card_border, muted,
                ),
                unsafe_allow_html=True,
            )
        with sc3:
            st.markdown(
                _dcf_scenario_card(
                    "🐐", "Bull Case", "#3DDC63", "rgba(61,220,99,0.18)",
                    f"{bull_result.assumptions['terminal_growth']:.1%}", f"{bull_result.assumptions['wacc']:.1%}",
                    f"${bull_iv:,.2f}" if bull_iv else "N/A",
                    f"{bull_upside:+.1f}% vs. price" if bull_upside is not None else "",
                    "#3DDC63", card_bg, card_border, muted,
                ),
                unsafe_allow_html=True,
            )
        if current_price:
            st.markdown(
                f'<p style="text-align:center; color:{muted}; font-size:0.78rem; margin-top:8px;">vs. Current Price: ${current_price:,.2f}</p>',
                unsafe_allow_html=True,
            )

    with sens_col:
        st.markdown("**Sensitivity Analysis** — Implied Equity Value per Share")
        wacc_range = [wacc_input + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        tg_range = [terminal_growth_input + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
        sensitivity_grid = sensitivity_table(
            base_revenue=base_revenue,
            base_assumptions=base_assumptions_dict,
            net_debt=net_debt_val,
            shares_outstanding=shares_out_val,
            projection_years=projection_years_input,
            wacc_range=wacc_range,
            terminal_growth_range=tg_range,
        )
        flat_values = [v for row in sensitivity_grid for v in row if v is not None]
        lo_val, hi_val = (min(flat_values), max(flat_values)) if flat_values else (0, 1)

        header_cells = "".join(
            f'<th style="padding:6px 8px; font-size:0.75rem; color:{muted}; font-weight:600;">{t:.1%}</th>'
            for t in tg_range
        )
        base_tg_col_idx = min(range(len(tg_range)), key=lambda i: abs(tg_range[i] - terminal_growth_input))
        rows_html = ""
        for w, row in zip(wacc_range, sensitivity_grid):
            is_base_row = abs(w - wacc_input) < 1e-9
            cells = ""
            for col_idx, v in enumerate(row):
                bg = _dcf_sensitivity_cell_color(v, lo_val, hi_val)
                is_base_cell = is_base_row and col_idx == base_tg_col_idx
                border = f"2px solid {muted}" if is_base_cell else "1px solid transparent"
                cells += (
                    f'<td style="background:{bg}; text-align:center; padding:7px 8px; '
                    f'font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; border-radius:4px; '
                    f'border:{border};">'
                    f'{f"${v:,.0f}" if v is not None else "N/A"}</td>'
                )
            rows_html += (
                f'<tr><td style="color:{muted}; font-size:0.75rem; padding:6px 8px; font-weight:600;">{w:.2%}</td>{cells}</tr>'
            )

        st.markdown(
            f"""
            <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px; padding:16px; overflow-x:auto;">
                <table style="border-collapse:separate; border-spacing:3px; width:100%;">
                    <tr><th style="color:{muted}; font-size:0.72rem;">WACC ↓ / TG →</th>{header_cells}</tr>
                    {rows_html}
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Base Case: WACC {wacc_input:.2%} | Terminal Growth {terminal_growth_input:.2%}")

    st.write("")

    # ----------------------------------------------------------------- #
    # Current Price vs. Intrinsic Value gradient bar
    # ----------------------------------------------------------------- #
    st.markdown(
        f'<div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px; padding:22px 24px 18px 24px;">'
        f'<div style="font-weight:700; margin-bottom:4px;">Current Price vs. Intrinsic Value</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_dcf_price_range_bar(bear_iv, current_price, base_iv, bull_iv, muted), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Deterministic Python computed every number above; this narrative only
    # DESCRIBES those already-computed results in plain English — it does
    # not perform any valuation math of its own.
    narrative = generate_dcf_narrative(company_name, symbol_input, scenarios, current_price)
    st.info(f"🤖 {narrative}")

    if education_mode:
        with st.expander("📚 See the actual FCFF calculation for this company"):
            f1 = base_result.forecast[0]
            st.markdown(
                f"**FCFF (Year 1) = EBIT × (1 − Tax Rate) + D&A − CapEx − ΔNWC**\n\n"
                f"= {fmt_large_number(f1.ebit)} × (1 − {tax_rate_input:.1%}) "
                f"+ {fmt_large_number(f1.da)} − {fmt_large_number(f1.capex)} "
                f"− {fmt_large_number(f1.change_in_nwc)}\n\n"
                f"= {fmt_large_number(f1.nopat)} (NOPAT) + {fmt_large_number(f1.da)} "
                f"− {fmt_large_number(f1.capex)} − {fmt_large_number(f1.change_in_nwc)}\n\n"
                f"= **{fmt_large_number(f1.fcff)}**"
            )
        st.markdown("##### 📖 DCF Concepts Explained")
        for _dcf_term in ["WACC", "NOPAT", "EBIT", "Terminal Value", "Net Working Capital (NWC)", "FCFF"]:
            render_metric_education(_dcf_term)

    st.info(
        "⚠️ DCF valuation is highly sensitive to its assumptions and should "
        "NOT be treated as a guaranteed prediction of future stock price. "
        "Small changes in growth, margin, WACC, or terminal growth can "
        "produce very different results — that's inherent to the model, "
        "not a flaw in the calculation. A basic DCF like this one is also "
        "generally less reliable for banks, insurers, companies with "
        "negative free cash flow, very young companies, highly cyclical "
        "businesses, and extremely high-growth companies (see the warnings "
        "above if any applied to this ticker)."
    )


def render_peers_page() -> None:
    """Peer / sector benchmarking against curated or user-supplied comparables."""
    st.markdown(f"#### 🏆 Peer & Sector Comparison — {symbol_input}")

    curated_peers = get_peer_symbols(symbol_input)
    default_peer_text = ", ".join(curated_peers)
    if not curated_peers:
        st.info(
            f"No curated peer list is available for {symbol_input} yet. "
            f"Enter peer tickers manually below (sector: {info.get('sector', 'N/A')})."
        )

    peer_text = st.text_input(
        "Peer tickers (comma-separated)",
        value=default_peer_text,
        key="peer_symbols_input",
    )
    peer_symbols = [p.strip().upper() for p in peer_text.split(",") if p.strip()][:6]

    if not peer_symbols:
        st.warning("Add at least one peer ticker to run a comparison.")
    else:
        peer_df = build_peer_comparison_table(symbol_input, peer_symbols)

        st.markdown("##### Benchmarking Table")
        st.dataframe(
            peer_df.style.format(
                {
                    "Market Cap": lambda x: fmt_large_number(x),
                    "Revenue Growth": lambda x: fmt_pct(x),
                    "Profit Margin": lambda x: fmt_pct(x),
                    "P/E Ratio": lambda x: fmt_ratio(x),
                    "ROE": lambda x: fmt_pct(x),
                },
                na_rep="N/A",
            ),
            use_container_width=True,
        )

        st.markdown("##### Compare a Metric")
        peer_metric = st.selectbox(
            "Metric",
            options=["Market Cap", "Revenue Growth", "Profit Margin", "P/E Ratio", "ROE"],
            key="peer_metric_select",
        )
        st.plotly_chart(
            peer_comparison_chart(peer_df, peer_metric, symbol_input),
            use_container_width=True,
            key="peer_chart",
        )

        main_row = peer_df[peer_df["Symbol"] == symbol_input].iloc[0]
        peer_rows = peer_df[peer_df["Symbol"] != symbol_input]

        st.markdown(f"##### {symbol_input} vs. Peer Average")
        bench_col1, bench_col2, bench_col3, bench_col4 = st.columns(4)
        for _bench_col, _bench_metric, _bench_fmt in zip(
            [bench_col1, bench_col2, bench_col3, bench_col4],
            ["Profit Margin", "P/E Ratio", "ROE", "Revenue Growth"],
            [fmt_pct, fmt_ratio, fmt_pct, fmt_pct],
        ):
            peer_avg = peer_rows[_bench_metric].mean() if not peer_rows.empty else None
            main_value = main_row[_bench_metric]
            delta = None
            if peer_avg is not None and main_value is not None and not pd.isna(peer_avg) and not pd.isna(main_value):
                delta = f"{main_value - peer_avg:+.4f} vs. peers"
            _bench_col.metric(
                _bench_metric,
                _bench_fmt(main_value) if main_value is not None else "N/A",
                delta=delta,
            )


def render_compare_page() -> None:
    """Multi-ticker relative performance and metrics comparison."""
    st.markdown("#### Multi-Stock Comparison")

    symbols_list = tuple(
        s.strip().upper()
        for s in compare_symbols.split(",")
        if s.strip()
    )

    if not symbols_list:
        st.info("Enter comma-separated ticker symbols in the sidebar.")
    else:
        with st.spinner("Loading comparison..."):
            histories = get_multi_price_history(
                symbols_list,
                period=period,
                interval=interval,
            )

        st.plotly_chart(
            comparison_chart(histories),
            use_container_width=True,
            key="compare_chart",
        )

        st.markdown("#### Fundamental Comparison")

        rows = []

        for sym in symbols_list:
            cinfo = get_company_info(sym)

            rows.append(
                {
                    "Symbol": sym,
                    "Name": cinfo.get("shortName", "—"),
                    "Price": cinfo.get("currentPrice")
                    or cinfo.get("regularMarketPrice"),
                    "Market Cap": cinfo.get("marketCap"),
                    "P/E": cinfo.get("trailingPE"),
                    "Forward P/E": cinfo.get("forwardPE"),
                    "ROE": cinfo.get("returnOnEquity"),
                    "Dividend Yield": cinfo.get("dividendYield"),
                    "Beta": cinfo.get("beta"),
                }
            )

        compare_df = pd.DataFrame(rows)

        st.dataframe(
            compare_df.style.format(
                {
                    "Price": "${:.2f}",
                    "Market Cap": lambda x: fmt_large_number(x),
                    "P/E": "{:.2f}",
                    "Forward P/E": "{:.2f}",
                    "ROE": lambda x: fmt_pct(x),
                    "Dividend Yield": lambda x: fmt_pct(x),
                    "Beta": "{:.2f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
        )


def render_news_page() -> None:
    """Latest company news with keyword-based sentiment tagging."""
    st.markdown("#### Latest News")
    raw_news = get_company_news(symbol_input, limit=10)
    normalized = normalize_news_list(raw_news)

    if not normalized:
        st.info("No recent news found for this ticker.")
    else:
        # ----- NEW: NEWS SENTIMENT ANALYSIS (Aug 2026 update) ----- #
        sentiment_df = analyze_headlines_sentiment(normalized)
        sentiment_lookup = dict(zip(sentiment_df["title"], sentiment_df["sentiment"]))

        with st.expander("🧭 News Sentiment Overview", expanded=True):
            pos_count = int((sentiment_df["sentiment"] == "Positive").sum())
            neu_count = int((sentiment_df["sentiment"] == "Neutral").sum())
            neg_count = int((sentiment_df["sentiment"] == "Negative").sum())

            sent_col1, sent_col2, sent_col3 = st.columns(3)
            sent_col1.metric("🟢 Positive Headlines", pos_count)
            sent_col2.metric("⚪ Neutral Headlines", neu_count)
            sent_col3.metric("🔴 Negative Headlines", neg_count)

            st.plotly_chart(
                sentiment_summary_chart(sentiment_df),
                use_container_width=True,
                key="news_sentiment_chart",
            )
            st.caption(
                "Sentiment is estimated with a simple keyword-based scan of each "
                "headline and is meant as a quick directional signal, not "
                "investment advice."
            )
        # ----- END NEW: NEWS SENTIMENT ANALYSIS ----- #

        _sentiment_badges = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
        for item in normalized:
            col_img, col_text = st.columns([1, 5])
            with col_img:
                if item["thumbnail"]:
                    try:
                        st.image(item["thumbnail"], use_container_width=True)
                    except Exception:
                        st.write("📰")
                else:
                    st.write("📰")
            with col_text:
                # ----- NEW: sentiment badge next to each headline ----- #
                _badge = _sentiment_badges.get(sentiment_lookup.get(item["title"], "Neutral"), "⚪")
                st.markdown(f"{_badge} **[{item['title']}]({item['link']})**")
                st.caption(f"{item['publisher']} • {item['published']}")
            st.divider()


def render_portfolio_overview_page() -> None:
    """Portfolio Overview — totals and allocation breakdown at a glance."""
    st.markdown("#### Portfolio Overview")
    st.caption("A snapshot of total value, cost basis, and how your holdings are allocated.")

    summary_df = get_portfolio_summary()

    if summary_df.empty:
        st.info("No holdings yet. Add a position on the **Holdings** page to get started.")
    else:
        totals = get_portfolio_totals(summary_df)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Value", fmt_large_number(totals["total_value"]))
        t2.metric("Total Cost", fmt_large_number(totals["total_cost"]))
        t3.metric("Unrealized P/L", fmt_large_number(totals["total_pl"]))
        t4.metric("Return", f"{totals['total_pl_pct']:+.2f}%")

        st.markdown("##### Allocation by Holding")
        fig_allocation = go.Figure(
            data=[
                go.Pie(
                    labels=summary_df["Symbol"],
                    values=summary_df["Weight (%)"],
                    hole=0.55,
                    marker=dict(colors=[COLOR_ACCENT, COLOR_SMA20, COLOR_SMA50, COLOR_UP, COLOR_DOWN, COLOR_SMA200]),
                    textinfo="label+percent",
                )
            ]
        )
        fig_allocation = _base_layout(fig_allocation, title="Portfolio Weight by Position", height=380)
        st.plotly_chart(fig_allocation, use_container_width=True, key="portfolio_allocation_chart")


def render_holdings_page() -> None:
    """Holdings — add, view, and remove individual positions."""
    st.markdown("#### Holdings")
    st.caption("Track hypothetical or real holdings. Data is kept for this browser session only.")

    with st.form("add_holding_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            holding_symbol = st.text_input("Ticker", value=symbol_input)
        with col2:
            holding_shares = st.number_input("Shares", min_value=0.0, value=10.0, step=1.0)
        with col3:
            holding_cost = st.number_input("Avg Cost / Share", min_value=0.0, value=100.0, step=1.0)
        with col4:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Add / Update", use_container_width=True)
        if submitted:
            add_holding(holding_symbol, holding_shares, holding_cost)
            st.toast(f"Added {holding_symbol} to portfolio")

    summary_df = get_portfolio_summary()

    if summary_df.empty:
        st.info("No holdings yet. Add a position above to get started.")
    else:
        st.dataframe(
            summary_df.style.format(
                {
                    "Shares": "{:.2f}",
                    "Avg Cost": "${:.2f}",
                    "Current Price": "${:.2f}",
                    "Market Value": "${:,.2f}",
                    "Total Cost": "${:,.2f}",
                    "Unrealized P/L ($)": "${:,.2f}",
                    "Unrealized P/L (%)": "{:+.2f}%",
                    "Weight (%)": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        remove_symbol = st.selectbox("Remove a holding", options=summary_df["Symbol"].tolist())
        if st.button("Remove Selected Holding"):
            remove_holding(remove_symbol)
            st.rerun()


def render_performance_page() -> None:
    """Performance — profit/loss breakdown per holding."""
    st.markdown("#### Performance")
    st.caption("Unrealized profit and loss for each position, plus overall portfolio return.")

    summary_df = get_portfolio_summary()

    if summary_df.empty:
        st.info("No holdings yet. Add a position on the **Holdings** page to get started.")
    else:
        totals = get_portfolio_totals(summary_df)
        p1, p2 = st.columns(2)
        p1.metric("Total Unrealized P/L", fmt_large_number(totals["total_pl"]))
        p2.metric("Overall Return", f"{totals['total_pl_pct']:+.2f}%")

        st.markdown("##### Unrealized P/L by Holding")
        pl_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in summary_df["Unrealized P/L ($)"]]
        fig_pl = go.Figure(
            data=[
                go.Bar(
                    x=summary_df["Symbol"],
                    y=summary_df["Unrealized P/L ($)"],
                    marker_color=pl_colors,
                )
            ]
        )
        fig_pl = _base_layout(fig_pl, title="Unrealized P/L ($) by Position", height=380)
        st.plotly_chart(fig_pl, use_container_width=True, key="portfolio_pl_chart")

        st.dataframe(
            summary_df[["Symbol", "Shares", "Avg Cost", "Current Price", "Unrealized P/L ($)", "Unrealized P/L (%)"]].style.format(
                {
                    "Shares": "{:.2f}",
                    "Avg Cost": "${:.2f}",
                    "Current Price": "${:.2f}",
                    "Unrealized P/L ($)": "${:,.2f}",
                    "Unrealized P/L (%)": "{:+.2f}%",
                }
            ),
            use_container_width=True,
        )


def render_risk_analysis_page() -> None:
    """Risk Analysis — the Risk Score gauge and its contributing factors, on its own."""
    st.markdown("#### Risk Analysis")
    st.caption("Volatility- and leverage-aware risk read for the currently selected ticker.")

    risk_result = risk_score(info, hist)
    risk_col1, risk_col2 = st.columns([1, 2])
    with risk_col1:
        st.plotly_chart(
            gauge_chart(risk_result.score, "Risk Score"),
            use_container_width=True,
            key="risk_analysis_gauge",
        )
    with risk_col2:
        st.markdown(f"##### Risk Breakdown — {risk_result.label}")
        for line in risk_result.breakdown:
            st.write(f"• {line}")


def render_watchlist_page() -> None:
    """Watchlist — live price snapshots for saved tickers."""
    st.markdown("#### Watchlist")
    st.caption("Track tickers without adding them as portfolio holdings.")
    watchlist_df = get_watchlist_snapshot()
    if watchlist_df.empty:
        st.info("Your watchlist is empty. Use the sidebar to add tickers.")
    else:
        st.dataframe(
            watchlist_df.style.format(
                {"Price": "${:.2f}", "Change": "{:+.2f}", "% Change": "{:+.2f}%", "Market Cap": lambda v: fmt_large_number(v)}
            ),
            use_container_width=True,
        )


def render_alerts_page() -> None:
    """Price alerts: set, track, and remove above/below targets."""
    st.markdown("#### 🔔 Price Alerts")
    st.caption("Set a target price for any ticker and see whether it's been triggered.")

    with st.form("add_alert_form", clear_on_submit=True):
        alert_col1, alert_col2, alert_col3 = st.columns([2, 1, 1])
        with alert_col1:
            alert_symbol = st.text_input("Ticker", value=symbol_input, key="alert_symbol_input")
        with alert_col2:
            alert_direction = st.selectbox("Direction", options=["above", "below"], key="alert_direction_input")
        with alert_col3:
            alert_target = st.number_input("Target Price ($)", min_value=0.0, step=1.0, key="alert_target_input")
        alert_note = st.text_input("Note (optional)", key="alert_note_input")
        submitted_alert = st.form_submit_button("➕ Add Alert")
        if submitted_alert:
            if alert_target <= 0:
                st.warning("Please enter a target price greater than 0.")
            else:
                add_price_alert(alert_symbol, alert_target, alert_direction, alert_note)
                st.toast(f"Alert added for {alert_symbol.strip().upper()}")

    st.divider()

    alerts_df = evaluate_price_alerts()
    if alerts_df.empty:
        st.info("No price alerts set yet. Add one above to get started.")
    else:
        triggered_count = int(alerts_df["Triggered"].sum())
        if triggered_count > 0:
            st.success(f"🔔 {triggered_count} alert(s) triggered!")

        for _, alert_row in alerts_df.iterrows():
            status_icon = "🔔 Triggered" if alert_row["Triggered"] else "⏳ Watching"
            current_price_display = (
                f"${alert_row['Current Price']:.2f}" if alert_row["Current Price"] is not None else "N/A"
            )
            with st.container(border=True):
                a_col1, a_col2, a_col3, a_col4 = st.columns([2, 2, 2, 1])
                a_col1.markdown(f"**{alert_row['Symbol']}** — {alert_row['Direction']} ${alert_row['Target']:.2f}")
                a_col2.markdown(f"Current: {current_price_display}")
                a_col3.markdown(status_icon)
                if a_col4.button("🗑️", key=f"remove_alert_{int(alert_row['Index'])}"):
                    remove_price_alert(int(alert_row["Index"]))
                    st.rerun()
                if alert_row["Note"]:
                    st.caption(f"Note: {alert_row['Note']}")


def render_backtest_page() -> None:
    """SMA-crossover strategy backtest vs. buy-and-hold."""
    st.markdown(f"#### 🔁 Backtest: SMA Crossover Strategy — {symbol_input}")
    st.caption(
        "Simulates going long whenever the short-term SMA crosses above the "
        "long-term SMA, and staying in cash otherwise, compared to simply "
        "buying and holding."
    )

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_short_window = st.number_input(
            "Short SMA Window", min_value=2, max_value=100, value=20, step=1, key="bt_short_window"
        )
    with bt_col2:
        bt_long_window = st.number_input(
            "Long SMA Window", min_value=5, max_value=300, value=50, step=1, key="bt_long_window"
        )
    with bt_col3:
        bt_capital = st.number_input(
            "Starting Capital ($)", min_value=100.0, value=10_000.0, step=500.0, key="bt_capital"
        )

    if bt_short_window >= bt_long_window:
        st.warning("The short SMA window should be smaller than the long SMA window.")
    else:
        backtest_result = run_sma_crossover_backtest(
            hist, short_window=int(bt_short_window), long_window=int(bt_long_window), initial_capital=bt_capital
        )
        equity_df = backtest_result["equity"]
        bt_stats = backtest_result["stats"]
        bt_trades = backtest_result["trades"]

        if equity_df.empty:
            st.info(
                "Not enough price history for this period to run the backtest. "
                "Try a longer chart period in the sidebar."
            )
        else:
            st.plotly_chart(
                backtest_equity_chart(equity_df, symbol_input),
                use_container_width=True,
                key="backtest_chart",
            )

            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            stat_col1.metric("Strategy Return", f"{bt_stats['total_return_pct']:+.2f}%")
            stat_col2.metric("Buy & Hold Return", f"{bt_stats['buyhold_return_pct']:+.2f}%")
            stat_col3.metric("Number of Trades", f"{bt_stats['num_trades']}")
            stat_col4.metric("Win Rate", f"{bt_stats['win_rate_pct']:.1f}%")
            stat_col5.metric("Max Drawdown", f"{bt_stats['max_drawdown_pct']:.2f}%")

            with st.expander(f"📋 Trade Log ({len(bt_trades)} trades)"):
                if not bt_trades:
                    st.info("No completed trades for this window combination.")
                else:
                    trades_df = pd.DataFrame(bt_trades)
                    st.dataframe(
                        trades_df.style.format(
                            {
                                "entry_price": "${:.2f}",
                                "exit_price": "${:.2f}",
                                "return_pct": "{:+.2f}%",
                            },
                            na_rep="N/A",
                        ),
                        use_container_width=True,
                    )


def render_dividends_page() -> None:
    """Dividend payment history, yield, payout ratio, and growth rate."""
    st.markdown(f"#### 💵 Dividend History — {symbol_input}")

    dividend_series = get_dividend_history(symbol_input)

    if dividend_series.empty:
        st.info(f"{symbol_input} has no recorded dividend payment history.")
    else:
        annual_dividend = info.get("dividendRate")
        dividend_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")
        growth_rate = compute_dividend_growth_rate(dividend_series)

        div_col1, div_col2, div_col3, div_col4 = st.columns(4)
        div_col1.metric(
            "Annual Dividend",
            fmt_currency_price(annual_dividend, _DISPLAY_FX_RATE, _DISPLAY_CURRENCY_SYMBOL)
            if annual_dividend
            else "N/A",
        )
        div_col2.metric("Dividend Yield", fmt_pct(dividend_yield))
        div_col3.metric("Payout Ratio", fmt_pct(payout_ratio))
        div_col4.metric(
            "Dividend Growth (CAGR)",
            f"{growth_rate * 100:+.2f}%" if growth_rate is not None else "N/A",
        )

        st.plotly_chart(
            dividend_history_chart(dividend_series, symbol_input),
            use_container_width=True,
            key="dividend_history_chart",
        )

        with st.expander(f"📋 Full Dividend Payment History ({len(dividend_series)} payments)"):
            dividend_table = dividend_series.sort_index(ascending=False).reset_index()
            dividend_table.columns = ["Ex-Dividend Date", "Amount ($ / share)"]
            st.dataframe(
                dividend_table.style.format({"Amount ($ / share)": "${:.4f}"}),
                use_container_width=True,
            )

        st.caption(
            "Dividend amounts are shown in the security's native currency "
            "per share, except for the Annual Dividend metric above, which "
            "reflects your selected Display Currency."
        )


def render_export_page() -> None:
    """PDF and CSV export of the current ticker's research and metrics."""
    st.markdown("#### Export Research Report")
    st.caption("Generate a downloadable PDF summary of the current ticker's key data and scores.")

    if st.button("🧾 Generate PDF Report", type="primary"):
        scores = {
            "AI Investment Score": ai_investment_score(info, hist),
            "Buffett Score": buffett_score(info),
            "Graham Score": graham_score(info),
            "Risk Score": risk_score(info, hist),
        }
        try:
            pdf_bytes = build_pdf_report(symbol_input, info, scores)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{symbol_input}_research_report.pdf",
                mime="application/pdf",
            )
            st.success("Report generated successfully.")
        except Exception as exc:
            st.error(f"Failed to generate PDF report: {exc}")

    st.divider()
    st.markdown("#### Export Data as CSV")
    st.caption("Download the raw price history and key metrics for the current ticker.")

    csv_col1, csv_col2 = st.columns(2)

    with csv_col1:
        if not hist.empty:
            price_csv = hist.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="⬇️ Download Price History (CSV)",
                data=price_csv,
                file_name=f"{symbol_input}_price_history.csv",
                mime="text/csv",
                key="download_price_csv",
            )
        else:
            st.info("No price history available to export.")

    with csv_col2:
        metrics_row = {
            "Symbol": symbol_input,
            "Market Cap": info.get("marketCap"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Profit Margin": info.get("profitMargins"),
            "Debt to Equity": info.get("debtToEquity"),
            "Current Ratio": info.get("currentRatio"),
            "Dividend Yield": info.get("dividendYield"),
        }
        metrics_csv = pd.DataFrame([metrics_row]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Key Metrics (CSV)",
            data=metrics_csv,
            file_name=f"{symbol_input}_key_metrics.csv",
            mime="text/csv",
            key="download_metrics_csv",
        )


def render_classroom_page() -> None:
    """Classroom tools: discussion questions, homework, quiz, teacher notes."""
    st.markdown(f"#### 📝 Classroom Tools: {company_name} ({symbol_input})")
    st.caption(
        "Predefined, ready-to-use classroom materials based on the selected company. "
        "(Template-based for now — no AI/LLM generation.)"
    )

    classroom_col1, classroom_col2 = st.columns(2)

    with classroom_col1:
        if st.button("💬 Generate Discussion Questions", use_container_width=True, key="gen_discussion"):
            st.session_state["classroom_output"] = ("Discussion Questions", generate_discussion_questions(company_name, symbol_input))
        if st.button("📓 Generate Homework", use_container_width=True, key="gen_homework"):
            st.session_state["classroom_output"] = ("Homework", generate_homework(company_name, symbol_input))
        if st.button("📝 Generate Quiz", use_container_width=True, key="gen_quiz"):
            st.session_state["classroom_output"] = ("Quiz", generate_quiz(company_name, symbol_input, info))
        if st.button("🎟️ Generate Exit Ticket", use_container_width=True, key="gen_exit"):
            st.session_state["classroom_output"] = ("Exit Ticket", generate_exit_ticket(company_name))

    with classroom_col2:
        if st.button("📖 Generate Vocabulary", use_container_width=True, key="gen_vocab"):
            st.session_state["classroom_output"] = (
                "Vocabulary Assignment",
                generate_vocabulary_assignment(
                    ["Market Capitalization", "P/E Ratio", "Revenue", "Net Margin", "ROE", "Free Cash Flow"]
                ),
            )
        if st.button("🧠 Generate Reflection Questions", use_container_width=True, key="gen_reflection"):
            st.session_state["classroom_output"] = ("Reflection Questions", generate_reflection_questions(company_name, info))
        if st.button("📂 Generate Case Study", use_container_width=True, key="gen_case_study"):
            st.session_state["classroom_output"] = ("Case Study", generate_case_study(company_name, symbol_input, info))

    st.divider()

    if st.session_state.get("classroom_output"):
        output_title, output_content = st.session_state["classroom_output"]
        st.markdown(f"##### 📄 {output_title}")
        if isinstance(output_content, list):
            for _item in output_content:
                if isinstance(_item, dict):
                    st.markdown(f"**Q:** {_item['question']}")
                    st.caption(f"Suggested answer: {_item['answer']}")
                else:
                    st.markdown(f"- {_item}")
        else:
            st.markdown(output_content)
    else:
        st.info("Click a button above to generate classroom material for this company.")

    st.divider()

    # ----- Teacher Notes (Requirement 6) ----- #
    st.markdown("##### 🧑‍🏫 Teacher Notes")
    st.caption("Notes are kept for this browser session only, using Streamlit session_state.")

    if "teacher_notes" not in st.session_state:
        st.session_state["teacher_notes"] = []

    new_note = st.text_area("Write a note for this lesson", key="teacher_note_input")
    if st.button("💾 Save Note", key="save_teacher_note"):
        if new_note.strip():
            st.session_state["teacher_notes"].append(
                {
                    "symbol": symbol_input,
                    "note": new_note.strip(),
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )
            st.toast("Note saved for this session.")
        else:
            st.warning("Please write a note before saving.")

    if st.session_state["teacher_notes"]:
        st.markdown("###### Saved Notes")
        for _note in reversed(st.session_state["teacher_notes"]):
            st.markdown(f"**[{_note['timestamp']}] {_note['symbol']}:** {_note['note']}")
    else:
        st.caption("No notes saved yet.")


def render_learn_page() -> None:
    """Learn — plain-language company profile, vocabulary, quizzes, and reflection questions."""
    learn_content = get_learn_content(symbol_input, info)
    st.markdown(f"#### 🎓 Learn: {company_name} ({symbol_input})")
    st.caption("Written in plain language for students — no finance background required.")

    with st.expander("🏢 Company Overview", expanded=True):
        st.write(learn_content["overview"])

    with st.expander("💡 Business Model"):
        st.write(learn_content["business_model"])

    with st.expander("📦 Products & Services"):
        st.write(learn_content["products"])

    with st.expander("💰 How the Company Makes Money"):
        st.write(learn_content["how_it_makes_money"])

    with st.expander("⚔️ Major Competitors"):
        st.write(learn_content["competitors"])

    with st.expander("🏭 Industry"):
        st.write(learn_content["industry"])

    with st.expander("🏆 Competitive Advantages"):
        st.write(learn_content["advantages"])

    with st.expander("⚠️ Potential Risks"):
        st.write(learn_content["risks"])

    # ----- Did You Know? card (Requirement 8) ----- #
    st.markdown("##### 💡 Did You Know?")
    did_you_know_fact = get_did_you_know(symbol_input, info)
    st.info(f"**Did you know?** {did_you_know_fact}")
    if st.button("🔄 Show Another Fact", key="dyk_refresh"):
        st.rerun()

    st.divider()

    # ----- Key Vocabulary (Requirement 5: vocabulary cards) ----- #
    st.markdown("##### 📖 Key Vocabulary")
    st.caption("Click each term below to see its definition, why it matters, an example, and a common mistake students make.")
    for _vocab_term in [
        "Market Capitalization",
        "P/E Ratio",
        "Revenue",
        "Net Margin",
        "ROE",
        "Free Cash Flow",
    ]:
        render_metric_education(_vocab_term)

    st.divider()

    # ----- Think Like an Investor (Requirement 9) ----- #
    st.markdown("##### 🧠 Think Like an Investor")
    st.caption("Reflect on these questions individually or discuss them as a class.")
    for _i, _question in enumerate(generate_reflection_questions(company_name, info), start=1):
        st.markdown(f"**{_i}.** {_question}")

    st.divider()

    # =============================================================== #
    # NEW: CLASSROOM ACTIVITIES EXPANDER
    # Added below the existing educational explanation content, per
    # Education Mode enhancement request. Purely additive — does not
    # touch any analytics, charts, or scoring logic.
    # =============================================================== #
    with st.expander("🏫 Classroom Activities", expanded=False):
        st.markdown("### Think–Pair–Share")
        st.markdown(
            f"Based on today's data, do you think **{company_name}** is overvalued or "
            f"undervalued? Support your answer using the P/E ratio."
        )

        st.markdown("### Small Group Activity")
        st.markdown(
            f"Have students compare **{company_name}** with another company of their "
            f"choice using:"
        )
        st.markdown(
            "- Market Cap\n"
            "- Revenue Growth\n"
            "- Profit Margin\n"
            "- P/E Ratio"
        )
        st.markdown("Students should decide which company they would invest in and explain why.")

        st.markdown("### Whole Class Discussion")
        st.markdown(
            "- Should investors rely more on financial statements or stock charts?\n"
            "- Can a great company still be a bad investment?\n"
            "- What financial metric surprised you the most today?"
        )
    # ----- END NEW: CLASSROOM ACTIVITIES EXPANDER ----- #

    # =============================================================== #
    # NEW: INVESTING VOCABULARY EXPANDER
    # =============================================================== #
    with st.expander("📚 Investing Vocabulary", expanded=False):
        _quick_vocab_terms = {
            "Market Capitalization": "The total value of a company's shares — share price multiplied by the number of shares outstanding.",
            "Revenue": "The total amount of money a company brings in from sales, before any expenses are subtracted.",
            "Net Income": "The company's actual profit after ALL expenses, taxes, and costs have been subtracted from revenue.",
            "Earnings Per Share (EPS)": "A company's profit divided by its number of shares — shows how much profit is earned per share.",
            "P/E Ratio": "A company's share price divided by its earnings per share — shows how much investors are paying for each dollar of profit.",
            "Dividend": "A cash payment some companies make to shareholders, usually from profits.",
            "Bull Market": "A period when stock prices are generally rising and investor confidence is high.",
            "Bear Market": "A period when stock prices are generally falling and investor confidence is low.",
            "Volatility": "How much and how quickly a stock's price moves up and down over time.",
            "Risk": "The chance that an investment could lose value or not perform as expected.",
            "Diversification": "Spreading investments across different assets to reduce overall risk.",
            "Return on Equity (ROE)": "A measure of how efficiently a company uses shareholders' money to generate profit.",
        }
        for _term, _definition in _quick_vocab_terms.items():
            st.markdown(f"**{_term}:** {_definition}")
    # ----- END NEW: INVESTING VOCABULARY EXPANDER ----- #

    # =============================================================== #
    # NEW: QUICK QUIZ EXPANDER
    # =============================================================== #
    with st.expander("📝 Quick Quiz", expanded=False):
        _quiz_questions = [
            {
                "question": "If a stock has a P/E ratio of 25, what does this generally mean?",
                "options": [
                    "Investors are paying $25 for every $1 of earnings",
                    "The company lost $25 million",
                    "The stock price will double in 25 days",
                    "The company pays a 25% dividend",
                ],
                "correct": "Investors are paying $25 for every $1 of earnings",
                "explanation": "The P/E ratio shows how much investors are willing to pay for each dollar of a company's earnings.",
            },
            {
                "question": "What is 'revenue'?",
                "options": [
                    "The total money a company earns from sales before expenses",
                    "The profit left after all expenses",
                    "The amount of debt a company owes",
                    "The number of employees a company has",
                ],
                "correct": "The total money a company earns from sales before expenses",
                "explanation": "Revenue is the total sales a company brings in — it doesn't account for costs yet.",
            },
            {
                "question": "What is a dividend?",
                "options": [
                    "A cash payment companies sometimes make to shareholders",
                    "A fee investors pay to buy stock",
                    "A type of stock market crash",
                    "A government tax on stock profits",
                ],
                "correct": "A cash payment companies sometimes make to shareholders",
                "explanation": "Dividends are a way companies share profits directly with their shareholders.",
            },
            {
                "question": "Why do investors diversify their portfolio?",
                "options": [
                    "To reduce risk by spreading investments across different assets",
                    "To guarantee higher returns every year",
                    "To avoid paying any taxes",
                    "To make the portfolio harder to track",
                ],
                "correct": "To reduce risk by spreading investments across different assets",
                "explanation": "Diversification helps reduce risk — if one investment performs poorly, others may offset the loss.",
            },
            {
                "question": "How is market capitalization calculated?",
                "options": [
                    "Share price multiplied by total number of shares outstanding",
                    "Total revenue minus total expenses",
                    "Total debt plus total equity",
                    "Stock price divided by earnings per share",
                ],
                "correct": "Share price multiplied by total number of shares outstanding",
                "explanation": "Market cap = share price × total shares outstanding, representing the company's total market value.",
            },
        ]

        _quiz_score = 0
        for _q_idx, _quiz_item in enumerate(_quiz_questions, start=1):
            st.markdown(f"**Question {_q_idx}:** {_quiz_item['question']}")
            _student_answer = st.radio(
                f"Select an answer for question {_q_idx}",
                options=_quiz_item["options"],
                index=None,
                key=f"student_quiz_q{_q_idx}",
                label_visibility="collapsed",
            )
            if _student_answer is not None:
                if _student_answer == _quiz_item["correct"]:
                    _quiz_score += 1
                    st.success(f"✅ Correct! {_quiz_item['explanation']}")
                else:
                    st.error(f"❌ Not quite. {_quiz_item['explanation']}")
            st.markdown("---")

        st.markdown("#### Quiz Score:")
        st.markdown(f"### {_quiz_score} / 5")
    # ----- END NEW: QUICK QUIZ EXPANDER ----- #

    # =============================================================== #
    # NEW: TEACHER NOTES (LESSON PLAN) EXPANDER
    # =============================================================== #
    with st.expander("👨‍🏫 Teacher Notes", expanded=False):
        st.markdown("**Recommended Grade:**")
        st.markdown("9–12")

        st.markdown("**Course:**")
        st.markdown("Emerging Financial Markets")

        st.markdown("**Estimated Lesson Time:**")
        st.markdown("25–40 minutes")

        st.markdown("**Learning Objectives:**")
        st.markdown(
            "- Interpret stock data\n"
            "- Understand valuation metrics\n"
            "- Compare companies\n"
            "- Build investment reasoning"
        )

        st.markdown("**Homework:**")
        st.markdown(
            "Choose another publicly traded company and write a one-page investment "
            "recommendation using at least five metrics from the app."
        )


def render_education_mode_page() -> None:
    """Education Mode — combines the Learn and Classroom content under one
    always-reachable nav page (previously gated behind a toggle)."""
    st.markdown("#### 🎓 Education Mode")
    st.caption(
        "Classroom-friendly explanations and tools for the currently selected "
        "ticker. The 'Show inline explanations' toggle in the sidebar also adds "
        "quick-reference term cards throughout Dashboard and Stock Research."
    )
    edu_inner_tabs = st.tabs(["🎓 Learn", "📝 Classroom"])
    with edu_inner_tabs[0]:
        render_learn_page()
    with edu_inner_tabs[1]:
        render_classroom_page()


def _render_coming_soon(title: str, icon: str, description: str) -> None:
    """Shared layout for not-yet-built features — honestly labeled rather
    than faked, so nothing in the nav claims functionality that doesn't exist."""
    st.markdown(f"#### {icon} {title}")
    st.info(f"🚧 **Coming soon.** {description}")
    st.caption("This page is a placeholder in the current build and isn't functional yet.")


def render_stock_screener_page() -> None:
    _render_coming_soon(
        "Stock Screener",
        "🖥️",
        "Filter stocks across the market by metric thresholds (P/E, market cap, "
        "sector, growth, and more) to build a custom candidate list.",
    )


def render_options_analyzer_page() -> None:
    _render_coming_soon(
        "Options Analyzer",
        "🎯",
        "Analyze options chains, Greeks, and payoff diagrams for the selected "
        "ticker.",
    )


def render_ai_tutor_page() -> None:
    _render_coming_soon(
        "AI Tutor",
        "🤖",
        "A conversational tutor for asking follow-up questions about financial "
        "concepts and the data shown elsewhere in the app, distinct from the "
        "static explanations already in Education Mode.",
    )


# =========================================================================== #
# SECTION 8: STREAMLIT APPLICATION (originally app.py)
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Stock Research Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# --------------------------------------------------------------------------- #
# Global styling — Bloomberg-inspired dark dashboard, dark-mode compatible
# --------------------------------------------------------------------------- #

# ----- NEW: DARK/LIGHT THEME TOGGLE -----
# Reads the persisted theme choice (set by the sidebar widget below) before
# the widget itself is re-instantiated later in this run, so the CSS for
# the correct theme is applied immediately.
_theme_mode = st.session_state.get("theme_mode_toggle", "Dark")
CUSTOM_CSS = build_custom_css(_theme_mode)
# ----- END NEW: DARK/LIGHT THEME TOGGLE -----
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# ----- NEW: MULTI-CURRENCY DISPLAY SUPPORT (v1.3) -----
# Module-level display-currency state, set by the sidebar currency selector
# further down in the script. Defaults preserve the original USD behavior
# exactly (symbol "$", rate 1.0) until the user picks a different currency.
_DISPLAY_CURRENCY_SYMBOL = "$"
_DISPLAY_FX_RATE = 1.0
# ----- END NEW -----


def fmt_large_number(value: Any) -> str:
    """
    Format large financial numbers with B/M/K suffixes, converted into the
    user's selected display currency (defaults to USD, unchanged from the
    original behavior, when Display Currency is left at USD).
    """
    try:
        value = float(value) * _DISPLAY_FX_RATE
    except (TypeError, ValueError):
        return "N/A"
    symbol = _DISPLAY_CURRENCY_SYMBOL
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{symbol}{value/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{symbol}{value/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{symbol}{value/1e6:.2f}M"
    if abs_v >= 1e3:
        return f"{symbol}{value/1e3:.2f}K"
    return f"{symbol}{value:.2f}"


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def build_pdf_report(symbol: str, info: dict, scores: dict) -> bytes:
    """
    Generate a lightweight PDF summary report using fpdf2.
    Returns the raw PDF bytes for use with st.download_button.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, f"{symbol} - Stock Research Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Company Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, str(info.get("longBusinessSummary", "No description available."))[:1200])
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Market Cap", fmt_large_number(info.get("marketCap"))),
        ("P/E (Trailing)", fmt_ratio(info.get("trailingPE"))),
        ("Forward P/E", fmt_ratio(info.get("forwardPE"))),
        ("PEG Ratio", fmt_ratio(info.get("pegRatio"))),
        ("ROE", fmt_pct(info.get("returnOnEquity"))),
        ("ROA", fmt_pct(info.get("returnOnAssets"))),
        ("Revenue", fmt_large_number(info.get("totalRevenue"))),
        ("Total Cash", fmt_large_number(info.get("totalCash"))),
        ("Total Debt", fmt_large_number(info.get("totalDebt"))),
        ("Dividend Yield", fmt_pct(info.get("dividendYield"))),
    ]
    for label, value in metrics:
        pdf.cell(0, 6, f"{label}: {value}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "AI-Generated Scores", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, result in scores.items():
        pdf.cell(0, 6, f"{label}: {result.score:.1f}/100 ({result.label})", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This report is generated for educational and informational "
        "purposes only and does not constitute financial advice."
    )

    return bytes(pdf.output(dest="S"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## 📈 AI Stock Research")
    st.caption("Bloomberg-inspired equity research dashboard")

    # ----- NEW: SIDEBAR NAVIGATION (UI overhaul) ----- #
    # Replaces the old flat/nested tab bar with Streamlit's native grouped
    # sidebar navigation, matching the v2 dashboard design. Each page below
    # is the exact same content that used to live in a `with tab_X:` block
    # (see SECTION 7E above) — Stock Screener, Options Analyzer, and AI
    # Tutor are honestly labeled "Coming Soon" placeholders since they
    # aren't built yet, rather than being faked as functional.
    pg = st.navigation(
        {
            "Research": [
                st.Page(render_dashboard_page, title="Dashboard", icon="📊", default=True),
                st.Page(render_stock_research_page, title="Stock Research", icon="🔍"),
                st.Page(render_dcf_valuation_page, title="DCF Valuation", icon="🧮"),
                st.Page(render_peers_page, title="Comparables", icon="🏆"),
                st.Page(render_compare_page, title="Compare Stocks", icon="⚖️"),
                st.Page(render_earnings_page, title="Earnings", icon="📅"),
                st.Page(render_news_page, title="News & Sentiment", icon="📰"),
            ],
            "Portfolio": [
                st.Page(render_portfolio_overview_page, title="Portfolio Overview", icon="💼"),
                st.Page(render_holdings_page, title="Holdings", icon="📋"),
                st.Page(render_performance_page, title="Performance", icon="📈"),
                st.Page(render_risk_analysis_page, title="Risk Analysis", icon="⚠️"),
            ],
            "Tools": [
                st.Page(render_stock_screener_page, title="Stock Screener", icon="🖥️"),
                st.Page(render_watchlist_page, title="Watchlist", icon="⭐"),
                st.Page(render_alerts_page, title="Alerts", icon="🔔"),
                st.Page(render_options_analyzer_page, title="Options Analyzer", icon="🎯"),
                st.Page(render_backtest_page, title="Backtest", icon="🔁"),
                st.Page(render_dividends_page, title="Dividends", icon="💵"),
                st.Page(render_export_page, title="Export", icon="📤"),
            ],
            "Learn": [
                st.Page(render_education_mode_page, title="Education Mode", icon="🎓"),
                st.Page(render_ai_tutor_page, title="AI Tutor", icon="🤖"),
            ],
        }
    )
    # ----- END NEW: SIDEBAR NAVIGATION ----- #

    st.divider()
    st.markdown("#### Session Settings")

    symbol_input = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, MSFT, TSLA").strip().upper()

    period = st.selectbox(
        "Chart Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
    )
    interval = st.selectbox(
        "Interval",
        options=["1d", "1wk", "1mo"],
        index=0,
    )

    with st.expander("Chart Overlays"):
        show_sma20 = st.checkbox("20-Day SMA", value=True)
        show_sma50 = st.checkbox("50-Day SMA", value=True)
        show_sma200 = st.checkbox("200-Day SMA", value=False)
        show_bollinger = st.checkbox("Bollinger Bands", value=False)

    with st.expander("Watchlist Quick Actions"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⭐ Add", use_container_width=True):
                add_to_watchlist(symbol_input)
                st.toast(f"{symbol_input} added to watchlist")
        with col_b:
            if st.button("🗑️ Remove", use_container_width=True):
                remove_from_watchlist(symbol_input)
                st.toast(f"{symbol_input} removed")
        if st.session_state.get(WATCHLIST_KEY):
            st.caption("Current watchlist:")
            st.write(", ".join(st.session_state[WATCHLIST_KEY]))

    with st.expander("Compare Stocks — Tickers"):
        compare_symbols = st.text_input(
            "Comma-separated tickers", value="AAPL, MSFT, GOOGL", key="compare_input"
        )

    # ----- NEW: EDUCATION MODE (Requirement 1) ----- #
    with st.expander("🎓 Classroom Tools"):
        education_mode = st.checkbox(
            "🎓 Show inline explanations",
            value=False,
            key="education_mode_toggle",
            help="Adds classroom-friendly metric explanation cards to "
            "Dashboard and Stock Research. The full Learn/Classroom content "
            "is always available from the Learn section of the sidebar nav "
            "above, regardless of this toggle.",
        )
        student_mode = False
        if education_mode:
            student_mode = st.checkbox(
                "👦 Student Mode",
                value=False,
                key="student_mode_toggle",
                help="Hides advanced metrics and simplifies terminology for "
                "student use.",
            )
    # ----- END NEW: EDUCATION MODE ----- #

    # ----- NEW: MULTI-CURRENCY DISPLAY SUPPORT (v1.3) ----- #
    with st.expander("💱 Display Currency"):
        display_currency = st.selectbox(
            "Currency",
            options=SUPPORTED_CURRENCIES,
            index=0,
            key="display_currency_select",
            label_visibility="collapsed",
            help="Converts market cap, revenue, and other dollar figures into "
            "the selected currency using a live FX rate. Underlying data is "
            "still sourced in the security's native currency.",
        )
        _DISPLAY_FX_RATE = get_fx_rate(display_currency)
        _DISPLAY_CURRENCY_SYMBOL = CURRENCY_SYMBOLS.get(display_currency, "$")
        if display_currency != "USD":
            st.caption(f"1 USD ≈ {_DISPLAY_FX_RATE:.4f} {display_currency}")
    # ----- END NEW: MULTI-CURRENCY DISPLAY SUPPORT ----- #

    st.divider()

    # ----- NEW: DARK/LIGHT THEME TOGGLE — moved to bottom of sidebar ----- #
    _theme_index = 0 if _theme_mode == "Dark" else 1
    st.radio(
        "🌓 Theme",
        options=["Dark", "Light"],
        index=_theme_index,
        key="theme_mode_toggle",
        horizontal=True,
        help="Switch between the dashboard's dark and light color schemes.",
    )
    # ----- END NEW: DARK/LIGHT THEME TOGGLE ----- #

    st.caption("⚠️ For educational purposes only. Not financial advice.")


# --------------------------------------------------------------------------- #
# Validate ticker & load core data
# --------------------------------------------------------------------------- #

if not symbol_input:
    st.info("👈 Enter a ticker symbol in the sidebar to begin.")
    st.stop()

with st.spinner(f"Loading data for {symbol_input}..."):
    info = get_company_info(symbol_input)
    hist = get_price_history(symbol_input, period=period, interval=interval)

if not info and hist.empty:
    st.error(
        f"Could not find data for ticker '{symbol_input}'. "
        "Please check the symbol and try again."
    )
    st.stop()

live = get_live_price(symbol_input)
logo_url = get_logo_url(symbol_input, info.get("website"))

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

header_col1, header_col2, header_col3 = st.columns([1, 4, 2])

with header_col1:
    if logo_url:
        try:
            st.image(logo_url, width=72)
        except Exception:
            st.markdown("### 🏢")
    else:
        st.markdown("### 🏢")

with header_col2:
    company_name = info.get("shortName") or info.get("longName") or symbol_input
    st.markdown(f"### {company_name}  &nbsp; <span class='ticker-badge'>{symbol_input}</span>", unsafe_allow_html=True)
    sector = info.get("sector", "—")
    industry = info.get("industry", "—")
    st.caption(f"{sector} • {industry} • {info.get('country', '—')}")

with header_col3:
    price = live.get("price")
    change = live.get("change")
    pct_change = live.get("pct_change")
    currency = live.get("currency", "USD")
    if price is not None:
        direction_class = "price-up" if (change or 0) >= 0 else "price-down"
        arrow = "▲" if (change or 0) >= 0 else "▼"
        st.markdown(f"### {price:,.2f} {currency}")
        if change is not None and pct_change is not None:
            st.markdown(
                f"<span class='{direction_class}'>{arrow} {change:+.2f} ({pct_change:+.2f}%)</span>",
                unsafe_allow_html=True,
            )
        # ----- NEW: MULTI-CURRENCY DISPLAY SUPPORT (v1.3) ----- #
        if display_currency != currency:
            st.caption(
                f"≈ {fmt_currency_price(price, _DISPLAY_FX_RATE, _DISPLAY_CURRENCY_SYMBOL)} "
                f"{display_currency}"
            )
        # ----- END NEW ----- #
    else:
        st.markdown("### Price unavailable")

st.divider()

# --------------------------------------------------------------------------- #
# Render the selected page
# --------------------------------------------------------------------------- #

pg.run()