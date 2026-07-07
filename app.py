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
@dataclass
class DCFResult:
    """Structured output of a DCF run."""
    projected_fcfs: list[float]
    discounted_fcfs: list[float]
    terminal_value: float
    discounted_terminal_value: float
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: Optional[float]


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


def run_dcf(
    base_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    projection_years: int,
    net_debt: float = 0.0,
    shares_outstanding: Optional[float] = None,
) -> DCFResult:
    """
    Run a straightforward two-stage Discounted Cash Flow model.

    Parameters
    ----------
    base_fcf : float
        Most recent trailing free cash flow (in the same currency units
        as the rest of the model, e.g. USD).
    growth_rate : float
        Annual FCF growth rate applied during the explicit projection
        window, expressed as a decimal (e.g. 0.08 for 8%).
    discount_rate : float
        Discount rate / WACC, expressed as a decimal (e.g. 0.09 for 9%).
    terminal_growth_rate : float
        Perpetual growth rate applied to the terminal value, as a decimal.
    projection_years : int
        Number of explicit forecast years (typically 5-10).
    net_debt : float
        Total debt minus cash & equivalents, used to bridge from
        enterprise value to equity value.
    shares_outstanding : float, optional
        Diluted shares outstanding, used to compute per-share value.

    Returns
    -------
    DCFResult
    """
    if discount_rate <= terminal_growth_rate:
        # Guard against a mathematically invalid (negative/infinite) terminal value
        terminal_growth_rate = max(0.0, discount_rate - 0.01)

    projected_fcfs = []
    discounted_fcfs = []
    fcf = base_fcf

    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + growth_rate)
        projected_fcfs.append(fcf)
        discount_factor = (1 + discount_rate) ** year
        discounted_fcfs.append(fcf / discount_factor)

    terminal_value = (
        projected_fcfs[-1] * (1 + terminal_growth_rate)
        / (discount_rate - terminal_growth_rate)
    )
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** projection_years

    enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
    equity_value = enterprise_value - net_debt

    intrinsic_value_per_share = None
    if shares_outstanding and shares_outstanding > 0:
        intrinsic_value_per_share = round(equity_value / shares_outstanding, 2)

    return DCFResult(
        projected_fcfs=[round(v, 2) for v in projected_fcfs],
        discounted_fcfs=[round(v, 2) for v in discounted_fcfs],
        terminal_value=round(terminal_value, 2),
        discounted_terminal_value=round(discounted_terminal_value, 2),
        enterprise_value=round(enterprise_value, 2),
        equity_value=round(equity_value, 2),
        intrinsic_value_per_share=intrinsic_value_per_share,
    )


def margin_of_safety(intrinsic_value: Optional[float], current_price: Optional[float]) -> Optional[float]:
    """
    Percentage margin of safety between an estimated intrinsic value and
    the current market price. Positive = undervalued, negative = overvalued.
    """
    if not intrinsic_value or not current_price or current_price <= 0:
        return None
    return round(((intrinsic_value - current_price) / current_price) * 100, 2)


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

CUSTOM_CSS = """
<style>
    .stApp { background-color: var(--background-color); }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.8; }

    /* Section headers */
    h1, h2, h3 { letter-spacing: -0.3px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* Ticker badge */
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(135deg, #F5A623, #F76B1C);
        color: #111;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .price-up { color: #00C805; font-weight: 700; }
    .price-down { color: #FF3B30; font-weight: 700; }

    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def fmt_large_number(value: Any) -> str:
    """Format large financial numbers with B/M/K suffixes."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"${value/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${value/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${value/1e6:.2f}M"
    if abs_v >= 1e3:
        return f"${value/1e3:.2f}K"
    return f"${value:.2f}"


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
    st.divider()

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

    st.divider()
    st.markdown("#### Chart Overlays")
    show_sma20 = st.checkbox("20-Day SMA", value=True)
    show_sma50 = st.checkbox("50-Day SMA", value=True)
    show_sma200 = st.checkbox("200-Day SMA", value=False)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⭐ Watchlist", use_container_width=True):
            add_to_watchlist(symbol_input)
            st.toast(f"{symbol_input} added to watchlist")
    with col_b:
        if st.button("🗑️ Remove", use_container_width=True):
            remove_from_watchlist(symbol_input)
            st.toast(f"{symbol_input} removed")

    if st.session_state.get(WATCHLIST_KEY):
        st.caption("Current watchlist:")
        st.write(", ".join(st.session_state[WATCHLIST_KEY]))

    st.divider()
    st.markdown("#### Compare Stocks")
    compare_symbols = st.text_input(
        "Comma-separated tickers", value="AAPL, MSFT, GOOGL", key="compare_input"
    )

    # ----- NEW: EDUCATION MODE (Requirement 1) ----- #
    st.divider()
    st.markdown("#### 🎓 Classroom Tools")
    education_mode = st.checkbox(
        "🎓 Education Mode",
        value=False,
        key="education_mode_toggle",
        help="Turns on classroom-friendly metric explanations plus new "
        "'Learn' and 'Classroom' tabs. When off, the app behaves exactly "
        "as it did before.",
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

    st.divider()
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
    else:
        st.markdown("### Price unavailable")

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

# ----- NEW: EDUCATION MODE — conditionally add "Learn" and "Classroom" -----
# tabs only when Education Mode is on, so tab layout is identical to the
# original app when Education Mode is off (Requirement 1).
_tab_labels = [
    "Overview",
    "Technical",
    "Fundamentals",
    "Financials",
    "Analyst",
    "News",
    "Valuation",
    "AI Scores",
    "Portfolio",
    "Compare",
]
if education_mode:
    _tab_labels += ["🎓 Learn", "📝 Classroom"]
_tab_labels += ["Export"]

_tabs = st.tabs(_tab_labels)

(
    tab_overview,
    tab_technical,
    tab_fundamentals,
    tab_financials,
    tab_analyst,
    tab_news,
    tab_valuation,
    tab_scores,
    tab_portfolio,
    tab_compare,
) = _tabs[:10]

if education_mode:
    tab_learn, tab_classroom = _tabs[10:12]
    tab_export = _tabs[12]
else:
    tab_learn, tab_classroom = None, None
    tab_export = _tabs[10]
# ----- END NEW: EDUCATION MODE -----

# --------------------------------------------------------------------------- #
# Overview tab
# --------------------------------------------------------------------------- #

with tab_overview:
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
    # ----- END NEW: EDUCATION MODE ----- #


# --------------------------------------------------------------------------- #
# Technical tab
# --------------------------------------------------------------------------- #

with tab_technical:
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

# --------------------------------------------------------------------------- #
# Financials tab
# --------------------------------------------------------------------------- #

with tab_financials:
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


# --------------------------------------------------------------------------- #
# Analyst tab
# --------------------------------------------------------------------------- #

with tab_analyst:
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


# --------------------------------------------------------------------------- #
# News tab
# --------------------------------------------------------------------------- #

with tab_news:
    st.markdown("#### Latest News")
    raw_news = get_company_news(symbol_input, limit=10)
    normalized = normalize_news_list(raw_news)

    if not normalized:
        st.info("No recent news found for this ticker.")
    else:
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
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['publisher']} • {item['published']}")
            st.divider()


# --------------------------------------------------------------------------- #
# Valuation tab
# --------------------------------------------------------------------------- #

with tab_valuation:
    st.markdown("#### Graham Intrinsic Value")
    eps = info.get("trailingEps")
    book_value = info.get("bookValue")
    g_number = graham_number(eps, book_value)

    col1, col2, col3 = st.columns(3)
    col1.metric("Trailing EPS", f"{eps:.2f}" if eps else "N/A")
    col2.metric("Book Value / Share", f"{book_value:.2f}" if book_value else "N/A")
    col3.metric("Graham Number", f"${g_number:.2f}" if g_number else "N/A")

    if g_number and live.get("price"):
        mos = margin_of_safety(g_number, live["price"])
        if mos is not None:
            st.metric("Margin of Safety vs. Current Price", f"{mos:+.2f}%")

    st.divider()
    st.markdown("#### Discounted Cash Flow (DCF) Calculator")
    st.caption(
        "Adjust the assumptions below to build your own DCF estimate. "
        "This is a simplified two-stage model intended for educational use."
    )

    base_fcf = info.get("freeCashflow") or 0
    shares_out = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0
    net_debt = (total_debt or 0) - (total_cash or 0)

    dcf_col1, dcf_col2, dcf_col3 = st.columns(3)
    with dcf_col1:
        input_fcf = st.number_input(
            "Base Free Cash Flow ($)", value=float(base_fcf) if base_fcf else 1_000_000_000.0, step=1_000_000.0
        )
        growth_rate = st.slider("Growth Rate (Years 1-5)", 0.0, 30.0, 8.0, 0.5) / 100
    with dcf_col2:
        discount_rate = st.slider("Discount Rate (WACC)", 4.0, 20.0, 9.0, 0.5) / 100
        terminal_growth = st.slider("Terminal Growth Rate", 0.0, 5.0, 2.5, 0.1) / 100
    with dcf_col3:
        projection_years = st.slider("Projection Years", 3, 10, 5, 1)
        input_net_debt = st.number_input("Net Debt ($)", value=float(net_debt))

    dcf_result = run_dcf(
        base_fcf=input_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth,
        projection_years=projection_years,
        net_debt=input_net_debt,
        shares_outstanding=shares_out,
    )

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Enterprise Value", fmt_large_number(dcf_result.enterprise_value))
    res_col2.metric("Equity Value", fmt_large_number(dcf_result.equity_value))
    if dcf_result.intrinsic_value_per_share:
        res_col3.metric("Intrinsic Value / Share", f"${dcf_result.intrinsic_value_per_share:.2f}")
        if live.get("price"):
            dcf_mos = margin_of_safety(dcf_result.intrinsic_value_per_share, live["price"])
            if dcf_mos is not None:
                st.metric("DCF Margin of Safety", f"{dcf_mos:+.2f}%")
    else:
        res_col3.metric("Intrinsic Value / Share", "N/A (shares outstanding unavailable)")

    with st.expander("📊 Projected Cash Flow Detail"):
        proj_df = pd.DataFrame(
            {
                "Year": list(range(1, projection_years + 1)),
                "Projected FCF": dcf_result.projected_fcfs,
                "Discounted FCF": dcf_result.discounted_fcfs,
            }
        )
        st.dataframe(proj_df, use_container_width=True)
        st.caption(
            f"Terminal Value: {fmt_large_number(dcf_result.terminal_value)} | "
            f"Discounted Terminal Value: {fmt_large_number(dcf_result.discounted_terminal_value)}"
        )

    st.info(
        "⚠️ Disclaimer: Valuation estimates are for educational purposes only "
        "and should not be considered investment advice."
    )


# --------------------------------------------------------------------------- #
# AI Scores tab
# --------------------------------------------------------------------------- #

with tab_scores:
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


# --------------------------------------------------------------------------- #
# Portfolio tab
# --------------------------------------------------------------------------- #

with tab_portfolio:
    st.markdown("#### Portfolio Calculator")
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
        totals = get_portfolio_totals(summary_df)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Value", fmt_large_number(totals["total_value"]))
        t2.metric("Total Cost", fmt_large_number(totals["total_cost"]))
        t3.metric("Unrealized P/L", fmt_large_number(totals["total_pl"]))
        t4.metric("Return", f"{totals['total_pl_pct']:+.2f}%")

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

    st.divider()
    st.markdown("#### Watchlist")
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


# --------------------------------------------------------------------------- #
# Compare tab
# --------------------------------------------------------------------------- #

with tab_compare:
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


# =========================================================================== #
# NEW: EDUCATION MODE — "🎓 Learn" and "📝 Classroom" tabs
# (Requirements 3, 4, 6, 8, 9). Only rendered when Education Mode is on;
# tab_learn / tab_classroom are None otherwise, matching the conditional
# tab creation above.
# =========================================================================== #

if education_mode:
    company_name = info.get("shortName") or info.get("longName") or symbol_input
    learn_content = get_learn_content(symbol_input, info)

    # ----------------------------------------------------------------- #
    # 🎓 Learn tab (Requirement 3)
    # ----------------------------------------------------------------- #
    with tab_learn:
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

    # ----------------------------------------------------------------- #
    # 📝 Classroom tab (Requirements 4 and 6)
    # ----------------------------------------------------------------- #
    with tab_classroom:
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


# --------------------------------------------------------------------------- #
# Export tab
# --------------------------------------------------------------------------- #

with tab_export:
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