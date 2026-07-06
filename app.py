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
    tab_export,
) = st.tabs(
    [
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
        "Export",
    ]
)

# --------------------------------------------------------------------------- #
# Overview tab
# --------------------------------------------------------------------------- #

with tab_overview:
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
    m1.metric("Market Cap", fmt_large_number(info.get("marketCap")))
    m2.metric("P/E (Trailing)", fmt_ratio(info.get("trailingPE")))
    m3.metric("Forward P/E", fmt_ratio(info.get("forwardPE")))
    m4.metric("PEG Ratio", fmt_ratio(info.get("pegRatio")))
    m5.metric("Dividend Yield", fmt_pct(info.get("dividendYield")))

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("ROE", fmt_pct(info.get("returnOnEquity")))
    m7.metric("ROA", fmt_pct(info.get("returnOnAssets")))
    m8.metric("Revenue (TTM)", fmt_large_number(info.get("totalRevenue")))
    m9.metric("Total Cash", fmt_large_number(info.get("totalCash")))
    m10.metric("Total Debt", fmt_large_number(info.get("totalDebt")))


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
    col1.metric("Market Cap", fmt_large_number(info.get("marketCap")))
    col2.metric("Trailing P/E", fmt_ratio(info.get("trailingPE")))
    col3.metric("Forward P/E", fmt_ratio(info.get("forwardPE")))
    col4.metric("PEG Ratio", fmt_ratio(info.get("pegRatio")))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Price / Book", fmt_ratio(info.get("priceToBook")))
    col6.metric("Price / Sales", fmt_ratio(info.get("priceToSalesTrailing12Months")))
    col7.metric("EV / EBITDA", fmt_ratio(info.get("enterpriseToEbitda")))
    col8.metric("Beta", fmt_ratio(info.get("beta")))

    st.markdown("#### Profitability & Returns")
    col9, col10, col11, col12 = st.columns(4)
    col9.metric("ROE", fmt_pct(info.get("returnOnEquity")))
    col10.metric("ROA", fmt_pct(info.get("returnOnAssets")))
    col11.metric("Profit Margin", fmt_pct(info.get("profitMargins")))
    col12.metric("Operating Margin", fmt_pct(info.get("operatingMargins")))

    st.markdown("#### Balance Sheet Snapshot")
    col13, col14, col15, col16 = st.columns(4)
    col13.metric("Total Cash", fmt_large_number(info.get("totalCash")))
    col14.metric("Total Debt", fmt_large_number(info.get("totalDebt")))
    col15.metric("Debt / Equity", fmt_ratio(info.get("debtToEquity")))
    col16.metric("Current Ratio", fmt_ratio(info.get("currentRatio")))

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