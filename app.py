import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

from predict import predict_stock
from data_fetcher import get_stock_data
from support_resistance import support_resistance
from breakout import detect_breakout
from sector_analysis import sector_strength
from stock_recommender import recommend_stocks
from news_sentiment import get_market_news, get_stock_news


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base background */
[data-testid="stAppViewContainer"] { background: #0e0e1a; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-size: 0.9rem;
    color: #999;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e0e0ff;
    border-bottom: 2px solid #7c6af7;
}

/* Divider colour */
hr { border-color: #2a2a3e !important; }

/* Input */
input { background: #1a1a2e !important; color: #e0e0ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _col(df, keyword):
    """Return the first column whose string representation contains keyword."""
    return next((c for c in df.columns if keyword in str(c)), keyword)


def _sentiment_badge(score):
    if score > 0.05:
        return "#00c853", "Positive"
    if score < -0.05:
        return "#ff1744", "Negative"
    return "#ffd600", "Neutral"


def _news_card(title, url, source, score):
    color, label = _sentiment_badge(score)
    href = f"href='{url}' target='_blank'" if url else ""
    link_style = "color:#e0e0ff; text-decoration:none;" if url else "color:#e0e0ff;"
    st.markdown(f"""
    <div style='background:#1a1a2e; padding:0.8rem 1rem; border-radius:10px;
                border-left:4px solid {color}; margin-bottom:0.5rem;'>
        <a {href} style='{link_style} font-size:0.95rem;'>{title}</a>
        <br><span style='color:#666; font-size:0.75rem;'>
            {source}&nbsp;·&nbsp;<span style='color:{color};'>{label}</span>
        </span>
    </div>""", unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_sectors():
    return sector_strength()


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_picks():
    return recommend_stocks()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_news():
    return get_market_news()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":    info.get("longName") or info.get("shortName") or ticker,
            "price":   info.get("currentPrice") or info.get("regularMarketPrice"),
            "change":  info.get("regularMarketChangePercent", 0) or 0,
            "cap":     info.get("marketCap"),
        }
    except:
        return {"name": ticker, "price": None, "change": 0, "cap": None}


# ── Header ─────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("## 📈 AI Stock Dashboard — India")
with hcol2:
    st.markdown(
        f"<div style='text-align:right; color:#555; padding-top:0.6rem; font-size:0.85rem;'>"
        f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Search bar ─────────────────────────────────────────────────────────────────
sc1, sc2 = st.columns([6, 1])
with sc1:
    ticker_input = st.text_input(
        "",
        placeholder="Enter ticker — e.g. RELIANCE, TCS, INFY, HDFCBANK.NS",
        label_visibility="collapsed",
    )
with sc2:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_clicked and ticker_input.strip():
    ticker = ticker_input.strip().upper()

    with st.spinner("Fetching market data…"):
        data = get_stock_data(ticker)

    if data is None or data.empty:
        st.error("⚠️ Invalid ticker or data unavailable. Try adding `.NS` (e.g. `TCS.NS`).")
        st.stop()

    used_ticker = data.attrs.get("used_ticker", ticker)

    # Company header card
    info = _cached_stock_info(used_ticker)
    ch_color  = "#00c853" if info["change"] >= 0 else "#ff1744"
    ch_arrow  = "▲" if info["change"] >= 0 else "▼"
    price_html = (
        f'<span style="font-size:1.6rem; color:#fff; font-weight:700;">₹{info["price"]:,.2f}</span>'
        f'&nbsp;<span style="color:{ch_color}; font-size:1rem;">{ch_arrow} {abs(info["change"]):.2f}%</span>'
        if info["price"] else ""
    )
    alias_note = (
        f'&nbsp;<span style="color:#555; font-size:0.8rem;">(resolved from {ticker})</span>'
        if used_ticker != ticker else ""
    )
    data_rows = len(data)
    st.markdown(f"""
    <div style='background:#1a1a2e; padding:1.2rem 1.5rem; border-radius:14px; margin-bottom:1rem;'>
        <div style='color:#e0e0ff; font-size:1.25rem; font-weight:600;'>
            {info["name"]}&nbsp;<span style='color:#555; font-size:0.9rem;'>({used_ticker})</span>{alias_note}
        </div>
        <div style='margin-top:0.4rem;'>{price_html}</div>
        <div style='color:#666; font-size:0.75rem; margin-top:0.3rem;'>
            Data: {data_rows} days • Last: {data.index[-1].strftime('%d %b %Y')}
        </div>
    </div>""", unsafe_allow_html=True)

    # AI Prediction - Use the SAME data we fetched for display
    with st.spinner("Running AI prediction…"):
        result = predict_stock(ticker, data)  # Pass the data we already have

    if result["bullish"] == 0 and result["bearish"] == 0:
        st.warning("⚠️ Not enough data for a prediction.")
    else:
        bull, bear, side = result["bullish"], result["bearish"], result["sideways"]
        top_signal = max(bull, bear, side)
        if top_signal == bull:
            signal_txt, signal_col = "🟢 BULLISH", "#00c853"
        elif top_signal == bear:
            signal_txt, signal_col = "🔴 BEARISH", "#ff1744"
        else:
            signal_txt, signal_col = "🟡 SIDEWAYS", "#ffd600"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI Signal", signal_txt)
        m2.metric("Bullish",  f"{bull:.1%}", delta=f"{bull - 0.33:+.1%}")
        m3.metric("Bearish",  f"{bear:.1%}", delta=f"{bear - 0.33:+.1%}", delta_color="inverse")
        m4.metric("Sideways", f"{side:.1%}")

    # ── Chart ──────────────────────────────────────────────────────────────────
    open_col  = _col(data, "Open")
    high_col  = _col(data, "High")
    low_col   = _col(data, "Low")
    close_col = _col(data, "Close")
    vol_col   = _col(data, "Volume")

    close_s = data[close_col].squeeze()
    open_s  = data[open_col].squeeze()
    sma20   = close_s.rolling(20).mean()
    sma50   = close_s.rolling(50).mean()

    bar_colors = [
        "#00c853" if float(c) >= float(o) else "#ff1744"
        for c, o in zip(close_s, open_s)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.73, 0.27],
        vertical_spacing=0.025,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data[open_col], high=data[high_col],
        low=data[low_col],   close=data[close_col],
        increasing_line_color="#00c853",
        decreasing_line_color="#ff1744",
        name="Price",
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=data.index, y=sma20,
        line=dict(color="#ffd600", width=1.3),
        name="SMA 20",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=sma50,
        line=dict(color="#40c4ff", width=1.3),
        name="SMA 50",
    ), row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data[vol_col].squeeze(),
        marker_color=bar_colors,
        opacity=0.55,
        name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e0e1a",
        plot_bgcolor="#0e0e1a",
        height=560,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        yaxis2=dict(title="Volume", title_font_size=11),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Support / Resistance / Breakout ────────────────────────────────────────
    sr_col, bo_col = st.columns([1, 1])

    with sr_col:
        supports, resistances = support_resistance(data)
        st.markdown("**🟩 Support Levels**")
        if supports:
            st.markdown(
                " &nbsp;·&nbsp; ".join(
                    f"<span style='color:#00c853; font-weight:600;'>₹{s:,.2f}</span>"
                    for s in supports
                ),
                unsafe_allow_html=True,
            )
        st.markdown("**🟥 Resistance Levels**")
        if resistances:
            st.markdown(
                " &nbsp;·&nbsp; ".join(
                    f"<span style='color:#ff1744; font-weight:600;'>₹{r:,.2f}</span>"
                    for r in resistances
                ),
                unsafe_allow_html=True,
            )

    with bo_col:
        breakout = detect_breakout(data)
        if "Bullish" in str(breakout):
            bo_color, bo_icon = "#00c853", "⬆️"
        elif "Bearish" in str(breakout):
            bo_color, bo_icon = "#ff1744", "⬇️"
        else:
            bo_color, bo_icon = "#ffd600", "↔️"
        st.markdown("**⚡ Breakout Signal**")
        st.markdown(
            f"<div style='background:#1a1a2e; padding:0.9rem 1.1rem; border-radius:10px; "
            f"border-left:4px solid {bo_color}; font-size:1.05rem; color:{bo_color}; font-weight:600;'>"
            f"{bo_icon} {breakout}</div>",
            unsafe_allow_html=True,
        )

    # ── Stock News ─────────────────────────────────────────────────────────────
    st.markdown("### 📰 Recent News")
    with st.spinner("Loading news…"):
        stock_news = get_stock_news(used_ticker)

    if stock_news:
        for item in stock_news:
            _news_card(item["title"], item["url"], item["source"], item["sentiment"])
    else:
        st.info("No recent news found for this stock.")


# ── Market Overview tabs ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌐 Market Overview")

tab_news, tab_sectors, tab_picks = st.tabs(["📰 Market News", "📊 Sector Strength", "🏆 Top Picks Today"])

# ── Market News tab ────────────────────────────────────────────────────────────
with tab_news:
    with st.spinner("Fetching market headlines…"):
        market_news = _cached_market_news()

    if market_news:
        for item in market_news:
            _news_card(item["title"], item["url"], item["source"], item["sentiment"])
    else:
        st.info("Market news unavailable right now.")

# ── Sector Strength tab ────────────────────────────────────────────────────────
with tab_sectors:
    with st.spinner("Loading sector data…"):
        try:
            sectors = _cached_sectors()
        except:
            sectors = []

    if sectors:
        max_abs = max(abs(v) for _, v in sectors) or 1
        for name, val in sectors:
            pct       = val * 100
            bar_w     = abs(val) / max_abs * 100
            s_color   = "#00c853" if val >= 0 else "#ff1744"
            sign      = "+" if val >= 0 else ""
            st.markdown(f"""
            <div style='margin-bottom:0.85rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                    <span style='color:#ccc; font-size:0.95rem;'>{name}</span>
                    <span style='color:{s_color}; font-weight:700;'>{sign}{pct:.2f}%</span>
                </div>
                <div style='height:10px; background:#222; border-radius:6px;'>
                    <div style='width:{bar_w:.1f}%; height:100%; background:{s_color}; border-radius:6px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Sector data unavailable.")

# ── Top Picks tab ──────────────────────────────────────────────────────────────
with tab_picks:
    with st.spinner("Scanning Nifty 50 for signals…"):
        try:
            picks = _cached_picks()
        except Exception as e:
            st.error(f"Error fetching picks: {e}")
            picks = []

    if picks:
        for i, s in enumerate(picks, 1):
            br_color  = "#00c853" if "Bullish" in str(s["breakout"]) else "#888"
            sent_color, sent_label = _sentiment_badge(s["sentiment"])
            bull_pct  = s["bullish"] * 100
            st.markdown(f"""
            <div style='background:#1a1a2e; padding:0.75rem 1.1rem; border-radius:10px;
                        border-left:4px solid #00c853; margin-bottom:0.5rem;
                        display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;'>
                <span style='color:#555; min-width:1.5rem; font-size:0.85rem;'>#{i}</span>
                <span style='color:#e0e0ff; font-weight:700; min-width:9rem;'>{s["stock"]}</span>
                <span style='color:#00c853;'>Bullish&nbsp;{bull_pct:.1f}%</span>
                <span style='color:#444;'>·</span>
                <span style='color:{sent_color};'>Sentiment&nbsp;{s["sentiment"]:+.2f}&nbsp;({sent_label})</span>
                <span style='color:#444;'>·</span>
                <span style='color:{br_color};'>{s["breakout"]}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("🔄 No strong bullish signals currently. The system scans all 50 stocks and may take a moment on first load.")
