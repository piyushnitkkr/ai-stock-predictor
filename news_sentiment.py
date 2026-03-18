import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file (for local development)
load_dotenv()

analyzer = SentimentIntensityAnalyzer()

# Try to get API key from Streamlit secrets first, then from .env, then use default
def get_api_key():
    try:
        # Streamlit Cloud uses st.secrets
        if hasattr(st, 'secrets') and st.secrets:
            api_key = st.secrets.get("NEWS_API_KEY", None)
            if api_key:
                return api_key
    except:
        pass
    
    # Fall back to .env file (local development)
    api_key = os.getenv("NEWS_API_KEY", None)
    if api_key and api_key != "YOUR_NEWS_API_KEY":
        return api_key
    
    # Return None if not found - graceful degradation
    return None

# Get API key lazily on first use
_API_KEY = None

def _get_api_key_cached():
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = get_api_key()
    return _API_KEY


def _parse_yf_item(item):
    """Handle both old and new yfinance news item formats."""
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content:
        title  = content.get("title", "")
        url    = content.get("canonicalUrl", {}).get("url", "") or ""
        source = content.get("provider", {}).get("displayName", "") or ""
    else:
        title  = item.get("title", "")
        url    = item.get("link", "") or ""
        source = item.get("publisher", "") or ""
    return title.strip(), url.strip(), source.strip()


def get_news(company):
    try:
        api_key = _get_api_key_cached()
        if not api_key:
            return []
        
        url = (
            f"https://newsapi.org/v2/everything?q={company}"
            f"&language=en&sortBy=publishedAt&apiKey={api_key}"
        )
        data = requests.get(url, timeout=5).json()
        return [a["title"] for a in data.get("articles", [])[:10]]
    except:
        return []


def analyze_sentiment(company):
    news = get_news(company)
    if not news:
        return 0
    scores = [analyzer.polarity_scores(h)["compound"] for h in news]
    return sum(scores) / len(scores)


def get_stock_news(ticker):
    """Return a list of news dicts with title, url, source, sentiment for a stock."""
    try:
        items = yf.Ticker(ticker).news or []
    except:
        return []

    results = []
    seen = set()
    for item in items[:8]:
        title, url, source = _parse_yf_item(item)
        if not title or title in seen:
            continue
        seen.add(title)
        score = analyzer.polarity_scores(title)["compound"]
        results.append({"title": title, "url": url, "source": source, "sentiment": score})
    return results


def get_market_news():
    """Return market-wide news by fetching from NSE & BSE indices via yfinance."""
    results = []
    seen = set()
    for symbol in ["^NSEI", "^BSESN"]:
        try:
            items = yf.Ticker(symbol).news or []
        except:
            continue
        for item in items[:8]:
            title, url, source = _parse_yf_item(item)
            if not title or title in seen:
                continue
            seen.add(title)
            score = analyzer.polarity_scores(title)["compound"]
            results.append({"title": title, "url": url, "source": source, "sentiment": score})

    return results[:12]
