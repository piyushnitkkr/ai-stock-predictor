# Added logging for debugging
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from predict import predict_stock
from breakout import detect_breakout
from news_sentiment import get_stock_news
from nifty50 import nifty50
from data_fetcher import get_stock_data


def _calculate_news_sentiment(ticker):
    """Calculate sentiment from yfinance news (more reliable than NewsAPI)."""
    try:
        news_items = get_stock_news(ticker)
        if not news_items:
            return 0.0
        scores = [item["sentiment"] for item in news_items]
        return sum(scores) / len(scores)
    except Exception as e:
        logging.warning(f"News sentiment failed for {ticker}: {e}")
        return 0.0


def recommend_stocks():

    picks = []
    processed = 0

    for stock in nifty50:

        try:

            data = get_stock_data(stock)
            if data is None or data.empty:
                logging.warning(f"No data for stock: {stock}")
                continue

            pred = predict_stock(stock, data)  # Use same data we fetched
            breakout = detect_breakout(data)
            sentiment = _calculate_news_sentiment(stock)

            # Dynamic threshold for bullish stocks
            bullish_threshold = 0.25  # Lowered from 0.30 to show more picks

            score = pred["bullish"] * 2 + sentiment
            if breakout == "Bullish Breakout":
                score += 1

            if pred["bullish"] > bullish_threshold:
                picks.append({
                    "stock": stock,
                    "bullish": pred["bullish"],
                    "sentiment": sentiment,
                    "breakout": breakout,
                    "score": score
                })
            
            processed += 1

        except Exception as e:
            logging.error(f"Error processing stock {stock}: {e}", exc_info=True)
            continue

    logging.info(f"Processed {processed}/{len(nifty50)} stocks, found {len(picks)} picks")
    picks = sorted(picks, key=lambda x: x["score"], reverse=True)
    return picks[:10]