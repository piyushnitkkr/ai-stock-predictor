import numpy as np
import joblib
import logging

from data_fetcher import get_stock_data, get_nifty_index
from features import add_indicators


transformer = None
xgb = None
scaler = None
feature_selection = None
models_ready = False


def _load_models_once():
    """Load models once at module startup."""
    global transformer, xgb, scaler, feature_selection, models_ready

    if models_ready:
        return

    try:
        from tensorflow.keras.models import load_model

        logging.info("Loading trained models...")
        transformer = load_model("transformer_model.keras", compile=False)
        xgb = joblib.load("xgb_model.save")
        scaler = joblib.load("scaler.save")
        feature_selection = joblib.load("feature_selection.save")
        models_ready = True
        logging.info("Models and feature selection loaded successfully.")
        logging.info(f"Using {len(feature_selection['selected_features'])} selected features: {feature_selection['selected_features']}")
    except Exception as e:
        logging.error(f"Failed to load models at startup: {e}")
        logging.warning("Will use heuristic predictions as fallback.")
        models_ready = False


# Load models once when module is imported
_load_models_once()


def _softmax(vals):
    arr = np.array(vals, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp)
    if denom == 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return exp / denom


def _heuristic_prediction(data):
    close = data["Close"].astype(float)

    # Use a short/medium momentum blend plus RSI/MACD when available.
    ret_5 = 0.0
    ret_20 = 0.0
    try:
        if len(close) > 5:
            c_last = float(close.iloc[-1]) if hasattr(close.iloc[-1], 'item') else float(close.iloc[-1])
            c_5 = float(close.iloc[-6]) if hasattr(close.iloc[-6], 'item') else float(close.iloc[-6])
            ret_5 = (c_last / c_5) - 1.0
        if len(close) > 20:
            c_last = float(close.iloc[-1]) if hasattr(close.iloc[-1], 'item') else float(close.iloc[-1])
            c_20 = float(close.iloc[-21]) if hasattr(close.iloc[-21], 'item') else float(close.iloc[-21])
            ret_20 = (c_last / c_20) - 1.0
    except Exception as e:
        logging.warning(f"Error computing price returns in heuristic: {e}")

    trend_score = (0.7 * ret_5) + (0.3 * ret_20)

    rsi_adj = 0.0
    if "rsi" in data.columns:
        try:
            rsi_val = data["rsi"].iloc[-1]
            rsi = float(rsi_val.item()) if hasattr(rsi_val, 'item') else float(rsi_val)
            if rsi < 35:
                rsi_adj = 0.15
            elif rsi > 65:
                rsi_adj = -0.15
        except Exception as e:
            logging.warning(f"Error extracting RSI in heuristic: {e}")

    macd_adj = 0.0
    if "macd" in data.columns:
        try:
            macd_raw = data["macd"].iloc[-1]
            macd_val = float(macd_raw.item()) if hasattr(macd_raw, 'item') else float(macd_raw)
            macd_adj = np.clip(macd_val / 10.0, -0.2, 0.2)
        except Exception as e:
            logging.warning(f"Error extracting MACD in heuristic: {e}")

    bullish_logit = (trend_score * 8.0) + rsi_adj + macd_adj
    bearish_logit = (-trend_score * 8.0) - rsi_adj - macd_adj
    sideways_logit = 0.35 - (abs(trend_score) * 6.0)

    probs = _softmax([bullish_logit, bearish_logit, sideways_logit])
    return {
        "bullish": float(np.array(probs[0]).item()),
        "bearish": float(np.array(probs[1]).item()),
        "sideways": float(np.array(probs[2]).item())
    }

def predict_stock(ticker, data=None):
    """
    Predict stock direction. If data is provided, use it directly.
    Otherwise, fetch fresh data (for backward compatibility).
    """
    if data is None:
        logging.info(f"Fetching fresh data for prediction: {ticker}")
        data = get_stock_data(ticker)
    else:
        used_ticker = data.attrs.get('used_ticker', ticker)
        logging.info(f"Using pre-fetched data for prediction: {used_ticker}")

    if data is None or data.empty:
        logging.warning(f"No data available for ticker: {ticker}")
        return {"bullish": 0.0, "bearish": 0.0, "sideways": 0.0}

    # Keep a reference to original data for fallback
    original_data = data.copy()

    nifty = get_nifty_index()

    try:
        data = add_indicators(data, nifty)
    except Exception as e:
        logging.warning(f"Indicator generation failed for {ticker}: {e}")
        # Keep raw data path for heuristic fallback

    if len(data) == 0:
        logging.warning(f"No usable rows after preprocessing for {ticker}")
        return _heuristic_prediction(original_data)

    # Prioritize model predictions if data is sufficient and models are available
    if len(data) >= 120 and models_ready and feature_selection is not None:
        try:
            logging.info(f"Using trained models for {ticker}")

            # Apply feature selection - only use selected features
            selected_indices = feature_selection['selected_indices']
            features = data.values[:, selected_indices]  # Select only the chosen features

            scaled = scaler.transform(features)
            seq = scaled[-120:]
            X = np.array(seq).reshape((1, 120, len(selected_indices)))

            t_probs = transformer.predict(X, verbose=0)[0]
            x_input = X.reshape(1, -1)
            x_probs = xgb.predict_proba(x_input)[0]
            probs = (t_probs + x_probs) / 2

            return {
                "bullish": float(probs[0].item()),
                "bearish": float(probs[1].item()),
                "sideways": float(probs[2].item())
            }
        except Exception as e:
            logging.error(f"Model inference failed for ticker {ticker}: {e}")
            logging.info(f"Falling back to heuristic prediction for {ticker}")
            return _heuristic_prediction(data)
    else:
        if len(data) < 120:
            logging.info(f"Insufficient data ({len(data)} rows) for model, using heuristic for {ticker}")
        elif not models_ready:
            logging.info(f"Models not available, using heuristic prediction for {ticker}")
        elif feature_selection is None:
            logging.info(f"Feature selection not available, using heuristic prediction for {ticker}")
        return _heuristic_prediction(data)