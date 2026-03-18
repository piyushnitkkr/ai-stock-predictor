import yfinance as yf
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s")


TICKER_ALIASES = {
    "HDFC.NS": "HDFCBANK.NS",
    "HDFC.BO": "HDFCBANK.BO",
}


def _candidate_tickers(ticker):

    base = (ticker or "").strip().upper()

    if base == "":
        return []

    candidates = []

    # direct alias
    if base in TICKER_ALIASES:
        candidates.append(TICKER_ALIASES[base])

    if "." in base:
        # Has explicit exchange suffix — use as-is
        candidates.append(base)
    else:
        # No suffix: go straight to NSE then BSE to avoid matching wrong US tickers
        candidates.append(base + ".NS")
        candidates.append(base + ".BO")

    # remove duplicates while preserving order
    seen = set()
    ordered = []

    for sym in candidates:
        if sym not in seen:
            seen.add(sym)
            ordered.append(sym)

    return ordered


def _valid_price(data):

    """Validate price range to detect incorrect symbols."""

    try:

        last_price = float(data["Close"].iloc[-1])

        # reject obviously wrong price ranges
        if last_price <= 0:
            return False

        if last_price > 200000:  # unrealistic for Indian stocks
            return False

        return True

    except:
        return False


def get_stock_data(ticker):

    symbols = _candidate_tickers(ticker)

    for symbol in symbols:

        try:

            logging.info(f"Fetching data for ticker: {symbol}")

            data = yf.download(
                symbol,
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if data is None or data.empty:
                continue

            cleaned = data.dropna()

            if cleaned.empty:
                continue

            if not _valid_price(cleaned):
                logging.warning(f"Rejected ticker due to invalid price: {symbol}")
                continue

            cleaned.attrs["used_ticker"] = symbol

            return cleaned

        except Exception as e:

            logging.warning(f"Error fetching {symbol}: {e}")
            continue

    logging.error(f"No valid data found for input ticker: {ticker}")

    return None


def get_nifty_index():

    try:

        data = yf.download(
            "^NSEI",
            period="5y",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if data is None or data.empty:
            return pd.Series(dtype=float)

        return data["Close"].dropna()

    except Exception as e:

        logging.warning(f"Failed to fetch NIFTY index: {e}")

        return pd.Series(dtype=float)