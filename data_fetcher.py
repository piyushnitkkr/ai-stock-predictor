import yfinance as yf
import logging
import pandas as pd
import warnings
import time
from typing import Optional

# Suppress yfinance download warnings
warnings.filterwarnings('ignore', message='.*download.*')

logging.basicConfig(level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s")

# Rate limiting configuration
REQUEST_DELAY = 0.5  # Minimum delay between requests (seconds)
RETRY_ATTEMPTS = 3
INITIAL_BACKOFF = 1  # Initial backoff time (seconds)
MAX_BACKOFF = 30  # Maximum backoff time (seconds)

# Track last request time for rate limiting
_last_request_time = 0


TICKER_ALIASES = {
    "HDFC.NS": "HDFCBANK.NS",
    "HDFC.BO": "HDFCBANK.BO",
}


def _enforce_request_throttle():
    """Enforce minimum delay between requests to avoid rate limiting."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - elapsed
        logging.debug(f"Throttling: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    _last_request_time = time.time()


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

        backoff = INITIAL_BACKOFF
        attempt = 0

        while attempt < RETRY_ATTEMPTS:
            attempt += 1

            try:
                # Enforce throttling to prevent rate limits
                _enforce_request_throttle()

                logging.debug(f"Fetching data for ticker: {symbol} (attempt {attempt}/{RETRY_ATTEMPTS})")

                # Suppress yfinance verbose logging
                yf_logger = logging.getLogger('yfinance')
                yf_logger.setLevel(logging.ERROR)

                data = yf.download(
                    symbol,
                    period="10y",
                    interval="1d",
                    progress=False,
                    auto_adjust=True
                )
                
                yf_logger.setLevel(logging.INFO)

                if data is None or data.empty:
                    logging.debug(f"No data returned for {symbol}")
                    break  # No data available, try next symbol

                cleaned = data.dropna()

                if cleaned.empty:
                    logging.debug(f"All data is NaN for {symbol}")
                    break

                if not _valid_price(cleaned):
                    logging.debug(f"Rejected ticker due to invalid price: {symbol}")
                    break

                cleaned.attrs["used_ticker"] = symbol
                return cleaned

            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_str or "too many" in error_str or "429" in error_str:
                    if attempt < RETRY_ATTEMPTS:
                        logging.warning(f"Rate limited for {symbol}, waiting {backoff}s before retry...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF)  # Exponential backoff
                        continue
                    else:
                        logging.error(f"[{symbol}]: {type(e).__name__}('{e}')")
                        break
                else:
                    # Other errors: log and move to next symbol
                    logging.debug(f"Error fetching {symbol}: {e}")
                    break

    logging.warning(f"No valid data found for ticker: {ticker}")
    return None


def get_nifty_index():

    backoff = INITIAL_BACKOFF
    attempt = 0

    while attempt < RETRY_ATTEMPTS:
        attempt += 1

        try:
            # Enforce throttling to prevent rate limits
            _enforce_request_throttle()

            yf_logger = logging.getLogger('yfinance')
            yf_logger.setLevel(logging.ERROR)

            data = yf.download(
                "^NSEI",
                period="5y",
                interval="1d",
                progress=False,
                auto_adjust=True
            )
            
            yf_logger.setLevel(logging.INFO)

            if data is None or data.empty:
                return pd.Series(dtype=float)

            return data["Close"].dropna()

        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate" in error_str or "too many" in error_str or "429" in error_str:
                if attempt < RETRY_ATTEMPTS:
                    logging.warning(f"Rate limited fetching NIFTY index, waiting {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                else:
                    logging.error(f"Failed to fetch NIFTY index after {RETRY_ATTEMPTS} attempts: {e}")
                    return pd.Series(dtype=float)
            else:
                logging.debug(f"Failed to fetch NIFTY index: {e}")
                return pd.Series(dtype=float)

    return pd.Series(dtype=float)