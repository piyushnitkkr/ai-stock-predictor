import ta
import numpy as np


def add_indicators(df, nifty):

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # ── Core indicators (proven to work) ──────────────────────────────────────
    df["rsi"]       = ta.momentum.RSIIndicator(close).rsi()
    df["macd"]      = ta.trend.MACD(close).macd()
    df["sma20"]     = close.rolling(20).mean()
    df["sma50"]     = close.rolling(50).mean()
    df["ema20"]     = ta.trend.EMAIndicator(close, 20).ema_indicator()
    df["momentum"]  = ta.momentum.ROCIndicator(close).roc()
    df["volume_ma"] = volume.rolling(20).mean()
    df["atr"]       = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["adx"]       = ta.trend.ADXIndicator(high, low, close).adx()
    df["stoch"]     = ta.momentum.StochasticOscillator(high, low, close).stoch()

    # ── Select high-value new indicators ──────────────────────────────────────
    # Bollinger %B - where price sits in the band (strong signal)
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_pct"] = bb.bollinger_pband()

    # Price returns - scale-invariant signals
    df["ret_5"]  = close.pct_change(5,  fill_method=None)
    df["ret_20"] = close.pct_change(20, fill_method=None)

    # Volume surge detection
    df["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    # ── Nifty index ───────────────────────────────────────────────────────────
    if nifty is None or getattr(nifty, "empty", True):
        df["nifty"] = close
    else:
        aligned = nifty.reindex(df.index).ffill().bfill()
        df["nifty"] = aligned.fillna(close)

    df["nifty_return"]  = df["nifty"].pct_change(fill_method=None).fillna(0.0)
    df["volume_change"] = volume.pct_change(fill_method=None)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    return df