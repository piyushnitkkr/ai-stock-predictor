import logging

def detect_breakout(data):
    try:
        recent = data.tail(20)

        high = float(recent["High"].max().item())
        low = float(recent["Low"].min().item())

        last = float(data["Close"].iloc[-1].item())

        if last > high:
            return "Bullish Breakout"

        if last < low:
            return "Bearish Breakdown"

        return "No Breakout"
    except Exception as e:
        logging.error(f"Breakout detection failed: {e}")
        return "Breakout detection unavailable"