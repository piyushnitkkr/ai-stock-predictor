import yfinance as yf


sectors = {
    "Banking": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY",
    "Pharma": "^CNXPHARMA"
}


def sector_strength():

    scores = {}

    for name, index in sectors.items():

        try:

            data = yf.download(index, period="3mo", progress=False)

            if data.empty:
                continue

            change = (
                data["Close"].iloc[-1] -
                data["Close"].iloc[0]
            ) / data["Close"].iloc[0]

            scores[name] = float(change.item())

        except:
            continue

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )