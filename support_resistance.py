import numpy as np


def support_resistance(data):

    # ensure 1D float array
    prices = np.array(data["Close"].tail(200)).flatten().astype(float)

    supports = []
    resistances = []

    for i in range(2, len(prices) - 2):

        if (
            prices[i] < prices[i - 1] and
            prices[i] < prices[i + 1] and
            prices[i] < prices[i - 2] and
            prices[i] < prices[i + 2]
        ):
            supports.append(prices[i])

        if (
            prices[i] > prices[i - 1] and
            prices[i] > prices[i + 1] and
            prices[i] > prices[i - 2] and
            prices[i] > prices[i + 2]
        ):
            resistances.append(prices[i])

    return supports[-3:], resistances[-3:]