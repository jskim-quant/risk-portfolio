import yfinance as yf, pandas as pd
from common import save

TICKERS = ["SPY","VGK","EWJ","SHY","IEF","TLT","LQD","HYG","GLD","DBC"]
for t in TICKERS:
    df = yf.download(t, start="1990-01-01", auto_adjust=False,
                     progress=False, multi_level_index=False)
    df.index.name = "date"
    df = df[["Open","High","Low","Close","Adj Close","Volume"]]
    df.columns = [f"{t}_{c.replace(' ', '')}" for c in df.columns]
    save(df[[f"{t}_AdjClose", f"{t}_Close"]].join(df.drop(columns=[f"{t}_AdjClose", f"{t}_Close"])),
         f"market/etf/{t}.csv", "Yahoo Finance", f"https://finance.yahoo.com/quote/{t}",
         "USD", "AdjClose = split+dividend adjusted")
