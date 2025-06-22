# indicators.py
# -*- coding: utf-8 -*-
"""Utility to attach 주요 기술적 지표 using pandas-ta."""
import pandas as pd
import pandas_ta as ta

def attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, RSI, MACD, Bollinger Bands, Volatility."""
    df = df.copy()
    # Moving Averages
    df.ta.sma(length=[20, 50, 200], append=True)
    # RSI
    df.ta.rsi(length=14, append=True)
    # MACD
    df.ta.macd(append=True)
    # Bollinger Bands (length 20, 2σ)
    df.ta.bbands(length=20, std=2, append=True)
    # 20-일 변동성(연율) 추가
    df["VOLATILITY_20d"] = (
        df["Close"].pct_change().rolling(20).std() * (252 ** 0.5) * 100
    )
    # Bollinger Band Position(0~1)
    df["BB_POS"] = (
        (df["Close"] - df["BBL_20_2.0"])
        / (df["BBU_20_2.0"] - df["BBL_20_2.0"])
    )
    return df
