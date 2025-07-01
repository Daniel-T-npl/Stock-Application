import pandas as pd
import numpy as np

def ema(series, span):
    """Exponential Moving Average (EMA)"""
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    """MACD indicator: returns MACD line, Signal line, and Histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': hist
    })

def stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['low'].rolling(window=k_window, min_periods=1).min()
    high_max = df['high'].rolling(window=k_window, min_periods=1).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window, min_periods=1).mean()
    return pd.DataFrame({'stoch_k': k, 'stoch_d': d})

def donchian_channel(df, window=20):
    upper = df['high'].rolling(window=window, min_periods=1).max()
    lower = df['low'].rolling(window=window, min_periods=1).min()
    return pd.DataFrame({'donchian_upper': upper, 'donchian_lower': lower})

def anchored_vwap(df, anchor_idx=0):
    price = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']
    cum_vol = volume.cumsum().shift(anchor_idx).fillna(0) + volume.iloc[anchor_idx:].cumsum()
    cum_pv = (price * volume).cumsum().shift(anchor_idx).fillna(0) + (price.iloc[anchor_idx:] * volume.iloc[anchor_idx:]).cumsum()
    vwap = cum_pv / cum_vol
    return pd.Series(vwap, name='anchored_vwap')

def volume_profile(df, bins=12):
    # Basic version: bin prices and sum volume in each bin
    price_bins = np.linspace(df['low'].min(), df['high'].max(), bins+1)
    df['price_bin'] = pd.cut(df['close'], bins=price_bins, include_lowest=True)
    profile = df.groupby('price_bin')['volume'].sum().reset_index()
    return profile

def ichimoku_cloud(df):
    # Placeholder stub
    return pd.DataFrame()

def fibonacci_retracement(df):
    # Placeholder stub
    return pd.DataFrame() 