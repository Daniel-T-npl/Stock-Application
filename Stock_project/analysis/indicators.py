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
    df['price_bin'] = df['price_bin'].apply(lambda x: str(x) if not pd.isnull(x) else "")
    profile = df.groupby('price_bin')['volume'].sum().reset_index()
    return profile

def ichimoku_cloud(df):
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    chikou_span = df['close'].shift(-26)
    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })

def fibonacci_retracement(df):
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = {f'fibonacci_{int(l*100)}': max_price - l * diff for l in levels}
    # Return a DataFrame with each level as a column, repeated for each row
    fib_df = pd.DataFrame({k: [v]*len(df) for k, v in fib_levels.items()})
    return fib_df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi 