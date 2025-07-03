import pandas as pd
import numpy as np
import logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

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
    price_min = df['low'].min()
    price_max = df['high'].max()
    if price_min == price_max:
        # All prices are the same, single bin
        total_volume = df['volume'].sum()
        return pd.DataFrame({'price_bin': [f'[{price_min},{price_max}]'], 'volume': [total_volume]})
    # If price range is very small, reduce bins
    if price_max - price_min < 1e-6:
        bins = 1
    price_bins = np.linspace(price_min, price_max, bins+1)
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

def adx(df, window=14):
    """Average Directional Index (ADX)"""
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(window=window, min_periods=window).mean()
    return pd.Series(adx_val, index=df.index, name='adx')

def parabolic_sar(df, step=0.02, max_step=0.2):
    """Parabolic SAR"""
    high = df['high']
    low = df['low']
    close = df['close']
    sar = [close.iloc[0]]
    ep = low.iloc[0]
    af = step
    uptrend = True
    for i in range(1, len(df)):
        prev_sar = sar[-1]
        if uptrend:
            sar_new = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar_new:
                uptrend = False
                sar_new = ep
                ep = high.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar_new = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar_new:
                uptrend = True
                sar_new = ep
                ep = low.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
        sar.append(sar_new)
    return pd.Series(sar, index=df.index, name='parabolic_sar')

def obv(df):
    """On-Balance Volume (OBV)"""
    close = df['close']
    volume = df['volume']
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index, name='obv')

def vpt(df):
    """Volume Price Trend (VPT)"""
    close = df['close']
    volume = df['volume']
    vpt = [0]
    for i in range(1, len(df)):
        pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] if close.iloc[i-1] != 0 else 0
        vpt.append(vpt[-1] + volume.iloc[i] * pct)
    return pd.Series(vpt, index=df.index, name='vpt')

def chaikin_money_flow(df, window=20):
    """Chaikin Money Flow (CMF)"""
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low + 1e-9) * volume
    cmf = mfv.rolling(window=window, min_periods=window).sum() / volume.rolling(window=window, min_periods=window).sum()
    return cmf.rename('cmf')

def atr(df, window=14):
    """Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=window, min_periods=window).mean()
    return pd.Series(atr_val, index=df.index, name='atr')

def keltner_channels(df, window=20, atr_mult=2):
    """Keltner Channels"""
    close = df['close']
    ema_mid = close.ewm(span=window, adjust=False).mean()
    atr_val = atr(df, window)
    upper = ema_mid + atr_mult * atr_val
    lower = ema_mid - atr_mult * atr_val
    return pd.DataFrame({'kc_middle': ema_mid, 'kc_upper': upper, 'kc_lower': lower})

def fractals(df):
    """Fractals (local highs/lows)"""
    high = df['high']
    low = df['low']
    n = 2
    fractal_high = [None]*len(df)
    fractal_low = [None]*len(df)
    for i in range(n, len(df)-n):
        if high.iloc[i] > max(high.iloc[i-n:i].tolist() + high.iloc[i+1:i+n+1].tolist()):
            fractal_high[i] = high.iloc[i]
        if low.iloc[i] < min(low.iloc[i-n:i].tolist() + low.iloc[i+1:i+n+1].tolist()):
            fractal_low[i] = low.iloc[i]
    return pd.DataFrame({'fractal_high': fractal_high, 'fractal_low': fractal_low}, index=df.index)

# Canonical list of all available indicators for the dashboard
INDICATOR_CHOICES = [
    ("ema_20", "EMA"),
    ("macd", "MACD"),
    ("stoch", "Stochastic Oscillator"),
    ("donchian", "Donchian Channel"),
    ("anchored_vwap", "Anchored VWAP"),
    ("bollinger", "Bollinger Bands"),
    ("rsi", "RSI"),
    ("ichimoku", "Ichimoku Cloud"),
    ("fibonacci", "Fibonacci Retracement"),
    ("volume_profile", "Volume Profile"),
    ("adx", "ADX (Average Directional Index)"),
    ("parabolic_sar", "Parabolic SAR"),
    ("obv", "On-Balance Volume (OBV)"),
    ("vpt", "Volume Price Trend (VPT)"),
    ("cmf", "Chaikin Money Flow (CMF)"),
    ("atr", "ATR (Average True Range)"),
    ("keltner", "Keltner Channels"),
    ("fractals", "Fractals"),
]

# REMOVE any logger.info("API response: %s", ...) 