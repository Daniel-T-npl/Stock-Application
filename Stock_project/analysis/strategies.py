import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stocksite.settings")

import pandas as pd
import numpy as np
from analysis.stock_service import StockService
from analysis import indicators
from analysis.influx_client import get_all_symbols

# --- Strategy Implementations ---
def compute_indicators(df):
    # Add all required indicators to df
    df = df.copy()
    df['ema_20'] = indicators.ema(df['close'], 20)
    macd_df = indicators.macd(df['close'])
    df = pd.concat([df, macd_df], axis=1)
    df['adx'] = indicators.adx(df)
    df['obv'] = indicators.obv(df)
    # Always use manual calculation for Bollinger Bands
    ma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = ma + 2*std
    df['bb_lower'] = ma - 2*std
    df['bb_mid'] = ma
    df['rsi'] = indicators.rsi(df['close'])
    df['cmf'] = indicators.chaikin_money_flow(df)
    kc = indicators.keltner_channels(df)
    df = pd.concat([df, kc], axis=1)
    donchian = indicators.donchian_channel(df)
    df = pd.concat([df, donchian], axis=1)
    df['vpt'] = indicators.vpt(df)
    ichimoku = indicators.ichimoku_cloud(df)
    df = pd.concat([df, ichimoku], axis=1)
    fractals = indicators.fractals(df)
    df = pd.concat([df, fractals], axis=1)
    df['parabolic_sar'] = indicators.parabolic_sar(df)
    stoch = indicators.stochastic_oscillator(df)
    df = pd.concat([df, stoch], axis=1)
    df['anchored_vwap'] = indicators.anchored_vwap(df)
    df['atr'] = indicators.atr(df)
    return df

def format_date(dt):
    return dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]

# --- Trend Following Strategy ---
def trend_following_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        buy = (
            df['close'].iloc[i] > df['ema_20'].iloc[i]
            and df['adx'].iloc[i] > 25
            and df['macd'].iloc[i] > df['macd_signal'].iloc[i]
            and df['obv'].iloc[i] > df['obv'].iloc[i-1]
        )
        sell = (
            df['close'].iloc[i] < df['ema_20'].iloc[i]
            or df['macd'].iloc[i] < df['macd_signal'].iloc[i]
            or df['adx'].iloc[i] < 20
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

def backtest_strategy(df, signals):
    trades = []
    position = None
    entry_price = 0
    profit = 0
    for sig in signals:
        if sig['signal'] == 'Buy' and position is None:
            position = 'long'
            entry_price = sig['price']
            trades.append({'date': sig['date'], 'price': sig['price'], 'action': 'Buy'})
        elif sig['signal'] == 'Sell' and position == 'long':
            trade_profit = sig['price'] - entry_price
            profit += trade_profit
            trades.append({'date': sig['date'], 'price': sig['price'], 'action': 'Sell', 'trade_profit': trade_profit, 'total_profit': profit})
            position = None
    return trades, profit

def run_trend_following_on_nlicl():
    # Fetch data
    service = StockService()
    df = service.get_stock_data_df('NLICL')
    print(f"Loaded NLICL data shape: {df.shape}")
    if df.empty:
        print('No data for NLICL')
        return
    df = compute_indicators(df)
    print(f"After indicators, data shape: {df.shape}")
    signals = trend_following_signals(df)
    print(f"Number of signals generated: {len(signals)}")
    trades, profit = backtest_strategy(df, signals)
    print(f"Number of trades: {len(trades)}")
    if not trades:
        print("No trades generated. Check indicator logic or data.")
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '../data/trend_following_NLICL.csv')
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Total profit: {profit}")
        print(f"Trade log saved to {out_path}")

# --- Mean Reversion Strategy: Bollinger Bounce ---
def mean_reversion_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        buy = (
            df['close'].iloc[i] < df['bb_lower'].iloc[i]
            and df['rsi'].iloc[i] < 30
            and df['cmf'].iloc[i] > 0
        )
        sell = (
            df['close'].iloc[i] > df['bb_mid'].iloc[i]
            or df['rsi'].iloc[i] > 60
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

def run_mean_reversion_on_nlicl():
    service = StockService()
    df = service.get_stock_data_df('NLICL')
    print(f"Loaded NLICL data shape: {df.shape}")
    if df.empty:
        print('No data for NLICL')
        return
    df = compute_indicators(df)
    print(f"After indicators, data shape: {df.shape}")
    signals = mean_reversion_signals(df)
    print(f"Number of signals generated: {len(signals)}")
    trades, profit = backtest_strategy(df, signals)
    print(f"Number of trades: {len(trades)}")
    if not trades:
        print("No trades generated. Check indicator logic or data.")
    out_path = os.path.join(os.path.dirname(__file__), '../data/mean_reversion_NLICL.csv')
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Total profit: {profit}")
        print(f"Trade log saved to {out_path}")

# --- Breakout Strategy: Volatility Squeeze ---
def breakout_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        bb_width = df['bb_upper'].iloc[i] - df['bb_lower'].iloc[i]
        kc_width = df['kc_upper'].iloc[i] - df['kc_lower'].iloc[i]
        vpt_increasing = df['vpt'].iloc[i] > df['vpt'].iloc[i-1]
        vpt_decreasing = df['vpt'].iloc[i] < df['vpt'].iloc[i-1]
        buy = (
            bb_width < kc_width
            and df['close'].iloc[i] > df['donchian_upper'].iloc[i]
            and vpt_increasing
        )
        sell = (
            df['close'].iloc[i] < df['donchian_lower'].iloc[i]
            or vpt_decreasing
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

def run_breakout_on_nlicl():
    service = StockService()
    df = service.get_stock_data_df('NLICL')
    print(f"Loaded NLICL data shape: {df.shape}")
    if df.empty:
        print('No data for NLICL')
        return
    df = compute_indicators(df)
    print(f"After indicators, data shape: {df.shape}")
    signals = breakout_signals(df)
    print(f"Number of signals generated: {len(signals)}")
    trades, profit = backtest_strategy(df, signals)
    print(f"Number of trades: {len(trades)}")
    if not trades:
        print("No trades generated. Check indicator logic or data.")
    out_path = os.path.join(os.path.dirname(__file__), '../data/breakout_NLICL.csv')
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Total profit: {profit}")
        print(f"Trade log saved to {out_path}")

# --- Reversal Strategy: Cloud + Fractals ---
def reversal_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        # Ichimoku cloud top/bottom
        cloud_top = max(df['senkou_span_a'].iloc[i], df['senkou_span_b'].iloc[i])
        cloud_bottom = min(df['senkou_span_a'].iloc[i], df['senkou_span_b'].iloc[i])
        is_local_low = pd.notnull(df['fractal_low'].iloc[i])
        is_local_high = pd.notnull(df['fractal_high'].iloc[i])
        parabolic_below = df['parabolic_sar'].iloc[i] < df['close'].iloc[i]
        parabolic_above = df['parabolic_sar'].iloc[i] > df['close'].iloc[i]
        buy = (
            df['close'].iloc[i] > cloud_top
            and is_local_low
            and parabolic_below
        )
        sell = (
            df['close'].iloc[i] < cloud_bottom
            or parabolic_above
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

def run_reversal_on_nlicl():
    service = StockService()
    df = service.get_stock_data_df('NLICL')
    print(f"Loaded NLICL data shape: {df.shape}")
    if df.empty:
        print('No data for NLICL')
        return
    df = compute_indicators(df)
    print(f"After indicators, data shape: {df.shape}")
    signals = reversal_signals(df)
    print(f"Number of signals generated: {len(signals)}")
    trades, profit = backtest_strategy(df, signals)
    print(f"Number of trades: {len(trades)}")
    if not trades:
        print("No trades generated. Check indicator logic or data.")
    out_path = os.path.join(os.path.dirname(__file__), '../data/reversal_NLICL.csv')
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Total profit: {profit}")
        print(f"Trade log saved to {out_path}")

# --- Momentum Divergence Strategy: Volume Disagreement ---
def divergence_signals(df):
    signals = []
    position = None
    for i in range(2, len(df)):
        macd_decreasing = df['macd'].iloc[i] < df['macd'].iloc[i-1] < df['macd'].iloc[i-2]
        obv_decreasing = df['obv'].iloc[i] < df['obv'].iloc[i-1] < df['obv'].iloc[i-2]
        adx_weak = df['adx'].iloc[i] < 20
        macd_increasing = df['macd'].iloc[i] > df['macd'].iloc[i-1] > df['macd'].iloc[i-2]
        obv_increasing = df['obv'].iloc[i] > df['obv'].iloc[i-1] > df['obv'].iloc[i-2]
        price_falling = df['close'].iloc[i] < df['close'].iloc[i-1] < df['close'].iloc[i-2]
        price_rising = df['close'].iloc[i] > df['close'].iloc[i-1] > df['close'].iloc[i-2]
        # Sell signal (bearish divergence)
        sell = macd_decreasing and obv_decreasing and adx_weak
        # Buy signal (reverse logic)
        buy = price_falling and macd_increasing and obv_increasing and adx_weak
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

def run_divergence_on_nlicl():
    service = StockService()
    df = service.get_stock_data_df('NLICL')
    print(f"Loaded NLICL data shape: {df.shape}")
    if df.empty:
        print('No data for NLICL')
        return
    df = compute_indicators(df)
    print(f"After indicators, data shape: {df.shape}")
    signals = divergence_signals(df)
    print(f"Number of signals generated: {len(signals)}")
    trades, profit = backtest_strategy(df, signals)
    print(f"Number of trades: {len(trades)}")
    if not trades:
        print("No trades generated. Check indicator logic or data.")
    out_path = os.path.join(os.path.dirname(__file__), '../data/divergence_NLICL.csv')
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Total profit: {profit}")
        print(f"Trade log saved to {out_path}")

# --- 1. Momentum Breakout Strategy ---
def momentum_breakout_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        obv_increasing = df['obv'].iloc[i] > df['obv'].iloc[i-1]
        buy = (
            df['close'].iloc[i] > df['donchian_upper'].iloc[i]
            and df['adx'].iloc[i] > 25
            and obv_increasing
        )
        sell = (
            df['close'].iloc[i] < df['donchian_lower'].iloc[i]
            or df['adx'].iloc[i] < 20
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- 2. VWAP Reversion Strategy ---
def vwap_reversion_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        # Detect parabolic SAR flip: from above to below price
        parabolic_flipped_bullish = (
            df['parabolic_sar'].iloc[i-1] > df['close'].iloc[i-1]
            and df['parabolic_sar'].iloc[i] < df['close'].iloc[i]
        )
        buy = (
            df['close'].iloc[i] < df['anchored_vwap'].iloc[i]
            and df['stoch_k'].iloc[i] < 20
            and parabolic_flipped_bullish
        )
        sell = (
            df['close'].iloc[i] > df['anchored_vwap'].iloc[i]
            and df['stoch_k'].iloc[i] > 80
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- 3. Smart Money Divergence Strategy ---
def smart_money_divergence_signals(df):
    signals = []
    position = None
    for i in range(2, len(df)):
        # Bearish divergence: price higher highs, RSI lower highs, CMF decreasing
        price_higher_highs = df['close'].iloc[i] > df['close'].iloc[i-1] > df['close'].iloc[i-2]
        rsi_lower_highs = df['rsi'].iloc[i] < df['rsi'].iloc[i-1] < df['rsi'].iloc[i-2]
        cmf_decreasing = df['cmf'].iloc[i] < df['cmf'].iloc[i-1]
        obv_div = df['obv'].iloc[i] < df['obv'].iloc[i-1]
        sell = price_higher_highs and rsi_lower_highs and cmf_decreasing and obv_div
        # Bullish divergence: price lower lows, RSI higher lows, CMF increasing
        price_lower_lows = df['close'].iloc[i] < df['close'].iloc[i-1] < df['close'].iloc[i-2]
        rsi_higher_lows = df['rsi'].iloc[i] > df['rsi'].iloc[i-1] > df['rsi'].iloc[i-2]
        cmf_increasing = df['cmf'].iloc[i] > df['cmf'].iloc[i-1]
        obv_div_bull = df['obv'].iloc[i] > df['obv'].iloc[i-1]
        buy = price_lower_lows and rsi_higher_lows and cmf_increasing and obv_div_bull
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- 4. Volatility Expansion Fade ---
def volatility_expansion_fade_signals(df):
    signals = []
    position = None
    for i in range(2, len(df)):
        atr_spike = df['atr'].iloc[i] > 1.5 * df['atr'].iloc[i-1]
        price_above_bb = df['close'].iloc[i] > df['bb_upper'].iloc[i]
        macd_hist_decreasing = df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
        sell = atr_spike and price_above_bb and macd_hist_decreasing
        if sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- 5. Trend-Pullback Strategy ---
def trend_pullback_signals(df):
    signals = []
    position = None
    for i in range(2, len(df)):
        ema_increasing = df['ema_20'].iloc[i] > df['ema_20'].iloc[i-1]
        rsi_pullback = 30 < df['rsi'].iloc[i] < 50
        fractal_low = pd.notnull(df['fractal_low'].iloc[i])
        buy = ema_increasing and rsi_pullback and fractal_low
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        # Sell: exit on RSI > 60 or EMA flattening
        sell = df['rsi'].iloc[i] > 60 or df['ema_20'].iloc[i] < df['ema_20'].iloc[i-1]
        if sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- 6. Cloud Crossover Strategy (Ichimoku Classic) ---
def cloud_crossover_signals(df):
    signals = []
    position = None
    for i in range(26, len(df)):
        price = df['close'].iloc[i]
        cloud_top = max(df['senkou_span_a'].iloc[i], df['senkou_span_b'].iloc[i])
        tenkan = df['tenkan_sen'].iloc[i]
        kijun = df['kijun_sen'].iloc[i]
        chikou = df['chikou_span'].iloc[i]
        price_n26 = df['close'].iloc[i-26] if i-26 >= 0 else np.nan
        buy = (
            price > cloud_top
            and tenkan > kijun
            and chikou > price_n26
        )
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': price, 'signal': 'Buy'})
            position = 'long'
        # Sell: price < cloud or tenkan < kijun
        sell = (price < cloud_top or tenkan < kijun)
        if sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': price, 'signal': 'Sell'})
            position = None
    return signals

# --- 7. Composite Score Strategy ---
def composite_score_signals(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        score = 0
        if df['rsi'].iloc[i] < 30:
            score += 1
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
            score += 1
        if df['close'].iloc[i] > df['ema_20'].iloc[i]:
            score += 1
        if df['adx'].iloc[i] > 25:
            score += 1
        if df['cmf'].iloc[i] > 0:
            score += 1
        if df['parabolic_sar'].iloc[i] < df['close'].iloc[i]:
            score += 1
        buy = score >= 4
        sell = score <= 2
        if buy and position != 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Buy'})
            position = 'long'
        elif sell and position == 'long':
            signals.append({'date': format_date(df.index[i]), 'price': df['close'].iloc[i], 'signal': 'Sell'})
            position = None
    return signals

# --- Batch Backtest for All Stocks ---
def batch_backtest_strategy(strategy_name, signal_func):
    service = StockService()
    symbols = get_all_symbols()
    print(f"Found {len(symbols)} symbols.")
    all_trades = []
    profit_list = []
    trade_count_list = []
    for symbol in symbols:
        df = service.get_stock_data_df(symbol)
        if df.empty:
            print(f"No data for {symbol}")
            continue
        df = compute_indicators(df)
        signals = signal_func(df)
        trades, profit = backtest_strategy(df, signals)
        profit_list.append(profit)
        trade_count_list.append(len(trades))
        for trade in trades:
            trade_row = trade.copy()
            trade_row['symbol'] = symbol
            all_trades.append(trade_row)
        # Add per-stock summary row (always include total_profit)
        summary_row = {
            'symbol': symbol,
            'date': '',
            'price': '',
            'action': 'STOCK_PROFIT',
            'trade_profit': '',
            'total_profit': str(profit)
        }
        all_trades.append(summary_row)
        print(f"{symbol}: profit={profit}, trades={len(trades)}")
    # Ensure all columns are present
    cols = ['symbol', 'date', 'price', 'action', 'trade_profit', 'total_profit']
    df_out = pd.DataFrame(all_trades)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ''
    df_out = df_out[cols]
    # Append summary rows
    avg_profit = sum(profit_list) / len(profit_list) if profit_list else 0
    avg_trades = sum(trade_count_list) / len(trade_count_list) if trade_count_list else 0
    summary = {k: '' for k in df_out.columns}
    summary['symbol'] = 'AVERAGE'
    summary['total_profit'] = str(avg_profit)
    df_out = pd.concat([df_out, pd.DataFrame([summary])], ignore_index=True)
    summary2 = summary.copy()
    summary2['symbol'] = 'AVG_TRADES'
    summary2['total_profit'] = str(avg_trades)
    df_out = pd.concat([df_out, pd.DataFrame([summary2])], ignore_index=True)
    out_path = os.path.join(os.path.dirname(__file__), f'../data/{strategy_name}_all.csv')
    df_out.to_csv(out_path, index=False)
    print(f"Saved {strategy_name} results for all stocks to {out_path}")

# --- Main ---
if __name__ == '__main__':
    print("Batch backtest for all stocks...")
    batch_backtest_strategy('trend_following', trend_following_signals)
    batch_backtest_strategy('mean_reversion', mean_reversion_signals)
    batch_backtest_strategy('breakout', breakout_signals)
    batch_backtest_strategy('reversal', reversal_signals)
    batch_backtest_strategy('divergence', divergence_signals)
    batch_backtest_strategy('momentum_breakout', momentum_breakout_signals)
    batch_backtest_strategy('vwap_reversion', vwap_reversion_signals)
    batch_backtest_strategy('smart_money_divergence', smart_money_divergence_signals)
    batch_backtest_strategy('volatility_expansion_fade', volatility_expansion_fade_signals)
    batch_backtest_strategy('trend_pullback', trend_pullback_signals)
    batch_backtest_strategy('cloud_crossover', cloud_crossover_signals)
    batch_backtest_strategy('composite_score', composite_score_signals) 