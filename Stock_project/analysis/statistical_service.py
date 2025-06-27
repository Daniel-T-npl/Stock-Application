import os
import sys
import django
import warnings
import base64
from io import BytesIO
from typing import Optional
import matplotlib
matplotlib.use('Agg')

# --- Standalone Script Setup ---
# This allows the script to be run from the terminal to test the logic.
# It configures Django to load project settings.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
    django.setup()
except Exception as e:
    print(f"WARNING: Could not set up Django. Standalone script may fail. Error: {e}")
# --- End Setup ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from analysis.stock_service import StockService

warnings.filterwarnings("ignore", category=Warning)

def nan_to_none(obj):
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, (list, np.ndarray)):
        return [nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    return obj

def test_stationarity(series, title="Series"):
    """Test stationarity using ADF test."""
    print(f"\nDEBUG: Testing stationarity for {title}")
    adf_result = adfuller(series.dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    
    # Safely access critical values
    if len(adf_result) > 4:
        print(f"Critical values: {adf_result[4]}")
    else:
        print("Critical values: Not available")
    
    is_stationary = adf_result[1] < 0.05
    print(f"Stationary: {is_stationary}")
    return is_stationary

def analyze_acf_pacf(series, title="Series"):
    """Analyze ACF and PACF patterns to suggest p and q values."""
    print(f"\nDEBUG: Analyzing ACF/PACF for {title}")
    
    # Calculate ACF and PACF values
    from statsmodels.tsa.stattools import acf, pacf
    
    acf_values = acf(series.dropna(), nlags=30, fft=False)
    pacf_values = pacf(series.dropna(), nlags=30)
    
    # Find where ACF and PACF cut off (first lag where |value| < 1.96/sqrt(n))
    n = len(series.dropna())
    threshold = 1.96 / np.sqrt(n)
    
    # Find first lag where ACF becomes insignificant
    acf_cutoff = None
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) < threshold:
            acf_cutoff = i
            break
    
    # Find first lag where PACF becomes insignificant  
    pacf_cutoff = None
    for i in range(1, len(pacf_values)):
        if abs(pacf_values[i]) < threshold:
            pacf_cutoff = i
            break
    
    print(f"DEBUG: ACF cutoff at lag: {acf_cutoff}")
    print(f"DEBUG: PACF cutoff at lag: {pacf_cutoff}")
    print(f"DEBUG: Significance threshold: {threshold:.4f}")
    
    # Suggest p and q based on cutoffs
    suggested_p = pacf_cutoff if pacf_cutoff else 1
    suggested_q = acf_cutoff if acf_cutoff else 1
    
    print(f"DEBUG: Suggested p (from PACF): {suggested_p}")
    print(f"DEBUG: Suggested q (from ACF): {suggested_q}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(series.dropna(), lags=30, ax=axes[0])
    axes[0].set_title(f"ACF for {title}")
    axes[0].axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    axes[0].axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
    
    plot_pacf(series.dropna(), lags=30, ax=axes[1])
    axes[1].set_title(f"PACF for {title}")
    axes[1].axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'acf_pacf_{title.lower().replace(" ", "_")}.png')
    plt.close()
    
    print(f"ACF/PACF plots saved to acf_pacf_{title.lower().replace(' ', '_')}.png")
    
    return suggested_p, suggested_q

def find_best_arima_model(close_series):
    """Find the best ARIMA model using systematic analysis."""
    print("DEBUG: Starting systematic ARIMA model selection...")
    
    # Step 1: Test stationarity of original series
    is_stationary = test_stationarity(close_series, "Close Prices")
    
    # Step 2: If not stationary, test first difference
    if not is_stationary:
        diff1 = close_series.diff().dropna()
        is_diff1_stationary = test_stationarity(diff1, "First Difference")
        
        if is_diff1_stationary:
            d = 1
            working_series = diff1
            print("DEBUG: Using d=1 (first difference)")
        else:
            # Test second difference
            diff2 = close_series.diff().diff().dropna()
            is_diff2_stationary = test_stationarity(diff2, "Second Difference")
            if is_diff2_stationary:
                d = 2
                working_series = diff2
                print("DEBUG: Using d=2 (second difference)")
            else:
                d = 1  # fallback
                working_series = diff1
                print("DEBUG: Fallback to d=1")
    else:
        d = 0
        working_series = close_series
        print("DEBUG: Using d=0 (series is stationary)")
    
    # Step 3: Analyze ACF/PACF for p and q suggestions
    suggested_p, suggested_q = analyze_acf_pacf(working_series, f"Series with d={d}")
    
    # Step 4: Focused grid search around suggested values
    best_aic = float('inf')
    best_order = (1, d, 1)  # default fallback
    best_model = None
    
    print(f"\nDEBUG: Focused grid search with d={d}")
    print(f"DEBUG: ACF/PACF suggested p={suggested_p}, q={suggested_q}")
    print("DEBUG: Testing ARIMA models...")
    
    # Create a focused search around the suggested values
    p_range = range(max(0, suggested_p-1), min(4, suggested_p+2))
    q_range = range(max(0, suggested_q-1), min(4, suggested_q+2))
    
    print(f"DEBUG: Testing p values: {list(p_range)}")
    print(f"DEBUG: Testing q values: {list(q_range)}")
    
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(close_series, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                
                print(f"DEBUG: ARIMA({p},{d},{q}) - AIC: {aic:.2f}")
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = fitted_model
                    print(f"DEBUG: New best model: ARIMA{best_order} with AIC: {best_aic:.2f}")
                    
            except Exception as e:
                print(f"DEBUG: ARIMA({p},{d},{q}) failed: {e}")
                continue
    
    print(f"\nDEBUG: Final selected model: ARIMA{best_order} with AIC: {best_aic:.2f}")
    
    # Print model summary for the best model
    if best_model:
        print(f"\nDEBUG: Best model summary:")
        print(best_model.summary())
    
    return best_order

def generate_moving_average_forecast(close_series, forecast_steps, ma_window=10):
    """Generate forecast using moving average approach."""
    print(f"DEBUG: Generating moving average forecast with window={ma_window}")
    
    # Calculate moving average
    ma_values = close_series.rolling(window=ma_window).mean()
    
    # Get the last few MA values to understand the trend
    last_ma_values = ma_values.dropna().tail(5)
    print(f"DEBUG: Last 5 MA values: {last_ma_values.values}")
    
    # Calculate the trend (slope) from the last few MA values
    if len(last_ma_values) >= 2:
        x = np.arange(len(last_ma_values))
        y = last_ma_values.values
        slope, intercept = np.polyfit(x, y, 1)
        print(f"DEBUG: MA trend slope: {slope:.4f}")
        print(f"DEBUG: MA trend intercept: {intercept:.4f}")
    else:
        slope = 0
        intercept = last_ma_values.iloc[-1] if len(last_ma_values) > 0 else close_series.iloc[-1]
    
    # Generate forecast using the trend
    forecast_values = []
    last_ma = last_ma_values.iloc[-1] if len(last_ma_values) > 0 else close_series.iloc[-1]
    
    for i in range(forecast_steps):
        # Extend the trend
        forecast_value = last_ma + slope * (i + 1)
        forecast_values.append(forecast_value)
    
    # Calculate confidence intervals based on historical volatility
    historical_volatility = close_series.pct_change().std()
    confidence_interval = []
    
    for i, forecast_val in enumerate(forecast_values):
        # Wider confidence intervals for longer forecasts
        multiplier = 1 + (i * 0.1)  # Increase uncertainty over time
        margin = forecast_val * historical_volatility * multiplier * 1.96  # 95% confidence
        confidence_interval.append([forecast_val - margin, forecast_val + margin])
    
    print(f"DEBUG: Moving average forecast values: {forecast_values[:5]}...")
    print(f"DEBUG: Historical volatility: {historical_volatility:.4f}")
    
    return np.array(forecast_values), np.array(confidence_interval)

def generate_forecast_data(symbol: str, model_start_date: str, model_end_date: str, forecast_end_date: str):
    """
    Generates ARIMA and GARCH forecasts and returns the data in a dict.
    """
    try:
        model_start = pd.to_datetime(model_start_date)
        model_end = pd.to_datetime(model_end_date)
        forecast_end = pd.to_datetime(forecast_end_date)
    except (ValueError, TypeError):
        return {"error": "Invalid date format or order."}

    stock_service = StockService()
    df = stock_service.get_stock_data_df(symbol, start=model_start_date, stop=model_end_date)

    if df is None or df.empty or 'close' not in df.columns or len(df) < 10:
        return {"error": f"No sufficient data found for symbol {symbol} to run the model."}

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df[['close']].copy()
    close_series = pd.Series(df['close'])

    if len(close_series) < 10 or close_series.nunique() < 2:
        return {"error": "Not enough unique or sufficient data for forecasting."}

    # 4. Forecasting
    forecast_steps = (forecast_end - model_end).days
    if forecast_steps <= 0:
        return {"error": "Forecast end date must be after the model end date."}

    # Use moving average forecast instead of ARIMA
    print("DEBUG: Using moving average forecast approach...")
    forecast_values, confidence_intervals = generate_moving_average_forecast(close_series, forecast_steps, ma_window=10)
    
    # DEBUG: Print forecast details
    print(f"DEBUG: Moving average forecast steps: {forecast_steps}")
    print(f"DEBUG: Moving average forecast values: {forecast_values}")
    print(f"DEBUG: Moving average confidence intervals: {confidence_intervals}")
    
    forecast_index = pd.date_range(start=model_end + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame({
        'forecast': forecast_values,
        'conf_int_lower': confidence_intervals[:, 0],
        'conf_int_upper': confidence_intervals[:, 1]
    }, index=forecast_index)
    
    # DEBUG: Print the forecast DataFrame
    print(f"DEBUG: Forecast DataFrame head: {forecast_df.head()}")
    print(f"DEBUG: Forecast DataFrame tail: {forecast_df.tail()}")

    # For GARCH, we'll use a simple volatility model based on historical volatility
    historical_volatility = close_series.pct_change().rolling(window=20).std().dropna()
    forecast_volatility = np.full(forecast_steps, historical_volatility.iloc[-1])
    
    # Pad historical volatility for alignment
    historical_vol_array = np.array(historical_volatility.values)
    padded_historical_volatility = np.insert(historical_vol_array, 0, np.nan)
    
    if isinstance(df.index, pd.DatetimeIndex):
      df.index = df.index.tz_localize(None)
    combined_df = pd.concat([df, forecast_df], axis=1)

    garch_dates_list = (pd.to_datetime(df.index).strftime('%Y-%m-%d').tolist() + forecast_index.strftime('%Y-%m-%d').tolist())
    garch_volatility_list = np.concatenate([padded_historical_volatility, forecast_volatility]).tolist()

    # Pad garch_dates_list and garch_volatility_list if needed, but avoid None for datetime conversion
    if len(garch_volatility_list) > len(garch_dates_list):
        # Use the first date minus one day as a dummy date
        first_date = pd.to_datetime(garch_dates_list[0])
        dummy_date = (first_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        garch_dates_list = [dummy_date] + garch_dates_list
        garch_volatility_list = [np.nan] + garch_volatility_list

    # Debug prints to verify lengths and first few elements
    print(f"DEBUG INFO: After padding, Length of garch_dates list: {len(garch_dates_list)}")
    print(f"DEBUG INFO: After padding, Length of garch_volatility list: {len(garch_volatility_list)}")
    print(f"DEBUG INFO: First 5 garch_dates: {garch_dates_list[:5]}")
    print(f"DEBUG INFO: First 5 garch_volatility: {garch_volatility_list[:5]}")

    # Defensive: ensure lists are not None
    valid_garch_dates = [d for d in garch_dates_list if isinstance(d, str)] if garch_dates_list else []
    valid_all_dates = [d for d in pd.to_datetime(combined_df.index).strftime('%Y-%m-%d').tolist() if isinstance(d, str)] if combined_df is not None else []
    garch_volatility_list = garch_volatility_list if garch_volatility_list else []

    result = {
        "dates": valid_all_dates,
        "actual_close": combined_df['close'].tolist() if combined_df is not None else [],
        "forecast": combined_df['forecast'].tolist() if combined_df is not None else [],
        "conf_int_lower": combined_df['conf_int_lower'].tolist() if combined_df is not None else [],
        "conf_int_upper": combined_df['conf_int_upper'].tolist() if combined_df is not None else [],
        "garch_dates": valid_garch_dates,
        "garch_volatility": garch_volatility_list,
        "forecast_start_date": model_end.strftime('%Y-%m-%d')
    }
    return nan_to_none(result)

def generate_forecast_graph(symbol: str, model_start_date: str, model_end_date: str, forecast_end_date: str):
    """
    This function is now deprecated in favor of generating data for D3.
    Kept for compatibility but should be removed later.
    """
    return None 

def flatten_and_filter_str(lst):
    if not lst:
        return []
    result = []
    for item in lst:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten_and_filter_str(item))
        # ignore None and dicts
    return result

def generate_and_save_forecast_image(symbol: str, model_start_date: str, model_end_date: str, forecast_end_date: str, filename: Optional[str] = None):
    """Generates forecast data and saves the plots to an image file with web app dark card theme styling."""
    # Set default output path if not provided
    if filename is None:
        output_dir = os.path.join(os.path.dirname(__file__), '../stocks/static/forecasts')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'forecast_{symbol}.png')
    
    print("Fetching data and generating forecast...")
    data = generate_forecast_data(symbol, model_start_date, model_end_date, forecast_end_date)

    # FIX: Add type check to resolve linter errors and handle error case safely
    if not isinstance(data, dict) or 'error' in data:
        error_message = data.get('error', 'Unknown error') if isinstance(data, dict) else 'Invalid data format'
        print(f"Error generating data: {error_message}")
        return

    print("Data generated successfully. Now creating plot...")
    
    # Use helper to flatten and filter date lists, then filter to only strings
    all_dates = [d for d in flatten_and_filter_str(data.get('dates', [])) if isinstance(d, str)]
    all_dates = [d for d in all_dates if isinstance(d, str)]
    garch_dates = [d for d in flatten_and_filter_str(data.get('garch_dates', [])) if isinstance(d, str)]
    garch_dates = [d for d in garch_dates if isinstance(d, str)]
    garch_volatility = data.get('garch_volatility', [])
    if isinstance(garch_volatility, str):
        garch_volatility = [garch_volatility]
    forecast_start_date = data.get('forecast_start_date', None)
    if not all_dates or not garch_dates or not garch_volatility or not forecast_start_date:
        print("Insufficient data to plot. Skipping plot generation.")
        return
    
    # Final type assertion to guarantee strings for pd.to_datetime
    all_dates = [str(d) for d in all_dates if d is not None]
    garch_dates = [str(d) for d in garch_dates if d is not None]
    forecast_start_date = str(forecast_start_date) if forecast_start_date is not None else None
    
    if not all_dates or not garch_dates or not forecast_start_date:
        print("Insufficient valid date data to plot. Skipping plot generation.")
        return
        
    all_dates_dt = pd.to_datetime(all_dates)
    garch_dates_dt = pd.to_datetime(garch_dates)
    forecast_start_date_dt = pd.to_datetime(forecast_start_date)

    plot_df = pd.DataFrame({
        'actual_close': data.get('actual_close', []),
        'forecast': data.get('forecast', []),
        'lower': data.get('conf_int_lower', []),
        'upper': data.get('conf_int_upper', [])
    }, index=all_dates_dt)

    # Align lengths for garch volatility and dates
    min_len = min(len(garch_volatility), len(garch_dates_dt))
    garch_df = pd.DataFrame({
        'volatility': list(garch_volatility)[-min_len:]
    }, index=garch_dates_dt[-min_len:])

    # --- WEB APP DARK CARD THEME STYLING ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, facecolor='#3b3846')
    fig.patch.set_facecolor('#3b3846')

    # ARIMA/MA Forecast Chart
    ax1.set_facecolor('#3b3846')
    ax1.plot(plot_df.index, plot_df['actual_close'], label='Actual Close', color='#4FC3F7', linewidth=2)
    ax1.plot(plot_df.index, plot_df['forecast'], label='Forecast', color='#FFA726', linestyle='--', linewidth=2)
    ax1.fill_between(plot_df.index, plot_df['lower'], plot_df['upper'], color='#FFA726', alpha=0.18, label='Confidence Interval')
    ax1.axvline(x=forecast_start_date_dt, color='#FF5252', linestyle='--', label='Forecast Start', linewidth=2)
    ax1.set_title(f'Price Forecast for {symbol}', color='white', fontsize=18, pad=15)
    ax1.set_ylabel('Price', color='white', fontsize=14)
    ax1.legend(facecolor='#23212b', edgecolor='white', fontsize=12)
    ax1.grid(True, color='#55536a', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    # GARCH Volatility Chart
    ax2.set_facecolor('#3b3846')
    historical_garch = garch_df[garch_df.index < forecast_start_date_dt]
    forecast_garch = garch_df[garch_df.index >= forecast_start_date_dt]
    ax2.plot(historical_garch.index, historical_garch['volatility'], label='Historical Volatility', color='#00E676', linewidth=2)
    ax2.plot(forecast_garch.index, forecast_garch['volatility'], label='Forecast Volatility', color='#AB47BC', linestyle='--', linewidth=2)
    ax2.axvline(x=forecast_start_date_dt, color='#FF5252', linestyle='--', label='Forecast Start', linewidth=2)
    ax2.set_title(f'Volatility Forecast for {symbol}', color='white', fontsize=18, pad=15)
    ax2.set_ylabel('Volatility', color='white', fontsize=14)
    ax2.set_xlabel('Date', color='white', fontsize=14)
    ax2.legend(facecolor='#23212b', edgecolor='white', fontsize=12)
    ax2.grid(True, color='#55536a', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Forecast graph saved to {filename}")

if __name__ == '__main__':
    symbol_to_test = 'BARUN'
    start = '2023-03-01'
    end = '2025-06-26'
    forecast_end = '2025-07-26'
    output_filename = f'forecast_{symbol_to_test}.png'
    
    print(f"--- Running Standalone Forecast Test for symbol: {symbol_to_test} ---")
    generate_and_save_forecast_image(
        symbol=symbol_to_test,
        model_start_date=start,
        model_end_date=end,
        forecast_end_date=forecast_end,
        filename=output_filename
    )
    print(f"--- Test complete. Check for '{output_filename}' ---") 