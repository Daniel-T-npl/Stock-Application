import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
from .stock_service import StockService
from io import BytesIO
import base64
import os

warnings.filterwarnings("ignore", category=Warning)

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
    close_prices = df['close']
    df['log_return'] = np.log(close_prices / close_prices.shift(1))
    df.dropna(inplace=True)

    if df.empty:
        return {"error": "Not enough data after pre-processing."}

    returns = df['log_return'] * 100

    # 2. ARIMA Model
    arima_model = ARIMA(df['close'], order=(5, 1, 0)) # Example order, can be optimized
    arima_result = arima_model.fit()
    
    # 3. GARCH Model on ARIMA residuals
    residuals = arima_result.resid.dropna()
    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
    garch_result = garch_model.fit(disp='off')

    # 4. Forecasting
    forecast_steps = (forecast_end - model_end).days
    if forecast_steps <= 0:
        return {"error": "Forecast end date must be after the model end date."}

    forecast = arima_result.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=model_end + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame({
        'forecast': forecast.predicted_mean,
        'conf_int_lower': forecast.conf_int().iloc[:, 0],
        'conf_int_upper': forecast.conf_int().iloc[:, 1]
    }, index=forecast_index)

    garch_forecast = garch_result.forecast(horizon=forecast_steps)
    forecast_volatility = np.sqrt(garch_forecast.variance.values.T)
    
    # Combine data for response
    if isinstance(df.index, pd.DatetimeIndex):
      df.index = df.index.tz_localize(None) # Remove timezone for clean merge
    combined_df = pd.concat([df, forecast_df], axis=1)

    return {
        "dates": combined_df.index.strftime('%Y-%m-%d').tolist() if isinstance(combined_df.index, pd.DatetimeIndex) else [],
        "actual_close": combined_df['close'].where(pd.notna(combined_df['close']), None).tolist(),
        "forecast": combined_df['forecast'].where(pd.notna(combined_df['forecast']), None).tolist(),
        "conf_int_lower": combined_df['conf_int_lower'].where(pd.notna(combined_df['conf_int_lower']), None).tolist(),
        "conf_int_upper": combined_df['conf_int_upper'].where(pd.notna(combined_df['conf_int_upper']), None).tolist(),
        "garch_dates": (df.index.strftime('%Y-%m-%d').tolist() if isinstance(df.index, pd.DatetimeIndex) else []) + (forecast_index.strftime('%Y-%m-%d').tolist() if isinstance(forecast_index, pd.DatetimeIndex) else []),
        "garch_volatility": np.concatenate([garch_result.conditional_volatility, forecast_volatility[0]]).tolist(),
        "forecast_start_date": model_end.strftime('%Y-%m-%d')
    }

def generate_forecast_graph(symbol: str, model_start_date: str, model_end_date: str, forecast_end_date: str):
    """
    This function is now deprecated in favor of generating data for D3.
    Kept for compatibility but should be removed later.
    """
    return None 