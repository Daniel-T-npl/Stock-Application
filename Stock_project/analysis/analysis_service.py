import pandas as pd
import numpy as np

class AnalysisService:
    """
    A service class for performing financial analysis on stock data.
    All methods are static and operate on Pandas DataFrames.
    """

    @staticmethod
    def calculate_moving_average(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.DataFrame:
        """Calculates the simple moving average."""
        if column not in df.columns:
            return df
        df[f'sma_{window}'] = df[column].rolling(window=window).mean()
        return df

    @staticmethod
    def calculate_log_returns(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """Calculates the logarithmic returns."""
        if column not in df.columns:
            return df
        df['log_return'] = np.log(df[column] / df[column].shift(1))
        return df

    @staticmethod
    def calculate_rolling_volatility(df: pd.DataFrame, window: int = 20, column: str = 'log_return') -> pd.DataFrame:
        """Calculates the rolling volatility of log returns."""
        if column not in df.columns:
            # Ensure log returns are calculated first
            df = AnalysisService.calculate_log_returns(df)
        df[f'volatility_{window}'] = df[column].rolling(window=window).std()
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, column: str = 'close', stddev: float = 2) -> pd.DataFrame:
        """Calculates Bollinger Bands with a variable stddev."""
        if column not in df.columns:
            return df
        ma = df[column].rolling(window=window).mean()
        std_dev = df[column].rolling(window=window).std()
        df['ma'] = ma
        df['bb_upper'] = ma + (std_dev * stddev)
        df['bb_lower'] = ma - (std_dev * stddev)
        return df

    