"""
ML Models for NEPSE Stock Forecasting

Feature Engineering Steps:
- log_return: log(close / close.shift(1))
- ma7: 7-day moving average of close
- ma21: 21-day moving average of close
- volatility: 7-day rolling std of log_return
- rsi: 14-day Relative Strength Index (RSI) using rolling means of up/down closes
- lag1, lag2, lag3: close price lagged by 1, 2, 3 days
- dayofweek: day of week (0=Monday)
- month: month of year (1-12)
- close_open_ratio: close / open
- next_return: log_return shifted -1 (target)

To use these models in production, apply the same feature engineering steps to new data before prediction.

Scaler/Normalization for LSTM or other models:
- If you use normalization (e.g., MinMaxScaler, StandardScaler), fit the scaler on the training data, save it with joblib, and load/apply it to new data before prediction.
- Example code is provided at the end of this file.
"""

import os
import sys
import json

# Setup Django environment for standalone script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')

import django
django.setup()

# Now import everything else
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from .stock_service import StockService

# --- CONFIG ---
N_RECURSIONS = 20  # Number of walk-forward iterations
TEST_SIZE = 0.2    # Fraction of data to reserve for final testing
VALIDATION_SIZE = 0.1  # Fraction of training data to use for validation in each window
MIN_TEST_SIZE = 30  # Minimum number of days to reserve for testing
MIN_VALIDATION_SIZE = 15  # Minimum number of days for validation

def fetch_stock_data(symbol, start_date, end_date):
    service = StockService()
    df = service.get_stock_data_df(symbol, start=start_date, stop=end_date)
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol} in range {start_date} to {end_date}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def feature_engineering(df):
    df = df.copy()
    
    # Basic price-based features
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change"] = df["close"] - df["close"].shift(1)
    df["price_change_pct"] = df["close"].pct_change()
    
    # Moving averages with different windows
    for window in [5, 7, 10, 14, 21, 30]:
        df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"ma_{window}_ratio"] = df["close"] / df[f"ma_{window}"]
        df[f"ma_{window}_slope"] = df[f"ma_{window}"].diff()
    
    # Volatility features
    df["volatility_5"] = df["log_return"].rolling(window=5).std()
    df["volatility_10"] = df["log_return"].rolling(window=10).std()
    df["volatility_20"] = df["log_return"].rolling(window=20).std()
    df["volatility_ratio"] = df["volatility_5"] / df["volatility_20"]
    
    # RSI with different periods
    for period in [7, 14, 21]:
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()
        rs = roll_up / (roll_down + 1e-9)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    for window in [10, 20]:
        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        df[f"bb_upper_{window}"] = ma + (2 * std)
        df[f"bb_lower_{window}"] = ma - (2 * std)
        df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"])
    
    # Price momentum features
    for period in [1, 3, 5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
        df[f"log_momentum_{period}"] = np.log(df["close"] / df["close"].shift(period))
    
    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
        df["volume_price_trend"] = df["volume"] * df["price_change_pct"]
    else:
        # Create dummy volume features if not available
        df["volume_ma_5"] = 1000
        df["volume_ma_20"] = 1000
        df["volume_ratio"] = 1.0
        df["volume_price_trend"] = 0.0
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag) if "volume" in df.columns else 1000
    
    # Time-based features
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["dayofyear"] = df.index.dayofyear
    
    # Price level features
    df["close_open_ratio"] = df["close"] / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"] if "high" in df.columns and "low" in df.columns else 1.0
    df["close_high_ratio"] = df["close"] / df["high"] if "high" in df.columns else 1.0
    
    # Trend features
    df["trend_5"] = np.where(df["close"] > df["close"].shift(5), 1, -1)
    df["trend_10"] = np.where(df["close"] > df["close"].shift(10), 1, -1)
    df["trend_20"] = np.where(df["close"] > df["close"].shift(20), 1, -1)
    
    # Multiple target variables for different prediction horizons
    df["next_return_1d"] = df["log_return"].shift(-1)
    df["next_return_3d"] = df["log_return"].shift(-3)
    df["next_return_5d"] = df["log_return"].shift(-5)
    df["next_price_1d"] = df["close"].shift(-1)
    df["next_price_3d"] = df["close"].shift(-3)
    df["next_price_5d"] = df["close"].shift(-5)
    
    # Direction prediction (easier than exact values)
    df["price_direction_1d"] = np.where(df["next_price_1d"] > df["close"], 1, 0)
    df["price_direction_3d"] = np.where(df["next_price_3d"] > df["close"], 1, 0)
    df["price_direction_5d"] = np.where(df["next_price_5d"] > df["close"], 1, 0)
    
    df = df.dropna()
    return df

def reconstruct_price(last_close, predicted_log_returns):
    # last_close: float, predicted_log_returns: np.array
    prices = [last_close]
    for r in predicted_log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

def ensure_model_dir(outdir, model_name):
    model_dir = os.path.join(outdir, model_name.replace(' ', '_').lower())
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def evaluate(model, X_test, y_test, name, outdir=None, actual_close=None, last_train_close=None, test_index=None, iteration=None):
    y_pred = model.predict(X_test)
    print(f"\n{name} R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"{name} MSE: {mean_squared_error(y_test, y_pred):.6f}")
    print(f"{name} MAE: {mean_absolute_error(y_test, y_pred):.6f}")
    # Plot log return prediction
    model_dir = ensure_model_dir(outdir, name) if outdir else None
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Actual Log Return')
    plt.plot(y_pred, label='Predicted Log Return')
    plt.title(f"{name} Log Return Prediction vs Actual")
    plt.legend()
    plt.grid()
    if model_dir:
        fname = f"log_return_pred_vs_actual"
        if iteration is not None:
            fname += f"_iter{iteration+1}"
        plt.savefig(os.path.join(model_dir, f"{fname}.png"))
    plt.close()
    # Reconstruct and plot price if actual_close and last_train_close are provided
    if actual_close is not None and last_train_close is not None and test_index is not None:
        pred_price = reconstruct_price(last_train_close, y_pred)
        plt.figure(figsize=(10,4))
        plt.plot(test_index, actual_close, label='Actual Price')
        plt.plot(test_index, pred_price, label='Predicted Price')
        plt.title(f"{name} Price Prediction (Reconstructed)")
        plt.legend()
        plt.grid()
        if model_dir:
            fname = f"price_pred_vs_actual"
            if iteration is not None:
                fname += f"_iter{iteration+1}"
            plt.savefig(os.path.join(model_dir, f"{fname}.png"))
        plt.close()

def plot_feature_importance(model, features, title, outdir=None, iteration=None, model_name=None):
    importance = model.feature_importances_
    model_dir = ensure_model_dir(outdir, model_name) if outdir and model_name else None
    sns.barplot(x=importance, y=features)
    plt.title(title)
    if model_dir:
        fname = f"feature_importance"
        if iteration is not None:
            fname += f"_iter{iteration+1}"
        plt.savefig(os.path.join(model_dir, f"{fname}.png"))
    plt.close()

def run_all_models(symbol, start_date, end_date, outdir="ml_outputs", n_recursions=N_RECURSIONS):
    os.makedirs(outdir, exist_ok=True)
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_stock_data(symbol, start_date, end_date)
    df = feature_engineering(df)
    
    print(f"Feature engineering completed. Total features: {len(df.columns)}")
    print(f"Available features: {list(df.columns)}")
    
    # Multiple target variables to try
    targets = {
        "price_direction_1d": "Price Direction (1 day)",
        "price_direction_3d": "Price Direction (3 days)", 
        "price_direction_5d": "Price Direction (5 days)",
        "next_return_1d": "Log Return (1 day)",
        "next_return_3d": "Log Return (3 days)",
        "next_return_5d": "Log Return (5 days)"
    }
    
    # Feature sets for different model types
    feature_sets = {
        "basic": [
            "log_return", "price_change_pct", "ma_7", "ma_21", "volatility_10", 
            "rsi_14", "volume_ratio", "close_lag_1", "close_lag_2", "dayofweek", "month"
        ],
        "technical": [
            "log_return", "price_change_pct", "ma_5", "ma_10", "ma_20", "ma_30",
            "volatility_5", "volatility_10", "volatility_20", "volatility_ratio",
            "rsi_7", "rsi_14", "rsi_21", "macd", "macd_signal", "macd_histogram",
            "bb_position_10", "bb_position_20", "momentum_5", "momentum_10", "momentum_20",
            "volume_ratio", "volume_price_trend", "trend_5", "trend_10", "trend_20"
        ],
        "comprehensive": [
            col for col in df.columns if col not in targets.keys() and col not in 
            ["open", "high", "low", "close", "volume"]  # Exclude raw price data
        ]
    }
    
    # Reserve data for final testing (out-of-sample)
    test_size = max(int(TEST_SIZE * len(df)), MIN_TEST_SIZE)
    train_size = len(df) - test_size
    
    print(f"\n=== DATA SPLIT ===")
    print(f"Total data points: {len(df)}")
    print(f"Training data: 0 to {train_size-1} ({train_size} points)")
    print(f"Test data: {train_size} to {len(df)-1} ({test_size} points)")
    print(f"Training period: {df.index[0]} to {df.index[train_size-1]}")
    print(f"Test period: {df.index[train_size]} to {df.index[-1]}")
    
    # Use only training data for walk-forward validation
    df_train_full = df.iloc[:train_size]
    df_test_final = df.iloc[train_size:]
    
    # Walk-forward validation on training data only
    validation_size = max(int(VALIDATION_SIZE * train_size), MIN_VALIDATION_SIZE)
    window_shift = max(1, int(validation_size / n_recursions))
    
    best_models = {}
    best_scores = {}
    
    print(f"\n=== WALK-FORWARD VALIDATION ===")
    print(f"Validation window size: {validation_size}")
    print(f"Window shift: {window_shift}")
    print(f"Number of iterations: {n_recursions}")
    
    # Test each target variable and feature set
    for target_name, target_desc in targets.items():
        print(f"\n{'='*50}")
        print(f"TESTING TARGET: {target_desc}")
        print(f"{'='*50}")
        
        if target_name not in df_train_full.columns:
            print(f"Target {target_name} not available, skipping...")
            continue
            
        y_train_full = df_train_full[target_name]
        y_test_final = df_test_final[target_name]
        
        best_models[target_name] = {}
        best_scores[target_name] = {}
        
        for feature_set_name, features in feature_sets.items():
            print(f"\n--- Feature Set: {feature_set_name} ---")
            
            # Filter available features
            available_features = [f for f in features if f in df_train_full.columns]
            if len(available_features) < 5:
                print(f"Not enough features available ({len(available_features)}), skipping...")
                continue
                
            X_train_full = df_train_full[available_features]
            X_test_final = df_test_final[available_features]
            
            best_rf_params = None
            best_rf_score = -np.inf
            best_xgb_params = None
            best_xgb_score = -np.inf
            
            # Walk-forward validation
            for i in range(min(n_recursions, 5)):  # Limit iterations for faster testing
                train_end = train_size - validation_size + i * window_shift
                val_start = train_end
                val_end = min(val_start + validation_size, train_size)
                
                if val_end - val_start < 5:
                    break
                    
                X_train, X_val = X_train_full.iloc[:train_end], X_train_full.iloc[val_start:val_end]
                y_train, y_val = y_train_full.iloc[:train_end], y_train_full.iloc[val_start:val_end]
                
                print(f"  Iteration {i+1}: Train {len(X_train)}, Val {len(X_val)}")
                
                # Random Forest with hyperparameter tuning
                rf_param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                try:
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    
                    # Choose classifier or regressor based on target
                    if "direction" in target_name:
                        rf_model = RandomForestClassifier(random_state=42)
                        rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, n_jobs=-1, scoring='accuracy')
                    else:
                        rf_model = RandomForestRegressor(random_state=42)
                        rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, n_jobs=-1, scoring='r2')
                    
                    rf_grid.fit(X_train, y_train)
                    rf_score = rf_grid.best_score_
                    
                    if rf_score > best_rf_score:
                        best_rf_score = rf_score
                        best_rf_params = rf_grid.best_params_
                        
                except Exception as e:
                    print(f"    RF error: {e}")
                    continue
                
                # XGBoost with hyperparameter tuning
                xgb_param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                
                try:
                    from xgboost import XGBClassifier, XGBRegressor
                    
                    if "direction" in target_name:
                        xgb_model = XGBClassifier(random_state=42, verbosity=0)
                        xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, n_jobs=-1, scoring='accuracy')
                    else:
                        xgb_model = XGBRegressor(random_state=42, verbosity=0)
                        xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, n_jobs=-1, scoring='r2')
                    
                    xgb_grid.fit(X_train, y_train)
                    xgb_score = xgb_grid.best_score_
                    
                    if xgb_score > best_xgb_score:
                        best_xgb_score = xgb_score
                        best_xgb_params = xgb_grid.best_params_
                        
                except Exception as e:
                    print(f"    XGB error: {e}")
                    continue
            
            # Store best models for this target and feature set
            best_models[target_name][feature_set_name] = {
                'rf_params': best_rf_params,
                'xgb_params': best_xgb_params,
                'rf_score': best_rf_score,
                'xgb_score': best_xgb_score,
                'features': available_features
            }
            
            best_scores[target_name][feature_set_name] = max(best_rf_score, best_xgb_score)
            
            print(f"  Best RF score: {best_rf_score:.4f}")
            print(f"  Best XGB score: {best_xgb_score:.4f}")
    
    # Train final models on best configurations
    print(f"\n=== FINAL MODEL TRAINING ===")
    final_models = {}
    
    for target_name, target_desc in targets.items():
        if target_name not in best_models:
            continue
            
        print(f"\nTraining final models for {target_desc}")
        
        # Find best feature set for this target
        best_feature_set = max(best_models[target_name].keys(), 
                              key=lambda x: best_scores[target_name][x])
        
        best_config = best_models[target_name][best_feature_set]
        features = best_config['features']
        
        X_train_full = df_train_full[features]
        X_test_final = df_test_final[features]
        y_train_full = df_train_full[target_name]
        y_test_final = df_test_final[target_name]
        
        # Train Random Forest
        if best_config['rf_params']:
            if "direction" in target_name:
                rf_final = RandomForestClassifier(random_state=42, **best_config['rf_params'])
            else:
                rf_final = RandomForestRegressor(random_state=42, **best_config['rf_params'])
            
            rf_final.fit(X_train_full, y_train_full)
            
            # Evaluate on test data
            y_pred = rf_final.predict(X_test_final)
            if "direction" in target_name:
                from sklearn.metrics import accuracy_score, classification_report
                accuracy = accuracy_score(y_test_final, y_pred)
                print(f"  RF Test Accuracy: {accuracy:.4f}")
                print(f"  RF Classification Report:")
                print(classification_report(y_test_final, y_pred))
            else:
                from sklearn.metrics import r2_score, mean_squared_error
                r2 = r2_score(y_test_final, y_pred)
                mse = mean_squared_error(y_test_final, y_pred)
                print(f"  RF Test R²: {r2:.4f}")
                print(f"  RF Test MSE: {mse:.6f}")
            
            # Save model
            model_filename = f"rf_{target_name}_{best_feature_set}.pkl"
            joblib.dump(rf_final, os.path.join(outdir, model_filename))
            print(f"  Saved RF model: {model_filename}")
        
        # Train XGBoost
        if best_config['xgb_params']:
            if "direction" in target_name:
                xgb_final = XGBClassifier(random_state=42, verbosity=0, **best_config['xgb_params'])
            else:
                xgb_final = XGBRegressor(random_state=42, verbosity=0, **best_config['xgb_params'])
            
            xgb_final.fit(X_train_full, y_train_full)
            
            # Evaluate on test data
            y_pred = xgb_final.predict(X_test_final)
            if "direction" in target_name:
                accuracy = accuracy_score(y_test_final, y_pred)
                print(f"  XGB Test Accuracy: {accuracy:.4f}")
            else:
                r2 = r2_score(y_test_final, y_pred)
                mse = mean_squared_error(y_test_final, y_pred)
                print(f"  XGB Test R²: {r2:.4f}")
                print(f"  XGB Test MSE: {mse:.6f}")
            
            # Save model
            model_filename = f"xgb_{target_name}_{best_feature_set}.pkl"
            joblib.dump(xgb_final, os.path.join(outdir, model_filename))
            print(f"  Saved XGB model: {model_filename}")
        
        final_models[target_name] = {
            'best_feature_set': best_feature_set,
            'features': features,
            'rf_model': rf_final if best_config['rf_params'] else None,
            'xgb_model': xgb_final if best_config['xgb_params'] else None
        }
    
    # Save comprehensive results
    results_summary = {
        'data_split': {
            'train_size': train_size,
            'test_size': test_size,
            'train_start_date': str(df.index[0]),
            'train_end_date': str(df.index[train_size-1]),
            'test_start_date': str(df.index[train_size]),
            'test_end_date': str(df.index[-1])
        },
        'best_models': best_models,
        'best_scores': best_scores,
        'final_models': {k: {'best_feature_set': v['best_feature_set'], 'features': v['features']} 
                        for k, v in final_models.items()}
    }
    
    with open(os.path.join(outdir, "comprehensive_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n=== COMPREHENSIVE RESULTS SAVED ===")
    print(f"Results saved to: {os.path.join(outdir, 'comprehensive_results.json')}")
    
    return final_models

if __name__ == '__main__':
    # Example usage
    symbol = "SHEL"  # Change as needed
    start_date = "2022-01-01"
    end_date = "2025-06-27"
    run_all_models(symbol, start_date, end_date)

"""
Note: For LSTM or any normalization, you should also save the scaler/transformer used for feature scaling.
Example:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(X_train)
    ...
    joblib.dump(scaler, 'ml_outputs/scaler.pkl')
Then, in production, load the scaler and apply it to new data before prediction:
    scaler = joblib.load('ml_outputs/scaler.pkl')
    X_new_scaled = scaler.transform(X_new)
"""