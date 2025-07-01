"""
Advanced ML Models for Stock Forecasting with 60% Directional Accuracy Target

Features:
- Advanced feature engineering (100+ features)
- Multiple target variables
- Walk-forward analysis
- Comprehensive evaluation with graphs
- Ensemble methods
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Setup Django environment
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')

import django
django.setup()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from datetime import datetime, timedelta

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available, skipping feature importance analysis")
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, skipping TensorFlow LSTM/GRU models")
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    from analysis.pytorch_models import PyTorchTrainer, create_pytorch_lstm_model, create_pytorch_gru_model
    PYTORCH_AVAILABLE = True
    print("PyTorch available, LSTM/GRU models will use PyTorch")
except ImportError:
    print("PyTorch not available, skipping LSTM/GRU models")
    PYTORCH_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available, skipping XGBoost models")
    XGBOOST_AVAILABLE = False

from analysis.stock_service import StockService

class AdvancedStockForecaster:
    def __init__(self, symbol, start_date="2021-01-01", end_date="2024-12-31", force_retrain=False):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.force_retrain = force_retrain
        self.data = None
        self.features = None
        self.targets = None
        self.models = {}
        self.results = {}
        self.model_dir = "ml_outputs_advanced"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def fetch_and_prepare_data(self):
        """Fetch and prepare data with advanced feature engineering."""
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        service = StockService()
        df = service.get_stock_data_df(self.symbol, start=self.start_date, stop=self.end_date)
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        self.data = self.advanced_feature_engineering(df)
        print(f"Data prepared: {len(self.data)} points, {len(self.data.columns)} features")
        
        return self.data
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering with 100+ features."""
        df = df.copy()
        
        # Basic price features
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["price_change"] = df["close"] - df["close"].shift(1)
        df["price_change_pct"] = df["close"].pct_change()
        
        # Volatility features (multiple timeframes)
        volatility_windows = [5, 10, 20, 30, 50]
        for window in volatility_windows:
            df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()
            df[f"realized_vol_{window}"] = df["log_return"].rolling(window=window).apply(lambda x: np.sqrt(np.sum(x**2)))
            df[f"vol_ratio_{window}"] = df[f"volatility_{window}"] / df[f"volatility_{window}"].rolling(window=window).mean()
        
        # Moving averages with ratios and slopes
        for window in [5, 7, 10, 14, 21, 30, 50]:
            df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
            df[f"ma_{window}_ratio"] = df["close"] / df[f"ma_{window}"]
            df[f"ma_{window}_slope"] = df[f"ma_{window}"].diff()
            df[f"ma_{window}_acceleration"] = df[f"ma_{window}_slope"].diff()
        
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            delta = df["close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.rolling(period).mean()
            roll_down = down.rolling(period).mean()
            rs = roll_up / (roll_down + 1e-9)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
            df[f"rsi_{period}_slope"] = df[f"rsi_{period}"].diff()
        
        # MACD with multiple configurations
        for fast, slow in [(12, 26), (8, 21), (5, 13)]:
            ema_fast = df["close"].ewm(span=fast).mean()
            ema_slow = df["close"].ewm(span=slow).mean()
            df[f"macd_{fast}_{slow}"] = ema_fast - ema_slow
            df[f"macd_signal_{fast}_{slow}"] = df[f"macd_{fast}_{slow}"].ewm(span=9).mean()
            df[f"macd_hist_{fast}_{slow}"] = df[f"macd_{fast}_{slow}"] - df[f"macd_signal_{fast}_{slow}"]
        
        # Bollinger Bands with multiple windows
        for window in [10, 20, 30]:
            ma = df["close"].rolling(window=window).mean()
            std = df["close"].rolling(window=window).std()
            df[f"bb_upper_{window}"] = ma + (2 * std)
            df[f"bb_lower_{window}"] = ma - (2 * std)
            df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"])
            df[f"bb_width_{window}"] = df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            df[f"bb_squeeze_{window}"] = df[f"bb_width_{window}"] / df[f"bb_width_{window}"].rolling(window=window).mean()
        
        # Price momentum and acceleration
        for period in [1, 3, 5, 10, 20, 30]:
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
            df[f"log_momentum_{period}"] = np.log(df["close"] / df["close"].shift(period))
            df[f"momentum_accel_{period}"] = df[f"momentum_{period}"].diff()
        
        # Volume features (if available)
        if "volume" in df.columns:
            for window in [5, 10, 20, 30]:
                df[f"volume_ma_{window}"] = df["volume"].rolling(window=window).mean()
                df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]
                df[f"volume_price_trend_{window}"] = df["volume"] * df["price_change_pct"]
                df[f"volume_volatility_{window}"] = df["volume"].rolling(window=window).std()
        else:
            # Create dummy volume features
            for window in [5, 10, 20, 30]:
                df[f"volume_ma_{window}"] = 1000
                df[f"volume_ratio_{window}"] = 1.0
                df[f"volume_price_trend_{window}"] = 0.0
                df[f"volume_volatility_{window}"] = 0.0
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag) if "volume" in df.columns else 1000
        
        # Time-based features
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["dayofyear"] = df.index.dayofyear
        df["weekofyear"] = df.index.isocalendar().week
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding for time features
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Price level features
        df["close_open_ratio"] = df["close"] / df["open"]
        df["high_low_ratio"] = df["high"] / df["low"] if "high" in df.columns and "low" in df.columns else 1.0
        df["close_high_ratio"] = df["close"] / df["high"] if "high" in df.columns else 1.0
        df["close_low_ratio"] = df["close"] / df["low"] if "low" in df.columns else 1.0
        
        # Trend features
        for period in [5, 10, 20, 30]:
            df[f"trend_{period}"] = np.where(df["close"] > df["close"].shift(period), 1, -1)
            df[f"trend_strength_{period}"] = (df["close"] - df["close"].shift(period)) / df["close"].shift(period)
        
        # Volatility-adjusted features
        df["vol_adjusted_return"] = df["log_return"] / (df["volatility_20"] + 1e-9)
        df["vol_adjusted_momentum"] = df["momentum_10"] / (df["volatility_20"] + 1e-9)
        
        # Market regime features
        df["bull_market"] = np.where(df["close"] > df["ma_50"], 1, 0)
        df["bear_market"] = np.where(df["close"] < df["ma_50"], 1, 0)
        df["sideways_market"] = np.where((df["close"] >= df["ma_50"] * 0.95) & (df["close"] <= df["ma_50"] * 1.05), 1, 0)
        
        # Advanced targets - use only available volatility windows
        for horizon in [1, 3, 5, 10]:
            # Direction targets
            df[f"direction_{horizon}d"] = np.where(df["close"].shift(-horizon) > df["close"], 1, 0)
            
            # Volatility-adjusted return targets - use closest available volatility window
            future_return = df["close"].shift(-horizon) / df["close"] - 1
            # Find closest volatility window
            closest_vol_window = min(volatility_windows, key=lambda x: abs(x - horizon))
            vol_adjustment = df[f"volatility_{closest_vol_window}"].rolling(window=closest_vol_window).mean()
            df[f"vol_adjusted_return_{horizon}d"] = future_return / (vol_adjustment + 1e-9)
            
            # Categorical targets (Up/Down/Sideways)
            future_change = (df["close"].shift(-horizon) - df["close"]) / df["close"]
            df[f"category_{horizon}d"] = np.where(future_change > 0.02, 2, np.where(future_change < -0.02, 0, 1))
        
        df = df.dropna()
        return df
    
    def prepare_targets_and_features(self):
        """Prepare targets and features for modeling."""
        # Define target variables with their types
        self.targets = {
            "direction_1d": {"description": "1-Day Direction", "type": "classification"},
            "direction_3d": {"description": "3-Day Direction", "type": "classification"}, 
            "category_1d": {"description": "1-Day Category", "type": "classification"},
            "category_3d": {"description": "3-Day Category", "type": "classification"}
        }
        
        # Define feature sets
        feature_columns = [col for col in self.data.columns 
                          if col not in self.targets.keys() 
                          and col not in ["open", "high", "low", "close", "volume"]
                          and not col.startswith("vol_adjusted_return_")  # Remove future-looking features
                          and not col.startswith("direction_")  # Remove direction features
                          and not col.startswith("category_")]  # Remove category features
        
        self.features = feature_columns
        print(f"Prepared {len(self.features)} features and {len(self.targets)} targets")
        
        return self.features, self.targets
    
    def walk_forward_analysis(self, target_name, feature_set="all", test_size=0.2):
        """Perform walk-forward analysis with comprehensive evaluation."""
        print(f"\n=== WALK-FORWARD ANALYSIS: {target_name} ===")
        
        # Get target type
        target_type = self.targets[target_name]["type"]
        print(f"Target type: {target_type}")
        
        # Prepare data
        if feature_set == "all":
            X = self.data[self.features]
        else:
            X = self.data[feature_set]
        
        y = self.data[target_name]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(self.data) * test_size))
        
        if target_type == "classification":
            results = {
                'rf_accuracy': [], 'rf_f1': [], 'rf_precision': [], 'rf_recall': [],
                'xgb_accuracy': [], 'xgb_f1': [], 'xgb_precision': [], 'xgb_recall': [],
                'lstm_accuracy': [], 'lstm_f1': [], 'lstm_precision': [], 'lstm_recall': [],
                'gru_accuracy': [], 'gru_f1': [], 'gru_precision': [], 'gru_recall': []
            }
        else:  # regression
            results = {
                'rf_r2': [], 'rf_rmse': [], 'rf_mae': [],
                'xgb_r2': [], 'xgb_rmse': [], 'xgb_mae': [],
                'lstm_r2': [], 'lstm_rmse': [], 'lstm_mae': [],
                'gru_r2': [], 'gru_rmse': [], 'gru_mae': []
            }
        
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            print(f"\n--- Fold {fold} ---")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if target_type == "classification":
                # Random Forest Classification
                rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                
                results['rf_accuracy'].append(accuracy_score(y_test, rf_pred))
                results['rf_f1'].append(f1_score(y_test, rf_pred, average='weighted'))
                results['rf_precision'].append(precision_score(y_test, rf_pred, average='weighted'))
                results['rf_recall'].append(recall_score(y_test, rf_pred, average='weighted'))
                
                # XGBoost Classification
                if XGBOOST_AVAILABLE:
                    try:
                        unique_classes = np.unique(y_train)
                        if len(unique_classes) < 2:
                            print(f"XGBoost skipped: insufficient classes ({len(unique_classes)})")
                        else:
                            if len(unique_classes) == 2:
                                xgb_model = XGBClassifier(
                                    n_estimators=300, max_depth=8, random_state=42, verbosity=0,
                                    objective='binary:logistic', eval_metric='logloss'
                                )
                            else:
                                xgb_model = XGBClassifier(
                                    n_estimators=300, max_depth=8, random_state=42, verbosity=0,
                                    objective='multi:softmax', eval_metric='mlogloss',
                                    num_class=len(unique_classes)
                                )
                            
                            xgb_model.fit(X_train, y_train)
                            xgb_pred = xgb_model.predict(X_test)
                            xgb_accuracy = accuracy_score(y_test, xgb_pred)
                            
                            results['xgb_accuracy'].append(xgb_accuracy)
                            results['xgb_f1'].append(f1_score(y_test, xgb_pred, average='weighted'))
                            results['xgb_precision'].append(precision_score(y_test, xgb_pred, average='weighted'))
                            results['xgb_recall'].append(recall_score(y_test, xgb_pred, average='weighted'))
                    except Exception as e:
                        print(f"XGBoost training failed: {e}")
                
                # LSTM/GRU Classification (if available)
                if PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
                    try:
                        sequence_length = 20
                        # Use the class method for sequence preparation
                        X_train_seq, y_train_seq = self.prepare_sequences(
                            pd.concat([X_train, y_train], axis=1), target_name, sequence_length
                        )
                        X_test_seq, y_test_seq = self.prepare_sequences(
                            pd.concat([X_test, y_test], axis=1), target_name, sequence_length
                        )
                        
                        # Check if we have enough data
                        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                            print("Insufficient data for sequence models, skipping LSTM/GRU")
                        else:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
                            X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
                            
                            if PYTORCH_AVAILABLE:
                                # Use PyTorch models
                                print("Training PyTorch LSTM...")
                                lstm_trainer = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, X_train.shape[1]), len(np.unique(y_train))))
                                lstm_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                lstm_pred = lstm_trainer.predict(X_test_scaled)
                                lstm_accuracy = accuracy_score(y_test_seq, lstm_pred)
                                
                                models['lstm'] = {
                                    'model': lstm_trainer,
                                    'scaler': scaler,
                                    'accuracy': lstm_accuracy,
                                    'predictions': lstm_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'classification',
                                    'framework': 'pytorch'
                                }
                                
                                print("Training PyTorch GRU...")
                                gru_trainer = PyTorchTrainer(create_pytorch_gru_model((sequence_length, X_train.shape[1]), len(np.unique(y_train))))
                                gru_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                gru_pred = gru_trainer.predict(X_test_scaled)
                                gru_accuracy = accuracy_score(y_test_seq, gru_pred)
                                
                                models['gru'] = {
                                    'model': gru_trainer,
                                    'scaler': scaler,
                                    'accuracy': gru_accuracy,
                                    'predictions': gru_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'classification',
                                    'framework': 'pytorch'
                                }
                        
                    except Exception as e:
                        print(f"LSTM/GRU training failed: {e}")
            
            else:  # Regression
                # Random Forest Regression
                rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                
                results['rf_r2'].append(r2_score(y_test, rf_pred))
                results['rf_rmse'].append(np.sqrt(mean_squared_error(y_test, rf_pred)))
                results['rf_mae'].append(mean_absolute_error(y_test, rf_pred))
                
                # XGBoost Regression
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = XGBRegressor(
                            n_estimators=200, max_depth=6, random_state=42, verbosity=0,
                            objective='reg:squarederror', eval_metric='rmse'
                        )
                        xgb_model.fit(X_train, y_train)
                        xgb_pred = xgb_model.predict(X_test)
                        
                        results['xgb_r2'].append(r2_score(y_test, xgb_pred))
                        results['xgb_rmse'].append(np.sqrt(mean_squared_error(y_test, xgb_pred)))
                        results['xgb_mae'].append(mean_absolute_error(y_test, xgb_pred))
                    except Exception as e:
                        print(f"XGBoost regression failed: {e}")
                
                # LSTM/GRU Regression (if available)
                if PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
                    try:
                        sequence_length = 20
                        X_train_seq, y_train_seq = self.prepare_sequences(
                            pd.concat([X_train, y_train], axis=1), target_name, sequence_length
                        )
                        X_test_seq, y_test_seq = self.prepare_sequences(
                            pd.concat([X_test, y_test], axis=1), target_name, sequence_length
                        )
                        
                        # Check if we have enough data
                        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                            print("Insufficient data for sequence models, skipping LSTM/GRU")
                        else:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
                            X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
                            
                            if PYTORCH_AVAILABLE:
                                # Use PyTorch models
                                print("Training PyTorch LSTM...")
                                lstm_trainer = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, X_train.shape[1]), 1))
                                lstm_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                lstm_pred = lstm_trainer.predict(X_test_scaled)
                                lstm_r2 = r2_score(y_test_seq, lstm_pred)
                                
                                models['lstm'] = {
                                    'model': lstm_trainer,
                                    'scaler': scaler,
                                    'r2': lstm_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
                                    'mae': mean_absolute_error(y_test_seq, lstm_pred),
                                    'predictions': lstm_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'regression',
                                    'framework': 'pytorch'
                                }
                                
                                print("Training PyTorch GRU...")
                                gru_trainer = PyTorchTrainer(create_pytorch_gru_model((sequence_length, X_train.shape[1]), 1))
                                gru_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                gru_pred = gru_trainer.predict(X_test_scaled)
                                gru_r2 = r2_score(y_test_seq, gru_pred)
                                
                                models['gru'] = {
                                    'model': gru_trainer,
                                    'scaler': scaler,
                                    'r2': gru_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, gru_pred)),
                                    'mae': mean_absolute_error(y_test_seq, gru_pred),
                                    'predictions': gru_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'regression',
                                    'framework': 'pytorch'
                                }
                                
                            elif TENSORFLOW_AVAILABLE:
                                # Use TensorFlow models
                                # LSTM for regression
                                lstm_model = Sequential([
                                    LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                                    Dropout(0.2),
                                    LSTM(64, return_sequences=False),
                                    Dropout(0.2),
                                    Dense(32, activation='relu'),
                                    Dropout(0.2),
                                    Dense(1, activation='linear')
                                ])
                                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                                
                                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                                lstm_model.fit(X_train_scaled, y_train_seq, epochs=100, batch_size=32, 
                                             validation_split=0.2, callbacks=[early_stopping], verbose=0)
                                
                                lstm_pred = lstm_model.predict(X_test_scaled).flatten()
                                lstm_r2 = r2_score(y_test_seq, lstm_pred)
                                
                                models['lstm'] = {
                                    'model': lstm_model,
                                    'scaler': scaler,
                                    'r2': lstm_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
                                    'mae': mean_absolute_error(y_test_seq, lstm_pred),
                                    'predictions': lstm_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'regression',
                                    'framework': 'tensorflow'
                                }
                                
                                # GRU for regression
                                gru_model = Sequential([
                                    GRU(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                                    Dropout(0.2),
                                    GRU(64, return_sequences=False),
                                    Dropout(0.2),
                                    Dense(32, activation='relu'),
                                    Dropout(0.2),
                                    Dense(1, activation='linear')
                                ])
                                gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                                
                                gru_model.fit(X_train_scaled, y_train_seq, epochs=100, batch_size=32, 
                                            validation_split=0.2, callbacks=[early_stopping], verbose=0)
                                
                                gru_pred = gru_model.predict(X_test_scaled).flatten()
                                gru_r2 = r2_score(y_test_seq, gru_pred)
                                
                                models['gru'] = {
                                    'model': gru_model,
                                    'scaler': scaler,
                                    'r2': gru_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, gru_pred)),
                                    'mae': mean_absolute_error(y_test_seq, gru_pred),
                                    'predictions': gru_pred,
                                    'sequence_length': sequence_length,
                                    'type': 'regression',
                                    'framework': 'tensorflow'
                                }
                            
                    except Exception as e:
                        print(f"LSTM/GRU regression training failed: {e}")
            
            fold += 1
        
        # Calculate average results
        avg_results = {}
        for metric in results:
            if results[metric]:
                avg_results[metric] = np.mean(results[metric])
        
        print(f"\nAverage Results for {target_name}:")
        if target_type == "classification":
            for model in ['rf', 'xgb', 'lstm', 'gru']:
                if f'{model}_accuracy' in avg_results:
                    print(f"{model.upper()}: Accuracy={avg_results[f'{model}_accuracy']:.4f}, "
                          f"F1={avg_results[f'{model}_f1']:.4f}")
        else:
            for model in ['rf', 'xgb', 'lstm', 'gru']:
                if f'{model}_r2' in avg_results:
                    print(f"{model.upper()}: R²={avg_results[f'{model}_r2']:.4f}, "
                          f"RMSE={avg_results[f'{model}_rmse']:.4f}")
        
        return avg_results
    
    def train_final_models(self, target_name, feature_set="all"):
        """Train final models with the best configuration."""
        print(f"\n=== TRAINING FINAL MODELS: {target_name} ===")
        
        # Get target type
        target_type = self.targets[target_name]["type"]
        print(f"Target type: {target_type}")
        
        # Prepare data
        if feature_set == "all":
            X = self.data[self.features]
        else:
            X = self.data[feature_set]
        
        y = self.data[target_name]
        
        # Split data (80% train, 20% test)
        split_idx = int(len(self.data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        models = {}
        
        if target_type == "classification":
            # Random Forest Classification
            if not self.force_retrain and self.model_exists('random_forest', target_name):
                print("Loading existing Random Forest model...")
                rf_model = self.load_model('random_forest', target_name)
                rf_pred = rf_model.predict(X_test)
                rf_accuracy = accuracy_score(y_test, rf_pred)
            else:
                print("Training new Random Forest model...")
                rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_accuracy = accuracy_score(y_test, rf_pred)
                # Save the model
                self.save_model(rf_model, 'random_forest', target_name, {
                    'accuracy': rf_accuracy,
                    'type': 'classification',
                    'training_date': datetime.now().isoformat()
                })
            
            models['random_forest'] = {
                'model': rf_model,
                'accuracy': rf_accuracy,
                'predictions': rf_pred,
                'type': 'classification'
            }
            
            # XGBoost Classification
            if XGBOOST_AVAILABLE:
                try:
                    unique_classes = np.unique(y_train)
                    if len(unique_classes) < 2:
                        print(f"XGBoost skipped: insufficient classes ({len(unique_classes)})")
                    else:
                        if not self.force_retrain and self.model_exists('xgboost', target_name):
                            print("Loading existing XGBoost model...")
                            xgb_model = self.load_model('xgboost', target_name)
                            xgb_pred = xgb_model.predict(X_test)
                            xgb_accuracy = accuracy_score(y_test, xgb_pred)
                        else:
                            print("Training new XGBoost model...")
                            if len(unique_classes) == 2:
                                xgb_model = XGBClassifier(
                                    n_estimators=300, max_depth=8, random_state=42, verbosity=0,
                                    objective='binary:logistic', eval_metric='logloss'
                                )
                            else:
                                xgb_model = XGBClassifier(
                                    n_estimators=300, max_depth=8, random_state=42, verbosity=0,
                                    objective='multi:softmax', eval_metric='mlogloss',
                                    num_class=len(unique_classes)
                                )
                            
                            xgb_model.fit(X_train, y_train)
                            xgb_pred = xgb_model.predict(X_test)
                            xgb_accuracy = accuracy_score(y_test, xgb_pred)
                            # Save the model
                            self.save_model(xgb_model, 'xgboost', target_name, {
                                'accuracy': xgb_accuracy,
                                'type': 'classification',
                                'training_date': datetime.now().isoformat()
                            })
                        
                        models['xgboost'] = {
                            'model': xgb_model,
                            'accuracy': xgb_accuracy,
                            'predictions': xgb_pred,
                            'type': 'classification'
                        }
                except Exception as e:
                    print(f"XGBoost training failed: {e}")
            
            # LSTM and GRU Classification
            if PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
                try:
                    sequence_length = 20
                    # Use the class method for sequence preparation
                    X_train_seq, y_train_seq = self.prepare_sequences(
                        pd.concat([X_train, y_train], axis=1), target_name, sequence_length
                    )
                    X_test_seq, y_test_seq = self.prepare_sequences(
                        pd.concat([X_test, y_test], axis=1), target_name, sequence_length
                    )
                    
                    # Check if we have enough data
                    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                        print("Insufficient data for sequence models, skipping LSTM/GRU")
                    else:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
                        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
                        
                        # LSTM
                        if not self.force_retrain and self.model_exists('lstm', target_name):
                            print("Loading existing LSTM model...")
                            lstm_trainer = self.load_model('lstm', target_name, num_classes=len(np.unique(y_train)), input_size=X_train.shape[1])
                            lstm_pred = lstm_trainer.predict(X_test_scaled)
                            lstm_accuracy = accuracy_score(y_test_seq, lstm_pred)
                        else:
                            print("Training new LSTM model...")
                            lstm_trainer = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, X_train.shape[1]), len(np.unique(y_train))))
                            lstm_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                            lstm_pred = lstm_trainer.predict(X_test_scaled)
                            lstm_accuracy = accuracy_score(y_test_seq, lstm_pred)
                            # Save the model
                            self.save_model(lstm_trainer, 'lstm', target_name, {
                                'accuracy': lstm_accuracy,
                                'type': 'classification',
                                'framework': 'pytorch',
                                'training_date': datetime.now().isoformat()
                            })
                        
                        models['lstm'] = {
                            'model': lstm_trainer,
                            'scaler': scaler,
                            'accuracy': lstm_accuracy,
                            'predictions': lstm_pred,
                            'sequence_length': sequence_length,
                            'type': 'classification',
                            'framework': 'pytorch'
                        }
                        
                        # GRU
                        if not self.force_retrain and self.model_exists('gru', target_name):
                            print("Loading existing GRU model...")
                            gru_trainer = self.load_model('gru', target_name, num_classes=len(np.unique(y_train)), input_size=X_train.shape[1])
                            gru_pred = gru_trainer.predict(X_test_scaled)
                            gru_accuracy = accuracy_score(y_test_seq, gru_pred)
                        else:
                            print("Training new GRU model...")
                            gru_trainer = PyTorchTrainer(create_pytorch_gru_model((sequence_length, X_train.shape[1]), len(np.unique(y_train))))
                            gru_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                            gru_pred = gru_trainer.predict(X_test_scaled)
                            gru_accuracy = accuracy_score(y_test_seq, gru_pred)
                            # Save the model
                            self.save_model(gru_trainer, 'gru', target_name, {
                                'accuracy': gru_accuracy,
                                'type': 'classification',
                                'framework': 'pytorch',
                                'training_date': datetime.now().isoformat()
                            })
                        
                        models['gru'] = {
                            'model': gru_trainer,
                            'scaler': scaler,
                            'accuracy': gru_accuracy,
                            'predictions': gru_pred,
                            'sequence_length': sequence_length,
                            'type': 'classification',
                            'framework': 'pytorch'
                        }
                        
                except Exception as e:
                    print(f"LSTM/GRU training failed: {e}")
        
        else:  # Regression
            # Random Forest Regression
            if not self.force_retrain and self.model_exists('random_forest', target_name):
                print("Loading existing Random Forest model...")
                rf_model = self.load_model('random_forest', target_name)
                rf_pred = rf_model.predict(X_test)
                rf_r2 = r2_score(y_test, rf_pred)
            else:
                print("Training new Random Forest model...")
                rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_r2 = r2_score(y_test, rf_pred)
                # Save the model
                self.save_model(rf_model, 'random_forest', target_name, {
                    'r2': rf_r2,
                    'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                    'mae': mean_absolute_error(y_test, rf_pred),
                    'type': 'regression',
                    'training_date': datetime.now().isoformat()
                })
            
            models['random_forest'] = {
                'model': rf_model,
                'r2': rf_r2,
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'mae': mean_absolute_error(y_test, rf_pred),
                'predictions': rf_pred,
                'type': 'regression'
            }
            
            # XGBoost Regression
            if XGBOOST_AVAILABLE:
                try:
                    if not self.force_retrain and self.model_exists('xgboost', target_name):
                        print("Loading existing XGBoost model...")
                        xgb_model = self.load_model('xgboost', target_name)
                        xgb_pred = xgb_model.predict(X_test)
                        xgb_r2 = r2_score(y_test, xgb_pred)
                    else:
                        print("Training new XGBoost model...")
                        xgb_model = XGBRegressor(
                            n_estimators=300, max_depth=8, random_state=42, verbosity=0,
                            objective='reg:squarederror', eval_metric='rmse'
                        )
                        xgb_model.fit(X_train, y_train)
                        xgb_pred = xgb_model.predict(X_test)
                        xgb_r2 = r2_score(y_test, xgb_pred)
                        # Save the model
                        self.save_model(xgb_model, 'xgboost', target_name, {
                            'r2': xgb_r2,
                            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                            'mae': mean_absolute_error(y_test, xgb_pred),
                            'type': 'regression',
                            'training_date': datetime.now().isoformat()
                        })
                    
                    models['xgboost'] = {
                        'model': xgb_model,
                        'r2': xgb_r2,
                        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                        'mae': mean_absolute_error(y_test, xgb_pred),
                        'predictions': xgb_pred,
                        'type': 'regression'
                    }
                except Exception as e:
                    print(f"XGBoost regression training failed: {e}")
            
            # LSTM and GRU Regression
            if PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
                try:
                    sequence_length = 20
                    X_train_seq, y_train_seq = self.prepare_sequences(
                        pd.concat([X_train, y_train], axis=1), target_name, sequence_length
                    )
                    X_test_seq, y_test_seq = self.prepare_sequences(
                        pd.concat([X_test, y_test], axis=1), target_name, sequence_length
                    )
                    
                    # Check if we have enough data
                    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                        print("Insufficient data for sequence models, skipping LSTM/GRU")
                    else:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
                        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
                        
                        if PYTORCH_AVAILABLE:
                            # LSTM
                            if not self.force_retrain and self.model_exists('lstm', target_name):
                                print("Loading existing LSTM model...")
                                lstm_trainer = self.load_model('lstm', target_name, num_classes=1, input_size=X_train.shape[1])
                                lstm_pred = lstm_trainer.predict(X_test_scaled)
                                lstm_r2 = r2_score(y_test_seq, lstm_pred)
                            else:
                                print("Training new LSTM model...")
                                lstm_trainer = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, X_train.shape[1]), 1))
                                lstm_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                lstm_pred = lstm_trainer.predict(X_test_scaled)
                                lstm_r2 = r2_score(y_test_seq, lstm_pred)
                                # Save the model
                                self.save_model(lstm_trainer, 'lstm', target_name, {
                                    'r2': lstm_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
                                    'mae': mean_absolute_error(y_test_seq, lstm_pred),
                                    'type': 'regression',
                                    'framework': 'pytorch',
                                    'training_date': datetime.now().isoformat()
                                })
                            
                            models['lstm'] = {
                                'model': lstm_trainer,
                                'scaler': scaler,
                                'r2': lstm_r2,
                                'rmse': np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
                                'mae': mean_absolute_error(y_test_seq, lstm_pred),
                                'predictions': lstm_pred,
                                'sequence_length': sequence_length,
                                'type': 'regression',
                                'framework': 'pytorch'
                            }
                            
                            # GRU
                            if not self.force_retrain and self.model_exists('gru', target_name):
                                print("Loading existing GRU model...")
                                gru_trainer = self.load_model('gru', target_name, num_classes=1, input_size=X_train.shape[1])
                                gru_pred = gru_trainer.predict(X_test_scaled)
                                gru_r2 = r2_score(y_test_seq, gru_pred)
                            else:
                                print("Training new GRU model...")
                                gru_trainer = PyTorchTrainer(create_pytorch_gru_model((sequence_length, X_train.shape[1]), 1))
                                gru_trainer.train_model(X_train_scaled, y_train_seq, epochs=100)
                                gru_pred = gru_trainer.predict(X_test_scaled)
                                gru_r2 = r2_score(y_test_seq, gru_pred)
                                # Save the model
                                self.save_model(gru_trainer, 'gru', target_name, {
                                    'r2': gru_r2,
                                    'rmse': np.sqrt(mean_squared_error(y_test_seq, gru_pred)),
                                    'mae': mean_absolute_error(y_test_seq, gru_pred),
                                    'type': 'regression',
                                    'framework': 'pytorch',
                                    'training_date': datetime.now().isoformat()
                                })
                            
                            models['gru'] = {
                                'model': gru_trainer,
                                'scaler': scaler,
                                'r2': gru_r2,
                                'rmse': np.sqrt(mean_squared_error(y_test_seq, gru_pred)),
                                'mae': mean_absolute_error(y_test_seq, gru_pred),
                                'predictions': gru_pred,
                                'sequence_length': sequence_length,
                                'type': 'regression',
                                'framework': 'pytorch'
                            }
                        
                except Exception as e:
                    print(f"LSTM/GRU regression training failed: {e}")
        
        # Find best model
        best_model_name = None
        best_metric = 0
        
        for model_name, model_info in models.items():
            if target_type == "classification":
                metric_value = model_info['accuracy']
            else:  # regression
                metric_value = model_info['r2']
            
            if metric_value > best_metric:
                best_metric = metric_value
                best_model_name = model_name
        
        print(f"\nBest Model: {best_model_name.upper()} with {best_metric:.4f} {'accuracy' if target_type == 'classification' else 'R²'}")
        
        return models, best_model_name
    
    def plot_final_results(self, target_name, y_test, models, best_model_name, target_type):
        """Plot final results comparing actual vs predicted."""
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (15, 12)
        plt.rcParams['font.size'] = 10
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Final Results: {target_name} ({target_type.title()})', fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted for best model
        best_model = models[best_model_name]
        y_pred = best_model['predictions']
        
        # Create time index for x-axis
        time_index = range(len(y_test))
        
        axes[0, 0].plot(time_index, y_test.values, label='Actual', alpha=0.7, linewidth=2)
        axes[0, 0].plot(time_index, y_pred, label='Predicted', alpha=0.7, linewidth=2)
        axes[0, 0].set_title(f'{best_model_name.upper()} - Actual vs Predicted')
        axes[0, 0].set_xlabel('Time')
        if target_type == "classification":
            axes[0, 0].set_ylabel('Class')
        else:
            axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix (classification) or Scatter Plot (regression)
        if target_type == "classification":
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
            axes[0, 1].set_title(f'{best_model_name.upper()} - Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
        else:
            # Scatter plot for regression
            axes[0, 1].scatter(y_test, y_pred, alpha=0.6)
            axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual')
            axes[0, 1].set_ylabel('Predicted')
            axes[0, 1].set_title(f'{best_model_name.upper()} - Actual vs Predicted Scatter')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model Comparison
        model_names = list(models.keys())
        if target_type == "classification":
            accuracies = [models[name]['accuracy'] for name in model_names]
            metric_name = 'Accuracy'
        else:
            accuracies = [models[name]['r2'] for name in model_names]
            metric_name = 'R²'
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue']
        bars = axes[1, 0].bar(model_names, accuracies, color=colors[:len(model_names)])
        axes[1, 0].set_title(f'Model {metric_name} Comparison')
        axes[1, 0].set_ylabel(metric_name)
        if target_type == "classification":
            axes[1, 0].set_ylim(0, 1)
            # Add target line for classification
            axes[1, 0].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Target (60%)')
        else:
            axes[1, 0].set_ylim(min(0, min(accuracies)), max(accuracies) + 0.1)
        
        # Add metric values on bars
        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        if target_type == "classification":
            axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (for tree-based models) or Residuals (for regression)
        if best_model_name in ['random_forest', 'xgboost']:
            if target_type == "classification":
                feature_importance = best_model['model'].feature_importances_
                feature_names = self.features if hasattr(self, 'features') else [f'Feature_{i}' for i in range(len(feature_importance))]
                
                # Get top 20 features
                top_indices = np.argsort(feature_importance)[-20:]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = feature_importance[top_indices]
                
                axes[1, 1].barh(range(len(top_features)), top_importance, color='lightgreen')
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features)
                axes[1, 1].set_title(f'{best_model_name.upper()} - Top 20 Feature Importance')
                axes[1, 1].set_xlabel('Importance')
            else:
                # Residuals plot for regression
                residuals = y_test - y_pred
                axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title(f'{best_model_name.upper()} - Residuals Plot')
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # For LSTM/GRU, show model info
            if target_type == "classification":
                axes[1, 1].text(0.5, 0.5, f'{best_model_name.upper()}\nDeep Learning Model\nNo feature importance available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title(f'{best_model_name.upper()} - Model Info')
            else:
                # Residuals for LSTM/GRU regression
                residuals = y_test - y_pred
                axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title(f'{best_model_name.upper()} - Residuals Plot')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot with high quality
        output_dir = "ml_outputs_advanced"
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = f"{output_dir}/final_results_{target_name}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save as PDF for better quality
        pdf_filename = f"{output_dir}/final_results_{target_name}.pdf"
        plt.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
        
        # Display the plot
        plt.show()
        
        print(f"Final results plot saved to:")
        print(f"  - {plot_filename}")
        print(f"  - {pdf_filename}")
        
        # Create additional detailed plot
        self.create_detailed_plot(target_name, y_test, models, best_model_name, output_dir, target_type)
    
    def create_detailed_plot(self, target_name, y_test, models, best_model_name, output_dir, target_type):
        """Create additional detailed plots for analysis."""
        plt.figure(figsize=(16, 10))
        
        # Plot all models predictions
        time_index = range(len(y_test))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_index, y_test.values, label='Actual', linewidth=3, color='black', alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, model_info) in enumerate(models.items()):
            if target_type == "classification":
                metric_value = model_info['accuracy']
                metric_name = 'Accuracy'
            else:
                metric_value = model_info['r2']
                metric_name = 'R²'
            
            plt.plot(time_index, model_info['predictions'], 
                    label=f'{model_name.upper()} ({metric_value:.3f})', 
                    alpha=0.7, linewidth=2, color=colors[i % len(colors)])
        
        plt.title(f'All Models Comparison: {target_name} ({target_type.title()})', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        if target_type == "classification":
            plt.ylabel('Class')
        else:
            plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot metric comparison
        plt.subplot(2, 1, 2)
        model_names = list(models.keys())
        if target_type == "classification":
            metric_values = [models[name]['accuracy'] for name in model_names]
            metric_name = 'Accuracy'
            target_line = 0.6
        else:
            metric_values = [models[name]['r2'] for name in model_names]
            metric_name = 'R²'
            target_line = 0.5  # Reasonable R² target
        
        bars = plt.bar(model_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue'][:len(model_names)])
        plt.title(f'Model {metric_name} Comparison', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name)
        
        if target_type == "classification":
            plt.ylim(0, 1)
            plt.axhline(y=target_line, color='red', linestyle='--', alpha=0.7, label=f'Target ({target_line*100}%)')
        else:
            plt.ylim(min(0, min(metric_values)), max(metric_values) + 0.1)
            plt.axhline(y=target_line, color='red', linestyle='--', alpha=0.7, label=f'Target ({target_line})')
        
        # Add metric values on bars
        for bar, val in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_filename = f"{output_dir}/detailed_comparison_{target_name}.png"
        plt.savefig(detailed_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Detailed comparison plot saved to: {detailed_filename}")
    
    def run_complete_analysis(self):
        """Run complete analysis with all improvements."""
        print("=" * 80)
        print("ADVANCED STOCK FORECASTING ANALYSIS")
        print("=" * 80)
        print(f"Target: 60% Directional Accuracy")
        print(f"Symbol: {self.symbol}")
        print(f"Data Period: {self.start_date} to {self.end_date}")
        print("=" * 80)
        
        # Step 1: Fetch and prepare data
        self.fetch_and_prepare_data()
        
        # Step 2: Prepare targets and features
        self.prepare_targets_and_features()
        
        # Step 3: Walk-forward analysis for each target
        walk_forward_results = {}
        for target_name, target_info in self.targets.items():
            print(f"\n{'='*60}")
            print(f"ANALYZING TARGET: {target_info['description']} ({target_info['type']})")
            print(f"{'='*60}")
            
            # Walk-forward analysis
            wf_results = self.walk_forward_analysis(target_name)
            walk_forward_results[target_name] = wf_results
            
            # Train final models
            models, best_model = self.train_final_models(target_name)
            
            # Save models
            output_dir = "ml_outputs_advanced"
            os.makedirs(output_dir, exist_ok=True)
            
            for model_name, model_info in models.items():
                self.save_model(model_info['model'], model_name, target_name, model_info)
            
            # Store results
            self.results[target_name] = {
                'walk_forward': wf_results,
                'final_models': models,
                'best_model': best_model
            }
        
        # Step 4: Generate comprehensive report
        self.generate_comprehensive_report()
        
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        report = {
            'symbol': self.symbol,
            'data_period': f"{self.start_date} to {self.end_date}",
            'total_data_points': len(self.data),
            'total_features': len(self.features),
            'targets_analyzed': list(self.targets.keys()),
            'results': {}
        }
        
        for target_name, target_info in self.targets.items():
            if target_name in self.results:
                target_results = self.results[target_name]
                target_type = target_info["type"]
                
                # Get best metric based on target type
                best_metric = 0
                best_model_name = None
                
                for model_name, model_info in target_results['final_models'].items():
                    if target_type == "classification":
                        metric_value = model_info['accuracy']
                    else:  # regression
                        metric_value = model_info['r2']
                    
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_model_name = model_name
                
                # Determine if target is met
                if target_type == "classification":
                    target_met = best_metric >= 0.60
                    metric_name = "accuracy"
                else:  # regression
                    target_met = best_metric >= 0.50  # Reasonable R² target
                    metric_name = "r2"
                
                report['results'][target_name] = {
                    'description': target_info['description'],
                    'type': target_type,
                    'best_model': best_model_name,
                    'best_metric': best_metric,
                    'metric_name': metric_name,
                    'target_met': target_met,
                    'walk_forward_results': target_results['walk_forward']
                }
        
        # Save report
        output_dir = "ml_outputs_advanced"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Symbol: {self.symbol}")
        print(f"Data Points: {len(self.data)}")
        print(f"Features: {len(self.features)}")
        print(f"Targets: {len(self.targets)}")
        
        targets_met = 0
        for target_name, target_results in report['results'].items():
            metric_value = target_results['best_metric']
            target_met = target_results['target_met']
            metric_name = target_results['metric_name']
            target_type = target_results['type']
            status = "✅" if target_met else "❌"
            
            print(f"{status} {target_name} ({target_type}): {metric_value:.4f} {metric_name} ({'MET' if target_met else 'NOT MET'})")
            
            if target_met:
                targets_met += 1
        
        print(f"\nOverall: {targets_met}/{len(self.targets)} targets met their goals")
        print(f"Report saved to: {output_dir}/comprehensive_report.json")

    def get_model_path(self, model_name, target_name):
        """Get the file path for a saved model."""
        return os.path.join(self.model_dir, f"{model_name}_{target_name}.pkl")
    
    def get_pytorch_model_path(self, model_name, target_name):
        """Get the file path for a saved PyTorch model."""
        return os.path.join(self.model_dir, f"{model_name}_{target_name}.pt")
    
    def save_model(self, model, model_name, target_name, metadata=None):
        """Save a trained model to disk."""
        if hasattr(model, 'save') and callable(getattr(model, 'save')):
            # PyTorch model
            model_path = self.get_pytorch_model_path(model_name, target_name)
            model.save(model_path)
            if metadata:
                model.save_metadata(model_path, metadata)
        else:
            # Scikit-learn/XGBoost model
            model_path = self.get_model_path(model_name, target_name)
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model {model_name} for {target_name} saved successfully")
    
    def load_model(self, model_name, target_name, model_class=None, **kwargs):
        """Load a trained model from disk."""
        if model_name in ['lstm', 'gru'] and PYTORCH_AVAILABLE:
            # PyTorch model
            model_path = self.get_pytorch_model_path(model_name, target_name)
            if os.path.exists(model_path):
                # Create a new trainer with the same architecture
                sequence_length = 20
                num_classes = kwargs.get('num_classes', 2)
                input_size = kwargs.get('input_size', 161)
                
                if model_name == 'lstm':
                    trainer = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, input_size), num_classes))
                else:  # gru
                    trainer = PyTorchTrainer(create_pytorch_gru_model((sequence_length, input_size), num_classes))
                
                trainer.load(model_path)
                return trainer
        else:
            # Scikit-learn/XGBoost model
            model_path = self.get_model_path(model_name, target_name)
            if os.path.exists(model_path):
                return joblib.load(model_path)
        
        return None
    
    def model_exists(self, model_name, target_name):
        """Check if a saved model exists."""
        if model_name in ['lstm', 'gru'] and PYTORCH_AVAILABLE:
            model_path = self.get_pytorch_model_path(model_name, target_name)
        else:
            model_path = self.get_model_path(model_name, target_name)
        
        return os.path.exists(model_path)
    
    def create_lstm_model(self, input_shape, num_classes=2):
        """Create LSTM model for sequence prediction."""
        if PYTORCH_AVAILABLE:
            # Use PyTorch LSTM
            input_size = input_shape[1]
            model = create_pytorch_lstm_model((input_shape[0], input_size), num_classes)
            return model
        elif TENSORFLOW_AVAILABLE:
            # Use TensorFlow LSTM
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        else:
            raise ImportError("Neither PyTorch nor TensorFlow available for LSTM models")
    
    def create_gru_model(self, input_shape, num_classes=2):
        """Create GRU model for sequence prediction."""
        if PYTORCH_AVAILABLE:
            # Use PyTorch GRU
            input_size = input_shape[1]
            model = create_pytorch_gru_model((input_shape[0], input_size), num_classes)
            return model
        elif TENSORFLOW_AVAILABLE:
            # Use TensorFlow GRU
            model = Sequential([
                GRU(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                GRU(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        else:
            raise ImportError("Neither PyTorch nor TensorFlow available for GRU models")
    
    def prepare_sequences(self, data, target_col, sequence_length=20):
        """Prepare sequences for LSTM/GRU models."""
        X, y = [], []
        
        # Ensure we have enough data
        if len(data) <= sequence_length:
            return np.array(X), np.array(y)
        
        for i in range(sequence_length, len(data)):
            # Extract features (exclude target column)
            features = data.iloc[i-sequence_length:i].drop(columns=[target_col], errors='ignore').values
            target = data.iloc[i][target_col]
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)

def main():
    """Main function to run advanced analysis."""
    # Test with API stock
    forecaster = AdvancedStockForecaster("API", "2021-01-01", "2024-12-31")
    results = forecaster.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main() 