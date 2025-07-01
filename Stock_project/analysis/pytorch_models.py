#!/usr/bin/env python3
"""
PyTorch-based LSTM and GRU models for stock prediction
Alternative to TensorFlow when not available
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import json

class PyTorchLSTM(nn.Module):
    """PyTorch LSTM model for sequence prediction."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        if num_classes == 2:
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
            self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        if self.num_classes == 2:
            out = self.sigmoid(out)
        else:
            out = self.softmax(out)
        
        return out

class PyTorchGRU(nn.Module):
    """PyTorch GRU model for sequence prediction."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.2):
        super(PyTorchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        if num_classes == 2:
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
            self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        if self.num_classes == 2:
            out = self.sigmoid(out)
        else:
            out = self.softmax(out)
        
        return out

class PyTorchTrainer:
    """Trainer class for PyTorch models."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.scaler = StandardScaler()
        
    def prepare_sequences(self, data, target_col, sequence_length=20):
        """Prepare sequences for PyTorch models."""
        X, y = [], []
        
        # Ensure we have enough data
        if len(data) <= sequence_length:
            print(f"Insufficient data for sequences: {len(data)} <= {sequence_length}")
            return None, None
        
        # If data is a DataFrame with target column included
        if isinstance(data, pd.DataFrame) and target_col in data.columns:
            for i in range(sequence_length, len(data)):
                # Extract features (exclude target column)
                features = data.iloc[i-sequence_length:i].drop(columns=[target_col], errors='ignore').values
                target = data.iloc[i][target_col]
                X.append(features)
                y.append(target)
        else:
            # If data is already feature-only (X) and target is separate
            for i in range(sequence_length, len(data)):
                features = data.iloc[i-sequence_length:i].values
                target = target_col[i]  # target_col is actually the target array
                X.append(features)
                y.append(target)
        
        if len(X) == 0:
            print("No sequences created")
            return None, None
            
        return np.array(X), np.array(y)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=100, batch_size=32, learning_rate=0.001, patience=10):
        """Train the PyTorch model."""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Loss function and optimizer
        if self.model.num_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_tensor)
            if self.model.num_classes == 2:
                loss = criterion(outputs.squeeze(), y_train_tensor)
            else:
                loss = criterion(outputs, y_train_tensor.long())
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    if self.model.num_classes == 2:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
                    else:
                        val_loss = criterion(val_outputs, y_val_tensor.long())
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
        
        # Load best model if validation was used
        if X_val is not None and y_val is not None and os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
            os.remove('best_model.pth')
    
    def predict(self, X):
        """Make predictions with the trained model."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.model.num_classes == 2:
                predictions = (outputs.squeeze() > 0.5).float().cpu().numpy()
            else:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
        
        return probabilities

    def save(self, path):
        """Save the trained model to disk."""
        # Save model state dict
        torch.save(self.model.state_dict(), path)
        
        # Save scaler if it exists
        scaler_path = path.replace('.pt', '_scaler.pkl')
        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to: {path}")
    
    def load(self, path):
        """Load a trained model from disk."""
        # Load model state dict
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        
        # Load scaler if it exists
        scaler_path = path.replace('.pt', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from: {path}")
    
    def save_metadata(self, path, metadata):
        """Save model metadata (training info, metrics, etc.)."""
        metadata_path = path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to: {metadata_path}")
    
    def load_metadata(self, path):
        """Load model metadata."""
        metadata_path = path.replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

def create_pytorch_lstm_model(input_shape, num_classes=2):
    """Create a PyTorch LSTM model."""
    input_size = input_shape[1]
    model = PyTorchLSTM(input_size=input_size, num_classes=num_classes)
    return model

def create_pytorch_gru_model(input_shape, num_classes=2):
    """Create a PyTorch GRU model."""
    input_size = input_shape[1]
    model = PyTorchGRU(input_size=input_size, num_classes=num_classes)
    return model

def train_pytorch_models(X_train, y_train, X_test, y_test, target_type="classification", 
                        sequence_length=20, epochs=100):
    """Train PyTorch LSTM and GRU models."""
    
    results = {}
    
    # Determine number of classes
    if target_type == "classification":
        num_classes = len(np.unique(y_train))
    else:
        num_classes = 1  # Regression
    
    # Prepare sequences
    trainer_lstm = PyTorchTrainer(create_pytorch_lstm_model((sequence_length, X_train.shape[1]), num_classes))
    trainer_gru = PyTorchTrainer(create_pytorch_gru_model((sequence_length, X_train.shape[1]), num_classes))
    
    # Prepare data
    X_train_seq, y_train_seq = trainer_lstm.prepare_sequences(
        pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1), 
        len(X_train.columns), sequence_length
    )
    X_test_seq, y_test_seq = trainer_lstm.prepare_sequences(
        pd.concat([pd.DataFrame(X_test), pd.Series(y_test)], axis=1), 
        len(X_test.columns), sequence_length
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
    X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
    
    # Train LSTM
    print("Training PyTorch LSTM...")
    trainer_lstm.train_model(X_train_scaled, y_train_seq, epochs=epochs)
    lstm_pred = trainer_lstm.predict(X_test_scaled)
    
    # Train GRU
    print("Training PyTorch GRU...")
    trainer_gru.train_model(X_train_scaled, y_train_seq, epochs=epochs)
    gru_pred = trainer_gru.predict(X_test_scaled)
    
    # Evaluate results
    if target_type == "classification":
        results['lstm'] = {
            'accuracy': accuracy_score(y_test_seq, lstm_pred),
            'f1': f1_score(y_test_seq, lstm_pred, average='weighted'),
            'precision': precision_score(y_test_seq, lstm_pred, average='weighted'),
            'recall': recall_score(y_test_seq, lstm_pred, average='weighted'),
            'predictions': lstm_pred
        }
        
        results['gru'] = {
            'accuracy': accuracy_score(y_test_seq, gru_pred),
            'f1': f1_score(y_test_seq, gru_pred, average='weighted'),
            'precision': precision_score(y_test_seq, gru_pred, average='weighted'),
            'recall': recall_score(y_test_seq, gru_pred, average='weighted'),
            'predictions': gru_pred
        }
    else:
        results['lstm'] = {
            'r2': r2_score(y_test_seq, lstm_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
            'mae': mean_absolute_error(y_test_seq, lstm_pred),
            'predictions': lstm_pred
        }
        
        results['gru'] = {
            'r2': r2_score(y_test_seq, gru_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_seq, gru_pred)),
            'mae': mean_absolute_error(y_test_seq, gru_pred),
            'predictions': gru_pred
        }
    
    return results, trainer_lstm, trainer_gru, scaler 