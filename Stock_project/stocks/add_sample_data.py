import os
from datetime import datetime, timedelta
import random
from .firebase_client import db

def generate_sample_data(symbol, start_time, num_points=100):
    """Generate sample time series data for a stock symbol"""
    base_price = random.uniform(100, 500)
    data_points = []
    
    for i in range(num_points):
        timestamp = start_time + timedelta(minutes=i)
        # Generate random price movements
        price_change = random.uniform(-2, 2)
        current_price = base_price + price_change
        base_price = current_price
        
        data_point = {
            'symbol': symbol,
            'timestamp': timestamp.timestamp(),
            'open_price': round(current_price, 2),
            'high_price': round(current_price + random.uniform(0, 1), 2),
            'low_price': round(current_price - random.uniform(0, 1), 2),
            'close_price': round(current_price + random.uniform(-0.5, 0.5), 2),
            'volume': random.randint(1000, 10000)
        }
        data_points.append(data_point)
    
    return data_points

def add_sample_data_to_firestore():
    """Add sample time series data to Firestore"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_time = datetime.now() - timedelta(days=1)
    
    for symbol in symbols:
        data_points = generate_sample_data(symbol, start_time)
        
        # Create a batch write
        batch = db.batch()
        
        # Add each data point to Firestore
        for data_point in data_points:
            doc_ref = db.collection('stock_time_series').document()
            batch.set(doc_ref, data_point)
        
        # Commit the batch
        batch.commit()
        print(f"Added {len(data_points)} data points for {symbol}")

if __name__ == "__main__":
    add_sample_data_to_firestore() 