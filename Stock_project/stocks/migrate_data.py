import os
from datetime import datetime
from dotenv import load_dotenv
from influxdb_client import Point
from .firebase_client import db
from .influx_client import write_api

# Load environment variables
load_dotenv()

def migrate_stock_data():
    """
    Migrate stock data from Firestore to InfluxDB
    """
    try:
        # Get all documents from the stocks collection
        stocks_ref = db.collection('stocks')
        stocks = stocks_ref.stream()
        
        # Counter for migrated documents
        migrated_count = 0
        error_count = 0
        
        for stock in stocks:
            try:
                stock_data = stock.to_dict()
                symbol = stock_data.get('symbol')
                
                if not symbol:
                    print(f"Skipping document {stock.id}: No symbol found")
                    continue
                
                # Create a point for each stock data entry
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .field("open", float(stock_data.get('open', 0))) \
                    .field("high", float(stock_data.get('high', 0))) \
                    .field("low", float(stock_data.get('low', 0))) \
                    .field("close", float(stock_data.get('close', 0))) \
                    .field("volume", int(stock_data.get('volume', 0)))
                
                # If there's a timestamp in the data, use it
                if 'timestamp' in stock_data:
                    point.time(datetime.fromtimestamp(stock_data['timestamp']))
                else:
                    point.time(datetime.utcnow())
                
                # Write to InfluxDB
                write_api.write(
                    bucket=os.getenv('INFLUX_BUCKET'),
                    record=point
                )
                
                migrated_count += 1
                print(f"Migrated data for {symbol}")
                
            except Exception as e:
                error_count += 1
                print(f"Error migrating document {stock.id}: {str(e)}")
        
        print(f"\nMigration completed:")
        print(f"Successfully migrated: {migrated_count} documents")
        print(f"Errors encountered: {error_count}")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_stock_data() 