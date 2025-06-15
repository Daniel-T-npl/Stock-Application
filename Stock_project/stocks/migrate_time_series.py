import os
from datetime import datetime
from dotenv import load_dotenv
from influxdb_client import Point
from stocks.firebase_client import db
from stocks.influx_client import write_api

# Load environment variables
load_dotenv()

def migrate_time_series_data():
    """
    Migrate time series stock data from Firestore to InfluxDB
    """
    try:
        # Get all documents from the stock_time_series collection
        time_series_ref = db.collection('stock_time_series')
        time_series_docs = time_series_ref.stream()
        
        # Counter for migrated documents
        migrated_count = 0
        error_count = 0
        
        for doc in time_series_docs:
            try:
                data = doc.to_dict()
                symbol = data.get('symbol')
                
                if not symbol:
                    print(f"Skipping document {doc.id}: No symbol found")
                    continue
                
                # Create a point for each time series data entry
                point = Point("stock_time_series") \
                    .tag("symbol", symbol) \
                    .field("open_price", float(data.get('open_price', 0))) \
                    .field("high_price", float(data.get('high_price', 0))) \
                    .field("low_price", float(data.get('low_price', 0))) \
                    .field("close_price", float(data.get('close_price', 0))) \
                    .field("volume", int(data.get('volume', 0)))
                
                # Use the timestamp from the data
                if 'timestamp' in data:
                    point.time(datetime.fromtimestamp(data['timestamp']))
                else:
                    point.time(datetime.utcnow())
                
                # Write to InfluxDB
                write_api.write(
                    bucket=os.getenv('INFLUX_BUCKET'),
                    record=point
                )
                
                migrated_count += 1
                if migrated_count % 100 == 0:
                    print(f"Migrated {migrated_count} documents...")
                
            except Exception as e:
                error_count += 1
                print(f"Error migrating document {doc.id}: {str(e)}")
        
        print(f"\nMigration completed:")
        print(f"Successfully migrated: {migrated_count} documents")
        print(f"Errors encountered: {error_count}")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_time_series_data() 