import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import re
from datetime import datetime
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
from dotenv import load_dotenv
from analysis.influx_client import InfluxDBHandler

# Load environment variables
load_dotenv()

def clean_numeric_value(value):
    """
    Remove commas from quoted numeric values and convert to float.
    Example: "1,100.50" -> 1100.50
    """
    if not value or value == '':
        return None
    
    # Remove quotes and commas, then convert to float
    cleaned = value.strip().replace('"', '').replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return None

def parse_date(date_str):
    """
    Parse date string in format YYYY/MM/DD to datetime object
    """
    try:
        return datetime.strptime(date_str, '%Y/%m/%d')
    except ValueError:
        print(f"Warning: Could not parse date: {date_str}")
        return None

def migrate_csv_to_influxdb(csv_file_path, batch_size=1000):
    """
    Migrate stock data from CSV to InfluxDB
    """
    # InfluxDB configuration
    url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    token = os.getenv('INFLUXDB_TOKEN')
    org = os.getenv('INFLUXDB_ORG')
    bucket = os.getenv('INFLUXDB_BUCKET', 'stock_data')
    
    if not all([url, token, org]):
        print("Error: Missing InfluxDB configuration. Please set INFLUXDB_URL, INFLUXDB_TOKEN, and INFLUXDB_ORG environment variables.")
        return
    
    # Initialize InfluxDB client
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    try:
        # Read CSV file
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            points = []
            total_rows = 0
            successful_rows = 0
            
            for row in csv_reader:
                total_rows += 1
                
                # Clean and parse data
                symbol = row['symbol'].strip()
                date_obj = parse_date(row['date'])
                
                if not date_obj:
                    print(f"Skipping row {total_rows}: Invalid date format")
                    continue
                
                # Clean numeric values
                open_price = clean_numeric_value(row['open'])
                high_price = clean_numeric_value(row['high'])
                low_price = clean_numeric_value(row['low'])
                close_price = clean_numeric_value(row['close'])
                volume = clean_numeric_value(row['volume'])
                turnover = clean_numeric_value(row['turnover'])
                
                # Skip row if essential data is missing
                if not all([open_price, high_price, low_price, close_price]):
                    print(f"Skipping row {total_rows}: Missing essential price data")
                    continue
                
                # Create InfluxDB point
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .field("open", open_price) \
                    .field("high", high_price) \
                    .field("low", low_price) \
                    .field("close", close_price) \
                    .time(date_obj)
                
                # Add optional fields if available
                if volume is not None:
                    point = point.field("volume", volume)
                if turnover is not None:
                    point = point.field("turnover", turnover)
                
                points.append(point)
                successful_rows += 1
                
                # Write batch when we reach batch_size
                if len(points) >= batch_size:
                    try:
                        write_api.write(bucket=bucket, record=points)
                        print(f"Written batch of {len(points)} records. Total successful: {successful_rows}")
                        points = []
                    except Exception as e:
                        print(f"Error writing batch: {e}")
                        points = []
            
            # Write remaining points
            if points:
                try:
                    write_api.write(bucket=bucket, record=points)
                    print(f"Written final batch of {len(points)} records")
                except Exception as e:
                    print(f"Error writing final batch: {e}")
            
            print(f"\nMigration completed!")
            print(f"Total rows processed: {total_rows}")
            print(f"Successful rows migrated: {successful_rows}")
            print(f"Failed rows: {total_rows - successful_rows}")
            
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        # Close InfluxDB client
        client.close()

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = "stock_data.csv"
    
    print("Starting CSV to InfluxDB migration...")
    print(f"Reading from: {csv_file_path}")
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} does not exist!")
    else:
        migrate_csv_to_influxdb(csv_file_path) 