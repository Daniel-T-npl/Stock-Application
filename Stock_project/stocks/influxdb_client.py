import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class InfluxDBHandler:
    def __init__(self):
        # Get InfluxDB configuration from environment variables
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN')
        self.org = os.getenv('INFLUXDB_ORG', 'my-org')
        self.bucket = os.getenv('INFLUXDB_BUCKET', 'stock_data')
        
        if not self.token:
            raise ValueError("INFLUXDB_TOKEN environment variable is required")
        
        # Initialize the client
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        logger.info("InfluxDB client initialized successfully")

    def write_stock_data(self, date, stock_symbol, data):
        """
        Write stock data to InfluxDB
        
        Args:
            date (str): Date in YYYY-MM-DD format
            stock_symbol (str): Stock symbol
            data (dict): Stock data containing open, high, low, close, volume, etc.
        """
        try:
            point = Point("stock_data") \
                .tag("symbol", stock_symbol) \
                .field("open", float(data['open'])) \
                .field("high", float(data['high'])) \
                .field("low", float(data['low'])) \
                .field("close", float(data['close'])) \
                .field("volume", float(data['volume']))
            
            if data.get('turnover') is not None:
                point = point.field("turnover", float(data['turnover']))
            
            # Convert date string to datetime and then to timestamp
            dt = datetime.strptime(date, '%Y-%m-%d')
            timestamp = int(dt.timestamp() * 1e9)  # Convert to nanoseconds
            point = point.time(timestamp)
            
            self.write_api.write(bucket=self.bucket, record=point)
            logger.debug(f"Wrote data for {stock_symbol} on {date}")
            
        except Exception as e:
            logger.error(f"Error writing data for {stock_symbol} on {date}: {str(e)}")
            raise

    def close(self):
        """Close the InfluxDB client connection"""
        self.client.close()
        logger.info("InfluxDB client connection closed") 