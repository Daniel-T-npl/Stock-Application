import os
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from django.conf import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfluxDBClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InfluxDBClientSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.url = settings.INFLUXDB_URL
        self.token = settings.INFLUXDB_TOKEN
        self.org = settings.INFLUXDB_ORG
        self.bucket = settings.INFLUXDB_BUCKET

        if not all([self.url, self.token, self.org, self.bucket]):
            raise ValueError("INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, and INFLUXDB_BUCKET must be set in .env file and configured in settings.py")

        try:
            assert self.url is not None
            assert self.token is not None
            assert self.org is not None
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.query_api = self.client.query_api()
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info("InfluxDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise

        self._initialized = True

    def get_query_api(self):
        return self.query_api

    def get_write_api(self):
        return self.write_api

    def get_bucket(self):
        return self.bucket

    def get_org(self):
        return self.org
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("InfluxDB client connection closed.")

# For ease of use, you can export an instance
influx_client = InfluxDBClientSingleton()

# You can keep your existing helper functions here, but they should use the singleton instance.
# For example:
from influxdb_client.client.write.point import Point
from datetime import datetime

def write_stock_data(date: str, stock_symbol: str, data: dict):
    """
    Write stock data to InfluxDB using the singleton client.
    """
    try:
        write_api = influx_client.get_write_api()
        bucket = influx_client.get_bucket()
        
        point = Point("stock_data") \
            .tag("symbol", stock_symbol) \
            .field("open", float(data['open'])) \
            .field("high", float(data['high'])) \
            .field("low", float(data['low'])) \
            .field("close", float(data['close'])) \
            .field("volume", float(data['volume']))

        if data.get('turnover') is not None:
            point = point.field("turnover", float(data['turnover']))

        dt = datetime.strptime(date, '%Y-%m-%d')
        timestamp = int(dt.timestamp() * 1e9)
        point = point.time(timestamp)

        write_api.write(bucket=bucket, record=point)
        logger.debug(f"Wrote data for {stock_symbol} on {date}")

    except Exception as e:
        logger.error(f"Error writing data for {stock_symbol} on {date}: {str(e)}")
        raise

def get_all_symbols():
    """
    Query InfluxDB to get all unique stock symbols.
    """
    try:
        query_api = influx_client.get_query_api()
        bucket = influx_client.get_bucket()
        org = influx_client.get_org()

        query = f'''
        import "influxdata/influxdb/schema"
        schema.tagValues(
            bucket: "{bucket}",
            tag: "symbol"
        )
        '''
        result = query_api.query(org=org, query=query)
        symbols = [record.get_value() for table in result for record in table.records]
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols from InfluxDB: {str(e)}")
        return []


class InfluxDBHandler:
    """
    Handler class that provides the interface expected by the scripts.
    """
    def __init__(self):
        self.client = influx_client
    
    def get_all_symbols(self):
        """Get all stock symbols from InfluxDB."""
        return get_all_symbols()
    
    def close(self):
        """Close the InfluxDB connection."""
        self.client.close() 