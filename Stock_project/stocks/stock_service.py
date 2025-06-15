import os
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from influxdb_client import Point
from .influx_client import client, write_api, query_api

# Load environment variables
load_dotenv()

class StockService:
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.bucket = os.getenv('INFLUX_BUCKET')
        self.org = os.getenv('INFLUX_ORG')

    def fetch_stock_data(self, symbol, interval='1min'):
        """
        Fetch stock data from Alpha Vantage API
        """
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={self.alpha_vantage_api_key}'
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(data['Error Message'])
                
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                raise Exception('Invalid response format')
                
            return data[time_series_key]
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None

    def store_stock_data(self, symbol, data):
        """
        Store stock data in InfluxDB
        """
        try:
            for timestamp, values in data.items():
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .field("open", float(values['1. open'])) \
                    .field("high", float(values['2. high'])) \
                    .field("low", float(values['3. low'])) \
                    .field("close", float(values['4. close'])) \
                    .field("volume", int(values['5. volume'])) \
                    .time(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))

                write_api.write(bucket=self.bucket, record=point)
            
            return True
        except Exception as e:
            print(f"Error storing stock data: {str(e)}")
            return False

    def get_stock_data(self, symbol, start_time=None, end_time=None):
        """
        Retrieve stock data from InfluxDB
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        if not end_time:
            end_time = datetime.utcnow()

        try:
            query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                    |> filter(fn: (r) => r["_measurement"] == "stock_data")
                    |> filter(fn: (r) => r["symbol"] == "{symbol}")
            '''
            
            result = query_api.query(query)
            return result
        except Exception as e:
            print(f"Error retrieving stock data: {str(e)}")
            return None

    def update_stock_data(self, symbol):
        """
        Fetch and store latest stock data
        """
        data = self.fetch_stock_data(symbol)
        if data:
            return self.store_stock_data(symbol, data)
        return False 