import os
from datetime import datetime, timedelta
import requests
from influxdb_client.client.write.point import Point
from analysis.influx_client import influx_client
import pandas as pd
from django.conf import settings

class StockService:
    def __init__(self):
        self.alpha_vantage_api_key = settings.ALPHA_VANTAGE_API_KEY
        self.bucket = settings.INFLUXDB_BUCKET
        self.org = settings.INFLUXDB_ORG
        self.query_api = influx_client.get_query_api()

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
            write_api = influx_client.get_write_api()
            assert self.bucket is not None
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
            query_api = influx_client.get_query_api()
            assert self.bucket is not None
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

    def get_stock_data_df(self, ticker: str, start: str = "2021-01-03", stop: str = "2025-06-20") -> pd.DataFrame:
        """
        Fetches stock data for a given ticker and returns it as a Pandas DataFrame.
        The DataFrame is indexed by time.
        """
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}T00:00:00Z, stop: {stop}T00:00:00Z)
          |> filter(fn: (r) => r["_measurement"] == "stock_data")
          |> filter(fn: (r) => r["symbol"] == "{ticker}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "open", "high", "low", "close", "volume", "turnover"])
          |> sort(columns: ["_time"])
        '''
        try:
            result = self.query_api.query_data_frame(query=query)
            if isinstance(result, list):
                if not result:
                    return pd.DataFrame()
                df = pd.concat(result)
            else:
                df = result

            if df.empty:
                return df
                
            df.rename(columns={"_time": "time"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            # Drop unnecessary columns often returned by InfluxDB
            df.drop(columns=['result', 'table'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def update_stock_data(self, symbol):
        """
        Fetch and store latest stock data
        """
        data = self.fetch_stock_data(symbol)
        if data:
            return self.store_stock_data(symbol, data)
        return False 