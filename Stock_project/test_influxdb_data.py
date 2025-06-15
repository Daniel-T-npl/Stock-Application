import os
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path('.') / '.env'
logger.info(f"Looking for .env file at: {env_path.absolute()}")
load_dotenv(dotenv_path=env_path)

def test_influxdb_data():
    try:
        # Get InfluxDB configuration
        url = os.getenv('INFLUXDB_URL')
        token = os.getenv('INFLUXDB_TOKEN')
        org = os.getenv('INFLUXDB_ORG')
        bucket = os.getenv('INFLUXDB_BUCKET', 'stock_data')

        # Log configuration (without token)
        logger.info(f"InfluxDB URL: {url}")
        logger.info(f"InfluxDB Org: {org}")
        logger.info(f"InfluxDB Bucket: {bucket}")
        logger.info("Token: [HIDDEN]")

        if not all([url, token, org]):
            missing = []
            if not url: missing.append('INFLUXDB_URL')
            if not token: missing.append('INFLUXDB_TOKEN')
            if not org: missing.append('INFLUXDB_ORG')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Initialize client
        logger.info("Initializing InfluxDB client...")
        client = InfluxDBClient(url=url, token=token, org=org)

        # Query for stock_data measurement in the specified date range
        start = "2021-01-03T00:00:00Z"
        stop = "2023-12-31T23:59:59Z"
        stock_query = f'''
        from(bucket: "stock_data")
            |> range(start: {start}, stop: {stop})
            |> filter(fn: (r) => r["_measurement"] == "stock_data")
            |> limit(n: 20)
        '''
        logger.info(f"\nQuerying stock_data measurement from {start} to {stop}...")
        stock_result = client.query_api().query(stock_query)
        
        # Log the stock data structure
        all_symbols = set()
        for table in stock_result:
            logger.info(f"\nTable columns: {table.columns}")
            for record in table.records:
                symbol = record.values.get('symbol', None)
                if symbol:
                    all_symbols.add(symbol)
                logger.info(f"Record values: {record.values}")
                logger.info(f"Record symbol: {symbol}")
                logger.info(f"Record measurement: {record.values.get('_measurement', 'No measurement')}")
                logger.info(f"Record field: {record.values.get('_field', 'No field')}")
                logger.info("---")
        logger.info(f"\nFound {len(all_symbols)} unique symbols in sample records.")
        if all_symbols:
            logger.info(f"First 10 symbols: {list(all_symbols)[:10]}")

        # Try to get distinct symbols from stock_data measurement in the date range
        symbols_query = f'''
        from(bucket: "stock_data")
            |> range(start: {start}, stop: {stop})
            |> filter(fn: (r) => r["_measurement"] == "stock_data")
            |> keep(columns: ["symbol"])
            |> group()
            |> distinct(column: "symbol")
            |> sort(columns: ["symbol"])
        '''
        logger.info("\nChecking for symbols in stock_data measurement (date range)...")
        symbols_result = client.query_api().query(symbols_query)
        symbols = []
        for table in symbols_result:
            for record in table.records:
                symbol = record.values.get('symbol')
                if symbol:
                    symbols.append(symbol)
        logger.info(f"Found {len(symbols)} symbols in date range.")
        if symbols:
            logger.info(f"First 10 symbols: {symbols[:10]}")

        return True
        
    except Exception as e:
        logger.error(f"Error testing InfluxDB: {str(e)}", exc_info=True)
        return False
    finally:
        if 'client' in locals():
            client.close()
            logger.info("InfluxDB client connection closed")

if __name__ == "__main__":
    success = test_influxdb_data()
    if not success:
        exit(1) 