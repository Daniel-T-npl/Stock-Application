import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path('.') / '.env'
logger.info(f"Looking for .env file at: {env_path.absolute()}")
load_dotenv(dotenv_path=env_path)

def test_influxdb_connection():
    """Test InfluxDB connection and basic operations"""
    try:
        # Get InfluxDB configuration
        url = os.getenv('INFLUXDB_URL')
        token = os.getenv('INFLUXDB_TOKEN')
        org = os.getenv('INFLUXDB_ORG')
        bucket = os.getenv('INFLUXDB_BUCKET')

        # Log configuration (without token)
        logger.info(f"InfluxDB URL: {url}")
        logger.info(f"InfluxDB Org: {org}")
        logger.info(f"InfluxDB Bucket: {bucket}")
        logger.info("Token: [HIDDEN]")

        if not all([url, token, org, bucket]):
            missing = []
            if not url: missing.append('INFLUXDB_URL')
            if not token: missing.append('INFLUXDB_TOKEN')
            if not org: missing.append('INFLUXDB_ORG')
            if not bucket: missing.append('INFLUXDB_BUCKET')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Initialize client
        logger.info("Initializing InfluxDB client...")
        client = InfluxDBClient(url=url, token=token, org=org)
        
        # Test write
        logger.info("Testing write operation...")
        write_api = client.write_api(write_options=SYNCHRONOUS)
        point = Point("test_measurement") \
            .tag("test", "true") \
            .field("value", 1.0)
        write_api.write(bucket=bucket, record=point)
        
        # Test query
        logger.info("Testing query operation...")
        query_api = client.query_api()
        query = f'from(bucket:"{bucket}") |> range(start: -1h) |> filter(fn: (r) => r["_measurement"] == "test_measurement")'
        result = query_api.query(query=query, org=org)
        
        if len(result) > 0:
            logger.info("Successfully wrote and read test data!")
            logger.info(f"Query result: {result[0].records[0].values}")
        else:
            logger.error("Failed to read test data")
            
        # Clean up test data
        logger.info("Cleaning up test data...")
        delete_api = client.delete_api()
        delete_api.delete(
            start="1970-01-01T00:00:00Z",
            stop="2024-12-31T23:59:59Z",
            predicate='_measurement="test_measurement"',
            bucket=bucket
        )
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing InfluxDB: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()
            logger.info("InfluxDB client connection closed")

if __name__ == "__main__":
    success = test_influxdb_connection()
    if not success:
        exit(1) 