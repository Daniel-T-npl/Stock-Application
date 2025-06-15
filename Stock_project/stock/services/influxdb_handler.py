import os
from influxdb_client import InfluxDBClient
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

class InfluxDBHandler:
    def __init__(self):
        # Get InfluxDB configuration
        self.url = os.getenv('INFLUXDB_URL')
        self.token = os.getenv('INFLUXDB_TOKEN')
        self.org = os.getenv('INFLUXDB_ORG')
        self.bucket = os.getenv('INFLUXDB_BUCKET', 'stock_data')

        # Log configuration (without token)
        logger.info(f"InfluxDB URL: {self.url}")
        logger.info(f"InfluxDB Org: {self.org}")
        logger.info(f"InfluxDB Bucket: {self.bucket}")
        logger.info("Token: [HIDDEN]")

        if not all([self.url, self.token, self.org]):
            missing = []
            if not self.url: missing.append('INFLUXDB_URL')
            if not self.token: missing.append('INFLUXDB_TOKEN')
            if not self.org: missing.append('INFLUXDB_ORG')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Initialize client
        logger.info("Initializing InfluxDB client...")
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()
        logger.info("InfluxDB client and query API initialized successfully")

    def get_client(self):
        return self.client

    def close(self):
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("InfluxDB client connection closed") 