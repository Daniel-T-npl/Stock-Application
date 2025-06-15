import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WriteApi
from influxdb_client.client.write_api import SYNCHRONOUS

# Load environment variables
load_dotenv()

client = InfluxDBClient(
    url=os.environ["INFLUXDB_URL"],
    token=os.environ["INFLUXDB_TOKEN"],
    org=os.environ["INFLUXDB_ORG"],
)

write_api: WriteApi = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api() 