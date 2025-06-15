import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

# Load environment variables
load_dotenv()

def test_connection():
    try:
        # Create client
        client = InfluxDBClient(
            url=os.environ["INFLUX_URL"],
            token=os.environ["INFLUX_TOKEN"],
            org=os.environ["INFLUX_ORG"]
        )
        
        # Create write API
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Create a test point
        point = Point("test_measurement") \
            .tag("test", "connection") \
            .field("value", 1.0) \
            .time(datetime.utcnow())
        
        # Write the point
        write_api.write(bucket=os.environ["INFLUX_BUCKET"], record=point)
        
        print("Successfully connected to InfluxDB and wrote test data!")
        
        # Query the data to verify
        query_api = client.query_api()
        query = f'from(bucket:"{os.environ["INFLUX_BUCKET"]}") |> range(start: -1h)'
        result = query_api.query(query)
        
        if result:
            print("Successfully queried data from InfluxDB!")
        else:
            print("No data found in the query result.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    test_connection() 