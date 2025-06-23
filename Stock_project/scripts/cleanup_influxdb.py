import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from influxdb_client.client.delete_api import DeleteApi
from influxdb_client.client.query_api import QueryApi
from datetime import datetime, timedelta
from stocks.influx_client import InfluxDBHandler

# Load environment variables
load_dotenv()

def cleanup_influxdb():
    """
    Clean up InfluxDB by deleting all data outside of stock_data bucket and stock_data measurement
    """
    # InfluxDB configuration
    url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    token = os.getenv('INFLUXDB_TOKEN')
    org = os.getenv('INFLUXDB_ORG')
    
    if not all([url, token, org]):
        print("Error: Missing InfluxDB configuration. Please set INFLUXDB_URL, INFLUXDB_TOKEN, and INFLUXDB_ORG environment variables.")
        return
    
    # Initialize InfluxDB client
    client = InfluxDBClient(url=url, token=token, org=org)
    delete_api = DeleteApi(client)
    query_api = QueryApi(client)
    
    try:
        print("Starting InfluxDB cleanup...")
        
        # First, let's see what buckets exist
        buckets_api = client.buckets_api()
        buckets_obj = buckets_api.find_buckets()
        buckets = buckets_obj.buckets if hasattr(buckets_obj, 'buckets') else []
        
        print(f"Found {len(buckets)} buckets:")
        for bucket in buckets:
            print(f"  - {bucket.name}")
        
        # Delete data from all buckets except stock_data
        for bucket in buckets:
            if bucket.name != "stock_data":
                print(f"\nDeleting all data from bucket: {bucket.name}")
                
                # Delete all data from this bucket
                # We'll delete data from a very wide time range to ensure everything is removed
                start_time = datetime(1970, 1, 1)
                end_time = datetime.now() + timedelta(days=365)  # Future date to catch all data
                
                try:
                    delete_api.delete(
                        start=start_time,
                        stop=end_time,
                        predicate='',
                        bucket=bucket.name,
                        org=org
                    )
                    print(f"  ✓ Successfully deleted all data from bucket: {bucket.name}")
                except Exception as e:
                    print(f"  ✗ Error deleting data from bucket {bucket.name}: {e}")
        
        # Now clean up the stock_data bucket - delete all measurements except stock_data
        print(f"\nCleaning up stock_data bucket...")
        
        # Query to find all measurements in stock_data bucket
        query = '''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "stock_data")
        '''
        
        try:
            result = query_api.query(query=query, org=org)
            measurements = []
            
            for table in result:
                for record in table.records:
                    measurements.append(record.get_value())
            
            print(f"Found measurements in stock_data bucket: {measurements}")
            
            # Delete data from measurements other than stock_data
            for measurement in measurements:
                if measurement != "stock_data":
                    print(f"Deleting data from measurement: {measurement}")
                    
                    start_time = datetime(1970, 1, 1)
                    end_time = datetime.now() + timedelta(days=365)
                    
                    try:
                        delete_api.delete(
                            start=start_time,
                            stop=end_time,
                            predicate=f'_measurement="{measurement}"',
                            bucket="stock_data",
                            org=org
                        )
                        print(f"  ✓ Successfully deleted data from measurement: {measurement}")
                    except Exception as e:
                        print(f"  ✗ Error deleting data from measurement {measurement}: {e}")
                else:
                    print(f"Keeping measurement: {measurement}")
            
        except Exception as e:
            print(f"Error querying measurements: {e}")
        
        print(f"\nCleanup completed!")
        print(f"Only data in bucket 'stock_data' and measurement 'stock_data' should remain.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        # Close InfluxDB client
        client.close()

if __name__ == "__main__":
    cleanup_influxdb() 