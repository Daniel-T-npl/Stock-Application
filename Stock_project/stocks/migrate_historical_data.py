import os
from datetime import datetime
from dotenv import load_dotenv
from influxdb_client import Point
from stocks.firebase_client import db
from stocks.influx_client import write_api
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_migration.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

BATCH_SIZE = 1000  # Number of documents to process at once
MAX_RETRIES = 3    # Maximum number of retries for failed operations

def get_total_documents():
    """Get total number of documents in the collection"""
    try:
        return len(list(db.collection('stock_time_series').stream()))
    except Exception as e:
        logging.error(f"Error getting total documents: {str(e)}")
        return 0

def process_batch(docs, write_api):
    """Process a batch of documents and write to InfluxDB"""
    points = []
    for doc in docs:
        try:
            data = doc.to_dict()
            symbol = data.get('symbol')
            
            if not symbol:
                logging.warning(f"Skipping document {doc.id}: No symbol found")
                continue
            
            # Create a point for each time series data entry
            point = Point("stock_time_series") \
                .tag("symbol", symbol) \
                .field("open_price", float(data.get('open_price', 0))) \
                .field("high_price", float(data.get('high_price', 0))) \
                .field("low_price", float(data.get('low_price', 0))) \
                .field("close_price", float(data.get('close_price', 0))) \
                .field("volume", int(data.get('volume', 0)))
            
            # Use the timestamp from the data
            if 'timestamp' in data:
                point.time(datetime.fromtimestamp(data['timestamp']))
            else:
                point.time(datetime.utcnow())
            
            points.append(point)
            
        except Exception as e:
            logging.error(f"Error processing document {doc.id}: {str(e)}")
            continue
    
    if points:
        try:
            write_api.write(
                bucket=os.getenv('INFLUX_BUCKET'),
                record=points
            )
            return len(points)
        except Exception as e:
            logging.error(f"Error writing batch to InfluxDB: {str(e)}")
            return 0
    
    return 0

def migrate_historical_data():
    """
    Migrate historical stock data from Firebase to InfluxDB
    with batch processing and error handling
    """
    try:
        total_docs = get_total_documents()
        if total_docs == 0:
            logging.error("No documents found in the collection")
            return
        
        logging.info(f"Starting migration of {total_docs} documents")
        
        # Get all documents from the stock_time_series collection
        time_series_ref = db.collection('stock_time_series')
        
        # Process in batches
        processed_count = 0
        error_count = 0
        last_doc = None
        
        with tqdm(total=total_docs, desc="Migrating data") as pbar:
            while True:
                # Get next batch of documents
                query = time_series_ref.limit(BATCH_SIZE)
                if last_doc:
                    query = query.start_after(last_doc)
                
                batch = list(query.stream())
                if not batch:
                    break
                
                # Process batch
                success_count = process_batch(batch, write_api)
                processed_count += success_count
                error_count += (len(batch) - success_count)
                
                # Update progress
                pbar.update(len(batch))
                last_doc = batch[-1]
                
                # Log progress
                if processed_count % 10000 == 0:
                    logging.info(f"Processed {processed_count} documents...")
        
        logging.info(f"\nMigration completed:")
        logging.info(f"Successfully migrated: {processed_count} documents")
        logging.info(f"Errors encountered: {error_count}")
        
    except Exception as e:
        logging.error(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_historical_data() 