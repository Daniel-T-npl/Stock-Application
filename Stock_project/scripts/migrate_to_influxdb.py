import os
import logging
from pathlib import Path
from stocks.firebase_client import get_stock_data, get_stock_history
from stocks.influx_client import InfluxDBHandler
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_data():
    """Migrate stock data from Firebase to InfluxDB"""
    try:
        # Initialize InfluxDB client
        influx_client = InfluxDBHandler()
        
        # Get all dates from Firebase
        logger.info("Getting latest data to find available dates...")
        latest_data = get_stock_data()
        latest_date = list(latest_data.keys())[0]
        
        # Get all dates by getting historical data for a known stock
        logger.info("Getting all available dates...")
        sample_stock = list(latest_data[latest_date].keys())[0]
        all_dates = get_stock_history(sample_stock).keys()
        total_dates = len(all_dates)
        
        logger.info(f"Found {total_dates} dates to process")
        
        # Process each date
        for i, date in enumerate(sorted(all_dates), 1):
            try:
                # Get data for this date
                date_data = get_stock_data(date)
                stocks_data = date_data[date]  # This is already the data we need
                
                logger.info(f"Processing date {date} ({i}/{total_dates})")
                
                # Process each stock for this date
                for stock_symbol, stock_data in stocks_data.items():
                    try:
                        # Skip the 'summaryFor' key as it's not a stock
                        if stock_symbol == 'summaryFor':
                            continue
                            
                        influx_client.write_stock_data(date, stock_symbol, stock_data)
                    except Exception as e:
                        logger.error(f"Error processing {stock_symbol} on {date}: {str(e)}")
                        continue
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{total_dates} dates")
                    
            except Exception as e:
                logger.error(f"Error processing date {date}: {str(e)}")
                continue
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        # Close InfluxDB connection
        influx_client.close()

if __name__ == "__main__":
    # Ensure we're in production mode
    os.environ['FIREBASE_ENV'] = 'production'
    
    # Run migration
    migrate_data() 