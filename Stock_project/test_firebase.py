import os
import sys
import logging
from pathlib import Path
from stocks.firebase_client import get_stock_data, get_stock_history
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_functions():
    """Test the Firebase client functions"""
    try:
        # Test getting latest data
        logger.info("\nTesting get_stock_data() with no parameters...")
        latest_data = get_stock_data()
        latest_date = list(latest_data.keys())[0]
        logger.info(f"Latest date: {latest_date}")
        logger.info(f"Number of stocks: {len(latest_data[latest_date])}")
        
        # Get a sample stock symbol
        sample_stock = list(latest_data[latest_date].keys())[0]
        logger.info(f"Sample stock: {sample_stock}")
        
        # Test getting data for specific date and stock
        logger.info(f"\nTesting get_stock_data() for {latest_date} and {sample_stock}...")
        stock_data = get_stock_data(latest_date, sample_stock)
        logger.info(f"Stock data: {stock_data}")
        
        # Calculate a date one year before the latest date
        latest_date_obj = datetime.strptime(latest_date, "%Y-%m-%d")
        start_date = (latest_date_obj - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Test getting stock history
        logger.info(f"\nTesting get_stock_history() for {sample_stock} from {start_date}...")
        history = get_stock_history(sample_stock, start_date=start_date)
        logger.info(f"Number of historical records: {len(history)}")
        if history:
            logger.info(f"Sample record: {list(history.items())[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing functions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_functions()
    if success:
        logger.info("All tests completed successfully")
    else:
        logger.error("Tests failed")
        sys.exit(1) 