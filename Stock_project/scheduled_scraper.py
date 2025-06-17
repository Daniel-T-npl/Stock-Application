import time
import schedule
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from stocks.influx_client import InfluxDBHandler
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduled_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

URL = "https://merolagani.com/LatestMarket.aspx"

def scrape_and_save():
    """Scrape stock data and save to InfluxDB"""
    try:
        logger.info("=" * 50)
        logger.info(f"Starting scheduled scrape at {datetime.now()}")
        
        # Scrape data
        stock_data = scrape_stock_data()
        
        if not stock_data:
            logger.error("No data scraped. Skipping this run.")
            return
        
        # Save to InfluxDB
        save_to_influxdb(stock_data)
        
        logger.info(f"Scheduled scrape completed at {datetime.now()}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in scheduled scrape: {str(e)}")

def scrape_stock_data():
    """Scrape stock data from the website"""
    try:
        logger.info("Starting web scraping...")
        r = requests.get(url=URL, timeout=30)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.content, 'html5lib')
        stock_data = []

        stock_table = soup.find('table', {'class': 'table table-hover live-trading sortable'})
        
        if not stock_table:
            logger.error("Stock table not found on the page")
            return []

        for row in stock_table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 9:
                try:
                    qty = int(cols[6].text.strip().replace(',', ''))
                    ltp = float(cols[1].text.strip().replace(',', ''))
                    stock = [
                        cols[0].a.text.strip() if cols[0].a else None,    # Symbol
                        float(cols[5].text.strip().replace(',', '')),     # Open
                        float(cols[3].text.strip().replace(',', '')),     # High
                        float(cols[4].text.strip().replace(',', '')),     # Low
                        ltp,                                              # Close / ltp
                        qty,                                              # Volume
                        ltp * qty,                                        # Turnover
                    ]
                    stock_data.append(stock)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

        logger.info(f"Successfully scraped {len(stock_data)} stock records")
        return stock_data
        
    except Exception as e:
        logger.error(f"Error during web scraping: {str(e)}")
        return []

def save_to_influxdb(stock_data):
    """Save stock data to InfluxDB"""
    try:
        # Initialize InfluxDB client
        influx_client = InfluxDBHandler()
        
        # Use current timestamp for more precise data
        current_time = datetime.now()
        date = current_time.strftime('%Y-%m-%d')
        
        logger.info(f"Writing {len(stock_data)} records to InfluxDB for date: {date}")
        
        success_count = 0
        error_count = 0
        
        for stock in stock_data:
            try:
                symbol, open_price, high, low, close_price, volume, turnover = stock
                
                # Skip if symbol is None
                if symbol is None:
                    continue
                
                # Create stock data dictionary
                stock_data_dict = {
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close_price),
                    'volume': float(volume),
                    'turnover': float(turnover)
                }
                
                # Write to InfluxDB with current timestamp
                influx_client.write_stock_data_with_timestamp(date, symbol, stock_data_dict, current_time)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error writing stock {stock[0] if stock[0] else 'Unknown'}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"InfluxDB write completed!")
        logger.info(f"Successfully written: {success_count} records")
        logger.info(f"Errors: {error_count} records")
        
    except Exception as e:
        logger.error(f"Error writing to InfluxDB: {str(e)}")
        raise
    finally:
        # Close InfluxDB connection
        influx_client.close()

def main():
    """Main function to run the scheduled scraper"""
    logger.info("Starting scheduled stock scraper...")
    logger.info("Scraper will run every 5 minutes")
    logger.info("Press Ctrl+C to stop the scraper")
    
    # Schedule the job to run every 5 minutes
    schedule.every(1).minutes.do(scrape_and_save)
    
    # Run immediately on startup
    scrape_and_save()
    
    # Keep the script running and execute scheduled jobs
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds for pending jobs
    except KeyboardInterrupt:
        logger.info("Scheduled scraper stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 