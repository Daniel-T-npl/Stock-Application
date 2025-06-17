import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime
from stocks.influx_client import InfluxDBHandler
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

URL = "https://merolagani.com/LatestMarket.aspx"

def scrape_stock_data():
    """Scrape stock data from the website"""
    try:
        logger.info("Starting web scraping...")
        r = requests.get(url=URL)
        r.raise_for_status()  # Raise an exception for bad status codes
        
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

def save_to_csv(stock_data, filename="stock_data.csv"):
    """Save stock data to CSV file"""
    try:
        df = pd.DataFrame(stock_data, columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to CSV: {filename}")
        return df
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        return None

def save_to_influxdb(stock_data, date=None):
    """Save stock data to InfluxDB"""
    try:
        # Initialize InfluxDB client
        influx_client = InfluxDBHandler()
        
        # Use current date if no date provided
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
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
                
                # Write to InfluxDB
                influx_client.write_stock_data(date, symbol, stock_data_dict)
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
    """Main function to run the web scraper with both CSV and InfluxDB output"""
    try:
        # Scrape data
        stock_data = scrape_stock_data()
        
        if not stock_data:
            logger.error("No data scraped. Exiting.")
            return
        
        # Save to CSV
        df = save_to_csv(stock_data)
        
        # Save to InfluxDB
        save_to_influxdb(stock_data)
        
        logger.info("Web scraping and data migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 