import os
import sys

# Dynamically determine the project root (the parent of the directory containing this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
import django
django.setup()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, UnexpectedAlertPresentException, StaleElementReferenceException, ElementClickInterceptedException, ElementNotInteractableException
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime
from analysis.influx_client import InfluxDBHandler
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def dismiss_alert(driver):
    try:
        alert = driver.switch_to.alert
        alert.dismiss()  # or alert.accept()
        logger.info("Alert dismissed.")
    except Exception:
        pass  # No alert to dismiss

def get_all_symbols():
    influx = InfluxDBHandler()
    symbols = influx.get_all_symbols()
    influx.close()
    return symbols

def scrape_stock_history(symbol, driver):
    stop_scraping = False  # Reset for each stock
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    driver.get(url)
    dismiss_alert(driver)  # Dismiss alert after loading page
    # Wait and click 'Price History' button
    try:
        price_history_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Price History"))
        )
        price_history_btn.click()
        dismiss_alert(driver)  # Dismiss alert after clicking button
        # Wait for at least one <td> cell to appear in the table
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-bordered.table-striped.table-hover td"))
            )
        except TimeoutException:
            logger.error(f"No data loaded in table for {symbol} after waiting.")
            return []
    except TimeoutException:
        logger.error(f"Price History button not found for {symbol}")
        return []
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        logger.warning("Unexpected alert handled after clicking Price History.")

    all_rows = []
    while True:
        try:
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-hover.table-bordered.table-striped"))
            )
            table_html = table.get_attribute('outerHTML')
        except TimeoutException:
            logger.error(f"Table not found for {symbol}")
            break
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
            continue
        except StaleElementReferenceException:
            logger.warning("Stale element, retrying...")
            continue

        soup = BeautifulSoup(table_html, 'html.parser')
        tbody = soup.find('tbody')
        if not tbody:
            logger.warning(f"{symbol}: No <tbody> found in table!")
            rows = []
        else:
            all_trs = tbody.find_all('tr')
            # Skip the first row if it contains <th> (header)
            if all_trs and all_trs[0].find('th'):
                rows = all_trs[1:]
            else:
                rows = all_trs
        if len(rows) == 0:
            logger.warning(f"{symbol}: No data rows found, table HTML: {table_html[:500]}")
        for row in rows:
            cols = [td.text.strip() for td in row.find_all('td')]
            if not cols or len(cols) < 9:
                continue
            date_str = cols[1]
            logger.debug(f"{symbol}: Found row with date {date_str}")
            try:
                row_date = datetime.strptime(date_str, "%Y/%m/%d")
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")
                continue
            # TEMP: Set cutoff to a very old date for testing
            if row_date <= datetime.strptime("2025/06/24", "%Y/%m/%d"):
                logger.info(f"{symbol}: Hit cutoff at {date_str}, stopping for this stock.")
                stop_scraping = True
                break  # Stop processing further rows on this page
            selected_cols = [
                cols[6],  # open
                cols[4],  # high
                cols[5],  # low
                cols[2],  # close (LTP)
                cols[7],  # volume (Qty.)
                cols[8],  # turnover
            ]
            all_rows.append([symbol, date_str] + selected_cols)
        if stop_scraping:
            break  # Exit the while loop and move to the next stock
        # Try to click 'Next'
        try:
            # Only select the correct Next button for pagination
            next_btns = driver.find_elements(By.XPATH, "//a[@title='Next Page' and @href='javascript:void(0);']")
            if not next_btns:
                break  # No Next button, so we're done
            next_btn = next_btns[0]
            # Check if the button is displayed and enabled before clicking
            if not next_btn.is_displayed() or not next_btn.is_enabled() or "disabled" in next_btn.get_attribute("class"):
                break
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
            time.sleep(0.2)
            try:
                next_btn = driver.find_element(By.XPATH, "//a[@title='Next Page' and @href='javascript:void(0);']")
                if not next_btn.is_displayed() or not next_btn.is_enabled():
                    break
                next_btn.click()
            except ElementClickInterceptedException:
                logger.warning("Click intercepted, trying JS click.")
                driver.execute_script("arguments[0].click();", next_btn)
            except ElementNotInteractableException:
                logger.warning("Next button not interactable, breaking loop.")
                break
            time.sleep(1)
            dismiss_alert(driver)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
            continue
        except StaleElementReferenceException:
            logger.warning("Stale Next button, retrying...")
            continue
    return all_rows

def save_to_csv(stock_data, filename="stock_data.csv", write_header=True):
    """Save stock data to CSV file, appending if needed"""
    try:
        df = pd.DataFrame(stock_data, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        if not df.empty:
            df.to_csv(filename, mode='a', index=False, header=write_header)
            logger.info(f"Appended {len(df)} rows to CSV: {filename}")
        return df
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        return None

def main():
    symbols = get_all_symbols()
    options = webdriver.ChromeOptions()
    # Comment out or remove headless for debugging
    # options.add_argument('--headless')
    # Set a realistic user-agent
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    prefs = {"profile.default_content_setting_values.notifications": 2}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)
    csv_file = "stock_data.csv"
    # Create or clear the CSV file and write the header
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('symbol,date,open,high,low,close,volume,turnover\n')
    for symbol in symbols:
        logger.info(f"Scraping {symbol}")
        rows = scrape_stock_history(symbol, driver)
        logger.info(f"Scraped {len(rows)} rows for {symbol}")
        # Append to CSV after each stock
        save_to_csv(rows, filename=csv_file, write_header=False)
    driver.quit()

if __name__ == "__main__":
    main() 