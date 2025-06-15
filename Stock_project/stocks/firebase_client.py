import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if we're in production mode
is_production = os.getenv('FIREBASE_ENV') == 'production'

# Use FIREBASE_KEY_PATH from .env, fallback to default
cred_path = os.getenv('FIREBASE_KEY_PATH', str(Path(__file__).resolve().parent.parent / 'firebase-key.json'))
cred_path = Path(cred_path)

def initialize_firebase():
    if not firebase_admin._apps:
        try:
            if is_production:
                if not cred_path.exists():
                    raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")
                logger.info(f"Initializing Firebase with credentials from {cred_path}")
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully (production)")
            else:
                logger.info("Attempting to connect to Firebase emulator...")
                os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8081"
                firebase_admin.initialize_app(options={
                    'projectId': 'buffett-e0174',
                })
                logger.info("Firebase initialized successfully (emulator)")
        except Exception as e:
            if not is_production:
                logger.error("Failed to connect to Firebase emulator. Make sure it's running with 'firebase emulators:start'")
                logger.error("Or set FIREBASE_ENV=production to use production credentials")
            raise

# Initialize Firebase
try:
    initialize_firebase()
    db = firestore.client()
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {str(e)}")
    raise

def get_stock_data(date_str=None, stock_symbol=None):
    """
    Get stock data for a specific date and/or stock symbol
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. If None, returns latest date.
        stock_symbol (str, optional): Stock symbol to filter. If None, returns all stocks.
    
    Returns:
        dict: Stock data for the specified date and/or stock
    """
    try:
        collection_ref = db.collection('marketDailySummaries')
        
        # If no date specified, get the latest date
        if date_str is None:
            # Get all documents and find the latest one
            docs = list(collection_ref.stream())
            if not docs:
                raise ValueError("No data found in marketDailySummaries collection")
            
            # Find the document with the latest date (document ID)
            latest_doc = max(docs, key=lambda doc: doc.id)
            date_str = latest_doc.id
            doc = latest_doc
        else:
            doc = collection_ref.document(date_str).get()
            if not doc.exists:
                raise ValueError(f"No data found for date {date_str}")
        
        data = doc.to_dict()
        
        # Filter by stock symbol if specified
        if stock_symbol:
            if stock_symbol not in data:
                raise ValueError(f"Stock symbol {stock_symbol} not found for date {date_str}")
            return {date_str: {stock_symbol: data[stock_symbol]}}
        
        return {date_str: data}
        
    except Exception as e:
        logger.error(f"Error getting stock data: {str(e)}")
        raise

def get_stock_history(stock_symbol, start_date=None, end_date=None):
    """
    Get historical data for a specific stock
    
    Args:
        stock_symbol (str): Stock symbol to get history for
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
    
    Returns:
        dict: Historical data for the stock
    """
    try:
        collection_ref = db.collection('marketDailySummaries')
        docs = list(collection_ref.stream())
        history = {}

        for doc in docs:
            doc_date = doc.id
            # Filter by date range if specified
            if start_date and doc_date < start_date:
                continue
            if end_date and doc_date > end_date:
                continue
            data = doc.to_dict()
            if stock_symbol in data:
                history[doc_date] = data[stock_symbol]

        if not history:
            raise ValueError(f"No data found for stock {stock_symbol}")

        return history

    except Exception as e:
        logger.error(f"Error getting stock history: {str(e)}")
        raise 