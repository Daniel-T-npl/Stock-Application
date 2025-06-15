import os
from dotenv import load_dotenv
from stocks.firebase_client import db

def test_connection():
    """Test connection to Firebase and print some basic stats"""
    try:
        # Set environment to production
        os.environ['FIREBASE_ENV'] = 'production'
        
        # Try to get a document count
        collection_ref = db.collection('stock_time_series')
        docs = list(collection_ref.limit(5).stream())
        
        print("\nFirebase Connection Test:")
        print("------------------------")
        print(f"Successfully connected to Firebase")
        print(f"Retrieved {len(docs)} sample documents")
        
        if docs:
            print("\nSample document structure:")
            sample_doc = docs[0].to_dict()
            for key, value in sample_doc.items():
                print(f"{key}: {type(value).__name__}")
        
    except Exception as e:
        print(f"Error connecting to Firebase: {str(e)}")

if __name__ == "__main__":
    test_connection() 