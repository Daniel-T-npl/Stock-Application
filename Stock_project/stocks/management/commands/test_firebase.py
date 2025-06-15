from django.core.management.base import BaseCommand
import os
import sys
from stocks.firebase_client import db

class Command(BaseCommand):
    help = 'Test connection to Firebase and print some basic stats'

    def handle(self, *args, **options):
        self.stdout.write("Starting Firebase connection test...")
        self.stdout.write(f"Python path: {sys.path}")
        self.stdout.write(f"Current working directory: {os.getcwd()}")
        
        try:
            # Set environment to production
            os.environ['FIREBASE_ENV'] = 'production'
            self.stdout.write("Set FIREBASE_ENV to production")
            
            # Try to get a document count
            self.stdout.write("Attempting to connect to Firestore...")
            collection_ref = db.collection('stock_time_series')
            self.stdout.write("Successfully got collection reference")
            
            docs = list(collection_ref.limit(5).stream())
            self.stdout.write("Successfully retrieved documents")
            
            self.stdout.write("\nFirebase Connection Test:")
            self.stdout.write("------------------------")
            self.stdout.write(self.style.SUCCESS(f"Successfully connected to Firebase"))
            self.stdout.write(f"Retrieved {len(docs)} sample documents")
            
            if docs:
                self.stdout.write("\nSample document structure:")
                sample_doc = docs[0].to_dict()
                for key, value in sample_doc.items():
                    self.stdout.write(f"{key}: {type(value).__name__}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error connecting to Firebase: {str(e)}"))
            import traceback
            self.stdout.write(self.style.ERROR(traceback.format_exc())) 