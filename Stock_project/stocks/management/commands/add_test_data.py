from django.core.management.base import BaseCommand
from stocks.firebase_client import db
from datetime import datetime

class Command(BaseCommand):
    help = 'Add test stock data to Firestore'

    def handle(self, *args, **options):
        # Sample stock data
        test_stocks = [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'price': 175.50,
                'volume': 50000000,
                'change': 2.50,
                'change_percent': 1.45,
                'last_updated': datetime.now()
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'price': 380.25,
                'volume': 25000000,
                'change': -1.75,
                'change_percent': -0.46,
                'last_updated': datetime.now()
            },
            {
                'symbol': 'GOOGL',
                'name': 'Alphabet Inc.',
                'price': 140.80,
                'volume': 15000000,
                'change': 3.20,
                'change_percent': 2.32,
                'last_updated': datetime.now()
            },
            {
                'symbol': 'AMZN',
                'name': 'Amazon.com Inc.',
                'price': 175.35,
                'volume': 30000000,
                'change': 1.25,
                'change_percent': 0.72,
                'last_updated': datetime.now()
            },
            {
                'symbol': 'META',
                'name': 'Meta Platforms Inc.',
                'price': 485.90,
                'volume': 20000000,
                'change': 5.60,
                'change_percent': 1.17,
                'last_updated': datetime.now()
            }
        ]

        try:
            # Get reference to the stocks collection
            stocks_ref = db.collection('stocks')
            
            # Add each stock to Firestore
            for stock in test_stocks:
                # Use the symbol as the document ID
                doc_ref = stocks_ref.document(stock['symbol'])
                doc_ref.set(stock)
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully added stock: {stock["symbol"]}')
                )

            self.stdout.write(
                self.style.SUCCESS(f'Successfully added {len(test_stocks)} stocks to Firestore')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error adding test data: {str(e)}')
            ) 