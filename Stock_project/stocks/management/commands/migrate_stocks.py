from django.core.management.base import BaseCommand
from stocks.migrate_data import migrate_stock_data

class Command(BaseCommand):
    help = 'Migrate stock data from Firestore to InfluxDB'

    def handle(self, *args, **options):
        self.stdout.write('Starting stock data migration...')
        migrate_stock_data()
        self.stdout.write(self.style.SUCCESS('Stock data migration completed')) 