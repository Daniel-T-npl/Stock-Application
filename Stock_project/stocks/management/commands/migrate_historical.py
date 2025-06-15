from django.core.management.base import BaseCommand
from stocks.migrate_historical_data import migrate_historical_data

class Command(BaseCommand):
    help = 'Migrate historical stock data from Firebase to InfluxDB'

    def handle(self, *args, **options):
        self.stdout.write('Starting historical data migration...')
        migrate_historical_data()
        self.stdout.write(self.style.SUCCESS('Historical data migration completed')) 