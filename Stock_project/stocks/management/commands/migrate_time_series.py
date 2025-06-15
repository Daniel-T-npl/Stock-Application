from django.core.management.base import BaseCommand
from stocks.migrate_time_series import migrate_time_series_data

class Command(BaseCommand):
    help = 'Migrate time series stock data from Firestore to InfluxDB'

    def handle(self, *args, **options):
        self.stdout.write('Starting time series data migration...')
        migrate_time_series_data()
        self.stdout.write(self.style.SUCCESS('Time series data migration completed')) 