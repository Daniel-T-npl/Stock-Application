from django.core.management.base import BaseCommand
from stocks.add_sample_data import add_sample_data_to_firestore

class Command(BaseCommand):
    help = 'Add sample time series stock data to Firestore'

    def handle(self, *args, **options):
        self.stdout.write('Adding sample time series data...')
        add_sample_data_to_firestore()
        self.stdout.write(self.style.SUCCESS('Sample data added successfully')) 