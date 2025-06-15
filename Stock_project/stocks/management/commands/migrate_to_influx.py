import os
from datetime import datetime
from django.core.management.base import BaseCommand
from influxdb_client import Point
from stocks.firebase_client import db
from stocks.influx_client import write_api

class Command(BaseCommand):
    help = 'Migrate stock data from Firebase to InfluxDB'

    def handle(self, *args, **options):
        try:
            # Get all stock documents from Firestore
            stocks_ref = db.collection('stocks')
            stocks = stocks_ref.stream()

            # Counter for migrated documents
            migrated_count = 0

            for stock in stocks:
                stock_data = stock.to_dict()
                
                # Create a point for each stock
                point = Point("stock_data") \
                    .tag("symbol", stock_data.get('symbol', '')) \
                    .tag("name", stock_data.get('name', '')) \
                    .field("price", float(stock_data.get('price', 0))) \
                    .field("volume", int(stock_data.get('volume', 0))) \
                    .field("change", float(stock_data.get('change', 0))) \
                    .field("change_percent", float(stock_data.get('change_percent', 0))) \
                    .time(datetime.utcnow())

                # Write the point to InfluxDB
                write_api.write(
                    bucket=os.environ["INFLUX_BUCKET"],
                    record=point
                )
                
                migrated_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully migrated stock: {stock_data.get("symbol")}')
                )

            self.stdout.write(
                self.style.SUCCESS(f'Successfully migrated {migrated_count} stocks to InfluxDB')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during migration: {str(e)}')
            ) 