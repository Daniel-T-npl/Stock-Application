from django.core.management.base import BaseCommand
from stocks.firebase_client import db
from stocks.influx_client import write_api
from influxdb_client import Point
import pytz, datetime
import os

class Command(BaseCommand):
    help = "Sync Firestore 'tickers' collection into InfluxDB"

    def handle(self, *args, **kwargs):
        docs = db.collection("tickers").stream()
        for doc in docs:
            d = doc.to_dict()
            # Map Firestore fields to tags/fields
            p = (
                Point("company_info")
                .tag("ticker", d.get("tickerSymbol", ""))
                .tag("sector", d.get("sector", ""))
                .field("status", d.get("status", ""))
                .field("instrument", d.get("instrument", ""))
                .time(datetime.datetime.now(tz=pytz.UTC))
            )
            write_api.write(
                bucket=os.environ["INFLUX_BUCKET"],
                record=p,
            )
        self.stdout.write(self.style.SUCCESS("Migration complete.")) 