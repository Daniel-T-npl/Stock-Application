import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
import django
django.setup()

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://merolagani.com/LatestMarket.aspx"

r = requests.get(url=URL)
print(r.content)
soup = BeautifulSoup(r.content, 'html5lib')

stock_data = []

stock_table = soup.find('table', {'class': 'table table-hover live-trading sortable'})

for row in stock_table.find_all('tr'):
    cols = row.find_all('td')
    if len(cols) >= 9:
        qty = int(cols[6].text.strip().replace(',', ''))
        ltp = float(cols[1].text.strip().replace(',', ''))
        stock = [
            cols[0].a.text.strip() if cols[0].a else None,    # Symbol
            float(cols[5].text.strip().replace(',', '')),     # Open
            float(cols[3].text.strip().replace(',', '')),     # High
            float(cols[4].text.strip().replace(',', '')),     # Low
            ltp,                                              # Close / ltp
            qty,                                              # Volume
            ltp * qty,                                        # Turnover
        ]
        stock_data.append(stock)

print(stock_data)
df = pd.DataFrame(stock_data, columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
df.to_csv("stock_data.csv", index=False)






