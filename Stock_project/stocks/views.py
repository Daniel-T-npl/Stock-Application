from django.shortcuts import render
from django.http import JsonResponse
from stocks.influx_client import query_api
import os
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .stock_service import StockService
import json

# Create your views here.

def price_history(request, ticker):
    flux = f'''
      from(bucket:"{os.environ["INFLUX_BUCKET"]}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "prices")
      |> filter(fn: (r) => r.ticker == "{ticker}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    tables = query_api.query(flux)
    data = [
        {"t": rec.get_time().isoformat(), "price": rec.values["close"]}
        for table in tables
        for rec in table.records
    ]
    return JsonResponse(data, safe=False)

stock_service = StockService()

@csrf_exempt
@require_http_methods(["POST"])
def update_stock(request):
    """
    Update stock data for a given symbol
    """
    try:
        data = json.loads(request.body)
        symbol = data.get('symbol')
        
        if not symbol:
            return JsonResponse({'error': 'Symbol is required'}, status=400)
            
        success = stock_service.update_stock_data(symbol)
        
        if success:
            return JsonResponse({'message': f'Successfully updated data for {symbol}'})
        else:
            return JsonResponse({'error': f'Failed to update data for {symbol}'}, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_stock_data(request):
    """
    Get stock data for a given symbol
    """
    try:
        symbol = request.GET.get('symbol')
        if not symbol:
            return JsonResponse({'error': 'Symbol is required'}, status=400)
            
        data = stock_service.get_stock_data(symbol)
        
        if data:
            # Convert InfluxDB result to a more readable format
            result = []
            for table in data:
                for record in table.records:
                    result.append({
                        'time': record.get_time().isoformat(),
                        'symbol': record.values.get('symbol'),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    })
            return JsonResponse({'data': result})
        else:
            return JsonResponse({'error': f'No data found for {symbol}'}, status=404)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
