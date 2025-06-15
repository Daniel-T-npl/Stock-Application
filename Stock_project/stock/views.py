from django.shortcuts import render
from datetime import datetime, timedelta
import logging
from .services.influxdb_handler import InfluxDBHandler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stock_dashboard(request):
    try:
        # Initialize InfluxDB client
        influx_handler = InfluxDBHandler()
        client = influx_handler.client
        
        # Get query parameters
        symbol = request.GET.get('symbol', 'NICL')
        days = int(request.GET.get('days', 1000))
        
        logger.info(f"Requested symbol: {symbol}, days: {days}")
        
        # Calculate date range
        end_date = datetime(2023, 12, 31)
        start_date = datetime(2021, 1, 3)  # Fixed start date
        if days < 1000:  # If user selects a specific range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        
        # Format dates for InfluxDB queries
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Format dates for display
        start_date_display = start_date.strftime('%Y-%m-%d')
        end_date_display = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Date range: {start_date_str} to {end_date_str}")
        
        # Query for distinct symbols from stock_data measurement in the date range
        symbols_query = f'''
        from(bucket: "stock_data")
            |> range(start: {start_date_str}, stop: {end_date_str})
            |> filter(fn: (r) => r["_measurement"] == "stock_data")
            |> filter(fn: (r) => r["_field"] == "volume")
            |> keep(columns: ["symbol"])
            |> group()
            |> distinct(column: "symbol")
            |> sort(columns: ["symbol"])
        '''
        logger.info("\nChecking for symbols in stock_data measurement (date range)...")
        symbols_result = client.query_api().query(symbols_query)
        symbols = []
        for table in symbols_result:
            for record in table.records:
                symbol = record.values.get('symbol')
                if symbol:
                    symbols.append(symbol)
        logger.info(f"Found {len(symbols)} symbols in date range.")
        if symbols:
            logger.info(f"First 10 symbols: {symbols[:10]}")
        
        if not symbols:
            logger.warning("No symbols found in the database")
            return render(request, 'stock/dashboard.html', {
                'error': 'No stock symbols found in the database',
                'symbols': [],
                'data': [],
                'start_date': start_date_display,
                'end_date': end_date_display
            })
        
        # If the requested symbol is not in the list, use the first available one
        if symbol not in symbols:
            logger.warning(f"Requested symbol {symbol} not found, using first available: {symbols[0]}")
            symbol = symbols[0]
        
        # Query for stock data
        query = f'''
        from(bucket: "stock_data")
            |> range(start: {start_date_str}, stop: {end_date_str})
            |> filter(fn: (r) => r["_measurement"] == "stock_data")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> filter(fn: (r) => r["_field"] == "volume" or r["_field"] == "open" or r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "close")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        logger.info(f"Executing data query for symbol {symbol}...")
        result = client.query_api().query(query)
        logger.info(f"Data query returned {len(result)} tables")
        
        # Process the results
        data = []
        for table in result:
            logger.info(f"Processing data table with {len(table.records)} records")
            for record in table.records:
                # Format the date for display
                date = record.values.get('_time')
                if date:
                    date = date.strftime('%Y-%m-%d')
                data_point = {
                    'date': date,
                    'open': record.values.get('open'),
                    'high': record.values.get('high'),
                    'low': record.values.get('low'),
                    'close': record.values.get('close'),
                    'volume': record.values.get('volume'),
                    'turnover': record.values.get('turnover')
                }
                data.append(data_point)
                logger.info(f"Processed data point: {data_point}")
        
        logger.info(f"Total data points processed: {len(data)}")
        
        context = {
            'symbol': symbol,
            'symbols': symbols,
            'data': data,
            'start_date': start_date_display,
            'end_date': end_date_display,
            'days': days
        }
        
        logger.info(f"Rendering template with context: {context}")
        return render(request, 'stock/dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Error in stock_dashboard: {str(e)}", exc_info=True)
        return render(request, 'stock/dashboard.html', {
            'error': f'Error: {str(e)}',
            'symbols': [],
            'data': [],
            'start_date': start_date_display if 'start_date_display' in locals() else None,
            'end_date': end_date_display if 'end_date_display' in locals() else None
        })
    finally:
        if 'influx_handler' in locals():
            influx_handler.close()

@csrf_exempt
def test_influxdb_view(request):
    """Test view to verify InfluxDB data retrieval"""
    try:
        # Initialize InfluxDB client
        client = InfluxDBHandler()
        
        # Get available symbols
        symbols_query = '''
        from(bucket: "stock_data")
            |> range(start: 2021-01-03T00:00:00Z, stop: 2023-12-31T23:59:59Z)
            |> filter(fn: (r) => r["_measurement"] == "stock_data")
            |> filter(fn: (r) => r["_field"] == "volume")
            |> distinct(column: "symbol")
        '''
        
        symbols_result = client.query_api.query_data_frame(symbols_query)
        symbols = symbols_result['symbol'].tolist() if not symbols_result.empty else []
        
        # Get data for first symbol if available
        data = []
        if symbols:
            symbol = symbols[0]
            data_query = '''
            from(bucket: "stock_data")
                |> range(start: 2021-01-03T00:00:00Z, stop: 2023-12-31T23:59:59Z)
                |> filter(fn: (r) => r["_measurement"] == "stock_data")
                |> filter(fn: (r) => r["symbol"] == "{}")
                |> filter(fn: (r) => r["_field"] =~ /^(volume|open|high|low|close)$/)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''.format(symbol)
            
            data_result = client.query_api.query_data_frame(data_query)
            if not data_result.empty:
                data = data_result.to_dict('records')
        
        return JsonResponse({
            'status': 'success',
            'symbols': symbols,
            'sample_data': data[:5] if data else [],  # Return first 5 records as sample
            'total_symbols': len(symbols),
            'total_records': len(data) if data else 0
        })
        
    except Exception as e:
        logger.error(f"Error in test_influxdb_view: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
    finally:
        if 'client' in locals():
            client.close() 