from django.shortcuts import render
from django.http import JsonResponse
from analysis.influx_client import influx_client, get_all_symbols
import os
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_GET
from analysis.stock_service import StockService
import json
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from analysis.analysis_service import AnalysisService
from analysis.statistical_service import generate_forecast_graph, generate_forecast_data


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create your views here.

def price_history(request, ticker):
    query_api = influx_client.get_query_api()
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


# --- Views from old 'stock' app ---

def stock_dashboard(request):
    """Interactive dashboard supporting single-symbol, and date-range selection."""
    try:
        all_symbols = get_all_symbols()

        # Handle query-string parameters
        selected_symbols = request.GET.getlist("symbols")

        # Date range: default 1 year before today
        start_param = request.GET.get("start")
        end_param = request.GET.get("end")

        if not start_param or not end_param:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
        else:
            try:
                start_date = datetime.strptime(start_param, "%Y-%m-%d")
                end_date = datetime.strptime(end_param, "%Y-%m-%d")
            except ValueError:
                # Fallback to defaults on parse error
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=365)
        
        data_by_symbol = {}

        # Only fetch data if a symbol is actually selected
        if selected_symbols:
            query_api = influx_client.get_query_api()
            start_iso = start_date.strftime("%Y-%m-%dT00:00:00Z")
            end_iso = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")  # inclusive

            # Candlestick only uses the first symbol
            sym = selected_symbols[0]
            
            query_fields = ["open", "high", "low", "close"]
            keep_fields = ", ".join([f'\"{f}\"' for f in query_fields])

            flux = f"""
            from(bucket: "{influx_client.get_bucket()}")
                |> range(start: {start_iso}, stop: {end_iso})
                |> filter(fn: (r) => r["_measurement"] == "stock_data")
                |> filter(fn: (r) => r["symbol"] == "{sym}")
                |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
                |> keep(columns:["_time", {keep_fields}])
                |> sort(columns:["_time"])
            """
            tables = query_api.query(flux)
            
            points = []
            for table in tables:
                for rec in table.records:
                    point = {"date": rec.get_time().strftime("%Y-%m-%dT%H:%M:%SZ")}
                    for fld in query_fields:
                        point[fld] = rec.values.get(fld)
                    points.append(point)

            data_by_symbol[sym] = points

        context = {
            "all_symbols": all_symbols,
            "selected_symbols": selected_symbols,
            "data_json": json.dumps(data_by_symbol),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "request": request,
        }

        return render(request, "stocks/dashboard.html", context)
    except Exception as e:
        logger.error(f"Error in stock_dashboard: {str(e)}", exc_info=True)
        return render(request, 'stocks/dashboard.html', {
            'error': f'Error: {str(e)}',
            'all_symbols': get_all_symbols(),
            'selected_symbols': [],
            'data_json': '{}',
            'start_date': (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d'),
            'end_date': datetime.utcnow().strftime('%Y-%m-%d')
        })


@csrf_exempt
def test_influxdb_view(request):
    """Test view to verify InfluxDB data retrieval"""
    try:
        query_api = influx_client.get_query_api()
        
        # Get available symbols
        symbols = get_all_symbols()
        
        # Get data for first symbol if available
        data = []
        if symbols:
            symbol = symbols[0]
            data_query = f'''
            from(bucket: "{influx_client.get_bucket()}")
                |> range(start: 2021-01-03T00:00:00Z, stop: 2023-12-31T23:59:59Z)
                |> filter(fn: (r) => r["_measurement"] == "stock_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> filter(fn: (r) => r["_field"] =~ /^(volume|open|high|low|close)$/)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            data_result = query_api.query_data_frame(data_query)
            if isinstance(data_result, pd.DataFrame) and not data_result.empty:
                data = data_result.to_dict('records')
        
        return JsonResponse({
            'status': 'success',
            'symbols': symbols,
            'sample_data': data[:5] if data else [],
            'total_symbols': len(symbols),
            'total_records': len(data) if data else 0
        })
        
    except Exception as e:
        logger.error(f"Error in test_influxdb_view: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def line_dashboard(request):
    """Interactive dashboard for line graphs."""
    try:
        all_symbols = get_all_symbols()
        
        # Handle query-string parameters
        selected_symbols = request.GET.getlist("symbols")
        selected_fields = request.GET.getlist("fields")
        start_param = request.GET.get("start")
        end_param = request.GET.get("end")
        ma_period = request.GET.get('ma', '1')

        # Set default date range to last year if not specified
        if not start_param or not end_param:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
        else:
            end_date = datetime.strptime(end_param, "%Y-%m-%d")
            start_date = datetime.strptime(start_param, "%Y-%m-%d")

        data_by_symbol = {}
        # Only fetch data if symbols are actually selected
        if selected_symbols and selected_fields:
            query_api = influx_client.get_query_api()
            start_iso = start_date.strftime("%Y-%m-%dT00:00:00Z")
            end_iso = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

            for sym in selected_symbols:
                keep_fields = ", ".join([f'\"{f}\"' for f in selected_fields])
                flux = f"""
                from(bucket: "{influx_client.get_bucket()}")
                    |> range(start: {start_iso}, stop: {end_iso})
                    |> filter(fn: (r) => r["_measurement"] == "stock_data")
                    |> filter(fn: (r) => r["symbol"] == "{sym}")
                    |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
                    |> keep(columns:["_time", {keep_fields}])
                    |> sort(columns:["_time"])
                """
                tables = query_api.query(flux)
                
                points = []
                for table in tables:
                    for rec in table.records:
                        point = {"date": rec.get_time().strftime("%Y-%m-%dT%H:%M:%SZ")}
                        for fld in selected_fields:
                            point[fld] = rec.values.get(fld)
                        points.append(point)
                data_by_symbol[sym] = points

        context = {
            "all_symbols": all_symbols,
            "selected_symbols": selected_symbols,
            "selected_fields": selected_fields,
            "fields": ["open","close","high","low","turnover","volume"],
            "data_json": json.dumps(data_by_symbol),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "ma": ma_period,
            "request": request,
        }

        return render(request, "stocks/line.html", context)
    except Exception as e:
        logger.error(f"Error in line_dashboard: {str(e)}", exc_info=True)
        return render(request, 'stocks/line.html', {'error': f'Error: {str(e)}'})


def rsi_dashboard(request):
    """Interactive dashboard for RSI analysis."""
    try:
        all_symbols = get_all_symbols()

        # Handle query-string parameters
        selected_symbols = request.GET.getlist("symbols")
        start_param = request.GET.get("start")
        end_param = request.GET.get("end")
        rsi_period = int(request.GET.get('period', 14))
        overbought = int(request.GET.get('overbought', 70))
        oversold = int(request.GET.get('oversold', 30))

        # Set default date range to last year if not specified
        if not start_param or not end_param:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
        else:
            end_date = datetime.strptime(end_param, "%Y-%m-%d")
            start_date = datetime.strptime(start_param, "%Y-%m-%d")

        data_by_symbol = {}
        # Only fetch data if symbols are actually selected
        if selected_symbols:
            query_api = influx_client.get_query_api()
            start_iso = start_date.strftime("%Y-%m-%dT00:00:00Z")
            end_iso = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

            for sym in selected_symbols:
                flux = f"""
                from(bucket: "{influx_client.get_bucket()}")
                    |> range(start: {start_iso}, stop: {end_iso})
                    |> filter(fn: (r) => r["_measurement"] == "stock_data" and r["symbol"] == "{sym}")
                    |> filter(fn: (r) => r["_field"] == "close")
                    |> sort(columns:["_time"])
                """
                tables = query_api.query(flux)
                
                points = []
                for table in tables:
                    for rec in table.records:
                        points.append({"date": rec.get_time().strftime("%Y-%m-%dT%H:%M:%SZ"), "close": rec.get_value()})
                data_by_symbol[sym] = points

        context = {
            "all_symbols": all_symbols,
            "selected_symbols": selected_symbols,
            "data_json": json.dumps(data_by_symbol),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "rsi_period": rsi_period,
            "overbought": overbought,
            "oversold": oversold,
            "request": request,
        }
        return render(request, "stocks/rsi.html", context)
    except Exception as e:
        logger.error(f"Error in rsi_dashboard: {str(e)}", exc_info=True)
        return render(request, 'stocks/rsi.html', {'error': f'Error: {str(e)}'})

def bollinger_dashboard(request):
    all_symbols = get_all_symbols()
    selected_symbols = request.GET.getlist("symbols")
    start_param = request.GET.get("start")
    end_param = request.GET.get("end")
    ma_window = int(request.GET.get("ma", 20))
    stddev = float(request.GET.get("stddev", 2))

    # Default date range: last year
    if not start_param or not end_param:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)
    else:
        end_date = datetime.strptime(end_param, "%Y-%m-%d")
        start_date = datetime.strptime(start_param, "%Y-%m-%d")

    context = {
        "all_symbols": all_symbols,
        "selected_symbols": selected_symbols,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "ma_window": ma_window,
        "stddev": stddev,
        "data_json": '{}',
        "request": request,
    }
    return render(request, "stocks/bollinger.html", context)

@require_GET
def api_bollinger_data(request):
    symbols = request.GET.get("symbols")
    start = request.GET.get("start")
    end = request.GET.get("end")
    ma_window = int(request.GET.get("ma", 20))
    stddev = float(request.GET.get("stddev", 2))
    logger.info(f"API called with: symbols={symbols}, start={start}, end={end}, ma={ma_window}, stddev={stddev}")
    if not symbols or not start or not end:
        logger.error("API call missing required parameters.")
        return JsonResponse({"error": "Missing required parameters."}, status=400)
    try:
        symbol_list = symbols.split(',')
        query_api = influx_client.get_query_api()
        start_iso = f"{start}T00:00:00Z"
        end_iso = f"{end}T23:59:59Z"
        result = {}
        for symbol in symbol_list:
            flux = f'''
            from(bucket: "{influx_client.get_bucket()}")
                |> range(start: {start_iso}, stop: {end_iso})
                |> filter(fn: (r) => r["_measurement"] == "stock_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> filter(fn: (r) => r["_field"] == "close")
                |> sort(columns:["_time"])
            '''
            logger.info(f"Flux query: {flux}")
            tables = query_api.query(flux)
            dates = []
            closes = []
            for table in tables:
                for rec in table.records:
                    dates.append(rec.get_time())
                    closes.append(rec.get_value())
            logger.info(f"Found {len(dates)} records from InfluxDB for {symbol}.")
            if not dates or not closes:
                result[symbol] = []
                continue
            df = pd.DataFrame({"date": dates, "close": closes})
            df = df.sort_values("date")
            df = AnalysisService.calculate_bollinger_bands(df, window=ma_window, column='close', stddev=stddev)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            output_df = df.replace({np.nan: None})
            symbol_result = output_df[["date", "close", "ma", "bb_upper", "bb_lower"]].rename(columns={"bb_upper": "upper", "bb_lower": "lower"}).to_dict('records')
            result[symbol] = symbol_result
        return JsonResponse({"data": result})
    except Exception as e:
        logger.error(f"Error in api_bollinger_data: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

def arima_dashboard(request):
    """
    Renders the ARIMA dashboard page with form controls.
    The data is fetched asynchronously by the frontend.
    """
    all_symbols = get_all_symbols()
    selected_symbol = request.GET.get('symbol', 'API')

    # Provide default dates for the form
    model_start_date = request.GET.get('model_start', (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d'))
    model_end_date = request.GET.get('model_end', datetime.now().strftime('%Y-%m-%d'))
    forecast_end_date = request.GET.get('forecast_end', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))

    context = {
        'all_symbols': all_symbols,
        'selected_symbol': selected_symbol,
        'model_start_date': model_start_date,
        'model_end_date': model_end_date,
        'forecast_end_date': forecast_end_date,
    }
    return render(request, 'stocks/arima.html', context)

def api_arima_data(request):
    """
    API endpoint to fetch ARIMA/GARCH model data.
    """
    symbol = request.GET.get('symbol', 'API')
    model_start = request.GET.get('model_start')
    model_end = request.GET.get('model_end')
    forecast_end = request.GET.get('forecast_end')

    logger.info(f"ARIMA API called with: symbol={symbol}, model_start={model_start}, model_end={model_end}, forecast_end={forecast_end}")

    # Basic validation
    if not all([symbol, model_start, model_end, forecast_end]):
        logger.warning('ARIMA API: Missing required parameters.')
        return JsonResponse({'error': 'Missing required parameters.'}, status=400)

    try:
        data = generate_forecast_data(
            symbol=symbol,
            model_start_date=model_start,
            model_end_date=model_end,
            forecast_end_date=forecast_end
        )
        if isinstance(data, dict):
            logger.info(f"ARIMA API: Data keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, list):
                    logger.info(f"ARIMA API: {k} length: {len(v)}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error generating ARIMA data for {symbol}: {e}", exc_info=True)
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)
