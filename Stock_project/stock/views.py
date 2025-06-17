from django.shortcuts import render
from datetime import datetime, timedelta
import logging
from .services.influxdb_handler import InfluxDBHandler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stock_dashboard(request):
    """Interactive dashboard supporting multi-symbol, field, and date-range selection."""
    try:
        influx_handler = InfluxDBHandler()
        client = influx_handler.client

        # ------------------------------------------------------------
        # 1. Handle query-string parameters
        # ------------------------------------------------------------
        # Multi-select list may arrive as ?symbols=AAA&symbols=BBB *or* as comma-separated
        symbols_raw = request.GET.getlist("symbols") or []
        if len(symbols_raw) == 1 and "," in symbols_raw[0]:
            symbols_raw = [s.strip() for s in symbols_raw[0].split(",") if s.strip()]

        selected_symbols = symbols_raw

        # --- Field selection -------------------------------------------------
        fields_raw = request.GET.getlist("fields") or []
        if len(fields_raw) == 1 and "," in fields_raw[0]:
            fields_raw = [f.strip() for f in fields_raw[0].split(",") if f.strip()]

        selected_fields = fields_raw if fields_raw else ["close"]  # default

        # Date range: default earliest 2021-01-03 to today
        try:
            start_param = request.GET.get("start", "2021-01-03")
            end_param = request.GET.get("end", datetime.utcnow().strftime("%Y-%m-%d"))
            start_date = datetime.strptime(start_param, "%Y-%m-%d")
            end_date = datetime.strptime(end_param, "%Y-%m-%d")
        except ValueError:
            # Fallback to defaults on parse error
            start_date = datetime(2021, 1, 3)
            end_date = datetime.utcnow()

        logger.info(
            f"Dashboard params – symbols: {selected_symbols}, fields: {selected_fields}, range: {start_date} → {end_date}"
        )

        # ------------------------------------------------------------
        # 2. Discover all distinct symbols (for selector options)
        # ------------------------------------------------------------
        symbol_query = f"""
        from(bucket: \"stock_data\")
            |> range(start: 2021-01-03T00:00:00Z)
            |> filter(fn: (r) => r[\"_measurement\"] == \"stock_data\")
            |> keep(columns: [\"symbol\"])
            |> group()
            |> distinct(column: \"symbol\")
            |> sort(columns: [\"symbol\"])
        """
        symbol_result = client.query_api().query(symbol_query)
        all_symbols = []
        for table in symbol_result:
            for rec in table.records:
                value = rec.get_value() if hasattr(rec, 'get_value') else None
                if not value:
                    value = rec.values.get('symbol') if isinstance(rec.values, dict) else None
                if value:
                    all_symbols.append(value)

        # Default selected symbol list if none chosen
        if not selected_symbols:
            selected_symbols = all_symbols[:1]  # pick first symbol as default

        # ------------------------------------------------------------
        # 3. Fetch data for each selected symbol & field
        # ------------------------------------------------------------
        start_iso = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_iso = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")  # inclusive

        data_by_symbol = {}

        for sym in selected_symbols:
            # Build list of field names for Flux keep()
            keep_fields = ", ".join([f'\"{f}\"' for f in selected_fields])

            flux = f"""
            from(bucket: \"stock_data\")
                |> range(start: {start_iso}, stop: {end_iso})
                |> filter(fn: (r) => r[\"_measurement\"] == \"stock_data\")
                |> filter(fn: (r) => r[\"symbol\"] == \"{sym}\")
                |> pivot(rowKey:[\"_time\"], columnKey:[\"_field\"], valueColumn:\"_value\")
                |> keep(columns:[\"_time\", {keep_fields}])
                |> sort(columns:[\"_time\"])
            """

            tables = client.query_api().query(flux)
            pts_by_field = {fld: [] for fld in selected_fields}
            for table in tables:
                for rec in table.records:
                    for fld in selected_fields:
                        val = rec.values.get(fld)
                        if val is None:
                            continue
                        pts_by_field[fld].append({
                            "date": rec.get_time().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "value": val
                        })
            data_by_symbol[sym] = pts_by_field
            logger.info(f"Fetched data for {sym}: {[len(pts_by_field[f]) for f in selected_fields]}")

        # ------------------------------------------------------------
        # 4. Build context & render
        # ------------------------------------------------------------
        context = {
            "all_symbols": all_symbols,
            "selected_symbols": selected_symbols,
            "selected_fields": selected_fields,
            "data_json": json.dumps(data_by_symbol),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "fields": ["open","close","high","low","turnover","volume"],
        }

        return render(request, "stock/dashboard.html", context)
    except Exception as e:
        logger.error(f"Error in stock_dashboard: {str(e)}", exc_info=True)
        return render(request, 'stock/dashboard.html', {
            'error': f'Error: {str(e)}',
            'all_symbols': [],
            'selected_symbols': [],
            'selected_fields': ['close'],
            'data_json': '{}',
            'start_date': start_date.strftime('%Y-%m-%d') if 'start_date' in locals() else None,
            'end_date': end_date.strftime('%Y-%m-%d') if 'end_date' in locals() else None
        })
    finally:
        if "influx_handler" in locals():
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