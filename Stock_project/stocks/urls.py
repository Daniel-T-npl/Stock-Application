from django.urls import path
from . import views

app_name = 'stocks'

urlpatterns = [
    # API endpoints
    path("api/history/<str:ticker>/", views.price_history, name="price_history"),
    path("api/stock/update/", views.update_stock, name="update_stock"),
    path("api/stock/data/", views.get_stock_data, name="get_stock_data"),

    # Dashboard and pages
    path('', views.stock_dashboard, name='stock_dashboard'),
    path('test-influxdb/', views.test_influxdb_view, name='test_influxdb'),
    path('dashboard/candlestick/', views.stock_dashboard, name='candlestick_dashboard'),
    path('dashboard/line/', views.line_dashboard, name='line_dashboard'),
    path('dashboard/rsi/', views.rsi_dashboard, name='rsi_dashboard'),
    path('dashboard/bollinger/', views.bollinger_dashboard, name='bollinger_dashboard'),
    path('dashboard/arima/', views.arima_dashboard, name='arima_dashboard'),
    path('api/bollinger/', views.api_bollinger_data, name='api_bollinger_data'),
    path('api/arima/', views.api_arima_data, name='api_arima_data'),
    path('api/ohlcv_indicators/', views.get_ohlcv_and_indicators, name='ohlcv_indicators_api'),
    path('dashboard/', views.interactive_dashboard, name='interactive_dashboard'),
] 