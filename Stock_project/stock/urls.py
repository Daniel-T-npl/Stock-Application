from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_dashboard, name='stock_dashboard'),
    path('test-influxdb/', views.test_influxdb_view, name='test_influxdb'),
    path('dashboard/candlestick/', views.stock_dashboard, name='candlestick_dashboard'),
    path('dashboard/line/', views.line_dashboard, name='line_dashboard'),
    path('dashboard/rsi/', views.rsi_dashboard, name='rsi_dashboard'),
] 