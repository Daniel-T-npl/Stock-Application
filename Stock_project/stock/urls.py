from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_dashboard, name='stock_dashboard'),
    path('test-influxdb/', views.test_influxdb_view, name='test_influxdb'),
] 